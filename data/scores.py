"""data/scores.py — ESPN + football-data.org + OpenLigaDB (v3).

Что изменено:
  * _same_team: мягкий матчинг (Brooklyn FC ↔ Brooklyn, Rhode Island FC ↔ Rhode Island)
  * _find_espn: логирование near-miss (для диагностики NOT FOUND)
  * Сохранены все защиты v2: токены, диакритика, алиасы, женские/юношеские
  * football-data.org: 1 запрос /v4/matches на дату
  * AET: отдельный статус (не путается со FT)
"""
from __future__ import annotations

import logging
import re
import time as _time
import unicodedata
from datetime import datetime, timedelta
from typing import Iterator, Optional

import requests

from security import cache_get, cache_put

log = logging.getLogger("scores")

NO_PROXY = {"http": None, "https": None, "all": None}
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"
FDORG_URL = "https://api.football-data.org/v4/matches"

FDORG_COMPETITIONS = ["PL", "ELC", "PD", "SA", "BL1", "FL1", "DED", "PPL", "CL"]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


_sess = _session()


def _get_token() -> str:
    try:
        import streamlit as st
        return (st.session_state.get("data", {}).get("meta", {})
                .get("fdorg_token", "") or "").strip()
    except Exception:
        return ""


# ============ СОПОСТАВЛЕНИЕ КОМАНД ============
_TR = str.maketrans({"ß": "ss", "ø": "o", "æ": "ae", "œ": "oe", "ł": "l",
                     "đ": "d", "ð": "d", "þ": "th", "ı": "i"})
_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "cd", "ud", "rcd",
         "sv", "fk", "bv", "1"}
_FLAGS = {"women", "w", "wfc", "fem", "feminin", "femenino", "femeni", "frauen",
          "u17", "u18", "u19", "u20", "u21", "u23", "ii", "b", "reserves",
          "res", "youth", "castilla"}
_ALIASES = {
    "man utd": "manchester united", "man united": "manchester united",
    "manchester utd": "manchester united", "man city": "manchester city",
    "spurs": "tottenham", "tottenham hotspur": "tottenham",
    "wolves": "wolverhampton wanderers", "internazionale": "inter",
    "inter milan": "inter", "paris saint germain": "psg", "paris sg": "psg",
    "bayern munich": "bayern", "bayern munchen": "bayern",
    "nott m forest": "nottingham forest",
    # USL (для тех, что не матчились)
    "brooklyn": "brooklyn fc",
    "rhode island": "rhode island fc",
    "detroit city": "detroit city fc",
    "birmingham legion": "birmingham legion fc",
}


def _tokens(name: str) -> tuple[frozenset, frozenset]:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.translate(_TR).replace("&", " and ")
    words = re.findall(r"[a-zа-яё0-9]+", s)
    flags = frozenset(w for w in words if w in _FLAGS)
    core = [w for w in words if w not in _STOP and w not in _FLAGS]
    phrase = " ".join(core)
    phrase = _ALIASES.get(phrase, phrase)
    return frozenset(phrase.split()), flags


def _same_team(a: str, b: str) -> bool:
    """Мягкий матчинг: ESPN часто даёт короткие названия USL-команд."""
    ca, fa = _tokens(a)
    cb, fb = _tokens(b)
    if not ca or not cb or fa != fb:
        return False
    if ca == cb:
        return True

    # Одно подмножество другого
    small, big = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    if small <= big:
        return True

    # 60% общих слов + хотя бы одно длинное (>=4)
    common = small & big
    if len(common) / max(1, len(small)) >= 0.6:
        if any(len(t) >= 4 for t in common):
            return True
    return False


def _parse_day(date_iso: str) -> Optional[datetime]:
    try:
        return datetime.strptime((date_iso or "")[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _day_offsets(date_iso: str) -> Iterator[str]:
    d = _parse_day(date_iso)
    if d is None:
        return
    for off in (0, -1, 1):
        yield (d + timedelta(days=off)).strftime("%Y-%m-%d")


def _ttl(date_iso: str) -> int:
    d = _parse_day(date_iso)
    if d is None or (datetime.now() - d).days <= 2:
        return 600
    return 7 * 86400


# ============ ESPN ============
def _espn_day(date_iso: str) -> list:
    ck = f"espn_day_v4_{date_iso}"
    cached = cache_get(ck, _ttl(date_iso))
    if isinstance(cached, list):
        return cached
    try:
        r = _sess.get(ESPN_URL,
                      params={"dates": date_iso.replace("-", ""), "limit": 500},
                      timeout=15, proxies=NO_PROXY)
    except requests.RequestException as e:
        log.warning("espn error %s: %s", date_iso, type(e).__name__)
        return []
    if r.status_code != 200:
        log.warning("espn HTTP %s %s", r.status_code, date_iso)
        return []
    out = []
    for e in (r.json() or {}).get("events") or []:
        try:
            comp = (e.get("competitions") or [{}])[0]
            stype = (comp.get("status") or {}).get("type") or {}
            if not stype.get("completed"):
                continue
            home = away = None
            for t in comp.get("competitors") or []:
                name = (t.get("team") or {}).get("displayName") or ""
                if t.get("homeAway") == "home":
                    home = (name, t.get("score"))
                elif t.get("homeAway") == "away":
                    away = (name, t.get("score"))
            if not home or not away or home[1] is None or away[1] is None:
                continue
            sname = str(stype.get("name") or "").upper()
            extra = any(x in sname for x in ("AET", "PEN", "EXTRA"))
            out.append({"home": home[0], "away": away[0],
                        "hs": int(home[1]), "as": int(away[1]),
                        "status": "AET" if extra else "FT"})
        except Exception:
            continue
    log.info("espn %s: %d events", date_iso, len(out))
    cache_put(ck, out)
    return out


def _find_espn(home: str, away: str, date_iso: str) -> Optional[dict]:
    """ESPN. Near-miss логирование для диагностики NOT FOUND."""
    for d_str in _day_offsets(date_iso):
        events = _espn_day(d_str)
        for e in events:
            if _same_team(home, e["home"]) and _same_team(away, e["away"]):
                return {"home": e["hs"], "away": e["as"],
                        "status": e["status"], "source": "espn"}
        # Диагностика — только когда дата = сегодня (не спамим на -1/+1)
        if d_str != date_iso:
            continue
        try:
            ht, _ = _tokens(home)
            at, _ = _tokens(away)
            near = []
            for e in events:
                eh_t, _ = _tokens(e["home"])
                ea_t, _ = _tokens(e["away"])
                if (ht & eh_t) or (at & ea_t):
                    near.append(f"{e['home']} vs {e['away']}")
            if near:
                log.info("NEAR MISS: want '%s vs %s' | ESPN has: %s",
                         home, away, "; ".join(near[:5]))
        except Exception:
            pass
    return None


# ============ FOOTBALL-DATA.ORG ============
_LAST_FDORG_CALL = [0.0]
_FDORG_COOLDOWN_UNTIL = [0.0]


def _fdorg_rate_limit() -> None:
    delta = _time.time() - _LAST_FDORG_CALL[0]
    if delta < 6.5:
        _time.sleep(6.5 - delta)
    _LAST_FDORG_CALL[0] = _time.time()


def _fdorg_range(date_iso: str) -> list:
    token = _get_token()
    d = _parse_day(date_iso)
    if not token or d is None or _time.time() < _FDORG_COOLDOWN_UNTIL[0]:
        return []
    ck = f"fdorg_rng_v5_{d:%Y-%m-%d}"
    cached = cache_get(ck, _ttl(date_iso))
    if isinstance(cached, list):
        return cached
    _fdorg_rate_limit()
    try:
        r = _sess.get(
            FDORG_URL, headers={"X-Auth-Token": token},
            params={"dateFrom": (d - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "dateTo": (d + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "competitions": ",".join(FDORG_COMPETITIONS)},
            timeout=15, proxies=NO_PROXY)
    except requests.RequestException as e:
        log.warning("fdorg error: %s", type(e).__name__)
        return []
    if r.status_code == 429:
        _FDORG_COOLDOWN_UNTIL[0] = _time.time() + 120
        log.warning("fdorg 429 — пауза 120с")
        return []
    if r.status_code != 200:
        log.warning("fdorg HTTP %s", r.status_code)
        return []
    out = []
    for m in (r.json() or {}).get("matches") or []:
        try:
            if m.get("status") != "FINISHED":
                continue
            h = (m.get("homeTeam") or {}).get("name") or ""
            a = (m.get("awayTeam") or {}).get("name") or ""
            score = m.get("score") or {}
            tag = "FT"
            full = score.get("fullTime") or {}
            if (score.get("duration") or "REGULAR") != "REGULAR":
                reg = score.get("regularTime") or {}
                if reg.get("home") is not None and reg.get("away") is not None:
                    full = reg
                else:
                    tag = "AET"
            if not h or not a or full.get("home") is None or full.get("away") is None:
                continue
            out.append({"home": h, "away": a, "hs": int(full["home"]),
                        "as": int(full["away"]), "status": tag,
                        "date": (m.get("utcDate") or "")[:10]})
        except Exception:
            continue
    log.info("fdorg %s: %d events", date_iso, len(out))
    cache_put(ck, out)
    return out


def _closest(cands: list, date_iso: str) -> Optional[dict]:
    d = _parse_day(date_iso)
    best, best_gap = None, None
    for e in cands:
        ed = _parse_day(e.get("date", ""))
        gap = abs((ed - d).days) if (ed and d) else 99
        if gap <= 1 and (best_gap is None or gap < best_gap):
            best, best_gap = e, gap
    return best


def _find_fdorg(home: str, away: str, date_iso: str) -> Optional[dict]:
    cands = [e for e in _fdorg_range(date_iso)
             if _same_team(home, e["home"]) and _same_team(away, e["away"])]
    e = _closest(cands, date_iso)
    if not e:
        return None
    return {"home": e["hs"], "away": e["as"],
            "status": e["status"], "source": "fdorg"}


# ============ OPENLIGADB ============
def _olb_season(date_iso: str) -> Optional[int]:
    d = _parse_day(date_iso)
    if d is None:
        return None
    return d.year if d.month >= 7 else d.year - 1


def _olb_league(league: str, season: int) -> list:
    ck = f"olb_{league}_{season}_v5"
    cached = cache_get(ck, 1800)
    if isinstance(cached, list):
        return cached
    try:
        r = _sess.get(f"https://api.openligadb.de/getmatchdata/{league}/{season}",
                      timeout=20, proxies=NO_PROXY)
    except requests.RequestException as e:
        log.warning("olb %s error: %s", league, type(e).__name__)
        return []
    if r.status_code != 200:
        log.warning("olb %s HTTP %s", league, r.status_code)
        return []
    out = []
    for m in r.json() or []:
        try:
            if not m.get("matchIsFinished"):
                continue
            h = (m.get("team1") or {}).get("teamName") or ""
            a = (m.get("team2") or {}).get("teamName") or ""
            hs = ags = None
            for res in m.get("matchResults") or []:
                if res.get("resultTypeID") == 2:
                    hs, ags = res.get("pointsTeam1"), res.get("pointsTeam2")
                    break
            if not h or not a or hs is None or ags is None:
                continue
            out.append({"home": h, "away": a, "hs": int(hs), "as": int(ags),
                        "status": "FT",
                        "date": (m.get("matchDateTime") or "")[:10]})
        except Exception:
            continue
    log.info("olb %s/%s: %d finished", league, season, len(out))
    cache_put(ck, out)
    return out


def _find_olb(home: str, away: str, date_iso: str) -> Optional[dict]:
    season = _olb_season(date_iso)
    if season is None:
        return None
    for league in ("bl1", "bl2", "bl3"):
        cands = [e for e in _olb_league(league, season)
                 if _same_team(home, e["home"]) and _same_team(away, e["away"])]
        e = _closest(cands, date_iso)
        if e:
            return {"home": e["hs"], "away": e["as"],
                    "status": e["status"], "source": "openligadb"}
    return None


# ============ ПУБЛИЧНАЯ ФУНКЦИЯ ============
def find_match_score(home: str, away: str, date_iso: str) -> Optional[dict]:
    """ESPN → football-data.org → OpenLigaDB.
    status: FT | AET (доп. время — не закрывается автоматически)."""
    if not home or not away or not date_iso:
        return None
    for name, fn in (("espn", _find_espn),
                     ("fdorg", _find_fdorg),
                     ("olb", _find_olb)):
        try:
            r = fn(home, away, date_iso)
        except Exception as e:
            log.warning("%s fail: %s", name, e)
            continue
        if r:
            log.info("FOUND %s: %s vs %s = %s:%s (%s)",
                     r["source"], home, away,
                     r["home"], r["away"], r["status"])
            return r
    log.info("NOT FOUND: %s vs %s @ %s", home, away, date_iso)
    return None
