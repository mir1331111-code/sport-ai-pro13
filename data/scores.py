"""data/scores.py — ESPN (все лиги) + football-data.org + OpenLigaDB."""
from __future__ import annotations
import re
import time as _time
from datetime import datetime, timedelta
from typing import Optional

import requests

from security import cache_get, cache_put


NO_PROXY = {"http": None, "https": None, "all": None}

ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"

FDORG_COMPETITIONS = [
    "PL", "ELC",
    "PD", "SD",
    "SA", "SB",
    "BL1", "BL2",
    "FL1", "FL2",
    "DED", "PPL",
    "CL",
]


def _log(msg: str):
    try:
        import sys
        print(f"[SCORES] {msg}", file=sys.stdout, flush=True)
    except Exception:
        pass


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


def _norm(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


def _get_token() -> str:
    try:
        import streamlit as st
        return st.session_state.get("data", {}).get("meta", {}).get(
            "fdorg_token", "").strip()
    except Exception:
        return ""


# ============ ESPN (все лиги + сборные) ============
def _espn_day(date_iso: str) -> list:
    ck = f"espn_day_v2_{date_iso}"    # ← v2 — новый кэш
    cached = cache_get(ck, 3600)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    date_compact = date_iso.replace("-", "")
    try:
        r = _sess.get(ESPN_URL, params={"dates": date_compact},
                      timeout=15, proxies=NO_PROXY)
        if r.status_code != 200:
            _log(f"espn HTTP {r.status_code} {date_iso}")
            cache_put(ck, [])
            return []
        data = r.json() or {}
        events = data.get("events") or []
        out = []
        for e in events:
            try:
                comp = (e.get("competitions") or [{}])[0]
                status = (comp.get("status") or {}).get("type") or {}
                if not status.get("completed"):
                    continue
                teams = comp.get("competitors") or []
                if len(teams) < 2:
                    continue
                home = None
                away = None
                for t in teams:
                    name = (t.get("team") or {}).get("displayName") or ""
                    score = t.get("score")
                    if t.get("homeAway") == "home":
                        home = (name, score)
                    elif t.get("homeAway") == "away":
                        away = (name, score)
                if not home or not away:
                    continue
                hs = home[1]
                ags = away[1]
                if hs is None or ags is None:
                    continue
                out.append({
                    "home": home[0],
                    "away": away[0],
                    "hs": int(hs),
                    "as": int(ags),
                    "league": (e.get("season") or {}).get("slug") or "",
                })
            except Exception:
                continue
        _log(f"espn {date_iso}: {len(out)} events")
        cache_put(ck, out)
        return out
    except Exception as e:
        _log(f"espn error {date_iso}: {type(e).__name__}")
        return []


def _find_espn(home: str, away: str, date_iso: str) -> Optional[dict]:
    hn, an = _norm(home), _norm(away)
    if not hn or not an:
        return None
    for off in (0, -1, 1):
        try:
            d = datetime.strptime(date_iso[:10], "%Y-%m-%d") + timedelta(days=off)
            d_str = d.strftime("%Y-%m-%d")
        except Exception:
            continue
        for e in _espn_day(d_str):
            eh, ea = _norm(e["home"]), _norm(e["away"])
            if (hn in eh or eh in hn) and (an in ea or ea in an):
                try:
                    return {"home": int(e["hs"]), "away": int(e["as"]),
                            "status": "FT", "source": "espn"}
                except Exception:
                    continue
    return None


# ============ FOOTBALL-DATA.ORG ============
_LAST_FDORG_CALL = [0.0]


def _fdorg_rate_limit():
    now = _time.time()
    delta = now - _LAST_FDORG_CALL[0]
    if delta < 6.5:
        _time.sleep(6.5 - delta)
    _LAST_FDORG_CALL[0] = _time.time()


def _fdorg_league_matches(comp: str, date_iso: str) -> list:
    token = _get_token()
    if not token:
        return []
    ck = f"fdorg_lg_v3_{comp}_{date_iso}"
    cached = cache_get(ck, 7200)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    _fdorg_rate_limit()

    url = f"https://api.football-data.org/v4/competitions/{comp}/matches"
    try:
        r = _sess.get(
            url,
            headers={"X-Auth-Token": token},
            params={"dateFrom": date_iso, "dateTo": date_iso},
            timeout=15, proxies=NO_PROXY)
        if r.status_code == 429:
            _log(f"fdorg {comp} {date_iso}: 429 — пауза 60с")
            _time.sleep(60)
            r = _sess.get(
                url,
                headers={"X-Auth-Token": token},
                params={"dateFrom": date_iso, "dateTo": date_iso},
                timeout=15, proxies=NO_PROXY)
        if r.status_code != 200:
            _log(f"fdorg {comp} {date_iso}: HTTP {r.status_code}")
            cache_put(ck, [])
            return []
        matches = (r.json() or {}).get("matches") or []
        out = []
        for m in matches:
            try:
                if m.get("status") != "FINISHED":
                    continue
                h = (m.get("homeTeam") or {}).get("name") or ""
                a = (m.get("awayTeam") or {}).get("name") or ""
                if not h or not a:
                    continue
                score = m.get("score") or {}
                full = score.get("fullTime") or {}
                hs = full.get("home")
                ags = full.get("away")
                if hs is None or ags is None:
                    continue
                out.append({"home": h, "away": a, "hs": hs, "as": ags})
            except Exception:
                continue
        _log(f"fdorg {comp} {date_iso}: {len(out)} events")
        cache_put(ck, out)
        return out
    except Exception as e:
        _log(f"fdorg {comp} error: {type(e).__name__}")
        return []


def _find_fdorg(home: str, away: str, date_iso: str) -> Optional[dict]:
    hn, an = _norm(home), _norm(away)
    if not hn or not an:
        return None
    for off in (0, -1, 1):
        try:
            d = datetime.strptime(date_iso[:10], "%Y-%m-%d") + timedelta(days=off)
            d_str = d.strftime("%Y-%m-%d")
        except Exception:
            continue
        for comp in FDORG_COMPETITIONS:
            for e in _fdorg_league_matches(comp, d_str):
                eh, ea = _norm(e["home"]), _norm(e["away"])
                if not ((hn in eh or eh in hn) and (an in ea or ea in an)):
                    continue
                try:
                    return {"home": int(e["hs"]), "away": int(e["as"]),
                            "status": "FT", "source": f"fdorg_{comp}"}
                except Exception:
                    continue
    return None


# ============ OPENLIGADB ============
def _olb_day(date_iso: str) -> list:
    ck = f"olb_day_v3_{date_iso}"
    cached = cache_get(ck, 3600)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    out = []
    for league in ("bl1", "bl2", "bl3"):
        try:
            r = _sess.get(
                f"https://api.openligadb.de/getmatchdata/{league}/2026",
                timeout=15, proxies=NO_PROXY)
            if r.status_code != 200:
                continue
            matches = r.json() or []
            for m in matches:
                try:
                    dt_str = m.get("matchDateTime") or ""
                    if not dt_str.startswith(date_iso[:10]):
                        continue
                    h = (m.get("team1") or {}).get("teamName") or ""
                    a = (m.get("team2") or {}).get("teamName") or ""
                    if not h or not a:
                        continue
                    finished = m.get("matchIsFinished")
                    results = m.get("matchResults") or []
                    hs, ags = None, None
                    for res in results:
                        if res.get("resultTypeID") == 2:
                            hs = res.get("pointsTeam1")
                            ags = res.get("pointsTeam2")
                            break
                    if not finished or hs is None or ags is None:
                        continue
                    out.append({"home": h, "away": a,
                                "hs": hs, "as": ags, "league": league})
                except Exception:
                    continue
        except Exception:
            continue

    _log(f"olb {date_iso}: {len(out)} events (bl1+bl2+bl3)")
    cache_put(ck, out)
    return out


def _find_olb(home: str, away: str, date_iso: str) -> Optional[dict]:
    hn, an = _norm(home), _norm(away)
    for off in (0, -1, 1):
        try:
            d = datetime.strptime(date_iso[:10], "%Y-%m-%d") + timedelta(days=off)
            d_str = d.strftime("%Y-%m-%d")
        except Exception:
            continue
        for e in _olb_day(d_str):
            eh, ea = _norm(e["home"]), _norm(e["away"])
            if (hn in eh or eh in hn) and (an in ea or ea in an):
                try:
                    return {"home": int(e["hs"]), "away": int(e["as"]),
                            "status": "FT", "source": "openligadb"}
                except Exception:
                    continue
    return None


# ============ ПУБЛИЧНАЯ ФУНКЦИЯ ============
def find_match_score(home: str, away: str, date_iso: str) -> Optional[dict]:
    """ESPN → football-data.org → OpenLigaDB."""
    if not home or not away or not date_iso:
        return None

    try:
        r = _find_espn(home, away, date_iso)
        if r:
            _log(f"FOUND espn: {home} vs {away} = {r['home']}:{r['away']}")
            return r
    except Exception as e:
        _log(f"espn fail: {e}")

    try:
        r = _find_fdorg(home, away, date_iso)
        if r:
            _log(f"FOUND {r['source']}: {home} vs {away} = {r['home']}:{r['away']}")
            return r
    except Exception as e:
        _log(f"fdorg fail: {e}")

    try:
        r = _find_olb(home, away, date_iso)
        if r:
            _log(f"FOUND olb: {home} vs {away} = {r['home']}:{r['away']}")
            return r
    except Exception as e:
        _log(f"olb fail: {e}")

    _log(f"NOT FOUND: {home} vs {away} @ {date_iso}")
    return None
