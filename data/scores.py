"""data/scores.py — автосчёты: football-data.org + fotmob + OpenLigaDB."""
from __future__ import annotations
import re
from datetime import datetime, timedelta
from typing import Optional

import requests

from security import cache_get, cache_put
from storage import usage


NO_PROXY = {"http": None, "https": None, "all": None}


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
        "Accept": "application/json, text/plain, */*",
    })
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


_sess = _session()


def _norm(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


# ============ ИСТОЧНИК 1: FOTMOB (все лиги) ============
def _fotmob_day(date_iso: str) -> list:
    """FotMob — все лиги, не блокируется Cloudflare."""
    ck = f"fotmob_{date_iso}"
    cached = cache_get(ck, 3600)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    date_compact = date_iso.replace("-", "")
    url = f"https://www.fotmob.com/api/matches?date={date_compact}"
    try:
        r = _sess.get(url, timeout=15, proxies=NO_PROXY,
                      headers={"Referer": "https://www.fotmob.com/"})
        if r.status_code != 200:
            _log(f"fotmob HTTP {r.status_code} {date_iso}")
            cache_put(ck, [])
            return []
        data = r.json() or {}
        leagues = data.get("leagues") or []
        out = []
        for lg in leagues:
            for m in (lg.get("matches") or []):
                try:
                    h = (m.get("home") or {}).get("name") or ""
                    a = (m.get("away") or {}).get("name") or ""
                    if not h or not a:
                        continue
                    status = (m.get("status") or {}).get("finished")
                    hs = (m.get("home") or {}).get("score")
                    ags = (m.get("away") or {}).get("score")
                    if not status or hs is None or ags is None:
                        continue
                    out.append({"home": h, "away": a,
                                "hs": hs, "as": ags})
                except Exception:
                    continue
        _log(f"fotmob {date_iso}: {len(out)} events")
        cache_put(ck, out)
        return out
    except Exception as e:
        _log(f"fotmob error {date_iso}: {type(e).__name__}")
        return []


def _find_fotmob(home: str, away: str, date_iso: str) -> Optional[dict]:
    hn, an = _norm(home), _norm(away)
    if not hn or not an:
        return None
    for off in (0, -1, 1):
        try:
            d = datetime.strptime(date_iso[:10], "%Y-%m-%d") + timedelta(days=off)
            d_str = d.strftime("%Y-%m-%d")
        except Exception:
            continue
        for e in _fotmob_day(d_str):
            eh, ea = _norm(e["home"]), _norm(e["away"])
            if not ((hn in eh or eh in hn) and (an in ea or ea in an)):
                continue
            hs, ags = e["hs"], e["as"]
            if hs is None or ags is None:
                continue
            try:
                return {"home": int(hs), "away": int(ags),
                        "status": "FT", "source": "fotmob"}
            except Exception:
                continue
    return None


# ============ ИСТОЧНИК 2: FOOTBALL-DATA.ORG (по дате) ============
def _fdorg_day(date_iso: str) -> list:
    """football-data.org — все матчи на дату."""
    token = ""
    try:
        import streamlit as st
        token = st.session_state.get("data", {}).get("meta", {}).get(
            "fdorg_token", "")
    except Exception:
        pass
    if not token:
        return []
    ck = f"fdorg_day_{date_iso}"
    cached = cache_get(ck, 3600)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    try:
        r = _sess.get(
            "https://api.football-data.org/v4/matches",
            headers={"X-Auth-Token": token},
            params={"dateFrom": date_iso, "dateTo": date_iso},
            timeout=15, proxies=NO_PROXY)
        if r.status_code != 200:
            _log(f"fdorg HTTP {r.status_code} {date_iso}")
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
        _log(f"fdorg {date_iso}: {len(out)} events")
        cache_put(ck, out)
        return out
    except Exception as e:
        _log(f"fdorg error {date_iso}: {type(e).__name__}")
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
        for e in _fdorg_day(d_str):
            eh, ea = _norm(e["home"]), _norm(e["away"])
            if not ((hn in eh or eh in hn) and (an in ea or ea in an)):
                continue
            try:
                return {"home": int(e["hs"]), "away": int(e["as"]),
                        "status": "FT", "source": "fdorg"}
            except Exception:
                continue
    return None


# ============ ИСТОЧНИК 3: OPENLIGADB (Бундеслига) ============
def _olb_day(date_iso: str) -> list:
    ck = f"olb_day_{date_iso}"
    cached = cache_get(ck, 3600)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    try:
        r = _sess.get("https://api.openligadb.de/getmatchdata/bl1/2026",
                      timeout=15, proxies=NO_PROXY)
        if r.status_code != 200:
            cache_put(ck, [])
            return []
        matches = r.json() or []
        out = []
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
                out.append({"home": h, "away": a, "hs": hs, "as": ags})
            except Exception:
                continue
        cache_put(ck, out)
        return out
    except Exception:
        return []


def _find_olb(home: str, away: str, date_iso: str) -> Optional[dict]:
    hn, an = _norm(home), _norm(away)
    for e in _olb_day(date_iso):
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
    """Ищет результат: fotmob → fdorg → openligadb."""
    if not home or not away or not date_iso:
        return None

    # 1. FotMob — все лиги
    try:
        r = _find_fotmob(home, away, date_iso)
        if r:
            _log(f"FOUND fotmob: {home} vs {away} = {r['home']}:{r['away']}")
            return r
    except Exception as e:
        _log(f"fotmob fail: {e}")

    # 2. football-data.org — топ-лиги
    try:
        r = _find_fdorg(home, away, date_iso)
        if r:
            _log(f"FOUND fdorg: {home} vs {away} = {r['home']}:{r['away']}")
            return r
    except Exception as e:
        _log(f"fdorg fail: {e}")

    # 3. OpenLigaDB — Бундеслига
    try:
        r = _find_olb(home, away, date_iso)
        if r:
            _log(f"FOUND olb: {home} vs {away} = {r['home']}:{r['away']}")
            return r
    except Exception as e:
        _log(f"olb fail: {e}")

    _log(f"NOT FOUND: {home} vs {away} @ {date_iso}")
    return None
