"""data/scores.py — бесплатные счёты через Sofascore (без ключа)."""
from __future__ import annotations
import re
from datetime import datetime, timedelta
from typing import Optional

import requests

from security import cache_get, cache_put


NO_PROXY = {"http": None, "https": None, "all": None}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.sofascore.com/",
    })
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


_sess = _session()


def _norm(s: str) -> str:
    """Нормализация имени команды."""
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


def _log(msg: str):
    try:
        import sys
        print(f"[SOFA] {msg}", file=sys.stdout, flush=True)
    except Exception:
        pass


def _get_scheduled_events(date_iso: str) -> list:
    """Все матчи на дату через Sofascore."""
    ck = f"sofa_day_{date_iso}"
    cached = cache_get(ck, 3600)  # 1 час
    if cached is not None:
        return cached if isinstance(cached, list) else []

    url = f"https://api.sofascore.com/api/v1/sport/football/scheduled-events/{date_iso}"
    try:
        r = _sess.get(url, timeout=15, proxies=NO_PROXY)
        if r.status_code != 200:
            _log(f"HTTP {r.status_code} for {date_iso}")
            cache_put(ck, [])
            return []
        events = (r.json() or {}).get("events") or []
        out = []
        for e in events:
            try:
                home = (e.get("homeTeam") or {}).get("name") or ""
                away = (e.get("awayTeam") or {}).get("name") or ""
                if not home or not away:
                    continue
                status = (e.get("status") or {}).get("type") or ""
                hg = (e.get("homeScore") or {}).get("current")
                ag = (e.get("awayScore") or {}).get("current")
                out.append({
                    "id": e.get("id"),
                    "home": home,
                    "away": away,
                    "status": status,
                    "home_score": hg,
                    "away_score": ag,
                    "start_ts": e.get("startTimestamp"),
                })
            except Exception:
                continue
        _log(f"scheduled-events/{date_iso}: {len(out)} events")
        cache_put(ck, out)
        return out
    except Exception as e:
        _log(f"ERROR {date_iso}: {type(e).__name__}: {str(e)[:100]}")
        return []


def find_match_score(home: str, away: str, date_iso: str) -> Optional[dict]:
    """Ищет матч по названиям команд + дате. Возвращает счёт или None."""
    if not home or not away or not date_iso:
        return None

    hn = _norm(home)
    an = _norm(away)

    # Ищем за 3 дня вокруг даты (матч мог быть перенесён)
    for offset in (0, -1, 1):
        try:
            d = datetime.strptime(date_iso[:10], "%Y-%m-%d") + timedelta(days=offset)
            d_str = d.strftime("%Y-%m-%d")
        except Exception:
            continue

        events = _get_scheduled_events(d_str)
        for e in events:
            eh = _norm(e.get("home", ""))
            ea = _norm(e.get("away", ""))
            # Точное совпадение или частичное
            match = False
            if hn and an:
                if (hn == eh and an == ea) or (hn == ea and an == eh):
                    match = True
                elif (hn in eh or eh in hn) and (an in ea or ea in an):
                    match = True
            if not match:
                continue

            status = e.get("status", "")
            if status not in ("finished", "FT", "AET", "PEN"):
                continue

            hg = e.get("home_score")
            ag = e.get("away_score")
            if hg is None or ag is None:
                continue

            _log(f"MATCH: {home} vs {away} = {hg}:{ag}")
            return {
                "home": int(hg),
                "away": int(ag),
                "status": "FT",
                "source": "sofascore",
            }

    _log(f"NOT FOUND: {home} vs {away} @ {date_iso}")
    return None
