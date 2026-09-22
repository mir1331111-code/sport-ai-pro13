"""data/sources.py — football-data.org + история + Odds API + logos."""
from __future__ import annotations
import csv, io, re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
    _HAS_RETRY = True
except Exception:
    _HAS_RETRY = False

from config import (CACHE_TTL, FOOTBALL_DATA_ORG_HOST,
                    DIV_TO_FDORG, FDORG_TO_DIV)
from security import cache_get, cache_put
from storage import usage


MSK_OFFSET_HOURS = 3


def _msk_date(utc_iso: str) -> str:
    try:
        dt = datetime.strptime(utc_iso[:19], "%Y-%m-%dT%H:%M:%S")
        msk = dt + timedelta(hours=MSK_OFFSET_HOURS)
        return msk.strftime("%Y-%m-%d")
    except Exception:
        return utc_iso[:10] if utc_iso else ""


def _msk_time(utc_iso: str) -> str:
    try:
        dt = datetime.strptime(utc_iso[:19], "%Y-%m-%dT%H:%M:%S")
        msk = dt + timedelta(hours=MSK_OFFSET_HOURS)
        return msk.strftime("%H:%M")
    except Exception:
        return utc_iso[11:16] if len(utc_iso) >= 16 else ""


def _session() -> requests.Session:
    s = requests.Session()
    if _HAS_RETRY:
        r = Retry(total=3, connect=3, read=3, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET", "HEAD"]),
                  raise_on_status=False)
        ad = HTTPAdapter(max_retries=r, pool_connections=20, pool_maxsize=20)
        s.mount("https://", ad)
        s.mount("http://", ad)
    s.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0"})
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


_sess = _session()
NO_PROXY = {"http": None, "https": None, "all": None}


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def parse_date(s) -> Optional[datetime]:
    if s is None:
        return None
    src = str(s).strip()
    if not src:
        return None
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d",
                "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            sc = src[:19] if "%z" in fmt or "T" in fmt else src
            return datetime.strptime(sc, fmt)
        except Exception:
            continue
    return None


def season_str(year: int) -> str:
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


def load_seasonal(div: str, season: str) -> list:
    ck = f"fd_{div}_{season}"
    cached = cache_get(ck, CACHE_TTL["seasonal"])
    if cached is not None:
        return cached if isinstance(cached, list) else []
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
    try:
        r = _sess.get(url, timeout=25, proxies=NO_PROXY)
        if r.status_code != 200 or not r.content:
            return []
        text = r.content.decode("utf-8", errors="ignore").lstrip("\ufeff")
        out = []
        for row in csv.DictReader(io.StringIO(text)):
            if not isinstance(row, dict):
                continue
            if not row.get("HomeTeam") or not row.get("AwayTeam"):
                continue
            if row.get("FTHG") in (None, "") or row.get("FTAG") in (None, ""):
                continue
            out.append(row)
        cache_put(ck, out)
        return out
    except Exception:
        return []


def _fdorg_get(endpoint: str, params: dict, token: str) -> Optional[dict]:
    if not token:
        return None
    if usage.fdorg_remaining() <= 0:
        return None
    url = f"https://{FOOTBALL_DATA_ORG_HOST}/v4/{endpoint}"
    headers = {"X-Auth-Token": token}
    try:
        r = _sess.get(url, headers=headers, params=params,
                      timeout=15, proxies=NO_PROXY)
        usage.fdorg_increment(1)
        if r.status_code == 429:
            import time
            time.sleep(60)
            return None
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def fdorg_matches(days: int, token: str, logs=None) -> list:
    if not token:
        return []
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    d_from = today.strftime("%Y-%m-%d")
    d_to = (today + timedelta(days=days)).strftime("%Y-%m-%d")

    out = []
    for comp in list(DIV_TO_FDORG.values()):
        if usage.fdorg_remaining() <= 0:
            if logs:
                logs.append("fdorg: soft-лимит исчерпан")
            break
        div_code = FDORG_TO_DIV.get(comp, "G")
        ck = f"fdorg_v7_{comp}_{d_from}_{d_to}"    # ← v7 — новый кэш
        cached = cache_get(ck, 1800)
        if cached is not None:
            if isinstance(cached, list):
                out += cached
                if logs:
                    logs.append(f"OK {comp}: {len(cached)} (cached)")
            continue
        data = _fdorg_get("matches",
                          {"dateFrom": d_from, "dateTo": d_to,
                           "competitions": comp},
                          token)
        if not data or not data.get("matches"):
            if logs:
                logs.append(f"-- {comp}: 0")
            cache_put(ck, [])
            continue
        rows = []
        for m in data["matches"]:
            try:
                home = (m.get("homeTeam") or {}).get("name")
                away = (m.get("awayTeam") or {}).get("name")
                if not home or not away:
                    continue
                utc_date = m.get("utcDate") or ""
                rows.append({
                    "Div": div_code,
                    "League": (m.get("competition") or {}).get("name") or comp,
                    "competition": comp,
                    "Date": _msk_date(utc_date),
                    "Time": _msk_time(utc_date),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "fixture_id": m.get("id"),
                    "home_badge": (m.get("homeTeam") or {}).get("crest") or "",
                    "away_badge": (m.get("awayTeam") or {}).get("crest") or "",
                    "league_badge": (m.get("competition") or {}).get("emblem") or "",
                    "status": m.get("status", ""),
                    "referee": ((m.get("referees") or [{}])[0].get("name", "")
                                if m.get("referees") else ""),
                })
            except Exception:
                continue
        out += rows
        cache_put(ck, rows)
        if logs:
            logs.append(f"OK {comp}: {len(rows)}")
    return out


def fdorg_match_result(match_id, token: str) -> Optional[dict]:
    if not match_id or not token:
        return None
    ck = f"fdorg_result_v7_{match_id}"
    cached = cache_get(ck, 86400)
    if cached is not None:
        return cached or None
    data = _fdorg_get(f"matches/{match_id}", {}, token)
    if not data or not data.get("match"):
        cache_put(ck, None)
        return None
    m = data["match"]
    if m.get("status") != "FINISHED":
        cache_put(ck, None)
        return None
    score = m.get("score") or {}
    full = score.get("fullTime") or {}
    hg = full.get("home")
    ag = full.get("away")
    if hg is None or ag is None:
        cache_put(ck, None)
        return None
    res = {"home": int(hg), "away": int(ag), "status": "FT"}
    cache_put(ck, res)
    return res


def odds_api_fixture(sport_key: str, home: str, away: str,
                     api_key: str) -> Optional[dict]:
    if not api_key or not sport_key:
        return None
    ck = f"odds_{sport_key}_{_norm_name(home)}_{_norm_name(away)}"
    cached = cache_get(ck, CACHE_TTL["odds"])
    if cached is not None:
        return cached or None
    if usage.odds_remaining() <= 0:
        return None
    try:
        r = _sess.get(
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
            params={"apiKey": api_key, "regions": "eu",
                    "markets": "h2h,totals,btts", "oddsFormat": "decimal"},
            timeout=20, proxies=NO_PROXY)
        usage.odds_increment(1)
        if r.status_code != 200:
            return None
        events = r.json() or []
        hn, an = _norm_name(home), _norm_name(away)
        target = None
        for ev in events:
            eh = _norm_name(ev.get("home_team", ""))
            ea = _norm_name(ev.get("away_team", ""))
            if eh == hn and ea == an:
                target = ev
                break
            if (eh in hn or hn in eh) and (ea in an or an in ea):
                target = ev
                break
        if not target:
            return None
        acc = defaultdict(list)
        for bmk in target.get("bookmakers") or []:
            for mkt in bmk.get("markets") or []:
                name = mkt.get("key")
                for out in mkt.get("outcomes") or []:
                    on = out.get("name") or ""
                    odd = _f(out.get("price"))
                    if odd is None or odd <= 1.0:
                        continue
                    if name == "h2h":
                        if on == target.get("home_team"):
                            acc["П1"].append(odd)
                        elif on == target.get("away_team"):
                            acc["П2"].append(odd)
                        elif on == "Draw":
                            acc["X"].append(odd)
                    elif name == "totals":
                        pt = _f(out.get("point"))
                        if pt is not None and abs(pt - 2.5) < 0.01:
                            if on == "Over":
                                acc["ТБ 2.5"].append(odd)
                            elif on == "Under":
                                acc["ТМ 2.5"].append(odd)
                    elif name == "btts":
                        if on == "Yes":
                            acc["BTTS да"].append(odd)
                        elif on == "No":
                            acc["BTTS нет"].append(odd)
        out = {k: sum(v) / len(v) for k, v in acc.items() if v}
        if not (("П1" in out and "X" in out and "П2" in out) or
                ("ТБ 2.5" in out and "ТМ 2.5" in out) or
                ("BTTS да" in out and "BTTS нет" in out)):
            out = {}
        cache_put(ck, out or None)
        return out or None
    except Exception:
        return None


def team_logo_url(team_name: str) -> Optional[str]:
    return None


def league_logo_url(div_code: str) -> Optional[str]:
    return None
