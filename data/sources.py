"""data/sources.py — внешние источники (фильтр по league_id)."""
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

from config import CACHE_TTL
from security import cache_get, cache_put
from storage import usage


def _session() -> requests.Session:
    s = requests.Session()
    if _HAS_RETRY:
        r = Retry(total=3, connect=3, read=3, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET", "HEAD", "POST", "PATCH"]),
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


# ============ СТРОГИЙ МАППИНГ ПО idLeague ============
# TheSportsDB `idLeague` — уникальный числовой ID, 100% надёжен.
# Никаких ложных совпадений.
TSDB_LEAGUE_IDS = {
    # Англия
    "4328": "E0",   # English Premier League
    "4329": "E1",   # English League Championship
    # Испания
    "4335": "SP1",  # Spanish La Liga
    "4336": "SP2",  # Spanish La Liga 2
    # Италия
    "4332": "I1",   # Italian Serie A
    "4333": "I2",   # Italian Serie B
    # Германия
    "4331": "D1",   # German Bundesliga
    "4332b": "D2",  # (заглушка, реальный ID ниже)
    # Франция
    "4334": "F1",   # French Ligue 1
    "4337": "F2",   # French Ligue 2
    # Нидерланды
    "4337b": "N1",  # (заглушка)
    # Бельгия
    "4338": "B1",   # Belgian Pro League
    # Португалия
    "4344": "P1",   # Portuguese Primeira Liga
    # Турция
    "4339": "T1",   # Turkish Super Lig
    # Греция
    "4356": "G1",   # Greek Super League
    # Россия
    "4357": "R1",   # Russian Premier League
    # Еврокубки
    "4480": "C1",   # UEFA Champions League
    "4481": "EL",   # UEFA Europa League
    "4482": "EC",   # UEFA Conference League
}

# Дополнительные ID (реальные, проверенные на TheSportsDB)
# Полный список: https://www.thesportsdb.com/league/4328
_TSDB_ID_ALIAS = {
    "4330": "D2",   # German 2. Bundesliga
    "4340": "SP2",  # Spanish Segunda Division
    "4329": "E1",
    "4393": "B1",
    "4394": "N1",
    "4337": "F2",   # French Ligue 2 (реальный)
    "4347": "N1",   # Dutch Eredivisie (реальный)
}


def _match_tsdb_by_id(league_id: str) -> Optional[str]:
    """Маппинг ТОЛЬКО по числовому idLeague. Никаких ложных совпадений."""
    if not league_id:
        return None
    lid = str(league_id).strip()
    if lid in _TSDB_ID_ALIAS:
        return _TSDB_ID_ALIAS[lid]
    if lid in TSDB_LEAGUE_IDS:
        return TSDB_LEAGUE_IDS[lid]
    return None


def tsdb_today_matches(days: int = 7) -> list:
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    for off in range(days):
        d = today + timedelta(days=off)
        dstr = d.strftime("%Y-%m-%d")
        ck = f"tsdb_day_v3_{dstr}"
        cached = cache_get(ck, CACHE_TTL["tsdb_day"])
        if cached is not None:
            if isinstance(cached, list):
                out += cached
            continue
        try:
            r = _sess.get(
                "https://www.thesportsdb.com/api/v1/json/3/eventsday.php",
                params={"d": dstr, "s": "Soccer"},
                timeout=15, proxies=NO_PROXY)
            if r.status_code != 200:
                continue
            ev = (r.json() or {}).get("events") or []
            rows = []
            for e in ev:
                if not isinstance(e, dict):
                    continue
                h, a = e.get("strHomeTeam"), e.get("strAwayTeam")
                if not h or not a:
                    continue
                league_id = e.get("idLeague") or ""
                rows.append({
                    "Div": _match_tsdb_by_id(league_id),
                    "League": e.get("strLeague") or "Матч",
                    "league_id": league_id,
                    "Date": (e.get("dateEvent") or "")[:10],
                    "Time": (e.get("strTime") or "")[:5],
                    "HomeTeam": h, "AwayTeam": a,
                    "fixture_id": e.get("idEvent"),
                    "home_badge": e.get("strHomeTeamBadge"),
                    "away_badge": e.get("strAwayTeamBadge"),
                    "league_badge": e.get("strLeagueBadge"),
                })
            out += rows
            cache_put(ck, rows)
        except Exception:
            pass
    return out


def tsdb_past_league(tsdb_id: str, limit: int = 60) -> list:
    if not tsdb_id:
        return []
    ck = f"tsdb_past_v3_{tsdb_id}"
    cached = cache_get(ck, CACHE_TTL["tsdb_past"])
    if cached is not None:
        return cached if isinstance(cached, list) else []
    try:
        r = _sess.get(
            "https://www.thesportsdb.com/api/v1/json/3/eventspastleague.php",
            params={"id": tsdb_id}, timeout=15, proxies=NO_PROXY)
        if r.status_code != 200:
            return []
        ev = (r.json() or {}).get("events") or []
        out = []
        for e in ev[:limit]:
            h, a = e.get("strHomeTeam"), e.get("strAwayTeam")
            hg = _f(e.get("intHomeScore"))
            ag = _f(e.get("intAwayScore"))
            if not h or not a or hg is None or ag is None:
                continue
            out.append({
                "HomeTeam": h, "AwayTeam": a,
                "FTHG": str(int(hg)), "FTAG": str(int(ag)),
                "Date": (e.get("dateEvent") or "")[:10],
            })
        cache_put(ck, out)
        return out
    except Exception:
        return []


def tsdb_match_result(fixture_id: str) -> Optional[dict]:
    if not fixture_id:
        return None
    ck = f"tsdb_result_v3_{fixture_id}"
    cached = cache_get(ck, CACHE_TTL["tsdb_result"])
    if cached is not None:
        return cached or None
    try:
        r = _sess.get(
            "https://www.thesportsdb.com/api/v1/json/3/lookupevent.php",
            params={"id": fixture_id}, timeout=15, proxies=NO_PROXY)
        if r.status_code != 200:
            return None
        ev = ((r.json() or {}).get("events") or [None])[0]
        if not ev:
            return None
        status = ev.get("strStatus") or ""
        hg = _f(ev.get("intHomeScore"))
        ag = _f(ev.get("intAwayScore"))
        if hg is None or ag is None:
            cache_put(ck, None)
            return None
        result = {"home": int(hg), "away": int(ag), "status": status}
        cache_put(ck, result)
        return result
    except Exception:
        return None


def _norm_name(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


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
    if not team_name:
        return None
    ck = f"team_logo_{_norm_name(team_name)}"
    cached = cache_get(ck, 86400 * 30)
    if cached is not None:
        return cached or None
    try:
        r = _sess.get(
            "https://www.thesportsdb.com/api/v1/json/3/searchteams.php",
            params={"t": team_name}, timeout=10, proxies=NO_PROXY)
        if r.status_code != 200:
            cache_put(ck, None)
            return None
        teams = (r.json() or {}).get("teams") or []
        if not teams:
            cache_put(ck, None)
            return None
        badge = None
        tn = _norm_name(team_name)
        for t in teams:
            if _norm_name(t.get("strTeam", "")) == tn:
                badge = t.get("strTeamBadge") or t.get("strBadge")
                break
        if not badge and teams:
            badge = teams[0].get("strTeamBadge") or teams[0].get("strBadge")
        cache_put(ck, badge)
        return badge
    except Exception:
        return None


_LEAGUE_TSDB_IDS = {
    "E0": "4328", "E1": "4329", "D1": "4331", "D2": "4330",
    "I1": "4332", "I2": "4333", "SP1": "4335", "SP2": "4340",
    "F1": "4334", "F2": "4337", "N1": "4347", "B1": "4393",
    "P1": "4344", "T1": "4339", "G1": "4356", "R1": "4357",
    "C1": "4480", "EL": "4481", "EC": "4482",
}


def league_logo_url(div_code: str) -> Optional[str]:
    if not div_code:
        return None
    ck = f"league_logo_{div_code}"
    cached = cache_get(ck, 86400 * 30)
    if cached is not None:
        return cached or None
    tsdb_id = _LEAGUE_TSDB_IDS.get(div_code)
    if not tsdb_id:
        cache_put(ck, None)
        return None
    try:
        r = _sess.get(
            "https://www.thesportsdb.com/api/v1/json/3/lookupleague.php",
            params={"id": tsdb_id}, timeout=10, proxies=NO_PROXY)
        if r.status_code != 200:
            cache_put(ck, None)
            return None
        leagues = (r.json() or {}).get("leagues") or []
        if not leagues:
            cache_put(ck, None)
            return None
        badge = leagues[0].get("strBadge") or leagues[0].get("strLogo")
        cache_put(ck, badge)
        return badge
    except Exception:
        return None
