"""data/sources.py — Fixture feeds + Highlightly odds + logos (v4).

Совместим с App.py v13:
  * parse_date, fdorg_match_result    — для авто-сеттла
  * fdorg_matches, tsdb_matches       — для сканера
  * load_seasonal, season_str         — для обучения модели
  * highlightly_odds_fixture           — единственный источник реальных кэфов

Что изменено в v4:
  * USL Championship (США) больше не мапится на English Championship (E1).
    Раньше "championship" в названии лиги = E1 — это ломало USL-матчи.
  * Добавлены USA-лиги: USL Championship, USL League One, MLS.
  * Дополнительно распознаются шотландский, австралийский, японский чемпионаты.
"""
from __future__ import annotations
import csv, io, re, logging, os, hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

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


log = logging.getLogger(__name__)

MSK_OFFSET_HOURS = 3


# ============ ВРЕМЯ ============
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


def parse_date(s) -> Optional[datetime]:
    """Парсит дату из строки. Нужна для App.py и scanner."""
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
    try:
        from data.team_names import normalize_team
        return normalize_team(s)
    except Exception:
        return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


# ============ SESSION ============
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

# Circuit breaker: Basic-plan 401 means odds are unavailable for this process.
# Avoid issuing one guaranteed-to-fail odds request per fixture.
_HIGHLIGHTLY_ODDS_DISABLED_REASON = ""
_HIGHLIGHTLY_ODDS_401_LOGGED = False
_HIGHLIGHTLY_ODDS_DISABLED_KEY = ""


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


# ============ FOOTBALL-DATA.CO.UK ============
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


# ============ THESPORTSDB ============
def _tsdb_league_meta(league_name: str):
    """Определяет (div, kind, women, national) по названию лиги.

    Порядок проверки:
      1) Сборные / U-возрасты   → NT_*
      2) Женские                → W_*
      3) НЕ-английские чемпионшипы (USL, SPL, A-League, J-League...) → CLUB_*
      4) Клубные лиги из mapping
    """
    if not league_name:
        return ("G", "club", False, False)
    ln = league_name.lower()

    # --- 1. СБОРНЫЕ / U-ВОЗРАСТЫ (приоритет над всем)
    national = any(n in ln for n in (
        "u21", "u19", "u23", "u20", "u18", "u17",
        "national", "international", "world cup", "euro",
        "nations league", "friendly", "fifa", "uefa nations",
        "copa america", "africa cup", "afc asian",
    ))
    if national:
        if "u21" in ln:
            div = "NT_U21"
        elif "u19" in ln:
            div = "NT_U19"
        elif "u23" in ln:
            div = "NT_U23"
        elif "world cup" in ln:
            div = "NT_WC"
        elif "euro" in ln:
            div = "NT_EURO"
        elif "nations league" in ln:
            div = "NT_UNL"
        elif "friendly" in ln:
            div = "NT_FR"
        else:
            div = "NT"
        return (div, "national", False, True)

    # --- 2. ЖЕНСКИЕ
    women = any(w in ln for w in (
        "women", "womens", "ladies", "female", "женск"
    ))
    if women:
        return ("W_", "women", True, False)

    # --- 3. НЕ-АНГЛИЙСКИЕ "ЧЕМПИОНШИПЫ" И ДРУГИЕ ЛИГИ (важно: ДО mapping!)
    # США
    if "usl" in ln or "united soccer league" in ln:
        if "league one" in ln:
            return ("US_L1", "club", False, False)
        return ("US_USL", "club", False, False)
    if "mls" in ln or "major league soccer" in ln or "usa major" in ln:
        return ("US_MLS", "club", False, False)
    # Шотландия
    if "scottish" in ln or "scotland" in ln:
        return ("SC_", "club", False, False)
    # Австралия
    if "a-league" in ln or "a league" in ln or "australia" in ln:
        return ("AU_", "club", False, False)
    # Япония
    if "j1" in ln or "j-league" in ln or "jleague" in ln or "japan" in ln:
        return ("JP_", "club", False, False)
    # Корея
    if "k league" in ln or "k-league" in ln or "korea" in ln:
        return ("KR_", "club", False, False)
    # Китай
    if "chinese super" in ln or "china" in ln:
        return ("CN_", "club", False, False)
    # Мексика
    if "liga mx" in ln or "mexico" in ln:
        return ("MX_", "club", False, False)
    # Бразилия
    if "brasileir" in ln or "brazil" in ln:
        return ("BR_", "club", False, False)
    # Аргентина
    if "argentina" in ln or "primera division arg" in ln:
        return ("AR_", "club", False, False)

    # --- 4. АНГЛИЙСКИЕ / ЕВРОПЕЙСКИЕ ЛИГИ (после USL!)
    mapping = [
        # Англия — ВАЖНО: точное совпадение "english championship"/"efl championship"
        (("english premier league", "epl"), "E0"),
        (("efl championship", "english championship",
          "sky bet championship"), "E1"),
        (("efl league one", "english league one", "sky bet league one"), "E2"),
        (("efl league two", "english league two", "sky bet league two"), "E3"),
        # Испания
        (("spanish la liga", "la liga", "laliga"), "SP1"),
        (("spanish segunda", "segunda division", "laliga 2"), "SP2"),
        # Италия
        (("italian serie a", "serie a"), "I1"),
        (("italian serie b", "serie b"), "I2"),
        # Германия
        (("german bundesliga", "bundesliga"), "D1"),
        (("2. bundesliga", "german 2"), "D2"),
        (("3. liga", "german 3"), "D3"),
        # Франция
        (("french ligue 1", "ligue 1"), "F1"),
        (("french ligue 2", "ligue 2"), "F2"),
        # Нидерланды
        (("dutch eredivisie", "eredivisie"), "N1"),
        # Бельгия
        (("belgian pro league", "belgian first division"), "B1"),
        # Португалия
        (("portuguese primeira", "primeira liga"), "P1"),
        # Турция
        (("turkish super", "super lig"), "T1"),
        # Греция
        (("greek super", "super league greece", "super league 1"), "G1"),
        # Россия
        (("russian premier", "russian football"), "R1"),
        # Еврокубки
        (("uefa champions", "champions league"), "C1"),
        (("uefa europa", "europa league"), "EL"),
        (("uefa conference", "conference league"), "EC"),
    ]
    for keys, code in mapping:
        if any(k in ln for k in keys):
            return (code, "club", False, False)

    return ("G", "club", False, False)


def _tsdb_venue_metadata(venue_id: str, logs=None) -> dict:
    """Return venue name and its own stadium image, cached to avoid repeat requests."""
    venue_id = str(venue_id or "").strip()
    if not venue_id:
        return {}
    key = "tsdb_venue_v1_" + venue_id
    cached = cache_get(key, 86400 * 14)
    if isinstance(cached, dict):
        return cached
    try:
        response = _sess.get(
            "https://www.thesportsdb.com/api/v1/json/3/lookupvenue.php",
            params={"id": venue_id}, timeout=(3, 6), proxies=NO_PROXY,
        )
        if response.status_code != 200:
            return {}
        venues = (response.json() or {}).get("venues") or []
        venue = venues[0] if venues and isinstance(venues[0], dict) else {}
        result = {
            "venue": str(venue.get("strVenue") or ""),
            "stadium_image": str(venue.get("strThumb") or venue.get("strFanart1") or ""),
        }
        if result.get("venue") or result.get("stadium_image"):
            cache_put(key, result)
        return result
    except Exception as exc:
        if logs is not None:
            logs.append(f"TSDB venue {venue_id}: {type(exc).__name__}")
        return {}


def tsdb_matches(days: int = 7, logs=None) -> list:
    """Сбор матчей из TheSportsDB на N дней вперёд."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    total_days = min(max(1, int(days)), 14)
    venue_lookup_budget = 6

    for off in range(total_days):
        d = today + timedelta(days=off)
        dstr = d.strftime("%Y-%m-%d")
        ck = f"tsdb_day_v10_{dstr}"
        cached = cache_get(ck, 1800)
        if cached is not None:
            if isinstance(cached, list):
                out += cached
                if logs:
                    logs.append(f"TSDB {dstr}: {len(cached)} (cached)")
            continue
        try:
            r = _sess.get(
                "https://www.thesportsdb.com/api/v1/json/3/eventsday.php",
                params={"d": dstr, "s": "Soccer"},
                timeout=15, proxies=NO_PROXY)
            if r.status_code != 200:
                if logs:
                    logs.append(f"TSDB {dstr}: HTTP {r.status_code}")
                cache_put(ck, [])
                continue
            ev = (r.json() or {}).get("events") or []
            rows = []
            for e in ev:
                if not isinstance(e, dict):
                    continue
                h = e.get("strHomeTeam")
                a = e.get("strAwayTeam")
                if not h or not a:
                    continue
                # Validate sport explicitly; invalidate old cache via v9 key.
                sport_name = str(e.get("strSport") or "").strip().lower()
                league = str(e.get("strLeague") or "Матч")
                league_lower = league.lower()
                if sport_name and sport_name not in ("soccer", "football"):
                    continue
                if any(marker in league_lower for marker in (
                    "basketball", "ncaa", "ncaab", "nba", "wnba",
                    "college basketball", "march madness",
                )):
                    continue
                div, kind, women, national = _tsdb_league_meta(league)
                rows.append({
                    "Div": div,
                    "League": league,
                    "Date": (e.get("dateEvent") or "")[:10],
                    "Time": (e.get("strTime") or "")[:5],
                    "HomeTeam": h,
                    "AwayTeam": a,
                    "fixture_id": e.get("idEvent"),
                    "venue": e.get("strVenue") or "",
                    "stadium_image": e.get("strVenueThumb") or "",
                    "home_badge": e.get("strHomeTeamBadge"),
                    "away_badge": e.get("strAwayTeamBadge"),
                    "league_badge": e.get("strLeagueBadge"),
                    "women": women,
                    "national": national,
                    "kind": kind,
                })
            # Enrich a small number of unique venues per scan with official stadium photos.
            # Successful results are cached for 14 days; the limit keeps the free feed responsive.
            enriched_venues = 0
            for row in rows:
                venue_id = str(next((event.get("idVenue") for event in ev
                                     if isinstance(event, dict)
                                     and str(event.get("idEvent") or "") == str(row.get("fixture_id") or "")), "") or "")
                if venue_id and not row.get("stadium_image") and venue_lookup_budget > 0:
                    venue_meta = _tsdb_venue_metadata(venue_id, logs)
                    row.update({k: v for k, v in venue_meta.items() if v and not row.get(k)})
                    enriched_venues += 1
                    venue_lookup_budget -= 1
            out += rows
            cache_put(ck, rows)
            if logs:
                logos_count = sum(bool(x.get("home_badge")) + bool(x.get("away_badge")) for x in rows)
                stadium_count = sum(bool(x.get("stadium_image")) for x in rows)
                logs.append(f"TSDB {dstr}: {len(rows)} | team badges={logos_count} | stadium photos={stadium_count}")
        except Exception as e:
            if logs:
                logs.append(f"TSDB {dstr}: ошибка {type(e).__name__}")
    return out


# ============ ESPN SOCCER FALLBACK ============
_ESPN_DIVS = {
    "eng.1": "E0", "esp.1": "SP1", "ita.1": "I1",
    "ger.1": "D1", "fra.1": "F1", "eng.2": "E1",
    "esp.2": "SP2", "ita.2": "I2", "ger.2": "D2",
    "fra.2": "F2", "uefa.champions": "C1",
    "uefa.europa": "EL", "uefa.europa.conf": "EC",
}

def espn_matches(days: int, logs=None) -> list:
    out = []
    start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    for i in range(max(1, min(int(days), 14))):
        d = start + timedelta(days=i)
        dstr = d.strftime("%Y%m%d")
        ck = f"espn_soccer_v3_{dstr}"
        cached = cache_get(ck, 900)
        if cached is not None:
            if isinstance(cached, list):
                out += cached
                if logs:
                    logs.append(f"ESPN {dstr}: {len(cached)} (cached)")
            continue
        try:
            r = _sess.get(
                "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard",
                params={"dates": dstr}, timeout=15, proxies=NO_PROXY,
            )
            if r.status_code != 200:
                if logs:
                    logs.append(f"⚠️ ESPN {dstr}: HTTP {r.status_code}")
                continue
            payload = r.json()
            rows = []
            for event in payload.get("events") or []:
                comp = (event.get("competitions") or [{}])[0]
                competitors = comp.get("competitors") or []
                if len(competitors) < 2:
                    continue
                home = next((x for x in competitors if x.get("homeAway") == "home"), competitors[0])
                away = next((x for x in competitors if x.get("homeAway") == "away"), competitors[1])
                hn = ((home.get("team") or {}).get("displayName") or home.get("displayName") or "").strip()
                an = ((away.get("team") or {}).get("displayName") or away.get("displayName") or "").strip()
                if not hn or not an:
                    continue
                league_obj = event.get("league") or {}
                league = league_obj.get("name") or league_obj.get("abbreviation") or "Soccer"
                slug = str(league_obj.get("slug") or "")
                # Fail closed on contaminated/stale non-soccer ESPN payloads.
                sport_obj = event.get("sport") or {}
                sport_name = str(sport_obj.get("name") or sport_obj.get("slug") or "").lower()
                league_text = f"{league} {slug} {event.get('name') or ''}".lower()
                non_soccer_markers = (
                    "basketball", "ncaa", "ncaab", "nba", "wnba",
                    "college basketball", "march madness", "euroleague",
                    "volleyball", "baseball", "ice hockey", "american football",
                )
                if any(marker in league_text for marker in non_soccer_markers):
                    continue
                if sport_name and sport_name not in ("soccer", "football"):
                    continue
                logos = league_obj.get("logos") or []
                rows.append({
                    "Div": _ESPN_DIVS.get(slug, "G"),
                    "League": league, "competition": slug,
                    "Date": _msk_date(event.get("date") or ""),
                    "Time": _msk_time(event.get("date") or ""),
                    "HomeTeam": hn, "AwayTeam": an,
                    "fixture_id": f"espn:{event.get('id')}",
                    "home_badge": (home.get("team") or {}).get("logo") or "",
                    "away_badge": (away.get("team") or {}).get("logo") or "",
                    "league_badge": logos[0].get("href", "") if logos else "",
                    "status": ((event.get("status") or {}).get("type") or {}).get("name", ""),
                    "women": False, "national": False, "kind": "club",
                })
            out += rows
            cache_put(ck, rows)
            if logs:
                logs.append(f"ESPN {dstr}: {len(rows)}")
        except Exception as exc:
            if logs:
                logs.append(f"⚠️ ESPN {dstr}: {type(exc).__name__}")
    return out

def tsdb_match_result(match_id, logs=None) -> Optional[dict]:
    """Финальный счёт TSDB event."""
    if not match_id:
        return None
    ck = f"tsdb_result_v2_{match_id}"
    cached = cache_get(ck, 86400)
    if cached is not None:
        return cached or None
    try:
        r = _sess.get(
            "https://www.thesportsdb.com/api/v1/json/3/lookupevent.php",
            params={"id": str(match_id)}, timeout=15, proxies=NO_PROXY,
        )
        if r.status_code != 200:
            return None
        event = ((r.json() or {}).get("events") or [None])[0]
        if not event:
            return None
        status = str(event.get("strStatus") or "").lower()
        hg, ag = event.get("intHomeScore"), event.get("intAwayScore")
        if hg is None or ag is None:
            return None
        if status not in ("match finished", "finished", "ft", "after extra time", "after penalties"):
            # TSDB sometimes leaves status empty but has final scores.
            if not event.get("strStatus") and event.get("strResult"):
                pass
            elif status and "finish" not in status and status not in ("ft",):
                return None
        res = {"home": int(hg), "away": int(ag), "status": "FT"}
        cache_put(ck, res)
        return res
    except Exception as e:
        if logs:
            logs.append(f"TSDB result: {type(e).__name__}")
        return None


def espn_match_result(fixture_id, date_iso="", logs=None) -> Optional[dict]:
    """Ищет финальный счёт ESPN по event id."""
    event_id = str(fixture_id or "")
    if event_id.startswith("espn:"):
        event_id = event_id.split(":", 1)[1]
    if not event_id:
        return None
    ck = f"espn_result_v2_{event_id}"
    cached = cache_get(ck, 86400)
    if cached is not None:
        return cached or None
    dates = []
    if date_iso:
        try:
            base = datetime.strptime(str(date_iso)[:10], "%Y-%m-%d")
            dates = [(base + timedelta(days=i)).strftime("%Y%m%d") for i in (-1, 0, 1)]
        except Exception:
            pass
    if not dates:
        dates = [datetime.now().strftime("%Y%m%d")]
    try:
        for dstr in dates:
            r = _sess.get(
                "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard",
                params={"dates": dstr}, timeout=15, proxies=NO_PROXY,
            )
            if r.status_code != 200:
                continue
            for event in (r.json() or {}).get("events") or []:
                if str(event.get("id")) != event_id:
                    continue
                status = ((event.get("status") or {}).get("type") or {})
                if not status.get("completed"):
                    return None
                comp = (event.get("competitions") or [{}])[0]
                cs = comp.get("competitors") or []
                home = next((x for x in cs if x.get("homeAway") == "home"), None)
                away = next((x for x in cs if x.get("homeAway") == "away"), None)
                if not home or not away:
                    return None
                hs, aws = home.get("score"), away.get("score")
                if hs is None or aws is None:
                    return None
                res = {"home": int(float(hs)), "away": int(float(aws)), "status": "FT"}
                cache_put(ck, res)
                return res
    except Exception as e:
        if logs:
            logs.append(f"ESPN result: {type(e).__name__}")
    return None


# ============ GROK WEB MARKET ODDS ============
def grok_web_odds(home: str, away: str, league: str, api_key: str,
                  logs=None) -> dict:
    """Ищет текущие bookmaker odds через xAI Web Search.

    Возвращает только структурированные decimal odds для модельных рынков.
    Это экспериментальный источник: UI явно помечает его как GROK WEB.
    """
    if not api_key or not home or not away:
        return {}

    ck = f"grok_odds_v1_{_norm_name(home)}_{_norm_name(away)}"
    cached = cache_get(ck, 600)
    if isinstance(cached, dict):
        if logs:
            logs.append(f"GROK WEB: {home} — {away} (cached)")
        return cached

    prompt = f"""Find current bookmaker football odds for this match:
{home} vs {away}
League: {league}

Use web search. Prefer current bookmaker/odds pages and identify the bookmaker
and timestamp/date when available. Return ONLY valid JSON, no markdown:
{{"bookmaker":"name or unknown","updated":"date/time or unknown",
"odds":{{"П1":0,"X":0,"П2":0,"ТБ 2.5":0,"ТМ 2.5":0,
"BTTS да":0,"BTTS нет":0}},
"comment":"one short Russian comment about the found price/value, max 240 chars"}}

Use decimal odds. Put 0 when a market is not found. Do not invent odds.
Only return odds that are explicitly visible in the searched web results."""

    try:
        r = _sess.post(
            "https://api.x.ai/v1/responses",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "grok-4.7",
                "input": prompt,
                "tools": [{"type": "web_search"}],
            },
            timeout=45,
            proxies=NO_PROXY,
        )
        usage.llm_increment(1)
        if r.status_code != 200:
            if logs:
                logs.append(f"GROK WEB: HTTP {r.status_code}")
            return {}

        data = r.json() or {}
        content = data.get("output_text") or ""
        if not content:
            # Responses API may expose text inside output content blocks.
            parts = []
            for item in data.get("output") or []:
                for part in item.get("content") or []:
                    if isinstance(part, dict) and part.get("text"):
                        parts.append(str(part["text"]))
            content = "".join(parts).strip()

        match = re.search(r"\{.*\}", content, re.S)
        if not match:
            if logs:
                logs.append("GROK WEB: JSON не найден")
            return {}

        parsed = __import__("json").loads(match.group(0))
        raw_odds = parsed.get("odds") or {}
        odds = {}
        for key in ("П1", "X", "П2", "ТБ 2.5", "ТМ 2.5", "BTTS да", "BTTS нет"):
            try:
                value = float(raw_odds.get(key) or 0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 1.01:
                odds[key] = value

        if not odds:
            if logs:
                logs.append("GROK WEB: подходящих кэфов не найдено")
            return {}

        result = {
            "odds": odds,
            "bookmaker": str(parsed.get("bookmaker") or "unknown"),
            "updated": str(parsed.get("updated") or "unknown"),
            "comment": str(parsed.get("comment") or "").strip()[:240],
            "source": "grok_web",
        }
        cache_put(ck, result)
        if logs:
            logs.append(
                f"GROK WEB: {len(odds)} рынков · "
                f"{result['bookmaker']} · {result['updated']}"
            )
        return result
    except Exception as exc:
        if logs:
            logs.append(f"GROK WEB: {type(exc).__name__}")
        return {}


# ============ FOOTBALL-DATA.ORG ============
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
            log.warning("fdorg HTTP 429 — rate limit")
            return None
        if r.status_code == 403:
            log.warning("fdorg HTTP 403 — token rejected or limit")
            return None
        if r.status_code != 200:
            log.warning("fdorg HTTP %s", r.status_code)
            return None
        return r.json()
    except Exception as exc:
        log.warning("fdorg network error: %s", type(exc).__name__)
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
        ck = f"fdorg_v13_{comp}_{d_from}_{d_to}"
        cached = cache_get(ck, 1800)
        if cached is not None:
            if isinstance(cached, list):
                out += cached
                if logs:
                    logs.append(f"fdorg {comp}: {len(cached)} (cached)")
            continue
        data = _fdorg_get("matches",
                          {"dateFrom": d_from, "dateTo": d_to,
                           "competitions": comp},
                          token)
        if data is None:
            if logs:
                logs.append(f"fdorg {comp}: request failed (not cached)")
            continue
        if not data.get("matches"):
            if logs:
                logs.append(f"fdorg {comp}: 0")
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
                    "kickoff_utc": utc_date,
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "fixture_id": m.get("id"),
                    "home_badge": (m.get("homeTeam") or {}).get("crest") or "",
                    "away_badge": (m.get("awayTeam") or {}).get("crest") or "",
                    "league_badge": (m.get("competition") or {}).get("emblem") or "",
                    "status": m.get("status", ""),
                    "women": False,
                    "national": False,
                    "kind": "club",
                    "referee": ((m.get("referees") or [{}])[0].get("name", "")
                                if m.get("referees") else ""),
                    # football-data.org exposes average pre-match 1X2 odds
                    # when the account has the Odds Add-On enabled.
                    "odds": {
                        "П1": _f((m.get("odds") or {}).get("homeWin")),
                        "X": _f((m.get("odds") or {}).get("draw")),
                        "П2": _f((m.get("odds") or {}).get("awayWin")),
                    },
                    "odds_source": "fdorg" if any(
                        _f((m.get("odds") or {}).get(k)) is not None
                        for k in ("homeWin", "draw", "awayWin")
                    ) else "",
                })
            except Exception:
                continue
        out += rows
        cache_put(ck, rows)
        if logs:
            logs.append(f"fdorg {comp}: {len(rows)}")
    return out


def fdorg_match_result(match_id, token: str) -> Optional[dict]:
    """Финальный результат для авто-сеттла. App.py v13 требует эту функцию."""
    if not match_id or not token:
        return None
    ck = f"fdorg_result_v11_{match_id}"
    cached = cache_get(ck, 86400)
    if cached is not None:
        return cached or None
    data = _fdorg_get(f"matches/{match_id}", {}, token)
    if not data or not data.get("match"):
        # Do not cache transient failures: a later retry may succeed.
        return None
    m = data["match"]
    if m.get("status") != "FINISHED":
        # A live/scheduled fixture can become final later the same day.
        # Caching None for 24 hours prevented forecast reconciliation.
        return None
    score = m.get("score") or {}
    if (score.get("duration") or "REGULAR") != "REGULAR":
        reg = score.get("regularTime") or {}
        hg = reg.get("home")
        ag = reg.get("away")
        if hg is None or ag is None:
            res = {"home": 0, "away": 0, "status": "AET"}
            cache_put(ck, res)
            return res
    else:
        full = score.get("fullTime") or {}
        hg = full.get("home")
        ag = full.get("away")
    if hg is None or ag is None:
        cache_put(ck, None)
        return None
    res = {"home": int(hg), "away": int(ag), "status": "FT"}
    cache_put(ck, res)
    return res



# ============ PROGNOZIST / MARATHON LINE ============
_PROGNOZIST_URL = "https://prognozist.ru/line/marathon/out.php"
_PROGNOZIST_MATCH_CENTRE = "https://prognozist.ru/mc/football/"


class _PrognozistHTMLParser(__import__("html.parser", fromlist=["HTMLParser"]).HTMLParser):
    """Collect table rows, keeping match-detail links attached to their odds."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None
        self.row_hrefs = []
        self.links, self.link = [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.row = []
            self.row_hrefs = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        if tag == "a":
            href = str(attrs.get("href", "") or "").strip()
            self.link = {"href": href, "text": []}
            # Keep the event detail URL with its parent table row. Odds often
            # live in sibling cells, not inside the anchor text itself.
            if self.row is not None and (
                href.startswith("/mc/football/")
                or href.startswith("https://prognozist.ru/mc/football/")
            ):
                self.row_hrefs.append(href)

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)
        if self.link is not None:
            self.link["text"].append(data)

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self.cell is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            if self.row:
                # Put the match-detail URL first so the fixture parser can
                # fetch the H1 for authoritative team names and kickoff date.
                self.rows.append(list(dict.fromkeys(self.row_hrefs)) + self.row)
            self.row = None
            self.row_hrefs = []
        elif tag == "a" and self.link is not None:
            self.link["text"] = " ".join(" ".join(self.link["text"]).split())
            self.links.append(self.link)
            self.link = None


def _prognozist_num(value):
    value = str(value or "").strip().replace(",", ".")
    if not re.fullmatch(r"\d{1,3}(?:\.\d{1,4})?", value):
        return None
    try:
        number = float(value)
        return number if 1.01 <= number <= 100 else None
    except (TypeError, ValueError):
        return None


def _prognozist_rows(body):
    body = (body or "").lstrip("\ufeff")
    rows = []
    try:
        payload = __import__("json").loads(body)
        if isinstance(payload, dict):
            payload = payload.get("matches") or payload.get("events") or payload.get("data") or []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    rows.append([str(v) for v in item.values() if v is not None])
    except Exception:
        pass
    if not rows and ("<tr" in body.lower() or "<a" in body.lower()):
        parser = _PrognozistHTMLParser()
        parser.feed(body)
        rows.extend(parser.rows)
        rows.extend([[x.get("href", ""), x.get("text", "")] for x in parser.links])
    if not rows:
        try:
            rows = list(csv.reader(io.StringIO(body), delimiter=";"))
            if not any(len(row) > 1 for row in rows):
                rows = list(csv.reader(io.StringIO(body)))
        except Exception:
            rows = []
    return rows


def _prognozist_extract_1x2(text):
    """Extract only explicitly labelled P1/X/P2 decimal prices."""
    patterns = (
        r"П1\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?)\s*X\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?)\s*П2\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?)",
        r"П1\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?).{0,12}?X\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?).{0,12}?П2\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?)",
    )
    for pattern in patterns:
        m = re.search(pattern, text or "", flags=re.I | re.S)
        if m:
            vals = [_prognozist_num(x) for x in m.groups()]
            if all(v is not None for v in vals):
                return {"П1": vals[0], "X": vals[1], "П2": vals[2]}
    return None


def _prognozist_match(row, home, away):
    """Match home/away order across English provider names and Russian public-page labels."""
    joined = " ".join(str(x) for x in row).lower()
    raw = re.sub(r"[^a-zа-я0-9]", "", joined)
    if not raw or not home or not away:
        return False

    try:
        from data.team_names import ALIASES, normalize_team
        from config import TEAM_TRANSLATIONS

        def clean(value):
            return re.sub(r"[^a-zа-я0-9]", "", str(value or "").casefold())

        def variants(team):
            canonical = normalize_team(team)
            values = {clean(team)}
            values.update(
                clean(alias)
                for alias, target in ALIASES.items() if target == canonical
            )
            # The public Prognozist match-centre is Russian, while ESPN/TSDB
            # fixture names are commonly English. Add both sides of every
            # configured translation whose English alias maps to this club.
            for english, russian in TEAM_TRANSLATIONS.items():
                try:
                    english_key = normalize_team(english)
                    russian_key = normalize_team(russian)
                except Exception:
                    english_key, russian_key = clean(english), clean(russian)
                if canonical in (english_key, russian_key):
                    values.add(clean(english))
                    values.add(clean(russian))
            # Longest names first; short aliases are prone to false positives.
            return sorted((v for v in values if len(v) >= 5), key=len, reverse=True)

        home_variants, away_variants = variants(home), variants(away)
    except Exception:
        home_variants = [re.sub(r"[^a-zа-я0-9]", "", str(home).casefold())]
        away_variants = [re.sub(r"[^a-zа-я0-9]", "", str(away).casefold())]

    home_positions = [(raw.find(v), len(v)) for v in home_variants if raw.find(v) >= 0]
    away_positions = [(raw.find(v), len(v)) for v in away_variants if raw.find(v) >= 0]
    # Odds are side-specific. Presence alone is insufficient: reversed fixtures
    # must not be accepted because that swaps home and away prices.
    return any(hpos + hlen <= apos for hpos, hlen in home_positions
               for apos, _ in away_positions)


def _prognozist_extract_row_odds(row):
    """Read 1X2 odds from a single structured table row, even without labels."""
    text = " ".join(str(x) for x in row)
    labelled = _prognozist_extract_1x2(text)
    if labelled:
        return labelled
    # The public line often renders team names and three prices in separate
    # table cells, while the P1/X/P2 labels live in a different header row.
    # Only accept a row that contains at least two textual team-like cells.
    cells = [str(x).strip() for x in row if str(x).strip()]
    textual = [x for x in cells if re.search(r"[A-Za-zА-Яа-я]", x)]
    if len(textual) < 2:
        return None
    prices = []
    for cell in cells:
        value = _prognozist_num(cell)
        if value is not None:
            prices.append(value)
    if len(prices) < 3:
        # Some feeds put multiple prices in one cell.
        nums = re.findall(r"(?<![\w.])\d{1,3}(?:[.,]\d{1,4})(?![\w.])", text)
        prices = [_prognozist_num(x) for x in nums]
        prices = [x for x in prices if x is not None]
    if len(prices) < 3:
        return None
    return {"П1": prices[-3], "X": prices[-2], "П2": prices[-1]}


def prognozist_permission_enabled() -> bool:
    """Read explicit opt-in from the app toggle, environment, or Streamlit Secrets."""
    try:
        import streamlit as st
        # Once the UI opt-in exists, it is the operator's current choice and
        # can turn even a configured source off for this session.
        if "prognozist_permission_input" in st.session_state:
            return bool(st.session_state.get("prognozist_permission_input", False))
        data = st.session_state.get("data") or {}
        meta = data.get("meta") if isinstance(data, dict) else {}
        if isinstance(meta, dict) and bool(meta.get("prognozist_permission_enabled", False)):
            return True
        if os.environ.get("PROGNOZIST_PERMISSION", "").strip() == "1":
            return True
        return str(st.secrets.get("PROGNOZIST_PERMISSION", "") or "").strip() == "1"
    except Exception:
        return os.environ.get("PROGNOZIST_PERMISSION", "").strip() == "1"


def _prognozist_allowed(url: str, permission_granted: bool = False) -> bool:
    """Require explicit permission and robots.txt allowance; cache robots rules once.

    permission_granted is captured on the Streamlit main thread before worker
    threads start; worker threads cannot safely read st.session_state.
    """
    if not permission_granted and not prognozist_permission_enabled():
        log.info("Prognozist odds disabled: explicit permission is off.")
        return False
    decision_key = "prognozist_robots_decision_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    cached_decision = cache_get(decision_key, 180)
    if isinstance(cached_decision, bool):
        return cached_decision
    try:
        parsed_url = urlparse(url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        robots_text = cache_get("prognozist_robots_body_v1", 3600)
        if not isinstance(robots_text, str) or not robots_text.strip():
            response = _sess.get(robots_url, timeout=(3, 6), proxies=NO_PROXY)
            if response.status_code != 200 or not response.text:
                log.warning("Prognozist robots.txt unavailable (HTTP %s); disabling source.", response.status_code)
                cache_put(decision_key, False)
                return False
            robots_text = response.text
            cache_put("prognozist_robots_body_v1", robots_text)
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(robots_text.splitlines())
        agent = "Sport-AI-Pro13/1.0 (+contact the repository maintainer; research client)"
        allowed = bool(parser.can_fetch(agent, url))
        cache_put(decision_key, allowed)
        return allowed
    except Exception as exc:
        log.warning("Prognozist robots.txt check failed (%s); disabling source.", type(exc).__name__)
        cache_put(decision_key, False)
        return False

def prognozist_source_status() -> str:
    """Return an operator-facing status without scraping match data."""
    if not prognozist_permission_enabled():
        return "disabled_permission"
    allowed = [
        url for url in (_PROGNOZIST_URL, _PROGNOZIST_MATCH_CENTRE)
        if _prognozist_allowed(url)
    ]
    return "ready" if allowed else "blocked_or_unavailable"


def _prognozist_parse_russian_date(text, fallback_year=None):
    months = {"января":1,"февраля":2,"марта":3,"апреля":4,"мая":5,"июня":6,
              "июля":7,"августа":8,"сентября":9,"октября":10,"ноября":11,"декабря":12}
    m = re.search(r"\b(\d{1,2})\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)(?:\s+(20\d{2}))?\b", str(text or "").casefold())
    if not m:
        return None
    day, month, year = m.groups()
    try:
        result = datetime(int(year or fallback_year or datetime.now().year), months[month], int(day)).date()
        if not year and result < datetime.now().date() - timedelta(days=180):
            result = result.replace(year=result.year + 1)
        return result
    except (ValueError, OverflowError):
        return None


def _prognozist_parse_detail_heading(heading, odds=None, today=None):
    heading = " ".join(str(heading or "").split())
    m = re.search(r"^\s*(.+?)\s+[—–-]\s+(.+?)\s+прогноз\s+на\s+(.+?)\s*$", heading, re.I)
    if not m:
        return None
    home, away, date_time = (x.strip() for x in m.groups())
    match_date = _prognozist_parse_russian_date(date_time, fallback_year=(today or datetime.now().date()).year)
    if not match_date or not home or not away or _norm_name(home) == _norm_name(away):
        return None
    tm = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", date_time)
    fixture = {
        "Div":"G", "League":"Prognozist / Marathon", "Date":match_date.isoformat(),
        "Time":f"{int(tm.group(1)):02d}:{tm.group(2)}" if tm else "",
        "HomeTeam":home, "AwayTeam":away,
        "fixture_id":"prognozist:" + hashlib.sha256(f"{match_date.isoformat()}|{home}|{away}".casefold().encode("utf-8")).hexdigest()[:20],
        "women":"(ж)" in f"{home} {away}".casefold(), "national":False, "kind":"club",
    }
    if isinstance(odds, dict):
        try:
            prices = {k:float(odds[k]) for k in ("П1","X","П2")}
            if all(1.01 < v <= 100 for v in prices.values()):
                fixture["prognozist_odds"] = prices
        except (KeyError, TypeError, ValueError):
            pass
    return fixture


def _prognozist_extract_extra_markets(html):
    """Extract only explicitly labelled prices for the exact full-match market."""
    from html import unescape

    # Keep boundaries between HTML elements so headings and adjacent odds do not
    # get accidentally concatenated into unrelated numbers.
    text = unescape(str(html or ""))
    text = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>|</(?:p|div|tr|td|th|li|h[1-6])\s*>", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    found = {"btts_market_odds": {}, "total_market_odds": {}, "stat_market_odds": {}}

    # Parse prices only inside one explicitly labelled bookmaker block. Never
    # collect prices page-wide and then attach an unrelated bookmaker name to them.
    source_headings = list(re.finditer(
        r"Коэффициенты\s+(Winline|Марафон|Marathon)\b", text, re.I
    ))
    bookmaker_sections = []
    for idx, heading in enumerate(source_headings):
        section_end = source_headings[idx + 1].start() if idx + 1 < len(source_headings) else len(text)
        section = text[heading.end():section_end]
        has_prices = bool(re.search(
            r"\b(?:Больше|Меньше|Бол\.?|Мен\.?|Да|Нет)\b.{0,24}\b\d{1,3}[.,]\d{1,4}\b",
            section, re.I
        ) or re.search(
            r"\b\d{1,3}[.,]\d{1,4}\b.{0,24}\b(?:Больше|Меньше|Бол\.?|Мен\.?)\b",
            section, re.I
        ))
        if has_prices:
            source = ("prognozist_winline" if heading.group(1).casefold() == "winline"
                      else "prognozist_marathon")
            bookmaker_sections.append((source, section))
    # The requested source is Marathon. If both bookmaker blocks are present,
    # prefer the explicitly labelled Marathon block regardless of page order.
    bookmaker_section = next(
        (section for source, section in bookmaker_sections if source == "prognozist_marathon"),
        None,
    )
    source_label = "prognozist_marathon" if bookmaker_section is not None else "prognozist_unverified"
    if bookmaker_section is None and bookmaker_sections:
        source_label, bookmaker_section = bookmaker_sections[0]
    if bookmaker_section is None:
        found["extra_market_source"] = "prognozist_unverified"
        return found
    text = bookmaker_section

    def valid_price(raw):
        odd = _prognozist_num(raw)
        return odd if odd is not None and 1.01 < odd <= 100 else None

    # Only pair the explicit Да/Нет prices belonging to the BTTS label.
    btts = re.search(
        r"Обе\s+забьют\??\s*.{0,140}?\bДа\b\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?)"
        r".{0,80}?\bНет\b\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?)",
        text, re.I,
    )
    if btts:
        yes, no = (valid_price(x) for x in btts.groups())
        if yes and no:
            found["btts_market_odds"] = {"BTTS Да": yes, "BTTS Нет": no}

    # Work within the relevant market section, not across the whole page. This
    # prevents first/second-half prices being attached to full-match totals.
    def collect_totals(section, target, prefix, lines):
        for line in lines:
            line_pattern = re.escape(str(line)).replace(r"\.5", r"[.,]5")
            # Support both common layouts:
            # "Тотал 2.5 Больше 1.90 Меньше 1.90" and
            # "Больше 2.5 1.90 Меньше 2.5 1.90".
            for side, label in (("Больше", "ТБ"), ("Меньше", "ТМ")):
                patterns = (
                    rf"\b{side}\b\s*\(?\s*{line_pattern}\s*\)?\s*[:\-]?\s*([0-9]{{1,3}}(?:[.,][0-9]{{1,4}})?)",
                    rf"\b{side}\b\s*[:\-]?\s*([0-9]{{1,3}}(?:[.,][0-9]{{1,4}})?)\s*\(?\s*{line_pattern}\s*\)?",
                )
                for pattern in patterns:
                    match = re.search(pattern, section, re.I)
                    if match:
                        odd = valid_price(match.group(1))
                        if odd:
                            target[f"{prefix} {label} {line}"] = odd
                            break

    # Main goals total, explicitly stop at period-specific or next-market headings.
    total_sections = list(re.finditer(
        r"\bТотал(?:\s+голов)?\b(?!\s*(?:угловых|углов|карточек|желтых|жёлтых))"
        r"(?P<body>.{0,900}?)(?=\b(?:1-й?\s*тайм|1\s*тайм|2-й?\s*тайм|2\s*тайм|"
        r"Обе\s+забьют|Угловые|Тотал\s+угловых|Карточки|Желтые карточки|Жёлтые карточки|"
        r"1X2|Двойной шанс|Фора|Точный счет)\b|$)",
        text, re.I,
    ))
    for section_match in total_sections:
        preceding = text[max(0, section_match.start() - 100):section_match.start()]
        if re.search(r"(?:1\s*[-.]?\s*тайм|перв(?:ый|ого)\s+тайм|2\s*[-.]?\s*тайм|втор(?:ой|ого)\s+тайм|перв(?:ая|ого)\s*половин|втор(?:ая|ого)\s*половин)", preceding, re.I):
            continue
        body = section_match.group("body")
        body = re.split(r"\b(?:1\s*[-.]?\s*тайм|перв(?:ый|ого)\s+тайм|2\s*[-.]?\s*тайм|втор(?:ой|ого)\s+тайм|перв(?:ая|ого)\s*половин|втор(?:ая|ого)\s*половин)\b", body, maxsplit=1, flags=re.I)[0]
        collect_totals(body, found["total_market_odds"], "", (0.5, 1.5, 2.5, 3.5, 4.5, 5.5))
    # Clean keys after using a shared helper.
    found["total_market_odds"] = {
        key.strip(): value for key, value in found["total_market_odds"].items()
    }

    # Total corners and cards are separate markets from corner 1X2. Parse only
    # explicit totals/lines and store keys identical to those used by the scanner.
    market_specs = (
        (r"(?:Тотал\s+угловых|Угловые\s*[-:]?\s*Тотал|Угловые\s+тотал)", "Угл", (8.5, 9.5, 10.5)),
        (r"(?:Тотал\s+(?:желтых|жёлтых)\s+карточек|Тотал\s+карточек|Желтые карточки\s*[-:]?\s*Тотал|Жёлтые карточки\s*[-:]?\s*Тотал)", "Карты", (3.5, 4.5, 5.5)),
    )
    for heading_pattern, prefix, lines in market_specs:
        for heading in re.finditer(heading_pattern, text, re.I):
            section = text[heading.end():heading.end() + 800]
            section = re.split(
                r"\b(?:Обе\s+забьют|Тотал\s+голов|1X2|Двойной шанс|Фора|Точный счет|"
                r"Угловые\s*[-:]?\s*Тотал|Тотал\s+угловых|Тотал\s+карточек|"
                r"Тотал\s+(?:желтых|жёлтых)\s+карточек)\b",
                section, maxsplit=1, flags=re.I,
            )[0]
            collect_totals(section, found["stat_market_odds"], prefix, lines)

    # Prognozist/Marathon's betting-line page often renders additional totals
    # as compact labels, e.g. "Дополнительный тотал: (2.5) Бол. 1.181; Мен. 3.95".
    # Parse those explicit paired quotes as well as the prose-style detail page.
    compact_total = re.compile(
        r"(?:Дополнительный\s+)?Тотал(?:\s+голов)?\s*[:,-]?\s*"
        r"\(?\s*(?P<line>\d{1,2}[.,]\d)\s*\)?\s*"
        r"(?:Бол(?:ьше)?\.?|ТБ)\s*[:=]?\s*(?P<over>\d{1,3}(?:[.,]\d{1,4})?)"
        r"\s*[;,|]?\s*(?:Мен(?:ьше)?\.?|ТМ)\s*[:=]?\s*(?P<under>\d{1,3}(?:[.,]\d{1,4})?)",
        re.I,
    )
    for match in compact_total.finditer(text):
        start = max(0, match.start() - 90)
        context = text[start:match.start()]
        # Do not assign first/second-half totals to the full-match market.
        if re.search(r"(?:1\s*[-.]?\s*(?:й\s*)?тайм|перв(?:ый|ого)\s+тайм|2\s*[-.]?\s*(?:й\s*)?тайм|втор(?:ой|ого)\s+тайм)", context, re.I):
            continue
        line = match.group("line").replace(",", ".")
        over, under = valid_price(match.group("over")), valid_price(match.group("under"))
        if over:
            found["total_market_odds"][f"ТБ {line}"] = over
        if under:
            found["total_market_odds"][f"ТМ {line}"] = under

    # A common layout puts "Больше/Меньше" before the line and price, while
    # another uses "Тотал бол. (3.5) 2.375; Тотал мен. (3.5) 1.60".
    compact_side = re.compile(
        r"Тотал\s*(?:голов\s*)?(?P<side>бол(?:ьше)?\.?|мен(?:ьше)?\.?|ТБ|ТМ)"
        r"\s*\(?\s*(?P<line>\d{1,2}[.,]\d)\s*\)?\s*"
        r"[:=]?\s*(?P<price>\d{1,3}(?:[.,]\d{1,4})?)",
        re.I,
    )
    for match in compact_side.finditer(text):
        context = text[max(0, match.start() - 100):match.start()]
        if re.search(r"(?:1\s*[-.]?\s*(?:й\s*)?тайм|перв(?:ый|ого)\s+тайм|2\s*[-.]?\s*(?:й\s*)?тайм|втор(?:ой|ого)\s+тайм)", context, re.I):
            continue
        line = match.group("line").replace(",", ".")
        side = match.group("side").casefold()
        key = f"ТБ {line}" if side.startswith(("бол", "тб")) else f"ТМ {line}"
        price = valid_price(match.group("price"))
        if price:
            found["total_market_odds"][key] = price

    # The source also offers corner 1X2 on some matches. Do not confuse this
    # with a total-corners market.
    corners = re.search(
        r"1X2\s+Угловые\s*1\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?)"
        r"\s*X\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?)"
        r"\s*2\s*([0-9]{1,3}(?:[.,][0-9]{1,4})?)",
        text, re.I,
    )
    if corners:
        values = [valid_price(x) for x in corners.groups()]
        if all(values):
            found["stat_market_odds"].update({
                "Угл П1": values[0], "Угл X": values[1], "Угл П2": values[2],
            })

    # Keep a stable response schema: downstream callers and tests expect every
    # market key to exist even when that market is absent from this bookmaker block.
    found["extra_market_source"] = source_label
    return found

def _prognozist_fetch_detail_fixture(href, odds=None, today=None, permission_granted=False):
    href = str(href or "").strip()
    if href.startswith("/"):
        from urllib.parse import urljoin
        href = urljoin("https://prognozist.ru", href)
    parsed = urlparse(href)
    if parsed.scheme != "https" or parsed.netloc.casefold() != "prognozist.ru" or not parsed.path.startswith("/mc/football/"):
        return None
    if not _prognozist_allowed(href, permission_granted=permission_granted):
        return None
    # Versioned cache: old builds cached fixtures without extra market prices.
    cache_key = "prognozist_detail_v2_" + hashlib.sha256(href.encode("utf-8")).hexdigest()[:24]
    cached = cache_get(cache_key, 1800)
    if isinstance(cached, dict):
        fixture = dict(cached)
        if isinstance(odds, dict):
            fixture["prognozist_odds"] = dict(odds)
        return fixture
    headers = {"User-Agent":"Sport-AI-Pro13/1.0 (+contact the repository maintainer; research client)",
               "Accept":"text/html,application/xhtml+xml", "Referer":_PROGNOZIST_MATCH_CENTRE}
    try:
        response = _sess.get(href, headers=headers, timeout=(3,6), proxies=NO_PROXY)
        if response.status_code != 200 or not response.text:
            log.warning("Prognozist detail HTTP %s: %s", response.status_code, href)
            return None
        m = re.search(r"<h1\b[^>]*>(.*?)</h1>", response.text, re.I | re.S)
        if not m:
            return None
        heading = re.sub(r"<[^>]+>", " ", m.group(1))
        from html import unescape
        fixture = _prognozist_parse_detail_heading(unescape(heading), odds, today=today)
        if fixture:
            extra = _prognozist_extract_extra_markets(response.text)
            fixture.update(extra)
            market_counts = {
                "totals": len(fixture.get("total_market_odds") or {}),
                "btts": len(fixture.get("btts_market_odds") or {}),
                "stats": len(fixture.get("stat_market_odds") or {}),
            }
            log.info("Prognozist detail markets %s: %s", href, market_counts)
            cache_put(cache_key, {k:v for k,v in fixture.items() if k != "prognozist_odds"})
        return fixture
    except Exception as exc:
        log.warning("Prognozist detail fetch failed (%s)", type(exc).__name__)
        return None


def _prognozist_line_index(days=1, logs=None):
    """Download and parse the line only when explicitly permitted by the operator and robots.txt."""
    if not prognozist_permission_enabled():
        if logs is not None:
            logs.append("Prognozist: источник выключен переключателем разрешения.")
        _prognozist_allowed(_PROGNOZIST_URL)
        return []
    # Check the current permission/robots decision before consulting the
    # odds cache; a previously cached line must not bypass a newly blocked source.
    today = datetime.now().date()
    page_urls = [_PROGNOZIST_URL, "https://prognozist.ru/line/marathon/output_line.php", _PROGNOZIST_MATCH_CENTRE]
    for offset in range(1, max(1, min(int(days), 14))):
        if offset == 1:
            page_urls.append("https://prognozist.ru/mc/football/tomorrow/")
        else:
            page_urls.append(f"https://prognozist.ru/mc/football/{(today + timedelta(days=offset)).isoformat()}/")
    allowed_urls = list(dict.fromkeys(url for url in page_urls if _prognozist_allowed(url)))
    if not allowed_urls:
        if logs is not None:
            logs.append("Prognozist: все адреса заблокированы robots.txt или недоступны.")
        return []
    key = f"prognozist_marathon_line_index_v8_{max(1, min(int(days), 14))}"
    cached = cache_get(key, 180)
    if isinstance(cached, list):
        if logs is not None:
            logs.append(f"Prognozist: кэш линии содержит {len(cached)} строк с коэффициентами.")
        return cached
    headers = {
        "User-Agent": "Sport-AI-Pro13/1.0 (+contact the repository maintainer; research client)",
        "Accept": "text/html,application/json,text/plain,*/*",
        "Referer": _PROGNOZIST_URL,
    }
    indexed = []
    for url in allowed_urls:
        try:
            response = _sess.get(url, headers=headers, timeout=(3, 6), proxies=NO_PROXY)
            if response.status_code != 200 or not response.text:
                msg = f"Prognozist: HTTP {response.status_code} на {url}, body={len(response.text or '')} bytes."
                log.warning(msg)
                if logs is not None:
                    logs.append(msg)
                continue
            body = response.text
            parser = _PrognozistHTMLParser()
            if "<" in body:
                parser.feed(body)
            candidates = _prognozist_rows(body)
            candidates.extend([[x.get("href", ""), x.get("text", "")] for x in parser.links])
            before_count = len(indexed)
            for row in candidates:
                text = " ".join(str(x) for x in row)
                odds = _prognozist_extract_row_odds(row)
                # Preserve only rows that have explicit 1X2 labels or a table-row
                # shape with team text plus three plausible decimal prices.
                if odds:
                    indexed.append({
                        "text": text,
                        "cells": [str(cell) for cell in row],
                        "odds": odds,
                    })
            parsed_count = len(indexed) - before_count
            if parsed_count == 0:
                msg = f"Prognozist: HTTP 200 на {url}, HTML={len(body)} bytes, строк={len(candidates)}, 1X2=0."
                log.warning(msg)
            else:
                msg = f"Prognozist: HTTP 200 на {url}, строк={len(candidates)}, 1X2={parsed_count}."
                log.info(msg)
            if logs is not None:
                logs.append(msg)
            # Keep rows from both feeds: the first endpoint can return a partial
            # line that does not include the fixture being searched.
        except Exception as exc:
            msg = f"Prognozist: запрос {url} завершился ошибкой {type(exc).__name__}."
            log.warning(msg)
            if logs is not None:
                logs.append(msg)
    cache_put(key, indexed)
    return indexed


def _prognozist_fixture_from_item(item, today=None, permission_granted=False):
    if not isinstance(item, dict):
        return None
    cells = item.get("cells")
    href = str(item.get("href") or (cells[0] if isinstance(cells, list) and cells else "")).strip()
    odds = item.get("odds")
    # Match-centre entries are compact anchors, not separate team/date cells.
    if href.startswith("/mc/football/") or href.startswith("https://prognozist.ru/mc/football/"):
        if not isinstance(odds, dict) or not all(k in odds for k in ("П1", "X", "П2")):
            return None
        try:
            prices = {k: float(odds[k]) for k in ("П1", "X", "П2")}
        except (TypeError, ValueError):
            return None
        if not all(1.01 < v <= 100 for v in prices.values()):
            return None
        return _prognozist_fetch_detail_fixture(href, prices, today=today, permission_granted=permission_granted)

    """Convert a parsed Prognozist 1X2 row into a fixture only when teams/date are clear."""
    if not isinstance(item, dict):
        return None
    cells = item.get("cells")
    if not isinstance(cells, list) or len(cells) < 3:
        return None
    odds = item.get("odds")
    if not isinstance(odds, dict) or not all(k in odds for k in ("П1", "X", "П2")):
        return None
    prices = []
    for key in ("П1", "X", "П2"):
        try:
            prices.append(float(odds[key]))
        except (TypeError, ValueError):
            return None

    price_indexes = []
    for idx, cell in enumerate(cells):
        number = _prognozist_num(cell)
        if number is not None and any(abs(number - price) < 0.0001 for price in prices):
            price_indexes.append(idx)
    cutoff = min(price_indexes) if price_indexes else len(cells)
    candidates = []
    generic = {
        "футбол", "soccer", "матч", "линия", "прогноз", "событие",
        "п1", "x", "п2", "1", "2", "коэффициенты", "коэф.",
        "скоро", "live", "prematch", "предматч",
    }
    for cell in cells[:cutoff]:
        value = " ".join(str(cell or "").split()).strip(" —–-")
        if not value or value.casefold() in generic:
            continue
        if re.search(r"\b\d{1,2}:\d{2}\b", value):
            continue
        if re.search(r"(?<!\d)\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?(?!\d)", value):
            continue
        if re.fullmatch(r"\d{1,3}(?:[.,]\d{1,4})?", value):
            continue
        if not re.search(r"[A-Za-zА-Яа-я]", value):
            continue
        candidates.append(value)
    if len(candidates) < 2:
        # Compact rows may concatenate both teams and odds into one cell;
        # do not guess a split: a missing fixture is safer than wrong odds.
        return None
    home, away = candidates[-2], candidates[-1]
    if _norm_name(home) == _norm_name(away):
        return None

    text = " ".join(str(x or "") for x in cells)
    iso_dates = re.findall(r"(?<!\d)(20\d{2})-(\d{1,2})-(\d{1,2})(?!\d)", text)
    dates = []
    for yy, mm, dd in iso_dates:
        try:
            dates.append(datetime(int(yy), int(mm), int(dd)).date())
        except ValueError:
            pass
    if not dates:
        for dd, mm, yy in re.findall(
            r"(?<!\d)(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?(?!\d)", text
        ):
            year = int(yy) if yy else (today or datetime.now().date()).year
            if year < 100:
                year += 2000
            try:
                dates.append(datetime(year, int(mm), int(dd)).date())
            except ValueError:
                pass
    if not dates:
        return None
    match_date = dates[0]
    now_date = today or datetime.now().date()
    if match_date < now_date - timedelta(days=180):
        try:
            match_date = match_date.replace(year=match_date.year + 1)
        except ValueError:
            pass
    time_match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    match_time = f"{int(time_match.group(1)):02d}:{time_match.group(2)}" if time_match else ""
    return {
        "Div": "G",
        "League": "Prognozist / Marathon",
        "Date": match_date.isoformat(),
        "Time": match_time,
        "HomeTeam": home,
        "AwayTeam": away,
        "fixture_id": "prognozist:" + hashlib.sha256(
            f"{match_date.isoformat()}|{home}|{away}".casefold().encode("utf-8")
        ).hexdigest()[:20],
        "women": False,
        "national": False,
        "kind": "club",
        "prognozist_odds": {"П1": prices[0], "X": prices[1], "П2": prices[2]},
    }


def prognozist_matches(days: int = 7, logs=None) -> list:
    """Use Prognozist's own parsed 1X2 line as the scanner's fixture list."""
    if not prognozist_permission_enabled():
        if logs is not None:
            logs.append("Prognozist: сбор матчей выключен — требуется явное разрешение/opt-in.")
        return []
    # Capture Streamlit-controlled permission on the main thread. Worker threads
    # receive this explicit snapshot and never touch st.session_state.
    permission_granted = prognozist_permission_enabled()
    today = datetime.now().date()
    end_date = today + timedelta(days=max(1, min(int(days), 14)) - 1)
    line_index = _prognozist_line_index(days=days, logs=logs)
    fixtures, seen = [], set()
    compact_items, structured_items = [], []
    for item in line_index:
        cells = item.get("cells") if isinstance(item, dict) else None
        href = str(item.get("href") or (cells[0] if isinstance(cells, list) and cells else "")).strip()
        if href.startswith("/mc/football/") or href.startswith("https://prognozist.ru/mc/football/"):
            # Only detail-fetch upcoming rows with a full line; avoid needless requests for live/finished events.
            label = str(item.get("text") or " ".join(str(x) for x in cells or [])).casefold()
            if "скоро" in label and isinstance(item.get("odds"), dict):
                compact_items.append(item)
        else:
            structured_items.append(item)

    # Match pages are fetched concurrently with a conservative cap; all URLs still pass robots.txt checks.
    parsed_fixtures = []
    for item in structured_items:
        fixture = _prognozist_fixture_from_item(item, today=today, permission_granted=permission_granted)
        if fixture:
            parsed_fixtures.append(fixture)
    if compact_items:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures = [pool.submit(_prognozist_fixture_from_item, item, today, permission_granted) for item in compact_items]
            for future in futures:
                try:
                    fixture = future.result(timeout=12)
                except Exception:
                    fixture = None
                if fixture:
                    parsed_fixtures.append(fixture)

    for fixture in parsed_fixtures:
        try:
            match_date = datetime.strptime(fixture["Date"], "%Y-%m-%d").date()
        except Exception:
            continue
        if not today <= match_date <= end_date:
            continue
        key = (match_date.isoformat(), _norm_name(fixture["HomeTeam"]), _norm_name(fixture["AwayTeam"]))
        if key in seen:
            continue
        seen.add(key)
        fixtures.append(fixture)
    fixtures.sort(key=lambda row: (row["Date"], row["Time"], row["HomeTeam"], row["AwayTeam"]))
    if logs is not None:
        logs.append(f"📡 Prognozist/Marathon: строк с коэффициентами {len(line_index)}; матчей распознано {len(fixtures)}.")
        if line_index and not fixtures:
            logs.append("⚠️ Коэффициенты найдены, но карточки матчей не разобраны. Проверьте доступ к страницам матчей и заголовки H1.")
        elif not line_index:
            logs.append("⚠️ В ответе Prognozist не распознано ни одной строки 1X2.")
    return fixtures


def prognozist_marathon_odds(
    home: str, away: str, kickoff=None, logs=None
) -> Optional[dict]:
    """Return a complete, unambiguous real 1X2 line; never use cache to bypass permission."""
    if not home or not away:
        return None
    # Check permission before reading match or line caches, so disabling the
    # source takes effect immediately and cached data cannot bypass the gate.
    if not prognozist_permission_enabled():
        if logs is not None:
            logs.append("Prognozist: отключён — включите opt-in переключатель в боковой панели (только если разрешено правилами сайта).")
        return None
    hn, an = _norm_name(home), _norm_name(away)
    if not hn or not an:
        return None
    day = ""
    if kickoff:
        try:
            day = kickoff.strftime("%Y-%m-%d") if hasattr(kickoff, "strftime") else str(kickoff)[:10]
        except Exception:
            day = ""
    # Re-check source access before the fixture-level cache too.
    if not any(_prognozist_allowed(url)
               for url in (_PROGNOZIST_URL, _PROGNOZIST_MATCH_CENTRE)):
        if logs is not None:
            logs.append("Prognozist: доступ заблокирован robots.txt или robots.txt недоступен; линия не загружается.")
        return None
    cache_key = f"prognozist_marathon_v5_{day}_{hn}_{an}"
    cached = cache_get(cache_key, 180)
    if cached is not None:
        return cached or None
    try:
        matches = []
        matched_items = []
        line_index = _prognozist_line_index()
        for item in line_index:
            row_text = item.get("text", "")
            if not _prognozist_match([row_text], home, away):
                continue
            odds = item.get("odds")
            if not isinstance(odds, dict) or not all(k in odds for k in ("П1", "X", "П2")):
                continue
            if not all(1.01 < float(odds[k]) <= 100 for k in ("П1", "X", "П2")):
                continue
            # If the line contains a parseable date and a kickoff was supplied,
            # do not accept a different calendar date. Rows without dates remain
            # eligible, but multiple matching rows are rejected below.
            if day:
                # Parse dates from cells after removing odds. Compact labels such as
                # "П1 2.12" resemble "12.12"; in separate-cell rows, prices like
                # "3.10" can also look like a day/month date.
                row_cells = item.get("cells")
                if not isinstance(row_cells, list) or not row_cells:
                    row_cells = [row_text]
                date_cells = []
                prices = []
                for value in odds.values():
                    try:
                        prices.append(float(value))
                    except (TypeError, ValueError):
                        continue
                for cell in row_cells:
                    cell_text = str(cell or "")
                    cell_number = _prognozist_num(cell_text)
                    if cell_number is not None and any(
                        abs(cell_number - price) < 0.0001 for price in prices
                    ):
                        continue
                    cell_text = re.sub(
                        r"(?:П1|П2|X)\s*\d{1,3}(?:[.,]\d{1,4})?",
                        " ",
                        cell_text,
                        flags=re.IGNORECASE,
                    )
                    date_cells.append(cell_text)
                date_text = " ".join(date_cells)
                date_tokens = re.findall(
                    r"(?<!\d)(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?(?!\d)",
                    date_text,
                )
                dated = []
                for dd, mm, yy in date_tokens:
                    year = int(yy) if yy else int(day[:4])
                    if year < 100:
                        year += 2000
                    try:
                        dated.append(datetime(year, int(mm), int(dd)).strftime("%Y-%m-%d"))
                    except ValueError:
                        continue
                if dated and day not in dated:
                    continue
            matches.append({"П1": float(odds["П1"]), "X": float(odds["X"]), "П2": float(odds["П2"])})
            matched_items.append(item)
        # Same team names can occur in youth/cup/league fixtures. Without a
        # unique fixture identity, fail closed instead of attaching the wrong line.
        unique = {tuple(round(item[k], 4) for k in ("П1", "X", "П2")) for item in matches}
        if len(unique) == 1 and matches:
            result = dict(matches[0])
            # Fetch the detail page linked to this exact 1X2 row.
            matched_item = matched_items[0] if matched_items else {}
            href = str(matched_item.get("href") or "").strip()
            cells = matched_item.get("cells")
            if not href and isinstance(cells, list) and cells:
                href = str(cells[0] or "").strip()
            if href.startswith("/mc/football/") or href.startswith("https://prognozist.ru/mc/football/"):
                detail = _prognozist_fetch_detail_fixture(
                    href, odds=result,
                    today=datetime.strptime(day, "%Y-%m-%d").date() if day else None,
                    permission_granted=True,
                )
                if isinstance(detail, dict):
                    for market_key in ("total_market_odds", "btts_market_odds", "stat_market_odds"):
                        market_values = detail.get(market_key)
                        if isinstance(market_values, dict) and market_values:
                            result[market_key] = market_values
                    if detail.get("extra_market_source"):
                        result["extra_market_source"] = str(detail["extra_market_source"])
            cache_put(cache_key, result)
            return result
        if len(unique) > 1:
            log.warning("Prognozist: ambiguous fixture match for %s — %s; refusing to attach odds", home, away)
            if logs is not None:
                logs.append(f"Prognozist: найдено несколько разных линий для {home} — {away}; безопасно пропускаю.")
        elif not matches and logs is not None:
            if line_index:
                logs.append(
                    f"Prognozist: получено строк с коэффициентами 1X2: {len(line_index)}, "
                    f"но совпадение для {home} — {away} не найдено."
                )
            else:
                logs.append(
                    "Prognozist: источник доступен, но парсер не извлёк строки с коэффициентами; "
                    "проверьте журнал источника/изменение HTML."
                )
    except Exception as exc:
        log.debug("Prognozist lookup failed: %s", type(exc).__name__)
    cache_put(cache_key, None)
    return None


# ============ THERUNDOWN FREE SOCCER ODDS ============
THERUNDOWN_SPORT_BY_DIV = {
    "E0": 11, "SP1": 14, "I1": 15, "D1": 13, "F1": 12,
    "C1": 16, "EL": 33, "MLS": 10, "J1": 19,
}


def _american_to_decimal(value):
    """Convert American moneyline prices to decimal odds."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value == 0:
        return None
    decimal = 1.0 + (value / 100.0 if value > 0 else 100.0 / abs(value))
    return decimal if 1.01 < decimal <= 100 else None


def therundown_odds_fixture(home: str, away: str, match_date: str,
                            div_code: str, api_key: str, logs=None, league_name: str = "") -> Optional[dict]:
    """Get a real pre-match 1X2 line and report why a lookup failed."""
    sport_id = THERUNDOWN_SPORT_BY_DIV.get((div_code or "").upper())
    if not sport_id:
        league = (league_name or "").lower()
        league_aliases = (
            (("premier league", "english premier", "epl"), "E0"),
            (("la liga", "laliga", "spanish la liga"), "SP1"),
            (("serie a", "italian serie a"), "I1"),
            (("bundesliga", "german bundesliga"), "D1"),
            (("ligue 1", "french ligue 1"), "F1"),
            (("champions league", "uefa champions"), "C1"),
            (("europa league", "uefa europa league"), "EL"),
            (("major league soccer", "mls"), "MLS"),
            (("j1 league", "j-league", "japan"), "J1"),
        )
        for markers, code in league_aliases:
            if any(marker in league for marker in markers):
                sport_id = THERUNDOWN_SPORT_BY_DIV.get(code)
                if sport_id:
                    break
    if not api_key:
        if logs is not None:
            logs.append("TheRundown: пропущен запрос — API-ключ пустой")
        return None
    if not sport_id:
        if logs is not None:
            logs.append(f"TheRundown: лига не поддерживается (Div={div_code!r}); для неё не запрашивались коэффициенты")
        return None
    if not match_date:
        if logs is not None:
            logs.append("TheRundown: пропущен запрос — дата матча не распознана")
        return None

    ck = f"therundown_v2_{sport_id}_{match_date}"
    events = cache_get(ck, 300)
    if events is None:
        try:
            response = _sess.get(
                f"https://therundown.io/api/v2/sports/{sport_id}/events/{match_date}",
                headers={"X-TheRundown-Key": api_key, "Accept": "application/json"},
                params={"market_ids": "1,2,3,41,42,43", "affiliate_ids": "3,19,23",
                        "main_line": "true", "hide_closed": "true"},
                timeout=(5, 15), proxies=NO_PROXY,
            )
            if response.status_code != 200:
                detail = (response.text or "").replace("\\n", " ")[:160]
                message = f"TheRundown HTTP {response.status_code} (sport={sport_id}, date={match_date})"
                if detail:
                    message += f": {detail}"
                log.warning(message)
                if logs is not None:
                    logs.append("⚠️ " + message)
                # Do not cache failures: a transient 429/5xx must not poison the next scan.
                return None
            payload = response.json()
            events = payload.get("events", []) if isinstance(payload, dict) else []
            events = events if isinstance(events, list) else []
            cache_put(ck, events)
            if logs is not None:
                logs.append(f"TheRundown: API ответил успешно, событий {len(events)} (sport={sport_id}, {match_date})")
        except Exception as exc:
            message = f"TheRundown request failed: {type(exc).__name__}: {str(exc)[:100]}"
            log.warning(message)
            if logs is not None:
                logs.append("⚠️ " + message)
            return None

    try:
        from data.marathon import _find_team_match
        hn, an = _norm_name(home), _norm_name(away)
        for event in events or []:
            teams = event.get("teams") or []
            home_name = next((t.get("name", "") for t in teams if t.get("is_home")), "")
            away_name = next((t.get("name", "") for t in teams if not t.get("is_home")), "")
            if not home_name or not away_name:
                continue
            if not (_find_team_match(home, home_name) and _find_team_match(away, away_name)):
                eh, ea = _norm_name(home_name), _norm_name(away_name)
                if not (hn == eh and an == ea):
                    continue
            # Use one sportsbook's complete 1X2 market, never combine books.
            for book_markets in (event.get("lines") or {}).values():
                moneyline = book_markets.get("moneyline") or {}
                p1 = _american_to_decimal(moneyline.get("home"))
                draw = _american_to_decimal(moneyline.get("draw", moneyline.get("tie")))
                p2 = _american_to_decimal(moneyline.get("away"))
                if p1 and draw and p2:
                    return {"П1": p1, "X": draw, "П2": p2, "source": "therundown"}
            if logs is not None:
                logs.append(f"TheRundown: матч найден ({home_name} — {away_name}), но полной линии 1X2 нет")
            return None
        if logs is not None:
            logs.append(f"TheRundown: среди {len(events or [])} событий не найден матч {home} — {away}")
    except Exception as exc:
        message = f"TheRundown parse failed: {type(exc).__name__}: {str(exc)[:100]}"
        log.warning(message)
        if logs is not None:
            logs.append("⚠️ " + message)
    return None


# ============ THE ODDS API ============
def odds_api_fixture(sport_key: str, home: str, away: str,
                     api_key: str, logs=None) -> Optional[dict]:
    """Fetch real market odds and explain misses in the scanner diagnostics."""
    if not api_key or not sport_key:
        if logs:
            logs.append("⚠️ The Odds API: отсутствует ключ или sport key")
        return None
    # Odds move much faster than historical fixtures; a two-hour cache
    # can make a visible "real" price misleadingly stale. Keep live lines short.
    odds_cache_ttl = 300
    ck = f"odds_v3_{sport_key}_{_norm_name(home)}_{_norm_name(away)}"
    cached = cache_get(ck, odds_cache_ttl)
    if cached is not None:
        if logs:
            logs.append(f"The Odds API: матч найден в кэше ({sport_key})" if cached else f"The Odds API: пустой кэш матча ({sport_key})")
        return cached or None
    # Check the per-sport cache before the local quota: one API request provides
    # the whole league's events, so every fixture must reuse that response.
    events_key = f"odds_events_v3_{sport_key}"
    events = cache_get(events_key, odds_cache_ttl)
    if events is None:
        if usage.odds_remaining() <= 0:
            if logs:
                logs.append("⚠️ The Odds API: локальный лимит запросов исчерпан")
            return None
        try:
            odds_url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
            # /sports/{sport}/odds supports featured markets. BTTS is an additional
            # market on event-specific endpoints and made the entire league request
            # fail with HTTP 422, including the otherwise valid 1X2 prices.
            params = {"apiKey": api_key, "regions": "eu",
                      "markets": "h2h,totals", "oddsFormat": "decimal"}
            r = _sess.get(odds_url, params=params, timeout=20, proxies=NO_PROXY)
            usage.odds_increment(1)
            if r.status_code == 422 and "INVALID_MARKET" in (r.text or ""):
                if logs is not None:
                    logs.append(
                        f"ℹ️ The Odds API ({sport_key}): featured markets h2h/totals rejected; "
                        "повторяю запрос только с h2h для восстановления линии 1X2."
                    )
                if usage.odds_remaining() <= 0:
                    if logs is not None:
                        logs.append("⚠️ The Odds API: лимит исчерпан до fallback-запроса h2h")
                    return None
                r = _sess.get(
                    odds_url,
                    params={"apiKey": api_key, "regions": "eu",
                            "markets": "h2h", "oddsFormat": "decimal"},
                    timeout=20, proxies=NO_PROXY,
                )
                usage.odds_increment(1)
            if r.status_code != 200:
                if logs:
                    detail = (r.text or "").replace("\\n", " ")[:180]
                    logs.append(f"⚠️ The Odds API HTTP {r.status_code} ({sport_key}): {detail}")
                return None
            events = r.json() or []
            if isinstance(events, list):
                cache_put(events_key, events)
                if logs:
                    logs.append(f"The Odds API: получено событий {len(events)} ({sport_key})")
            else:
                events = []
        except Exception as exc:
            if logs:
                logs.append(f"⚠️ The Odds API network/JSON error: {type(exc).__name__}: {str(exc)[:100]}")
            return None
    try:
        hn, an = _norm_name(home), _norm_name(away)
        target = None
        for ev in events:
            eh = _norm_name(ev.get("home_team", ""))
            ea = _norm_name(ev.get("away_team", ""))
            # Require canonical equality: substring matching can attach
            # senior-team odds to U21/reserve fixtures with similar names.
            if eh and ea and eh == hn and ea == an:
                target = ev
                break
        if not target:
            if logs:
                logs.append(f"The Odds API: матч не найден среди {len(events)} событий ({sport_key}) — {home} / {away}")
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
                        # Preserve the exact line supplied by the bookmaker API;
                        # never collapse alternate totals into the 2.5 market.
                        if pt is not None and 0.5 <= pt <= 10.5:
                            line = f"{pt:g}"
                            if on == "Over":
                                acc[f"ТБ {line}"].append(odd)
                            elif on == "Under":
                                acc[f"ТМ {line}"].append(odd)
                    elif name == "btts":
                        if on == "Yes":
                            acc["BTTS да"].append(odd)
                        elif on == "No":
                            acc["BTTS нет"].append(odd)
        out = {k: sum(v) / len(v) for k, v in acc.items() if v}
        # Keep any complete exact totals line, not only 2.5: otherwise a
        # valid market such as 3.5 would be silently discarded by this guard.
        has_totals_pair = any(
            key.startswith("ТБ ") and f"ТМ {key[3:]}" in out
            for key in out
        )
        if not (("П1" in out and "X" in out and "П2" in out) or
                has_totals_pair or
                ("BTTS да" in out and "BTTS нет" in out)):
            out = {}
        cache_put(ck, out or None)
        if logs:
            logs.append(
                f"The Odds API: {len(out)} рынков для {home} — {away}"
                if out else
                f"The Odds API: матч найден, но нужные рынки не пришли — {home} / {away}"
            )
        return out or None
    except Exception as exc:
        if logs:
            logs.append(f"⚠️ The Odds API parse error: {type(exc).__name__}: {str(exc)[:100]}")
        return None


# ============ LOGOS ============
def team_logo_url(team_name: str) -> Optional[str]:
    return None


def league_logo_url(div_code: str) -> Optional[str]:
    return None


# ============ HIGHLIGHTLY ODDS API ============
def _highlightly_base_urls():
    """Try the dedicated Football API first, then the All Sports football route."""
    configured = str(os.environ.get("HIGHLIGHTLY_BASE_URL", "") or "").strip().rstrip("/")
    if configured:
        return [configured]
    return [
        "https://soccer.highlightly.net",
        "https://sports.highlightly.net/football",
    ]


def _highlightly_api_get(path: str, params: dict, api_key: str, preferred_base: str = ""):
    """GET through the dedicated Football API, with All Sports fallback."""
    bases = _highlightly_base_urls()
    if preferred_base:
        bases = [preferred_base] + [base for base in bases if base != preferred_base]
    headers = {"x-rapidapi-key": api_key, "Accept": "application/json"}
    last_response = None
    last_base = bases[0] if bases else ""
    for base in bases:
        last_base = base
        try:
            response = _sess.get(
                f"{base.rstrip('/')}/{path.lstrip('/')}",
                params=params,
                headers=headers,
                timeout=(5, 20),
                proxies=NO_PROXY,
            )
            if response.status_code == 200:
                return response, base
            last_response = response
            body = (response.text or "").casefold()
            # A Basic/Free plan denial is subscription-wide; another host will not
            # unlock odds, so stop rather than spending another request.
            if "not available in basic plan" in body or "not available on basic" in body:
                return response, base
            if response.status_code not in (401, 403, 404):
                return response, base
        except requests.RequestException:
            continue
        except Exception:
            continue
    return last_response, last_base


def _highlightly_fetch_pages(path: str, params: dict, api_key: str,
                             preferred_base: str = "", max_pages: int = 15):
    """Fetch paginated Highlightly data without silently dropping later pages."""
    all_items = []
    current = dict(params or {})
    limit = int(current.get("limit", 100) or 100)
    offset = int(current.get("offset", 0) or 0)
    selected_base = preferred_base
    first_payload = None
    last_response = None
    for _ in range(max_pages):
        page_params = dict(current)
        page_params["limit"] = limit
        page_params["offset"] = offset
        response, used_base = _highlightly_api_get(
            path, page_params, api_key, preferred_base=selected_base
        )
        selected_base = used_base or selected_base
        last_response = response
        if response is None or response.status_code != 200:
            return all_items, first_payload, selected_base, response
        try:
            payload = response.json() or {}
        except Exception:
            return all_items, first_payload, selected_base, response
        if first_payload is None:
            first_payload = payload
        data = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(data, list):
            data = []
        all_items.extend(item for item in data if isinstance(item, dict))
        pagination = payload.get("pagination") or {}
        try:
            total_count = int(pagination.get("totalCount"))
        except (TypeError, ValueError, AttributeError):
            total_count = None
        if not data:
            break
        offset += len(data)
        if total_count is not None:
            if offset >= total_count:
                break
        elif len(data) < limit:
            break
    return all_items, first_payload or {}, selected_base, last_response


def _market_number(market_name: str, fallback=None):
    """Read the line from Highlightly names such as 'Total Goals 2.5'."""
    patterns = (
        r"total\s+goals?\s*[:\-]?\s*([+-]?\d+(?:[.,]\d+)?)",
        r"total\s+corners?\s*[:\-]?\s*([+-]?\d+(?:[.,]\d+)?)",
        r"total\s+cards?\s*[:\-]?\s*([+-]?\d+(?:[.,]\d+)?)",
        r"(?:totals?|over/under|corners?|cards?)\s*[:\-]?\s*([+-]?\d+(?:[.,]\d+)?)",
    )
    text = str(market_name or "").casefold()
    for pattern in patterns:
        found = re.search(pattern, text)
        if found:
            try:
                return float(found.group(1).replace(",", "."))
            except ValueError:
                pass
    try:
        return float(fallback)
    except (TypeError, ValueError):
        return None


def _parse_highlightly_odds(odds_entries: list, wanted_ids: set, logs=None) -> Optional[dict]:
    """Normalize documented Highlightly markets into NEURO BET PRO market keys.

    If multiple bookmakers quote the same selection, keep the best available price.
    Never combine prices from different market names or guess a missing outcome.
    """
    prices = {
        "П1": [], "X": [], "П2": [],
        "total_market_odds": defaultdict(list),
        "btts_market_odds": defaultdict(list),
        "stat_market_odds": defaultdict(list),
        "asian_handicap_market_odds": defaultdict(list),
    }
    found_match = False

    for entry in odds_entries or []:
        if not isinstance(entry, dict) or str(entry.get("matchId", "")) not in wanted_ids:
            continue
        found_match = True
        for market in entry.get("odds") or []:
            if not isinstance(market, dict):
                continue
            market_type = str(market.get("type") or "prematch").casefold()
            if market_type not in ("prematch", "pre-match", ""):
                continue
            market_name = str(market.get("market") or market.get("name") or "").strip()
            mname = market_name.casefold()
            selections = market.get("values") or market.get("odds") or market.get("outcomes") or []
            for outcome in selections:
                if not isinstance(outcome, dict):
                    continue
                label = str(
                    outcome.get("value") or outcome.get("label") or outcome.get("name") or ""
                ).strip().casefold()
                raw_price = outcome.get("odd", outcome.get("price", outcome.get("odds")))
                try:
                    price = float(raw_price)
                except (TypeError, ValueError):
                    continue
                if not 1.01 < price <= 100:
                    continue

                # Full Time Result is the documented 1X2 market.
                if any(term in mname for term in (
                    "full time result", "match result", "1x2", "3-way", "winner"
                )):
                    if label in ("home", "1", "home win"):
                        prices["П1"].append(price)
                    elif label in ("draw", "x", "tie"):
                        prices["X"].append(price)
                    elif label in ("away", "2", "away win"):
                        prices["П2"].append(price)
                    continue

                # Both Teams to Score has Yes/No selections, not a goals line.
                if "both teams" in mname or "btts" in mname or "обе заб" in mname:
                    if label in ("yes", "да", "both", "true"):
                        prices["btts_market_odds"]["BTTS Да"].append(price)
                    elif label in ("no", "нет", "false"):
                        prices["btts_market_odds"]["BTTS Нет"].append(price)
                    continue

                # Asian Handicap variants carry the home and away line in the
                # market label, e.g. "Asian Handicap -1.25/+1.25".
                if "asian handicap" in mname or "asian handicap" in label:
                    nums = re.findall(r"[+-]?\d+(?:[.,]\d+)?", market_name)
                    if nums and label in ("home", "1", "home win"):
                        line = nums[0].replace(",", ".")
                        prices["asian_handicap_market_odds"][f"Ф1 {line}"].append(price)
                    elif nums and label in ("away", "2", "away win"):
                        line = (nums[1] if len(nums) > 1 else str(-float(nums[0].replace(",", ".")))).replace(",", ".")
                        prices["asian_handicap_market_odds"][f"Ф2 {line}"].append(price)
                    continue

                is_over = label in ("over", "more", "больше", "o") or label.startswith(("over ", "больше "))
                is_under = label in ("under", "less", "меньше", "u") or label.startswith(("under ", "меньше "))
                if not (is_over or is_under):
                    continue

                fallback_line = outcome.get("point", outcome.get("handicap", market.get("point")))
                line = _market_number(market_name, fallback_line)
                if line is None:
                    continue
                line_text = f"{line:g}"
                side = "ТБ " if is_over else "ТМ "
                if "corner" in mname or "углов" in mname:
                    prices["stat_market_odds"][("Угл ТБ " if is_over else "Угл ТМ ") + line_text].append(price)
                elif "card" in mname or "booking" in mname or "карточ" in mname:
                    prices["stat_market_odds"][("Карты ТБ " if is_over else "Карты ТМ ") + line_text].append(price)
                elif any(term in mname for term in ("total", "goal", "over/under", "гол")):
                    prices["total_market_odds"][side + line_text].append(price)

    if not found_match:
        return None

    result = {
        "total_market_odds": {},
        "btts_market_odds": {},
        "stat_market_odds": {},
        "asian_handicap_market_odds": {},
        "extra_market_source": "highlightly",
        "source": "highlightly",
    }
    for key in ("П1", "X", "П2"):
        if prices[key]:
            result[key] = max(prices[key])
    for group in (
        "total_market_odds", "btts_market_odds", "stat_market_odds",
        "asian_handicap_market_odds",
    ):
        result[group] = {key: max(values) for key, values in prices[group].items() if values}

    has_1x2 = all(key in result for key in ("П1", "X", "П2"))
    has_other = any(result[key] for key in (
        "total_market_odds", "btts_market_odds", "stat_market_odds",
        "asian_handicap_market_odds",
    ))
    if not has_1x2 and not has_other:
        return None
    if logs is not None:
        market_count = sum(len(result[key]) for key in (
            "total_market_odds", "btts_market_odds", "stat_market_odds",
            "asian_handicap_market_odds",
        )) + (3 if has_1x2 else 0)
        logs.append(f"Highlightly: распознано {market_count} исходов/линий для найденного матча")
    return result


def highlightly_connection_check(api_key: str) -> dict:
    """Check key authentication and real /odds access without exposing the key.

    Also clears a previous Basic-plan circuit breaker after a successful
    post-upgrade check using the same API key.
    """
    global _HIGHLIGHTLY_ODDS_DISABLED_REASON, _HIGHLIGHTLY_ODDS_401_LOGGED
    global _HIGHLIGHTLY_ODDS_DISABLED_KEY

    api_key = str(api_key or "").strip()
    if not api_key:
        return {"ok": False, "matches_ok": False, "message": "API-ключ не найден."}

    key_fingerprint = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    timezone_name = str(os.environ.get("APP_TIMEZONE", "Europe/Warsaw") or "Europe/Warsaw")
    today = datetime.now().strftime("%Y-%m-%d")
    match_response, match_base = _highlightly_api_get(
        "matches",
        {"date": today, "timezone": timezone_name, "limit": 1, "offset": 0},
        api_key,
    )
    matches_ok = bool(match_response is not None and match_response.status_code == 200)

    odds_response, odds_base = _highlightly_api_get(
        "odds",
        {"date": today, "timezone": timezone_name, "oddsType": "prematch", "limit": 1, "offset": 0},
        api_key,
        preferred_base=match_base,
    )
    if odds_response is None:
        return {
            "ok": False, "matches_ok": matches_ok,
            "message": "Не удалось соединиться с Highlightly. Проверь доступность API-хоста.",
        }

    if odds_response.status_code != 200:
        body = (odds_response.text or "").strip().replace("\n", " ")[:220]
        is_basic_denial = (
            odds_response.status_code in (401, 403)
            and any(term in body.casefold() for term in (
                "not available in basic plan", "not available on basic",
                "not available in free plan", "upgrade your plan",
            ))
        )
        if is_basic_denial:
            _HIGHLIGHTLY_ODDS_DISABLED_REASON = "basic_plan"
            _HIGHLIGHTLY_ODDS_DISABLED_KEY = key_fingerprint
            _HIGHLIGHTLY_ODDS_401_LOGGED = True
            return {
                "ok": False, "matches_ok": matches_ok,
                "http_status": odds_response.status_code,
                "base": odds_base,
                "message": (
                    "Ключ дошёл до Highlightly, но тариф Basic/Free не разрешает /odds. "
                    "Нужен платный план, где доступ к bookmaker odds явно включён. "
                    "После активации нажми эту проверку ещё раз."
                ),
            }
        return {
            "ok": False, "matches_ok": matches_ok,
            "http_status": odds_response.status_code,
            "base": odds_base,
            "message": f"Доступ к /odds не подтверждён (HTTP {odds_response.status_code}). {body}",
        }

    try:
        payload = odds_response.json() or {}
    except Exception:
        payload = {}
    plan = payload.get("plan") or {} if isinstance(payload, dict) else {}
    tier = str(plan.get("tier") or "не указан")
    data = payload.get("data", []) if isinstance(payload, dict) else []
    count = len(data) if isinstance(data, list) else 0
    if tier.casefold() in ("free", "basic"):
        _HIGHLIGHTLY_ODDS_DISABLED_REASON = "basic_plan"
        _HIGHLIGHTLY_ODDS_DISABLED_KEY = key_fingerprint
        _HIGHLIGHTLY_ODDS_401_LOGGED = True
        return {
            "ok": False, "matches_ok": matches_ok, "http_status": 200,
            "base": odds_base, "tier": tier, "odds_count": count,
            "message": f"Ключ работает, но тариф {tier} не открывает коэффициенты /odds.",
        }

    # A successful post-upgrade check unlocks fetching again even when the
    # API key itself did not change.
    _HIGHLIGHTLY_ODDS_DISABLED_REASON = ""
    _HIGHLIGHTLY_ODDS_401_LOGGED = False
    _HIGHLIGHTLY_ODDS_DISABLED_KEY = key_fingerprint

    return {
        "ok": True, "matches_ok": matches_ok, "http_status": 200,
        "base": odds_base, "tier": tier, "odds_count": count,
        "message": (
            f"Highlightly подключён; /odds отвечает HTTP 200, тариф {tier}, "
            f"записей на сегодня в первой странице: {count}."
        ),
    }

def highlightly_odds_fixture(match_date: str, home: str, away: str,
                             api_key: str, logs=None) -> Optional[dict]:
    """Fetch prematch odds for one exact fixture from Highlightly Football API."""
    global _HIGHLIGHTLY_ODDS_DISABLED_REASON, _HIGHLIGHTLY_ODDS_401_LOGGED
    global _HIGHLIGHTLY_ODDS_DISABLED_KEY

    if not api_key or not match_date or not home or not away:
        if logs is not None and not api_key:
            logs.append("ℹ️ Highlightly: API-ключ не задан")
        return None

    # If the user changes the API key in Streamlit Secrets or the sidebar, do
    # not keep a circuit-breaker state created by the previous credential.
    key_fingerprint = hashlib.sha256(str(api_key).encode("utf-8")).hexdigest()
    if _HIGHLIGHTLY_ODDS_DISABLED_KEY != key_fingerprint:
        _HIGHLIGHTLY_ODDS_DISABLED_REASON = ""
        _HIGHLIGHTLY_ODDS_401_LOGGED = False
        _HIGHLIGHTLY_ODDS_DISABLED_KEY = key_fingerprint
    if _HIGHLIGHTLY_ODDS_DISABLED_REASON:
        return None

    hn, an = _norm_name(home), _norm_name(away)
    timezone_name = str(os.environ.get("APP_TIMEZONE", "Europe/Warsaw") or "Europe/Warsaw")

    matches_key = f"highlightly_matches_v2_{match_date}_{timezone_name}"
    matches = cache_get(matches_key, 900)
    matches_base = ""
    if matches is None:
        matches, _match_payload, matches_base, match_response = _highlightly_fetch_pages(
            "matches",
            {"date": match_date, "timezone": timezone_name, "limit": 100, "offset": 0},
            api_key,
        )
        if match_response is None or match_response.status_code != 200:
            status = getattr(match_response, "status_code", "connection error")
            body = (getattr(match_response, "text", "") or "")[:140]
            if logs is not None:
                logs.append(f"⚠️ Highlightly matches HTTP {status}: {body}")
            return None
        cache_put(matches_key, matches)
    else:
        matches_base = "https://soccer.highlightly.net"

    match_ids = []
    for match in matches:
        if not isinstance(match, dict):
            continue
        home_obj = match.get("homeTeam") or {}
        away_obj = match.get("awayTeam") or {}
        home_name = home_obj.get("name") or match.get("homeTeamName") or ""
        away_name = away_obj.get("name") or match.get("awayTeamName") or ""
        if home_name and away_name and _norm_name(home_name) == hn and _norm_name(away_name) == an:
            match_id = match.get("id", match.get("matchId"))
            if match_id is not None:
                match_ids.append(str(match_id))

    if not match_ids:
        if logs is not None:
            logs.append(f"Highlightly: матч не найден в /matches за {match_date}: {home} — {away}")
        return None

    odds_key = f"highlightly_odds_v4_{match_date}_{timezone_name}"
    odds_entries = cache_get(odds_key, 900)
    if odds_entries is None:
        odds_entries, odds_payload, odds_base, odds_response = _highlightly_fetch_pages(
            "odds",
            {"date": match_date, "timezone": timezone_name,
             "oddsType": "prematch", "limit": 100, "offset": 0},
            api_key,
            preferred_base=matches_base,
        )
        if odds_response is None or odds_response.status_code != 200:
            status = getattr(odds_response, "status_code", "connection error")
            body = (getattr(odds_response, "text", "") or "")[:180]
            if status in (401, 403) or (
                odds_response is not None and
                "not available in basic plan" in (getattr(odds_response, "text", "") or "").casefold()
            ):
                _HIGHLIGHTLY_ODDS_DISABLED_REASON = "odds_access_denied"
            if logs is not None and not _HIGHLIGHTLY_ODDS_401_LOGGED:
                logs.append(f"⛔ Highlightly /odds не дал доступ (HTTP {status}): {body}")
                _HIGHLIGHTLY_ODDS_401_LOGGED = True
            return None
        plan = (odds_payload or {}).get("plan") or {}
        tier = str(plan.get("tier") or "")
        if tier.casefold() in ("free", "basic"):
            _HIGHLIGHTLY_ODDS_DISABLED_REASON = "basic_plan"
            if logs is not None and not _HIGHLIGHTLY_ODDS_401_LOGGED:
                logs.append(f"⛔ Highlightly: тариф {tier} не предоставляет /odds.")
                _HIGHLIGHTLY_ODDS_401_LOGGED = True
            return None
        cache_put(odds_key, odds_entries)

    wanted_ids = set(match_ids)
    returned_ids = {
        str(entry.get("matchId", ""))
        for entry in odds_entries
        if isinstance(entry, dict) and entry.get("matchId") is not None
    }
    for missing_id in sorted(wanted_ids - returned_ids):
        per_match_key = f"highlightly_odds_match_v2_{missing_id}"
        per_match = cache_get(per_match_key, 900)
        if per_match is None:
            per_match, per_payload, _per_base, per_response = _highlightly_fetch_pages(
                "odds",
                {"matchId": missing_id, "oddsType": "prematch", "limit": 100, "offset": 0},
                api_key,
                preferred_base=matches_base,
            )
            if per_response is not None and per_response.status_code in (401, 403):
                _HIGHLIGHTLY_ODDS_DISABLED_REASON = "odds_access_denied"
                if logs is not None and not _HIGHLIGHTLY_ODDS_401_LOGGED:
                    body = (per_response.text or "")[:180]
                    logs.append(
                        f"⛔ Highlightly /odds matchId={missing_id}: "
                        f"HTTP {per_response.status_code}: {body}"
                    )
                    _HIGHLIGHTLY_ODDS_401_LOGGED = True
                return None
            if per_response is None or per_response.status_code != 200:
                if logs is not None:
                    status = getattr(per_response, "status_code", "connection error")
                    logs.append(f"⚠️ Highlightly odds matchId={missing_id}: HTTP {status}")
                continue
            per_plan = (per_payload or {}).get("plan") or {}
            if str(per_plan.get("tier") or "").casefold() in ("free", "basic"):
                _HIGHLIGHTLY_ODDS_DISABLED_REASON = "basic_plan"
                return None
            cache_put(per_match_key, per_match)
        odds_entries.extend(
            entry for entry in per_match
            if isinstance(entry, dict) and str(entry.get("matchId", "")) == missing_id
        )

    result = _parse_highlightly_odds(odds_entries, wanted_ids, logs=logs)
    if not result:
        if logs is not None:
            logs.append(f"Highlightly: в ответе нет коэффициентов для {home} — {away}")
        return None
    if logs is not None:
        markets = (
            len(result.get("total_market_odds") or {})
            + len(result.get("btts_market_odds") or {})
            + len(result.get("stat_market_odds") or {})
            + len(result.get("asian_handicap_market_odds") or {})
        )
        has_1x2 = all(key in result for key in ("П1", "X", "П2"))
        logs.append(
            f"✅ Highlightly: {home} — {away} | "
            f"{'П1/X/П2 + ' if has_1x2 else ''}{markets} доп. рынков/линий"
        )
    return result



# ============ SSTATS.NET FREE FOOTBALL API ============
SSTATS_API_BASE = "https://api.sstats.net"


def _sstats_unwrap(payload):
    """Unwrap the documented ApiResponse envelope without assuming non-null data."""
    if isinstance(payload, dict):
        if str(payload.get("status") or "").casefold() == "error":
            raise ValueError(str(payload.get("message") or "SStats API returned an error"))
        return payload.get("data")
    return payload


def _sstats_timezone():
    """Return app-local date and integer UTC offset required by SStats."""
    zone_name = str(os.environ.get("APP_TIMEZONE", "Europe/Warsaw") or "Europe/Warsaw")
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo(zone_name))
        offset = now.utcoffset()
        offset_hours = int(round(offset.total_seconds() / 3600)) if offset else 3
        return now, max(-12, min(12, offset_hours))
    except Exception:
        return datetime.now(), 3


def _sstats_float(value):
    try:
        number = float(str(value).replace(",", "."))
        return number if 1.01 < number <= 100 else None
    except (TypeError, ValueError):
        return None


def _sstats_line(text):
    found = re.search(r"(?<!\d)(\d{1,2}(?:[.,]\d{1,2})?)(?!\d)", str(text or ""))
    if not found:
        return None
    try:
        return float(found.group(1).replace(",", "."))
    except ValueError:
        return None


def _sstats_flat_bets(odds_payload):
    """Yield (market, bookmaker) from compact Game.odds or full Odds response."""
    if isinstance(odds_payload, dict):
        odds_payload = odds_payload.get("data", odds_payload.get("odds", []))
    if not isinstance(odds_payload, list):
        return
    for item in odds_payload:
        if not isinstance(item, dict):
            continue
        # Compact /Games/list response: item itself is ApiSaBet.
        if "marketName" in item or "marketId" in item:
            yield item, ""
            continue
        # Detailed /Odds/{gameId} response: bookmaker object contains ApiSaBet[].
        nested = item.get("odds")
        bookmaker = str(item.get("bookmakerName") or "").strip()
        if isinstance(nested, list):
            for bet in nested:
                if isinstance(bet, dict):
                    yield bet, bookmaker


def _sstats_parse_markets(odds_payload):
    """Normalize SStats marketName + Price(name,value) records to app market keys.

    For detailed bookmaker responses, bookmaker_by_pick records the bookmaker
    offering the selected best decimal price. Compact fixtures may not include it.
    """
    prices = {
        "П1": [], "X": [], "П2": [],
        "total_market_odds": {},
        "btts_market_odds": {},
        "stat_market_odds": {},
        "asian_handicap_market_odds": {},
    }
    bookmaker_by_pick = {}

    def _record(selection, odd, bookmaker, group=None):
        target = prices[group] if group else prices
        values = target.setdefault(selection, []) if group else target[selection]
        values.append(odd)
        if bookmaker:
            best = max(values)
            if odd >= best:
                bookmaker_by_pick[selection] = bookmaker

    for bet, bookmaker in _sstats_flat_bets(odds_payload):
        market_name = str(bet.get("marketName") or bet.get("name") or "").strip()
        market = market_name.casefold()
        price_rows = bet.get("odds") or bet.get("values") or bet.get("outcomes") or []
        if not isinstance(price_rows, list):
            continue

        for price_row in price_rows:
            if not isinstance(price_row, dict):
                continue
            label = str(price_row.get("name") or price_row.get("label") or price_row.get("valueName") or "").strip()
            label_norm = label.casefold()
            odd = _sstats_float(price_row.get("value", price_row.get("odd", price_row.get("price"))))
            if odd is None:
                continue

            # 1X2 / match-result market.
            is_1x2 = any(term in market for term in (
                "1x2", "match result", "full time result", "match winner",
                "winner", "исход матча", "результат матча",
            ))
            if is_1x2:
                if label_norm in ("1", "home", "home win", "п1", "1 (home)"):
                    _record("П1", odd, bookmaker)
                elif label_norm in ("x", "х", "draw", "tie", "ничья"):
                    _record("X", odd, bookmaker)
                elif label_norm in ("2", "away", "away win", "п2", "2 (away)"):
                    _record("П2", odd, bookmaker)
                continue

            # Both teams to score.
            if any(term in market for term in ("both teams", "btts", "обе заб", "обе команды забьют")):
                if label_norm in ("yes", "да", "both", "true"):
                    _record("BTTS Да", odd, bookmaker, "btts_market_odds")
                elif label_norm in ("no", "нет", "false"):
                    _record("BTTS Нет", odd, bookmaker, "btts_market_odds")
                continue

            # Totals, including corners/cards; the line may be encoded in either
            # marketName or Price.name (e.g. "Total Goals" + "Over 2.5").
            is_over = (
                label_norm in ("over", "more", "больше", "o")
                or label_norm.startswith(("over ", "больше ", "o "))
            )
            is_under = (
                label_norm in ("under", "less", "меньше", "u")
                or label_norm.startswith(("under ", "меньше ", "u "))
            )
            if is_over or is_under:
                line = _sstats_line(label) or _sstats_line(market_name)
                if line is None:
                    continue
                line_text = f"{line:g}"
                if any(term in market for term in ("corner", "corners", "углов")):
                    key = ("Угл ТБ " if is_over else "Угл ТМ ") + line_text
                    _record(key, odd, bookmaker, "stat_market_odds")
                elif any(term in market for term in ("card", "booking", "карточ")):
                    key = ("Карты ТБ " if is_over else "Карты ТМ ") + line_text
                    _record(key, odd, bookmaker, "stat_market_odds")
                elif any(term in market for term in ("total", "goal", "over/under", "тотал", "гол")):
                    key = ("ТБ " if is_over else "ТМ ") + line_text
                    _record(key, odd, bookmaker, "total_market_odds")
                continue

            # Asian/European handicap — kept separately; never treated as 1X2.
            if any(term in market for term in ("handicap", "asian", "фора")):
                line = _sstats_line(label) or _sstats_line(market_name)
                if line is None:
                    continue
                if any(term in label_norm for term in ("home", " хозя", "п1", "1")):
                    key = f"Ф1 {line:g}"
                elif any(term in label_norm for term in ("away", " гост", "п2", "2")):
                    key = f"Ф2 {line:g}"
                else:
                    continue
                _record(key, odd, bookmaker, "asian_handicap_market_odds")

    result = {
        "total_market_odds": {},
        "btts_market_odds": {},
        "stat_market_odds": {},
        "asian_handicap_market_odds": {},
        "bookmaker_by_pick": bookmaker_by_pick,
        "extra_market_source": "sstats",
        "source": "sstats",
    }
    for key in ("П1", "X", "П2"):
        if prices[key]:
            result[key] = max(prices[key])
    for group in (
        "total_market_odds", "btts_market_odds", "stat_market_odds",
        "asian_handicap_market_odds",
    ):
        result[group] = {
            key: max(values) for key, values in prices[group].items() if values
        }
    return result


def _sstats_request(path: str, api_key: str, params=None, timeout=(5, 25)):
    """GET SStats API using its documented ?apikey= authentication.

    Never include request URLs or exception strings in logs: both can expose
    the query-string API key.
    """
    query = dict(params or {})
    query["apikey"] = str(api_key or "").strip()
    return _sess.get(
        f"{SSTATS_API_BASE}/{str(path).lstrip('/')}",
        params=query,
        headers={"Accept": "application/json"},
        timeout=timeout,
        proxies=NO_PROXY,
    )


def sstats_account_check(api_key: str) -> dict:
    """Verify key, upcoming fixtures endpoint and prematch odds endpoint."""
    key = str(api_key or "").strip()
    if not key:
        return {"ok": False, "message": "SStats API-ключ не найден."}
    try:
        response = _sstats_request("Account/Info", key)
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Не удалось соединиться с SStats: {type(exc).__name__}.",
        }
    if response.status_code != 200:
        return {
            "ok": False, "http_status": response.status_code,
            "message": f"SStats отклонил ключ или запрос: HTTP {response.status_code}.",
        }
    try:
        account_payload = response.json() or {}
        account_data = _sstats_unwrap(account_payload) or {}
    except Exception as exc:
        return {
            "ok": False,
            "http_status": response.status_code,
            "message": f"SStats не подтвердил ключ ({type(exc).__name__}).",
        }
    if not isinstance(account_data, dict) or not account_data.get("userName"):
        message = str(account_payload.get("message") or "SStats не подтвердил аккаунт.")
        return {"ok": False, "message": message[:180]}

    now, timezone_offset = _sstats_timezone()
    params = {
        "From": now.date().isoformat(),
        "To": (now.date() + timedelta(days=14)).isoformat(),
        "Upcoming": "true",
        "Limit": 3,
        "Offset": 0,
        "TimeZone": timezone_offset,
    }
    try:
        games_response = _sstats_request("Games/list", key, params=params)
        if games_response.status_code != 200:
            return {
                "ok": True, "games_ok": False,
                "message": (
                    f"Ключ SStats подтверждён ({account_data.get('userName')}); "
                    f"но /Games/list ответил HTTP {games_response.status_code}."
                ),
            }
        games_payload = games_response.json() or {}
        games = _sstats_unwrap(games_payload) or []
        if not isinstance(games, list):
            games = []
    except Exception as exc:
        return {
            "ok": True, "games_ok": False,
            "message": (
                f"Ключ SStats подтверждён ({account_data.get('userName')}), "
                f"но проверка расписания завершилась ошибкой {type(exc).__name__}."
            ),
        }

    odds_count = 0
    odds_checked = False
    if games and isinstance(games[0], dict):
        game_id = games[0].get("id")
        if game_id is not None:
            try:
                odds_response = _sstats_request(f"Odds/{game_id}", key, params={"opening": "false"})
                if odds_response.status_code == 200:
                    odds_payload = _sstats_unwrap(odds_response.json() or {}) or []
                    if isinstance(odds_payload, list):
                        odds_count = sum(
                            len(item.get("odds") or [])
                            for item in odds_payload if isinstance(item, dict)
                        )
                    odds_checked = True
            except Exception:
                pass

    return {
        "ok": True, "games_ok": True, "games_count": len(games),
        "odds_checked": odds_checked, "odds_count": odds_count,
        "message": (
            f"SStats подключён: ключ подтверждён для {account_data.get('userName')}; "
            f"расписание отвечает, ближайших матчей в тестовой выборке: {len(games)}; "
            + (
                f"в Odds найдены {odds_count} рынков/линий."
                if odds_checked and odds_count
                else "проверка Odds прошла, но в тестовом матче котировки не найдены."
                if odds_checked
                else "проверка полного Odds для тестового матча пока не подтверждена."
            )
        ),
    }


def sstats_upcoming_matches(days: int, api_key: str, logs=None) -> list:
    """Load upcoming fixtures and compact 1X2/total odds in one SStats request."""
    key = str(api_key or "").strip()
    if not key:
        if logs is not None:
            logs.append("ℹ️ SStats: API-ключ SSTATS_API_KEY не задан")
        return []
    now, timezone_offset = _sstats_timezone()
    horizon = max(1, min(30, int(days or 3)))
    cache_key = (
        f"sstats_upcoming_v1_{now.date().isoformat()}_{horizon}_"
        f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"
    )
    cached = cache_get(cache_key, 900)
    if cached is not None:
        if logs is not None:
            logs.append(f"SStats: использован кэш расписания ({len(cached)} матчей)")
        return cached if isinstance(cached, list) else []

    params = {
        "From": now.date().isoformat(),
        "To": (now.date() + timedelta(days=horizon)).isoformat(),
        "Upcoming": "true",
        "Limit": 1000,
        "Offset": 0,
        "TimeZone": timezone_offset,
        "Order": 1,
    }
    try:
        response = _sstats_request("Games/list", key, params=params, timeout=(5, 30))
        if response.status_code != 200:
            if logs is not None:
                logs.append(f"⚠️ SStats /Games/list HTTP {response.status_code}")
            return []
        payload = response.json() or {}
        records = _sstats_unwrap(payload)
        if not isinstance(records, list):
            if logs is not None:
                logs.append("⚠️ SStats /Games/list: неожиданный формат data")
            return []
    except Exception as exc:
        if logs is not None:
            logs.append(f"⚠️ SStats /Games/list: {type(exc).__name__}")
        return []

    rows = []
    for game in records:
        if not isinstance(game, dict):
            continue
        home_obj = game.get("homeTeam") or {}
        away_obj = game.get("awayTeam") or {}
        home = str(home_obj.get("name") or game.get("homeTeamName") or "").strip()
        away = str(away_obj.get("name") or game.get("awayTeamName") or "").strip()
        raw_date = str(game.get("date") or "")
        if not home or not away or not raw_date:
            continue
        kickoff = parse_date(raw_date)
        if not kickoff:
            continue
        if not (now.replace(tzinfo=None) - timedelta(minutes=5) <= kickoff.replace(tzinfo=None)
                < now.replace(tzinfo=None) + timedelta(days=horizon)):
            continue

        league_obj = (game.get("season") or {}).get("league") or {}
        league_name = str(
            league_obj.get("name")
            or game.get("leagueName")
            or game.get("league")
            or "SStats"
        ).strip()
        div = _sstats_div_code(league_name)
        normalized = _sstats_parse_markets(game.get("odds") or [])
        compact_1x2 = {
            key: normalized[key]
            for key in ("П1", "X", "П2")
            if normalized.get(key) is not None
        }
        start_time = raw_date[11:16] if len(raw_date) >= 16 else ""
        row = {
            "Date": raw_date[:19],
            "Time": start_time,
            "HomeTeam": home,
            "AwayTeam": away,
            "Div": div,
            "League": league_name,
            "fixture_id": f"sstats:{game.get('id')}",
            "sstats_game_id": game.get("id"),
            "source": "sstats",
            "odds_source": "sstats" if compact_1x2 else "no_market",
            "market_odds": compact_1x2,
            "bookmaker_by_pick": normalized.get("bookmaker_by_pick") or {},
            "total_market_odds": normalized.get("total_market_odds") or {},
            "btts_market_odds": normalized.get("btts_market_odds") or {},
            "stat_market_odds": normalized.get("stat_market_odds") or {},
            "asian_handicap_market_odds": normalized.get("asian_handicap_market_odds") or {},
            "total_odds_source": (
                "sstats" if normalized.get("total_market_odds") else "no_market"
            ),
            "extra_market_source": (
                "sstats"
                if any(normalized.get(name) for name in (
                    "total_market_odds", "btts_market_odds", "stat_market_odds",
                    "asian_handicap_market_odds",
                ))
                else ""
            ),
            "sstats_status": game.get("status"),
        }
        rows.append(row)

    # Deduplicate repeated records by kickoff date and ordered teams.
    unique = {}
    for row in rows:
        key_tuple = (
            str(row.get("Date") or "")[:10],
            _norm_name(row.get("HomeTeam") or ""),
            _norm_name(row.get("AwayTeam") or ""),
        )
        if key_tuple not in unique:
            unique[key_tuple] = row
            continue
        current = unique[key_tuple]
        current["total_market_odds"].update(row.get("total_market_odds") or {})
        current["market_odds"].update(row.get("market_odds") or {})
        current["btts_market_odds"].update(row.get("btts_market_odds") or {})
        current["stat_market_odds"].update(row.get("stat_market_odds") or {})
    rows = list(unique.values())
    cache_put(cache_key, rows)
    if logs is not None:
        quoted = sum(bool(row.get("market_odds")) for row in rows)
        totals = sum(bool(row.get("total_market_odds")) for row in rows)
        logs.append(
            f"✅ SStats: загружено {len(rows)} предстоящих матчей; "
            f"1X2 у {quoted}, тоталы у {totals}."
        )
    return rows


def _sstats_div_code(league_name: str) -> str:
    """Map common SStats league labels to the model's historical division codes."""
    name = str(league_name or "").casefold()
    mappings = (
        (("premier league", "premier league of england", "england premier"), "E0"),
        (("championship", "england championship"), "E1"),
        (("league one", "england league one"), "E2"),
        (("league two", "england league two"), "E3"),
        (("la liga", "primera division", "spain primera"), "SP1"),
        (("segunda division", "la liga 2", "spain segunda"), "SP2"),
        (("serie a", "italy serie a"), "I1"),
        (("serie b", "italy serie b"), "I2"),
        (("bundesliga", "germany bundesliga"), "D1"),
        (("2. bundesliga", "2 bundesliga"), "D2"),
        (("ligue 1", "france ligue 1"), "F1"),
        (("ligue 2", "france ligue 2"), "F2"),
        (("eredivisie", "netherlands eredivisie"), "N1"),
        (("primeira liga", "portugal primeira"), "P1"),
        (("jupiler pro league", "belgian pro league"), "B1"),
        (("super lig", "turkey super lig"), "T1"),
    )
    for tokens, code in mappings:
        if any(token in name for token in tokens):
            return code
    return "G"


def sstats_odds_fixture(game_id, api_key: str, logs=None) -> Optional[dict]:
    """Fetch full prematch bookmaker markets for one SStats fixture."""
    key = str(api_key or "").strip()
    if not key or game_id in (None, ""):
        return None
    cache_key = (
        f"sstats_full_odds_v1_{game_id}_"
        f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"
    )
    cached = cache_get(cache_key, 1800)
    if cached is not None:
        return cached if isinstance(cached, dict) else None
    try:
        response = _sstats_request(
            f"Odds/{game_id}", key, params={"opening": "false"}, timeout=(5, 25)
        )
        if response.status_code != 200:
            if logs is not None:
                logs.append(f"⚠️ SStats Odds/{game_id}: HTTP {response.status_code}")
            return None
        payload = response.json() or {}
        data = _sstats_unwrap(payload)
        if not isinstance(data, list):
            return None
        result = _sstats_parse_markets(data)
        result["bookmakers"] = [
            str(item.get("bookmakerName") or "")
            for item in data
            if isinstance(item, dict) and item.get("bookmakerName")
        ]
        market_count = sum(
            len(result.get(group) or {})
            for group in (
                "total_market_odds", "btts_market_odds", "stat_market_odds",
                "asian_handicap_market_odds",
            )
        ) + sum(key_name in result for key_name in ("П1", "X", "П2"))
        cache_put(cache_key, result)
        if logs is not None:
            logs.append(
                f"SStats Odds/{game_id}: {len(data)} букмекеров, "
                f"{market_count} исходов/линий."
            )
        return result
    except Exception as exc:
        if logs is not None:
            logs.append(f"⚠️ SStats Odds/{game_id}: {type(exc).__name__}")
        return None
