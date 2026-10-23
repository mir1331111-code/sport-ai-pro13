"""data/sources.py — TheSportsDB + football-data.org + Odds API + logos (v4).

Совместим с App.py v13:
  * parse_date, fdorg_match_result    — для авто-сеттла
  * fdorg_matches, tsdb_matches       — для сканера
  * load_seasonal, season_str         — для обучения модели
  * odds_api_fixture                  — для реальных кэфов

Что изменено в v4:
  * USL Championship (США) больше не мапится на English Championship (E1).
    Раньше "championship" в названии лиги = E1 — это ломало USL-матчи.
  * Добавлены USA-лиги: USL Championship, USL League One, MLS.
  * Дополнительно распознаются шотландский, австралийский, японский чемпионаты.
"""
from __future__ import annotations
import csv, io, re, logging, os
from collections import defaultdict
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


def tsdb_matches(days: int = 7, logs=None) -> list:
    """Сбор матчей из TheSportsDB на N дней вперёд."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    out = []
    total_days = min(max(1, int(days)), 14)

    for off in range(total_days):
        d = today + timedelta(days=off)
        dstr = d.strftime("%Y-%m-%d")
        ck = f"tsdb_day_v9_{dstr}"
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
                    "home_badge": e.get("strHomeTeamBadge"),
                    "away_badge": e.get("strAwayTeamBadge"),
                    "league_badge": e.get("strLeagueBadge"),
                    "women": women,
                    "national": national,
                    "kind": kind,
                })
            out += rows
            cache_put(ck, rows)
            if logs:
                logs.append(f"TSDB {dstr}: {len(rows)}")
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
    """Collect table rows and links from Prognozist pages."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows, self.row, self.cell = [], None, None
        self.links, self.link = [], None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        if tag == "a":
            self.link = {"href": attrs.get("href", ""), "text": []}

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
                self.rows.append(self.row)
            self.row = None
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
    joined = " ".join(str(x) for x in row)
    normalized = _norm_name(joined)
    hn, an = _norm_name(home), _norm_name(away)
    # Compare both full row and URL slug; tolerate transliteration and
    # common short/long team-name variants without matching single letters.
    if not hn or not an:
        return False
    home_ok = hn in normalized or (len(hn) >= 5 and hn in normalized)
    away_ok = an in normalized or (len(an) >= 5 and an in normalized)
    if home_ok and away_ok:
        return True
    # Token overlap handles source names such as "Man United" vs "Manchester United".
    # _norm_name removes spaces, so split() on normalized names never works.
    # Build meaningful tokens from the original team names instead.
    ht = {_norm_name(x) for x in re.findall(r"[A-Za-zА-Яа-я0-9]+", home)
          if len(_norm_name(x)) >= 3}
    at = {_norm_name(x) for x in re.findall(r"[A-Za-zА-Яа-я0-9]+", away)
          if len(_norm_name(x)) >= 3}
    return bool(ht and at and any(x in normalized for x in ht)
                and any(x in normalized for x in at))


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


def _prognozist_allowed(url: str) -> bool:
    """Require explicit permission and robots.txt allowance; cache the robots check."""
    if os.environ.get("PROGNOZIST_PERMISSION", "").strip() != "1":
        log.info("Prognozist odds disabled: set PROGNOZIST_PERMISSION=1 only after checking site terms/permission.")
        return False
    cache_key = "prognozist_robots_allowed_v1"
    cached = cache_get(cache_key, 180)
    if isinstance(cached, dict) and url in cached:
        return bool(cached[url])
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        response = _sess.get(robots_url, timeout=(3, 6), proxies=NO_PROXY)
        if response.status_code != 200:
            log.warning("Prognozist robots.txt unavailable (HTTP %s); disabling source.", response.status_code)
            return False
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())
        agent = "Sport-AI-Pro13/1.0 (+contact the repository maintainer; research client)"
        result = {
            target: bool(parser.can_fetch(agent, target))
            for target in (_PROGNOZIST_URL, _PROGNOZIST_MATCH_CENTRE)
        }
        cache_put(cache_key, result)
        return result.get(url, False)
    except Exception as exc:
        log.warning("Prognozist robots.txt check failed (%s); disabling source.", type(exc).__name__)
        return False

def prognozist_source_status() -> str:
    """Return an operator-facing status without scraping match data."""
    if os.environ.get("PROGNOZIST_PERMISSION", "").strip() != "1":
        return "disabled_permission"
    allowed = [
        url for url in (_PROGNOZIST_URL, _PROGNOZIST_MATCH_CENTRE)
        if _prognozist_allowed(url)
    ]
    return "ready" if allowed else "blocked_or_unavailable"


def _prognozist_line_index():
    """Download and parse the line only when explicitly permitted by the operator and robots.txt."""
    if os.environ.get("PROGNOZIST_PERMISSION", "").strip() != "1":
        _prognozist_allowed(_PROGNOZIST_URL)
        return []
    key = "prognozist_marathon_line_index_v5"
    cached = cache_get(key, 180)
    if isinstance(cached, list):
        return cached
    allowed_urls = [
        url for url in (_PROGNOZIST_URL, _PROGNOZIST_MATCH_CENTRE)
        if _prognozist_allowed(url)
    ]
    if not allowed_urls:
        return []
    headers = {
        "User-Agent": "Sport-AI-Pro13/1.0 (+contact the repository maintainer; research client)",
        "Accept": "text/html,application/json,text/plain,*/*",
        "Referer": "https://prognozist.ru/mc/football/",
    }
    indexed = []
    for url in allowed_urls:
        try:
            response = _sess.get(url, headers=headers, timeout=(3, 6), proxies=NO_PROXY)
            if response.status_code != 200 or not response.text:
                log.warning("Prognozist source %s: HTTP %s, body_len=%s",
                            url, response.status_code, len(response.text or ""))
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
                    indexed.append({"text": text, "odds": odds})
            parsed_count = len(indexed) - before_count
            if parsed_count == 0:
                log.warning("Prognozist source %s: HTTP %s, body_len=%s, rows=%s, odds_rows=0",
                            url, response.status_code, len(body), len(candidates))
            else:
                log.info("Prognozist source %s: HTTP %s, body_len=%s, rows=%s, odds_rows=%s",
                         url, response.status_code, len(body), len(candidates), parsed_count)
            # Keep rows from both feeds: the first endpoint can return a partial
            # line that does not include the fixture being searched.
        except Exception as exc:
            log.warning("Prognozist request failed (%s): %s", url, type(exc).__name__)
    cache_put(key, indexed)
    return indexed


def prognozist_marathon_odds(home: str, away: str, kickoff=None) -> Optional[dict]:
    """Return a complete, unambiguous real 1X2 line; never use cache to bypass permission."""
    if not home or not away:
        return None
    # Check permission before reading match or line caches, so disabling the
    # source takes effect immediately and cached data cannot bypass the gate.
    if os.environ.get("PROGNOZIST_PERMISSION", "").strip() != "1":
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
    cache_key = f"prognozist_marathon_v4_{day}_{hn}_{an}"
    cached = cache_get(cache_key, 180)
    if cached is not None:
        return cached or None
    try:
        matches = []
        for item in _prognozist_line_index():
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
                date_tokens = re.findall(r"(?<!\d)(\d{1,2})[./-](\d{1,2})(?:[./-](\d{2,4}))?(?!\d)", row_text)
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
        # Same team names can occur in youth/cup/league fixtures. Without a
        # unique fixture identity, fail closed instead of attaching the wrong line.
        unique = {tuple(round(item[k], 4) for k in ("П1", "X", "П2")) for item in matches}
        if len(unique) == 1 and matches:
            result = matches[0]
            cache_put(cache_key, result)
            return result
        if len(unique) > 1:
            log.warning("Prognozist: ambiguous fixture match for %s — %s; refusing to attach odds", home, away)
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
    ck = f"odds_{sport_key}_{_norm_name(home)}_{_norm_name(away)}"
    cached = cache_get(ck, CACHE_TTL["odds"])
    if cached is not None:
        if logs:
            logs.append(f"The Odds API: матч найден в кэше ({sport_key})" if cached else f"The Odds API: пустой кэш матча ({sport_key})")
        return cached or None
    # Check the per-sport cache before the local quota: one API request provides
    # the whole league's events, so every fixture must reuse that response.
    events_key = f"odds_events_v1_{sport_key}"
    events = cache_get(events_key, CACHE_TTL["odds"])
    if events is None:
        if usage.odds_remaining() <= 0:
            if logs:
                logs.append("⚠️ The Odds API: локальный лимит запросов исчерпан")
            return None
        try:
            r = _sess.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={"apiKey": api_key, "regions": "eu",
                        "markets": "h2h,totals,btts", "oddsFormat": "decimal"},
                timeout=20, proxies=NO_PROXY)
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
            if eh == hn and ea == an:
                target = ev
                break
            if (eh in hn or hn in eh) and (ea in an or an in ea):
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
