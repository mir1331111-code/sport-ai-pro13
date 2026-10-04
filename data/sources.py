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
import csv, io, re, logging
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
        ck = f"tsdb_day_v8_{dstr}"
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
                league = e.get("strLeague") or "Матч"
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
        ck = f"espn_soccer_v2_{dstr}"
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
        ck = f"fdorg_v12_{comp}_{d_from}_{d_to}"
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
        cache_put(ck, None)
        return None
    m = data["match"]
    if m.get("status") != "FINISHED":
        cache_put(ck, None)
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


# ============ THE ODDS API ============
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


# ============ LOGOS ============
def team_logo_url(team_name: str) -> Optional[str]:
    return None


def league_logo_url(div_code: str) -> Optional[str]:
    return None
