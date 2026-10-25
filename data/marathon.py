"""data/marathon.py — парсер линии Марафон с prognozist.ru."""
from __future__ import annotations
import os, re, sys
from datetime import datetime
from typing import Optional
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse

import requests

from security import cache_get, cache_put


NO_PROXY = {"http": None, "https": None, "all": None}


def _log(msg: str):
    try:
        print(f"[MARATHON] {msg}", file=sys.stdout, flush=True)
    except Exception:
        pass


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Sport-AI-Pro13/1.0 (+contact the repository maintainer; research client)",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "ru-RU,ru;q=0.9",
        "Referer": "https://prognozist.ru/",
    })
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


_sess = _session()


def _permission_enabled() -> bool:
    """Require explicit opt-in from the UI, environment, or Streamlit Secrets."""
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


def _permitted_by_robots(url: str) -> bool:
    """Fail closed: require explicit permission and cached robots.txt allowance."""
    if not _permission_enabled():
        _log("Disabled: set PROGNOZIST_PERMISSION=1 only after confirming site terms/permission.")
        return False
    cache_key = f"marathon_robots_allowed_v1_{url}"
    cached = cache_get(cache_key, 300)
    if cached is not None:
        return bool(cached)
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        response = _sess.get(robots_url, timeout=(2, 4), proxies=NO_PROXY)
        if response.status_code != 200:
            _log(f"Disabled: robots.txt unavailable (HTTP {response.status_code}); fail-closed.")
            cache_put(cache_key, False)
            return False
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.parse(response.text.splitlines())
        allowed = bool(parser.can_fetch(_sess.headers["User-Agent"], url))
        cache_put(cache_key, allowed)
        return allowed
    except Exception as exc:
        _log(f"Disabled: robots.txt check failed ({type(exc).__name__}); fail-closed.")
        cache_put(cache_key, False)
        return False


DIV_TO_CATEGORY = {
    "E0": 21520, "E1": 22807,
    "SP1": 8736, "SP2": 48300,
    "I1": 22434, "I2": 46723,
    "D1": 22436, "D2": 42528,
    "F1": 21533, "F2": 46785,
    "N1": 38090, "P1": 43058,
    "T1": 46180, "R1": 22433,
    "C1": 19715337, "EL": 19689253, "EC": 19689393,
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


def _parse_events(html: str, div_code: str = "") -> list:
    """Parse Prognozist/Marathon event rows while tolerating HTML attribute order."""
    events = []
    # Use double-quoted raw strings so both quote styles in HTML attributes are safe.
    pattern = re.compile(
        r"<tbody\b(?=[^>]*\bdata-event-treeid=['\"]?(\d+)['\"]?)[^>]*>(.*?)</tbody>",
        re.I | re.S,
    )
    blocks = pattern.findall(html or "")
    _log(f"PARSING {div_code}: HTML len={len(html or '')}, blocks={len(blocks)}")

    def text_of(fragment):
        fragment = re.sub(r"<[^>]+>", " ", fragment or "")
        return re.sub(r"\s+", " ", fragment).strip()

    def has_class(tag_html, class_name):
        m = re.search(r"\bclass\s*=\s*(['\"])(.*?)\1", tag_html, re.I | re.S)
        return bool(m and class_name in m.group(2).split())

    for event_id, block in blocks:
        try:
            names = []
            # Prognozist has changed its team markup: accept common class names
            # on spans, links and divs rather than requiring span.member-name.
            team_marker = re.compile(
                r"(member|team|participant|competitor)[-_ ]?(name|link|title)?",
                re.I,
            )
            ignored_marker = re.compile(
                r"price|odd|coef|selection|score|date|time|league|country|sport",
                re.I,
            )
            for m in re.finditer(
                r"<(span|a|div|strong|b)\b([^>]*)>(.*?)</\1>",
                block, re.I | re.S,
            ):
                attrs, inner = m.group(2) or "", m.group(3) or ""
                cls_m = re.search(r'\bclass\s*=\s*([\'"])(.*?)\1', attrs, re.I | re.S)
                cls = cls_m.group(2) if cls_m else ""
                marked = bool(team_marker.search(cls)) or bool(re.search(
                    r'\bdata-(team|member|participant|competitor)(-name)?\s*=',
                    attrs, re.I,
                )) or bool(re.search(
                    r'\bhref\s*=\s*([\'"])[^\'"]*/(?:team|member|participant|competitor)s?/[^\'"]*\1',
                    attrs, re.I,
                ))
                if not marked or ignored_marker.search(cls):
                    continue
                name = re.sub(r"\s+", " ", text_of(inner)).strip(" -:|")
                if (name and len(name) >= 2
                        and not re.fullmatch(r"[\d.,:+-]+", name)
                        and name not in names):
                    names.append(name)

            # Last fallback: some versions expose names only as data attributes.
            if len(names) < 2:
                for attr_name in ("data-team-name", "data-member-name",
                                  "data-participant-name"):
                    for match in re.finditer(
                        r'\b' + re.escape(attr_name) + r'\s*=\s*([\'"])(.*?)\1',
                        block, re.I | re.S,
                    ):
                        name = re.sub(r"\s+", " ", text_of(match.group(2))).strip(" -:|")
                        if (name and len(name) >= 2
                                and re.search(r"[A-Za-zА-Яа-я]", name)
                                and name not in names):
                            names.append(name)

            if len(names) < 2:
                _log(f"SKIP event {event_id}: found only {len(names)} team names")
                continue
            names = names[:2]

            date_str = ""
            for m in re.finditer(r"<td\b[^>]*>(.*?)</td>", block, re.I | re.S):
                opening = block[m.start():block.find(">", m.start()) + 1]
                if has_class(opening, "date"):
                    date_str = text_of(m.group(1))
                    break

            prices = []
            for m in re.finditer(r"<td\b[^>]*>(.*?)</td>", block, re.I | re.S):
                opening = block[m.start():block.find(">", m.start()) + 1]
                if not has_class(opening, "js-price"):
                    continue
                cell = m.group(1)
                for sm in re.finditer(r"<span\b[^>]*>(.*?)</span>", cell, re.I | re.S):
                    span_open = cell[sm.start():cell.find(">", sm.start()) + 1]
                    if not has_class(span_open, "selection-link"):
                        continue
                    value = text_of(sm.group(1)).replace(",", ".")
                    if re.fullmatch(r"\d+(?:\.\d+)?", value):
                        prices.append(value)

            if len(prices) < 3:
                _log(f"SKIP {names[0]} vs {names[1]}: prices={len(prices)}")
                continue

            odds = {}
            for key, value in zip(("П1", "X", "П2"), prices[:3]):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if 1.0 < number <= 100:
                    odds[key] = number
            if len(odds) != 3:
                _log(f"SKIP {names[0]} vs {names[1]}: invalid odds {odds}")
                continue

            events.append({
                "event_id": event_id,
                "home": names[0],
                "away": names[1],
                "date_str": date_str,
                "odds": odds,
            })
            _log(f"PARSED {names[0]} vs {names[1]}: {odds}")
        except Exception as e:
            _log(f"parse block error: {type(e).__name__}: {e}")
    return events

def fetch_league_events(div_code: str) -> list:
    """Загружает и парсит события одной лиги."""
    cat_id = DIV_TO_CATEGORY.get(div_code)
    if not cat_id:
        return []

    url = f"https://prognozist.ru/line/marathon/event_line.php?category={cat_id}"
    # Permission must be checked before cache reads: cached odds must never
    # bypass an operator's decision to disable this external source.
    if not _permitted_by_robots(url):
        return []

    ck = f"marathon_cat_v7_{cat_id}"
    cached = cache_get(ck, 300)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    try:
        r = _sess.get(url, timeout=(2, 4), proxies=NO_PROXY)
        if r.status_code != 200:
            _log(f"{div_code} HTTP {r.status_code}")
            cache_put(ck, [])
            return []

        # Отладка: сохраняем превью HTML
        _log(f"{div_code} (cat={cat_id}): HTML len={len(r.text)}, "
             f"first 200 chars: {r.text[:200]}")

        events = _parse_events(r.text, div_code)
        _log(f"{div_code}: parsed {len(events)} events")
        cache_put(ck, events)
        return events
    except Exception as e:
        _log(f"{div_code} error: {type(e).__name__}: {str(e)[:100]}")
        # Cache the miss too: otherwise every fixture in this league retries
        # the same failing HTTP request and the scan appears frozen.
        cache_put(ck, [])
        return []


RU_TO_EN_TEAM = {
    "эвертон": "everton", "челси": "chelsea",
    "манчестер сити": "manchester city",
    "манчестер юнайтед": "manchester united",
    "ливерпуль": "liverpool", "арсенал": "arsenal",
    "тоттенхэм": "tottenham", "ньюкасл": "newcastle",
    "астон вилла": "aston villa", "брайтон": "brighton",
    "кристал пэлас": "crystal palace", "фулхэм": "fulham",
    "вест хэм": "west ham", "борнмут": "bournemouth",
    "волверхэмптон": "wolverhampton",
    "ноттингем форест": "nottingham forest",
    "лестер": "leicester", "ипсвич": "ipswich",
    "саутгемптон": "southampton", "бернли": "burnley",
    "уотфорд": "watford", "норвич": "norwich",
    "лидс": "leeds", "халл сити": "hull",
    "ковентри сити": "coventry", "сандерленд": "sunderland",
    "брентфорд": "brentford",
    "шеффилд юнайтед": "sheffield united",
    "шеффилд уэнсдей": "sheffield wednesday",
    "мидлсбро": "middlesbrough", "сток сити": "stoke",
    "суонси сити": "swansea", "кардифф сити": "cardiff",
    "престон норт энд": "preston",
    "куинз парк рейнджерс": "qpr",
    "миллуолл": "millwall",
    "блекберн роверс": "blackburn",
    "портсмут": "portsmouth", "дребери каунти": "derby",
    "линкольн сити": "lincoln",
    "реал мадрид": "real madrid", "барселона": "barcelona",
    "атлетико мадрид": "atletico madrid", "севилья": "sevilla",
    "бетис": "real betis", "реал сосьедад": "real sociedad",
    "атлетик бильбао": "athletic bilbao",
    "валенсия": "valencia", "вильярреал": "villarreal",
    "жирона": "girona", "сельта": "celta vigo",
    "осасуна": "osasuna", "кадис": "cadiz",
    "мальорка": "mallorca", "лас пальмас": "las palmas",
    "алавес": "alaves",
    "райо вальекано": "rayo vallecano",
    "расинг сантандер": "racing santander",
    "хетафе": "getafe", "эспаньол": "espanyol",
    "гранада": "granada", "альмерия": "almeria",
    "интер": "inter", "милан": "ac milan",
    "ювентус": "juventus", "наполи": "napoli",
    "рома": "roma", "лацио": "lazio",
    "аталанта": "atalanta", "фиорентина": "fiorentina",
    "болонья": "bologna", "торино": "torino",
    "удинезе": "udinese", "кальяри": "cagliari",
    "дженоа": "genoa", "сампдория": "sampdoria",
    "сассуоло": "sassuolo", "эмполи": "empoli",
    "верона": "verona", "лечче": "lecce",
    "венеция": "venezia", "монца": "monza",
    "комо": "como", "парма": "parma",
    "бавария": "bayern munich",
    "боруссия дортмунд": "borussia dortmund",
    "дортмунд": "borussia dortmund",
    "рб лейпциг": "rb leipzig",
    "байер леверкузен": "bayer leverkusen",
    "айнтрахт франкфурт": "eintracht frankfurt",
    "фрайбург": "freiburg", "штутгарт": "stuttgart",
    "вольфсбург": "wolfsburg",
    "боруссия мёнхенгладбах": "borussia monchengladbach",
    "вердер бремен": "werder bremen",
    "аугсбург": "augsburg", "майнц": "mainz",
    "хоффенхайм": "hoffenheim",
    "унион берлин": "union berlin",
    "хольштайн киль": "holstein kiel",
    "санкт-паули": "st pauli",
    "хейденхайм": "heidenheim",
    "бохум": "bochum", "дармштадт": "darmstadt",
    "кёльн": "koln", "гамбург": "hamburger sv",
    "ганновер": "hannover",
    "псж": "psg", "пари сен-жермен": "psg",
    "марсель": "marseille", "лион": "lyon",
    "олимпик лион": "lyon", "монако": "monaco",
    "лилль": "lille", "рен": "rennes",
    "страсбур": "strasbourg", "ницца": "nice",
    "ланс": "lens", "тулуза": "toulouse",
    "монпелье": "montpellier", "нант": "nantes",
    "реймс": "reims", "брест": "brest",
    "гавр": "le havre", "метц": "metz",
    "лорьян": "lorient", "анже": "angers",
    "клермон": "clermont", "париж": "paris fc",
    "зенит": "zenit", "спартак": "spartak moscow",
    "спартак москва": "spartak moscow",
    "цска": "cska moscow", "цска москва": "cska moscow",
    "локомотив": "lokomotiv moscow",
    "локомотив москва": "lokomotiv moscow",
    "динамо": "dynamo moscow", "динамо москва": "dynamo moscow",
    "краснодар": "krasnodar", "ростов": "rostov",
    "рубин": "rubin kazan", "рубин казань": "rubin kazan",
    "ахмат": "akhmat grozny", "ахмат грозный": "akhmat grozny",
    "крылья советов": "krylia sovetov",
    "балтика": "baltika", "факел": "fakel",
    "пари нн": "paris nn", "пари нижегород": "paris nn",
    "оренбург": "orenburg", "сочи": "sochi",
    "торпедо": "torpedo moscow",
    "урал": "ural", "енисей": "yenisey",
    "арсенал тула": "arsenal tula",
    "аякс": "ajax", "псв": "psv", "фейеноорд": "feyenoord",
    "алмере сити": "almere city",
    "адо ден хааг": "ado den haag",
    "камбюр": "cambuur", "sc камбюр-леуварден": "cambuur",
    "sparta rotterdam": "sparta rotterdam",
    "heerenveen": "heerenveen", "хееренвен": "heerenveen",
    "willem ii tilburg": "willem ii",
    "фортуна ситтард": "fortuna sittard",
    "эксельсиор": "excelsior",
    "порту": "fc porto", "бенфика": "benfica",
    "спортинг": "sporting cp", "спортинг лиссабон": "sporting cp",
    "брага": "braga", "витория": "vitoria",
    "боавишта": "boavista", "маритиму": "maritimo",
    "насьонал": "nacional", "фейренсе": "feirense",
    "фамаликан": "famalicao", "жил виcente": "gil vicente",
    "риу аве": "rio ave", "арока": "arouca",
    "алверка": "alverca",
    "галатасарай": "galatasaray",
    "фенербахче": "fenerbahce",
    "бешикташ": "besiktas",
    "башакшехир": "istanbul basaksehir",
    "истанбул башакшехир": "istanbul basaksehir",
    "трабзонспор": "trabzonspor",
}


def _find_team_match(team_en: str, team_ru: str) -> bool:
    """Match provider names without collapsing youth/reserve sides into senior teams."""
    te, tr = _norm(team_en), _norm(team_ru)
    if not te or not tr:
        return False
    def age_marker(value):
        raw = (value or "").lower()
        match = re.search(r"\b(u\s*\d{2}|under\s*[- ]?\s*\d{2}|youth|reserves?|ii)\b", raw)
        return re.sub(r"\s+", "", match.group(1)) if match else ""
    age_en, age_ru = age_marker(team_en), age_marker(team_ru)
    if age_en != age_ru and (age_en or age_ru):
        return False
    aliases = {
        "athleticclub": "athleticbilbao", "athleticbilba": "athleticbilbao",
        "levanteud": "levante", "levanteuniondeportiva": "levante",
        "internazionalemilano": "inter", "fcinternazionale": "inter",
        "intermilan": "inter", "manutd": "manchesterunited",
        "manchesterutd": "manchesterunited",
        "manchestercityfc": "manchestercity", "lilleosc": "lille",
        "como1907": "como", "sportinglisbon": "sportingcp",
        "fcporto": "porto", "galatasarayysk": "galatasaray",
    }
    te, tr = aliases.get(te, te), aliases.get(tr, tr)
    if te == tr:
        return True
    # Use the shared alias registry for exact canonical matches across feeds.
    try:
        from data.team_names import normalize_team
        canonical_en = normalize_team(team_en)
        canonical_provider = normalize_team(team_ru)
        if canonical_en and canonical_en == canonical_provider:
            return True
    except Exception:
        pass
    mapped = RU_TO_EN_TEAM.get((team_ru or "").strip().lower(), "")
    if mapped and aliases.get(_norm(mapped), _norm(mapped)) == te:
        return True
    # Deliberately avoid substring/token-overlap matches here: e.g. "Manchester"
    # is shared by Manchester United and Manchester City and can attach wrong odds.
    return False


def _event_calendar_date(value: str, reference_date: str = "") -> str:
    """Extract a calendar date; infer the year for DD.MM when a fixture date is known."""
    raw = (value or "").strip()
    if not raw:
        return ""
    patterns = (
        (r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", "ymd"),
        (r"(?<!\d)(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})(?!\d)", "dmy"),
        (r"(?<!\d)(\d{1,2})[./-](\d{1,2})(?!\d)", "dm"),
    )
    for pattern, kind in patterns:
        match = re.search(pattern, raw)
        if not match:
            continue
        parts = [int(x) for x in match.groups()]
        if kind == "ymd":
            year, month, day = parts
        elif kind == "dmy":
            day, month, year = parts
            if year < 100:
                year += 2000
        else:
            day, month = parts
            try:
                year = int((reference_date or "")[:4])
            except ValueError:
                return ""
            if year < 2000:
                return ""
        try:
            return datetime(year, month, day).strftime("%Y-%m-%d")
        except ValueError:
            return ""
    return ""


def marathon_odds(home_en: str, away_en: str, date_iso: str = "",
                  div_code: str = "") -> Optional[dict]:
    if not home_en or not away_en:
        return None

    # Query only the fixture's likely league; do not attach a line from another
    # date or choose arbitrarily if multiple same-name fixtures are present.
    if div_code not in DIV_TO_CATEGORY:
        _log(f"SKIP category lookup for unmapped league {div_code!r}")
        return None

    target_date = (date_iso or "")[:10]
    events = fetch_league_events(div_code)
    matches = []
    for ev in events:
        if not (_find_team_match(home_en, ev.get("home", ""))
                and _find_team_match(away_en, ev.get("away", ""))):
            continue
        event_date = _event_calendar_date(ev.get("date_str", ""), target_date)
        if target_date and event_date and event_date != target_date:
            continue
        odds = ev.get("odds")
        if not isinstance(odds, dict) or not all(
            _valid_marathon_odd(odds.get(key)) for key in ("П1", "X", "П2")
        ):
            continue
        matches.append(ev)

    if not matches:
        _log(f"❌ NOT FOUND in {div_code}: {home_en} vs {away_en} ({target_date or 'date unknown'})")
        return None
    if len(matches) > 1:
        # If a single dated match can be identified, prefer it over undated rows.
        dated = [ev for ev in matches if _event_calendar_date(ev.get("date_str", ""), target_date) == target_date] if target_date else []
        if len(dated) == 1:
            matches = dated
        else:
            unique_lines = {
                tuple(round(float(ev["odds"][key]), 4) for key in ("П1", "X", "П2"))
                for ev in matches
            }
            if len(unique_lines) != 1:
                _log(f"⚠️ AMBIGUOUS: multiple lines for {home_en} vs {away_en}; refusing to choose")
                return None
    ev = matches[0]
    _log(f"✅ FOUND: {ev['home']} vs {ev['away']} = "
         f"П1 {ev['odds']['П1']} / X {ev['odds']['X']} / П2 {ev['odds']['П2']}")
    return {"П1": float(ev["odds"]["П1"]), "X": float(ev["odds"]["X"]),
            "П2": float(ev["odds"]["П2"]), "source": "marathon"}


def _valid_marathon_odd(value) -> bool:
    try:
        number = float(value)
        return 1.01 < number <= 100
    except (TypeError, ValueError):
        return False
