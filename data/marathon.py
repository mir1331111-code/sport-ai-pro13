"""data/marathon.py — парсер линии Марафон с prognozist.ru (fallback кэфов)."""
from __future__ import annotations
import re, sys
from typing import Optional

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
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "ru-RU,ru;q=0.9",
        "Referer": "https://prognozist.ru/",
    })
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


_sess = _session()


DIV_TO_CATEGORY = {
    "E0": 21520, "E1": 22807, "SP1": 8736, "SP2": 48300,
    "I1": 22434, "I2": 46723, "D1": 22436, "D2": 42528,
    "F1": 21533, "F2": 46785, "N1": 38090, "P1": 43058,
    "T1": 46180, "R1": 22433,
    "C1": 19715337, "EL": 19689253, "EC": 19689393,
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


def _parse_events(html: str) -> list:
    events = []
    blocks = re.split(r'<tbody\s+data-event-treeid="(\d+)"', html)
    for i in range(1, len(blocks), 2):
        try:
            event_id = blocks[i]
            block = blocks[i + 1] if i + 1 < len(blocks) else ""
            names = re.findall(
                r'<span\s+class="member-name">([^<]+)</span>', block)
            if len(names) < 2:
                continue
            home_ru = names[0].strip()
            away_ru = names[1].strip()
            date_m = re.search(r'<td\s+class="date">([^<]+)</td>', block)
            if not date_m:
                continue
            date_str = date_m.group(1).strip()
            prices = re.findall(
                r'<td[^>]*class="js-price[^"]*"[^>]*>\s*'
                r'(?:\([^)]*\)\s*<br\s*/?>\s*)?'
                r'<span\s+class="selection-link[^"]*">'
                r'([\d.,]+)</span>',
                block)
            if len(prices) < 3:
                continue

            def _f(s):
                try:
                    return float(str(s).replace(",", "."))
                except Exception:
                    return None

            odds = {
                "П1": _f(prices[0]),
                "X": _f(prices[1]),
                "П2": _f(prices[2]),
            }
            if len(prices) > 3:
                odds["1X"] = _f(prices[3])
            if len(prices) > 4:
                odds["12"] = _f(prices[4])
            if len(prices) > 5:
                odds["X2"] = _f(prices[5])

            odds = {k: v for k, v in odds.items() if v and v > 1.0}
            if not (odds.get("П1") and odds.get("X") and odds.get("П2")):
                continue
            events.append({
                "event_id": event_id,
                "home": home_ru,
                "away": away_ru,
                "date_str": date_str,
                "odds": odds,
            })
        except Exception as e:
            _log(f"parse block error: {e}")
            continue
    return events


def fetch_league_events(div_code: str) -> list:
    cat_id = DIV_TO_CATEGORY.get(div_code)
    if not cat_id:
        return []
    ck = f"marathon_cat_{cat_id}"
    cached = cache_get(ck, 1800)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    url = (f"https://prognozist.ru/line/marathon/"
           f"event_line.php?category={cat_id}")
    try:
        r = _sess.get(url, timeout=20, proxies=NO_PROXY)
        if r.status_code != 200:
            _log(f"{div_code} HTTP {r.status_code}")
            cache_put(ck, [])
            return []
        events = _parse_events(r.text)
        _log(f"{div_code} (cat={cat_id}): {len(events)} events")
        cache_put(ck, events)
        return events
    except Exception as e:
        _log(f"{div_code} error: {type(e).__name__}: {str(e)[:100]}")
        return []


RU_TO_EN_TEAM = {
    # Англия
    "эвертон": "everton", "челси": "chelsea",
    "манчестер сити": "manchester city",
    "манчестер юнайтед": "manchester united",
    "ливерпуль": "liverpool", "арсенал": "arsenal",
    "тоттенхэм": "tottenham", "тоттенхэм хотспур": "tottenham",
    "ньюкасл": "newcastle", "ньюкасл юнайтед": "newcastle",
    "астон вилла": "aston villa", "брайтон": "brighton",
    "кристал пэлас": "crystal palace", "фулхэм": "fulham",
    "вест хэм": "west ham", "веста хэм": "west ham",
    "борнмут": "bournemouth", "волверхэмптон": "wolverhampton",
    "ноттингем форест": "nottingham forest", "лестер": "leicester",
    "ипсвич": "ipswich", "ипсвич таун": "ipswich",
    "лидс": "leeds", "лидс юнайтед": "leeds",
    "саутгемптон": "southampton", "бернли": "burnley",
    "уотфорд": "watford", "норвич": "norwich",
    "халл сити": "hull", "ковентри сити": "coventry",
    "сандерленд": "sunderland", "брентфорд": "brentford",
    "портсмут": "portsmouth",
    "шеффилд юнайтед": "sheffield united",
    "шеффилд уэнсдей": "sheffield wednesday",
    "блэкберн": "blackburn", "сток сити": "stoke",
    "дерби каунти": "derby", "миллуолл": "millwall",
    "суонси": "swansea", "престон": "preston",
    "куинз парк рейнджерс": "queens park rangers",
    "qpr": "queens park rangers", "линкольн сити": "lincoln",
    # Испания
    "реал мадрид": "real madrid", "барселона": "barcelona",
    "атлетико мадрид": "atletico madrid", "севилья": "sevilla",
    "бетис": "real betis", "реал бетис": "real betis",
    "реал сосьедад": "real sociedad",
    "атлетик бильбао": "athletic bilbao",
    "атлетик клуб": "athletic bilbao",
    "валенсия": "valencia", "вильярреал": "villarreal",
    "жирона": "girona", "осасуна": "osasuna",
    "райо вальекано": "rayo vallecano",
    "сельта": "celta vigo",
    "расинг сантандер": "racing santander",
    "алавес": "alaves",
    # Италия
    "интер": "inter", "inter milan": "inter",
    "fc internazionale milano": "inter",
    "милан": "ac milan", "ac milan": "ac milan",
    "ювентус": "juventus", "наполи": "napoli",
    "рома": "roma", "as roma": "roma", "лацио": "lazio",
    "аталанта": "atalanta", "фиорентина": "fiorentina",
    "удинезе": "udinese", "кальяри": "cagliari",
    "болонья": "bologna", "bologna fc 1909": "bologna",
    "торино": "torino", "torino fc": "torino",
    "венеция": "venezia", "venezia fc": "venezia",
    "como 1907": "como", "como": "como",
    # Германия
    "бавария": "bayern munich",
    "боруссия дортмунд": "borussia dortmund",
    "рб лейпциг": "rb leipzig",
    "байер леверкузен": "bayer leverkusen",
    "айнтрахт франкфурт": "eintracht frankfurt",
    "фрайбург": "freiburg",
    "боруссия мёнхенгладбах": "borussia monchengladbach",
    "майнц": "mainz",
    "вердер бремен": "werder bremen",
    "аугсбург": "augsburg",
    "гамбург": "hamburger sv",
    "кёльн": "koln",
    "штутгарт": "vfb stuttgart",
    # Франция
    "псж": "psg", "марсель": "marseille",
    "лион": "lyon", "монако": "monaco",
    "лилль": "lille", "ренн": "rennes",
    "страсбур": "strasbourg", "лорьян": "lorient",
    "гавр": "le havre", "тулуза": "toulouse",
    "анже": "angers", "троа": "troyes",
    "le mans": "le mans",
    # Россия
    "зенит": "zenit", "спартак": "spartak moscow",
    "спартак москва": "spartak moscow",
    "цска": "cska moscow", "локомотив": "lokomotiv moscow",
    "динамо": "dynamo moscow", "динамо москва": "dynamo moscow",
    "краснодар": "krasnodar", "ростов": "rostov",
    # Нидерланды
    "аякс": "ajax", "псв": "psv", "фейеноорд": "feyenoord",
    "адо ден хааг": "ado den haag",
    "спарта роттердам": "sparta rotterdam",
    "heerenveen": "sc heerenveen",
    "виллем ii": "willem ii",
    "фортуна ситтард": "fortuna sittard",
    "excelsior": "sbv excelsior",
    # Португалия
    "порту": "fc porto", "бенфика": "benfica",
    "спортинг": "sporting cp",
    "спортинг клубе де португал": "sporting cp",
    "насионал": "cd nacional", "фамаликан": "fc famalicao",
    "жил vicente": "gil vicente",
    "маритиму": "cs maritimo", "алверка": "fc alverca",
    "риу аве": "rio ave", "ароука": "fc arouca",
    # Турция
    "галатасарай": "galatasaray", "фенербахче": "fenerbahce",
    # США
    "atlanta united": "atlanta united",
    "charlotte fc": "charlotte fc",
    "columbus crew": "columbus crew",
    "cf montreal": "cf montreal",
    "d.c. united": "dc united",
    "red bull new york": "new york red bulls",
    "fc cincinnati": "fc cincinnati",
    "new england revolution": "new england revolution",
    "inter miami": "inter miami",
    "new york city": "new york city",
    "toronto fc": "toronto fc",
    "orlando city": "orlando city",
    "chicago fire": "chicago fire",
    "philadelphia union": "philadelphia union",
    "fc dallas": "fc dallas",
    "seattle sounders": "seattle sounders",
    "nashville sc": "nashville sc",
    "sporting kansas city": "sporting kansas city",
    # ЛЧ, ЛЕ (топ-клубы)
    "lask linz": "lask linz", "liverpool fc": "liverpool",
    "shakhtar donetsk": "shakhtar donetsk",
    "aek": "aek athens", "pae aek": "aek athens",
    "slovan bratislava": "slovan bratislava",
    "bodo/glimt": "bodo glimt", "bodø/glimt": "bodo glimt",
}


def marathon_odds(home_en: str, away_en: str, date_iso: str = "",
                  div_code: str = "") -> Optional[dict]:
    """Ищет кэфы Марафона для матча."""
    if not home_en or not away_en:
        return None
    divs_to_try = [div_code] if div_code and div_code in DIV_TO_CATEGORY \
                  else list(DIV_TO_CATEGORY.keys())
    hn = _norm(home_en)
    an = _norm(away_en)

    for dv in divs_to_try:
        events = fetch_league_events(dv)
        for ev in events:
            ev_home_ru = ev["home"]
            ev_away_ru = ev["away"]

            ev_home_en = RU_TO_EN_TEAM.get(
                _norm(ev_home_ru), _norm(ev_home_ru))
            ev_away_en = RU_TO_EN_TEAM.get(
                _norm(ev_away_ru), _norm(ev_away_ru))

            home_match = (hn in ev_home_en or ev_home_en in hn)
            away_match = (an in ev_away_en or ev_away_en in an)

            if home_match and away_match:
                _log(f"FOUND: {ev_home_ru} vs {ev_away_ru} "
                     f"= П1 {ev['odds'].get('П1')} / "
                     f"X {ev['odds'].get('X')} / "
                     f"П2 {ev['odds'].get('П2')}")
                return {
                    "П1": ev["odds"].get("П1"),
                    "X": ev["odds"].get("X"),
                    "П2": ev["odds"].get("П2"),
                    "1X": ev["odds"].get("1X"),
                    "12": ev["odds"].get("12"),
                    "X2": ev["odds"].get("X2"),
                    "source": "marathon",
                }

    _log(f"NOT FOUND: {home_en} vs {away_en} @ {date_iso}")
    return None
