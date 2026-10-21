"""data/marathon.py — парсер линии Марафон с prognozist.ru."""
from __future__ import annotations
import re, sys
from datetime import datetime
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
    """Парсит HTML страницы category."""
    events = []
    blocks = re.split(r'<tbody\s+data-event-treeid="(\d+)"', html)

    _log(f"PARSING {div_code}: HTML len={len(html)}, blocks={len(blocks)//2}")

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

            date_m = re.search(
                r'<td\s+class="date">([^<]+)</td>', block)
            if not date_m:
                continue
            date_str = date_m.group(1).strip()

            prices = re.findall(
                r'<td[^>]*class="js-price[^"]*"[^>]*>\s*'
                r'(?:\([^)]*\)\s*<br\s*/?>\s*)?'
                r'<span\s+class="selection-link[^"]*">'
                r'([\d.,]+)</span>',
                block
            )

            if len(prices) < 3:
                _log(f"SKIP {home_ru} vs {away_ru}: prices={len(prices)}")
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
            odds = {k: v for k, v in odds.items() if v and v > 1.0}

            if not (odds.get("П1") and odds.get("X") and odds.get("П2")):
                _log(f"SKIP {home_ru} vs {away_ru}: invalid odds {odds}")
                continue

            _log(f"✅ PARSE: {home_ru} vs {away_ru} → П1={odds['П1']} X={odds['X']} П2={odds['П2']}")

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
    """Загружает и парсит события одной лиги."""
    cat_id = DIV_TO_CATEGORY.get(div_code)
    if not cat_id:
        return []

    ck = f"marathon_cat_v4_{cat_id}"
    cached = cache_get(ck, 1800)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    url = f"https://prognozist.ru/line/marathon/event_line.php?category={cat_id}"
    try:
        r = _sess.get(url, timeout=20, proxies=NO_PROXY)
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
    te = _norm(team_en)
    tr = _norm(team_ru)
    if te == tr:
        return True
    en_from_ru = RU_TO_EN_TEAM.get(tr, "")
    if en_from_ru:
        if te == _norm(en_from_ru):
            return True
        if te in en_from_ru or en_from_ru in te:
            return True
    if len(te) > 4 and (te in tr or tr in te):
        return True
    return False


def marathon_odds(home_en: str, away_en: str, date_iso: str = "",
                  div_code: str = "") -> Optional[dict]:
    if not home_en or not away_en:
        return None

    divs_to_try = [div_code] if div_code else list(DIV_TO_CATEGORY.keys())

    for dv in divs_to_try:
        events = fetch_league_events(dv)
        for ev in events:
            if (_find_team_match(home_en, ev["home"])
                    and _find_team_match(away_en, ev["away"])):
                _log(f"✅ FOUND: {ev['home']} vs {ev['away']} = "
                     f"П1 {ev['odds'].get('П1')} / X {ev['odds'].get('X')} / "
                     f"П2 {ev['odds'].get('П2')}")
                return {
                    "П1": ev["odds"].get("П1"),
                    "X": ev["odds"].get("X"),
                    "П2": ev["odds"].get("П2"),
                    "source": "marathon",
                }

    _log(f"❌ NOT FOUND: {home_en} vs {away_en}")
    return None
