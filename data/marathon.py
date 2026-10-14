"""data/marathon.py — парсер линии Марафон с prognozist.ru + словарь команд."""
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


# Маппинг наших лиг → category ID на Прогнозист.ру
DIV_TO_CATEGORY = {
    "E0": 21520,    # Англия. Премьер-лига
    "E1": 22807,    # Англия. Чемпионшип
    "SP1": 8736,    # Испания. Ла Лига
    "SP2": 48300,   # Испания. Сегунда
    "I1": 22434,    # Италия. Серия A
    "I2": 46723,    # Италия. Серия B
    "D1": 22436,    # Германия. Бундеслига
    "D2": 42528,    # Германия. 2-я Бундеслига
    "F1": 21533,    # Франция. Лига 1
    "F2": 46785,    # Франция. Лига 2
    "N1": 38090,    # Нидерланды. Эредивизи
    "P1": 43058,    # Португалия. Примейра
    "T1": 46180,    # Турция. Суперлига
    "R1": 22433,    # Россия. Премьер-лига
    "C1": 19715337, # ЛЧ УЕФА. Общий этап
    "EL": 19689253, # ЛЕ УЕФА. Общий этап
    "EC": 19689393, # ЛК УЕФА. Общий этап
}


# ============ СЛОВАРЬ РУССКИХ НАЗВАНИЙ КОМАНД МАРАФОНА ============
# Из HTML берём реальные названия: "Эвертон", "Челси", "Тоттенхэм Хотспур" и т.д.
RU_TO_EN_TEAM = {
    # ===== Англия. АПЛ =====
    "эвертон": "everton",
    "челси": "chelsea",
    "манчестер сити": "manchester city",
    "манчестер юнайтед": "manchester united",
    "ливерпуль": "liverpool",
    "арсенал": "arsenal",
    "тоттенхэм": "tottenham",
    "тоттенхэм хотспур": "tottenham",
    "ньюкасл": "newcastle",
    "ньюкасл юнайтед": "newcastle",
    "астон вилла": "aston villa",
    "брайтон": "brighton",
    "брайтон энд хав альбион": "brighton",
    "кристал пэлас": "crystal palace",
    "фулхэм": "fulham",
    "вест хэм": "west ham",
    "веста хэм юнайтед": "west ham",
    "вест хэм юнайтед": "west ham",
    "борнмут": "bournemouth",
    "волверхэмптон": "wolverhampton",
    "волверхэмптон уондерерс": "wolverhampton",
    "ноттингем форест": "nottingham forest",
    "лестер": "leicester",
    "лестер сити": "leicester",
    "ипсвич": "ipswich",
    "ипсвич таун": "ipswich",
    "лидс": "leeds",
    "лидс юнайтед": "leeds",
    "саутгемптон": "southampton",
    "бернли": "burnley",
    "уотфорд": "watford",
    "норвич": "norwich",
    "норвич сити": "norwich",
    "халл сити": "hull",
    "халл": "hull",
    "ковентри сити": "coventry",
    "ковентри": "coventry",
    "сандерленд": "sunderland",
    "брентфорд": "brentford",
    "стока": "stoke",
    "сток сити": "stoke",
    "стoke сити": "stoke",
    "престон норт энд": "preston",
    "престон": "preston",
    "миллуолл": "millwall",
    "миллуол": "millwall",
    "кардифф сити": "cardiff",
    "кардифф": "cardiff",
    "суонси сити": "swansea",
    "суонси": "swansea",
    "линкольн сити": "lincoln",
    "линкольн": "lincoln",
    "куинз парк рейнджерс": "qpr",
    "кп": "qpr",
    "блекберн роверс": "blackburn",
    "блекберн": "blackburn",
    "портсмут": "portsmouth",
    "портсмout": "portsmouth",
    "дерби каунти": "derby",
    "дерби": "derby",
    "рейксем": "wrexham",
    "чарльтон атлетик": "charlton",
    "чарльтон": "charlton",
    "шеффилд юнайтед": "sheffield united",
    "шеффилд уэнсдей": "sheffield wednesday",
    "бирмингем сити": "birmingham",
    "бирмингем": "birmingham",
    "мидлсбро": "middlesbrough",
    "астон вилла": "aston villa",
    "бернли": "burnley",

    # ===== Испания. Ла Лига =====
    "реал мадрид": "real madrid",
    "барселона": "barcelona",
    "атлетико мадрид": "atletico madrid",
    "севилья": "sevilla",
    "бетис": "real betis",
    "реал бетис": "real betis",
    "реал сосьедад": "real sociedad",
    "атлетик бильбао": "athletic bilbao",
    "атлетик": "athletic bilbao",
    "валенсия": "valencia",
    "вильярреал": "villarreal",
    "жирона": "girona",
    "осасуна": "osasuna",
    "ка оссасуна": "osasuna",
    "райо вальекано": "rayo vallecano",
    "райо вальекано де мадрид": "rayo vallecano",
    "сельта": "celta vigo",
    "сельта де виго": "celta vigo",
    "рс де виго": "celta vigo",
    "расинг сантандер": "racing santander",
    "реал расинг клуб де сантандер": "racing santander",
    "депортиво алавес": "alaves",
    "алавес": "alaves",
    "мальорка": "mallorca",
    "хетафе": "getafe",
    "леганес": "leganes",
    "эспаньол": "espanyol",
    "лас пальмас": "las palmas",
    "реал вальядолид": "valladolid",
    "сельта де виго": "celta vigo",
    "райо": "rayo vallecano",

    # ===== Италия. Серия A =====
    "интер": "inter",
    "интер милан": "inter",
    "интернационале": "inter",
    "fc интернационале милано": "inter",
    "милан": "ac milan",
    "ac милан": "ac milan",
    "ювентус": "juventus",
    "наполи": "napoli",
    "рома": "roma",
    "лацио": "lazio",
    "аталанта": "atalanta",
    "фиорентина": "fiorentina",
    "удинезе": "udinese",
    "удинезе кальчо": "udinese",
    "кальяри": "cagliari",
    "кальяри кальчо": "cagliari",
    "болонья": "bologna",
    "болонья 1909": "bologna",
    "bologna fc 1909": "bologna",
    "торino": "torino",
    "торіно": "torino",
    "торіно": "torino",
    "венеция": "venezia",
    "венеция fc": "venezia",
    "дженоа": "genoa",
    "сампдория": "sampdoria",
    "сассуоло": "sassuolo",
    "эмполи": "empoli",
    "лечче": "lecce",
    "верона": "hellas verona",
    "эллас верона": "hellas verona",
    "монца": "monza",
    "фрозиноне": "frosinone",
    "салернитана": "salernitana",

    # ===== Германия. Бундеслига =====
    "бавария": "bayern munich",
    "бавария мюнхен": "bayern munich",
    "боруссия дортмунд": "borussia dortmund",
    "боруссия д": "borussia dortmund",
    "дортмунд": "borussia dortmund",
    "рб лейпциг": "rb leipzig",
    "байер леверкузен": "bayer leverkusen",
    "байер 04": "bayer leverkusen",
    "айнтрахт франкфурт": "eintracht frankfurt",
    "айнтрахт ф": "eintracht frankfurt",
    "франкфурт": "eintracht frankfurt",
    "штутгарт": "vfb stuttgart",
    "вфб штутгарт": "vfb stuttgart",
    "боруссия менхенгладбах": "borussia monchengladbach",
    "боруссия м": "borussia monchengladbach",
    "менхенгладбах": "borussia monchengladbach",
    "вольфсбург": "wolfsburg",
    "фольксваген вольфсбург": "wolfsburg",
    "фрайбург": "sc freiburg",
    "sc фрайбург": "sc freiburg",
    "вердер бремен": "werder bremen",
    "вердер": "werder bremen",
    "бремен": "werder bremen",
    "аугсбург": "fc augsburg",
    "fc аугсбург": "fc augsburg",
    "майнц": "mainz 05",
    "майнц 05": "mainz 05",
    "1 fsv mainz 05": "mainz 05",
    "1 fsv майнц 05": "mainz 05",
    "хoffenheim": "hoffenheim",
    "хоффенхайм": "hoffenheim",
    "унион берлин": "union berlin",
    "унион": "union berlin",
    "бochum": "bochum",
    "бохум": "bochum",
    "хольштайн киль": "holstein kiel",
    "хольштайн": "holstein kiel",
    "киль": "holstein kiel",
    "санкт-паули": "st pauli",
    "st pauli": "st pauli",
    "гамбург": "hamburger sv",
    "гамбургер св": "hamburger sv",
    "гамбургер сv": "hamburger sv",
    "hamburger sv": "hamburger sv",
    "1 fc köln": "fc koln",
    "1 fc кельн": "fc koln",
    "кельн": "fc koln",
    "fc köln": "fc koln",

    # ===== Франция. Лига 1 =====
    "псж": "psg",
    "пари сен-жермен": "psg",
    "paris sg": "psg",
    "париж": "paris fc",
    "париж fc": "paris fc",
    "марсель": "marseille",
    "олимпик марсель": "marseille",
    "лион": "lyon",
    "олимпик лион": "lyon",
    "олимпик л": "lyon",
    "олимпик лионне": "lyon",
    "stade rennais fc 1901": "rennes",
    "ренн": "rennes",
    "stade rennais": "rennes",
    "монако": "monaco",
    "лилль": "lille",
    "никола": "nice",
    "нисса": "nice",
    "nice": "nice",
    "лans": "lens",
    "ланс": "lens",
    "rc ланс": "lens",
    "rc lens": "lens",
    "страсбург": "strasbourg",
    "rc страсбург альзас": "strasbourg",
    "rc strasbourg alsace": "strasbourg",
    "нат": "nantes",
    "нант": "nantes",
    "fc нант": "nantes",
    "рeims": "reims",
    "реймс": "reims",
    "stade de reims": "reims",
    "brest": "brest",
    "брест": "brest",
    "stade brestois": "brest",
    "монпелье": "montpellier",
    "тулуза": "toulouse",
    "toulouse fc": "toulouse",
    "гавр": "le havre",
    "le havre ac": "le havre",
    "le гавр ac": "le havre",
    "анжe": "angers",
    "анжер": "angers",
    "angers sco": "angers",
    "анжер sco": "angers",
    "труа": "troyes",
    "es труа ac": "troyes",
    "es troyes ac": "troyes",
    "лорьent": "lorient",
    "лорьян": "lorient",
    "fc лорьян": "lorient",
    "fc lorient": "lorient",
    "le манс": "le mans",
    "le манс fc": "le mans",
    "le mans fc": "le mans",
    "ле ман": "le mans",

    # ===== Россия. РПЛ =====
    "зенит": "zenit",
    "зенит спб": "zenit",
    "спартак": "spartak moscow",
    "спартак москва": "spartak moscow",
    "цска": "cska moscow",
    "цска москва": "cska moscow",
    "локомотив": "lokomotiv moscow",
    "локомотив москва": "lokomotiv moscow",
    "динамо": "dynamo moscow",
    "динамо москва": "dynamo moscow",
    "краснодар": "krasnodar",
    "ростов": "rostov",
    "рубин": "rubin kazan",
    "рубин казань": "rubin kazan",
    "ахмат": "akhmat grozny",
    "ахмат грозный": "akhmat grozny",
    "крылья советов": "krylia sovetov",
    "балтика": "baltika",
    "факел": "fakel",
    "оренбург": "orenburg",
    "пари нн": "paris nn",
    "пари нижний новгород": "paris nn",
    "акрон": "akron togliatti",
    "акрон тольятти": "akron togliatti",
    "динамо махачкала": "dynamo makhachkala",
    "хімки": "khimki",
    "химки": "khimki",

    # ===== Португалия. Примейра =====
    "порту": "fc porto",
    "fc порту": "fc porto",
    "бенфика": "benfica",
    "спортинг": "sporting cp",
    "спортинг лиссабон": "sporting cp",
    "sporting clube de portugal": "sporting cp",
    "брага": "braga",
    "sc braga": "braga",
    "витория": "vitoria guimaraes",
    "витория гимарайнш": "vitoria guimaraes",
    "бoавишта": "boavista",
    "боавишта": "boavista",
    "nacional": "cd nacional",
    "cd nacional": "cd nacional",
    "cd насьональ": "cd nacional",
    "насьональ": "cd nacional",
    "фamalican": "fc famalicao",
    "фамаликан": "fc famalicao",
    "fc фамаликан": "fc famalicao",
    "fc famalicão": "fc famalicao",
    "жил vicente": "gil vicente",
    "жил виcente": "gil vicente",
    "gil vicente fc": "gil vicente",
    "gil vicente": "gil vicente",
    "cs маритиму": "cs maritimo",
    "маритиму": "cs maritimo",
    "cs marítimo": "cs maritimo",
    "maritimo": "cs maritimo",
    "fc alverca": "fc alverca",
    "fc алверка": "fc alverca",
    "алверка": "fc alverca",
    "rio ave": "rio ave fc",
    "rio ave fc": "rio ave fc",
    "риу аве": "rio ave fc",
    "риу аве fc": "rio ave fc",
    "sporting clube de portugal": "sporting cp",
    "sporting cp": "sporting cp",
    "fc arouca": "fc arouca",
    "ароука": "fc arouca",

    # ===== Нидерланды. Эредивизи =====
    "аякс": "ajax",
    "psv": "psv",
    "псв": "psv",
    "psv eindhoven": "psv",
    "фейеноорд": "feyenoord",
    "фейеноорд роттердам": "feyenoord",
    "az alkmaar": "az alkmaar",
    "az алкмар": "az alkmaar",
    "az": "az alkmaar",
    "utrecht": "fc utrecht",
    "fc utrecht": "fc utrecht",
    "утрехт": "fc utrecht",
    "твенте": "fc twente",
    "fc twente": "fc twente",
    "sparta rotterdam": "sparta rotterdam",
    "спарта роттердам": "sparta rotterdam",
    "heerenveen": "sc heerenveen",
    "sc heerenveen": "sc heerenveen",
    "хееренвен": "sc heerenveen",
    "херенвен": "sc heerenveen",
    "sc херенвен": "sc heerenveen",
    "ado den haag": "ado den haag",
    "адо ден хааг": "ado den haag",
    "sc cambuur-leeuwarden": "sc cambuur",
    "sc cambuur": "sc cambuur",
    "камбюр": "sc cambuur",
    "sc камбюр": "sc cambuur",
    "sc камбюр-леуварден": "sc cambuur",
    "willem ii tilburg": "willem ii",
    "willem ii": "willem ii",
    "виллем ii": "willem ii",
    "фортуна ситтард": "fortuna sittard",
    "fortuna sittard": "fortuna sittard",
    "sbv excelsior": "excelsior",
    "excelsior": "excelsior",
    "эксельсиор": "excelsior",

    # ===== Турция. Суперлига =====
    "галатасарай": "galatasaray",
    "galatasaray": "galatasaray",
    "фенербахче": "fenerbahce",
    "fenerbahçe": "fenerbahce",
    "besiktas": "besiktas",
    "бешикташ": "besiktas",
    "trabzonspor": "trabzonspor",
    "трабзонспор": "trabzonspor",
    "башакшехир": "istanbul basaksehir",
    "istanbul basaksehir": "istanbul basaksehir",
    "kasimpasa": "kasimpasa",
    "касымпаша": "kasimpasa",
    "konyaspor": "konyaspor",
    "коньяспор": "konyaspor",
    "alanyaspor": "alanyaspor",
    "аланияспор": "alanyaspor",
    "antalyaspor": "antalyaspor",
    "антальяспор": "antalyaspor",
    "sivasspor": "sivasspor",
    "сивасспор": "sivasspor",
    "kayserispor": "kayserispor",
    "кайсериспор": "kayserispor",
    "gaziantep": "gaziantep fk",
    "газиантеп": "gaziantep fk",
    "rizespor": "caykur rizespor",
    "ризеспор": "caykur rizespor",
    "samsunspor": "samsunspor",
    "самсунспор": "samsunspor",
    "bodrum": "bodrum fk",
    "бодрум": "bodrum fk",
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


def _norm_team_ru(s: str) -> str:
    """Нормализация русских названий команд."""
    return re.sub(r"[^а-я0-9]", "", (s or "").lower())


def _parse_events(html: str) -> list:
    """Парсит HTML страницы category → список матчей."""
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
    """Загружает и парсит события одной лиги."""
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


def _translate_ru_to_en(team_ru: str) -> str:
    """Переводит русское название команды в английское."""
    if not team_ru:
        return ""
    key = _norm_team_ru(team_ru)
    if key in RU_TO_EN_TEAM:
        return RU_TO_EN_TEAM[key]
    # Поиск по подстроке
    for ru, en in RU_TO_EN_TEAM.items():
        if ru in key or key in ru:
            return en
    return key


def marathon_odds(home_en: str, away_en: str, date_iso: str = "",
                  div_code: str = "") -> Optional[dict]:
    """
    Ищет кэфы Марафона для матча.
    home_en, away_en — английские названия из нашей базы.
    div_code — код лиги (E0, SP1, ...).
    """
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

            ev_home_en = _translate_ru_to_en(ev_home_ru)
            ev_away_en = _translate_ru_to_en(ev_away_ru)

            # Если словарь не нашёл — используем нормализованное русское
            if not ev_home_en:
                ev_home_en = _norm(ev_home_ru)
            if not ev_away_en:
                ev_away_en = _norm(ev_away_ru)

            home_match = (hn == ev_home_en or hn in ev_home_en
                          or ev_home_en in hn)
            away_match = (an == ev_away_en or an in ev_away_en
                          or ev_away_en in an)

            if home_match and away_match:
                _log(f"FOUND: {ev_home_ru} ({ev_home_en}) vs "
                     f"{ev_away_ru} ({ev_away_en}) = "
                     f"П1 {ev['odds'].get('П1')} / "
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
