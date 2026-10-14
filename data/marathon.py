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


# Маппинг наших лиг → category ID на Прогнозист.ру
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
    """Нормализация имён команд."""
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


def _parse_events(html: str) -> list:
    """Парсит HTML страницы category и возвращает список матчей."""
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

    ck = f"marathon_cat_v3_{cat_id}"
    cached = cache_get(ck, 1800)
    if cached is not None:
        return cached if isinstance(cached, list) else []

    url = f"https://prognozist.ru/line/marathon/event_line.php?category={cat_id}"
    try:
        r = _sess.get(url, timeout=20, proxies=NO_PROXY)
        if r.status_code != 200:
            # Не кэшируем HTTP-сбой как успешный пустой ответ.
            _log(f"{div_code} HTTP {r.status_code}")
            return []
        events = _parse_events(r.text)
        _log(f"{div_code} (cat={cat_id}): {len(events)} events")
        cache_put(ck, events)
        return events
    except Exception as e:
        _log(f"{div_code} error: {type(e).__name__}: {str(e)[:100]}")
        return []


# ============ РАСШИРЕННЫЙ МАППИНГ РУССКИХ КОМАНД ============
RU_TO_EN_TEAM = {
    # ===== АПЛ (E0) =====
    "эвертон": "everton", "челси": "chelsea",
    "манчестер сити": "manchester city",
    "манчестер юнайтед": "manchester united",
    "ливерпуль": "liverpool", "арсенал": "arsenal",
    "тоттенхэм": "tottenham", "тоттенхэм хотспур": "tottenham",
    "ньюкасл": "newcastle", "ньюкасл юнайтед": "newcastle",
    "астон вилла": "aston villa", "брайтон": "brighton",
    "брайтон энд хоув альбион": "brighton",
    "кристал пэлас": "crystal palace", "фулхэм": "fulham",
    "вест хэм": "west ham", "вест хэм юнайтед": "west ham",
    "борнмут": "bournemouth", "волверхэмптон": "wolverhampton",
    "волверхэмптон уондерерс": "wolverhampton",
    "ноттингем форест": "nottingham forest",
    "лестер": "leicester", "лестер сити": "leicester",
    "ипсвич": "ipswich", "ипсвич таун": "ipswich",
    "саутгемптон": "southampton", "бернли": "burnley",
    "уотфорд": "watford", "норвич": "norwich",
    "норвич сити": "norwich", "лидс": "leeds",
    "лидс юнайтед": "leeds", "халл сити": "hull",
    "ковентри сити": "coventry", "сандерленд": "sunderland",
    "брентфорд": "brentford",
    # ===== Чемпионшип (E1) =====
    "шеффилд юнайтед": "sheffield united",
    "шеффилд уэнсдей": "sheffield wednesday",
    "мидлсбро": "middlesbrough", "сток сити": "stoke",
    "суонси сити": "swansea", "кардифф сити": "cardiff",
    "престон норт энд": "preston",
    "куинз парк рейнджерс": "qpr", "кп": "qpr",
    "миллуолл": "millwall",
    "блекберн роверс": "blackburn",
    "портсмут": "portsmouth", "дребери каунти": "derby",
    "линкольн сити": "lincoln",
    # ===== Ла Лига (SP1) =====
    "реал мадрид": "real madrid", "барселона": "barcelona",
    "атлетико мадрид": "atletico madrid", "севилья": "sevilla",
    "бетис": "real betis", "реал бетис": "real betis",
    "реал сосьедад": "real sociedad",
    "атлетик бильбао": "athletic bilbao",
    "атлетик клуб": "athletic bilbao",
    "валенсия": "valencia", "вильярреал": "villarreal",
    "жирона": "girona", "сельта": "celta vigo",
    "сельта де виго": "celta vigo",
    "осасуна": "osasuna", "кадис": "cadiz",
    "мальорка": "mallorca", "лас пальмас": "las palmas",
    "алавес": "alaves", "депортиво алавес": "alaves",
    "райо вальекано": "rayo vallecano",
    "расинг сантандер": "racing santander",
    "хетафе": "getafe", "эспаньол": "espanyol",
    "гранада": "granada", "альмерия": "almeria",
    # ===== Серия A (I1) =====
    "интер": "inter", "интер милан": "inter",
    "милан": "ac milan", "ювентус": "juventus",
    "наполи": "napoli", "рома": "roma", "лацио": "lazio",
    "аталанта": "atalanta", "фиорентина": "fiorentina",
    "болонья": "bologna", "торино": "torino",
    "удинезе": "udinese", "кальяри": "cagliari",
    "дженоа": "genoa", "сампдория": "sampdoria",
    "сассуоло": "sassuolo", "эмполи": "empoli",
    "верона": "verona", "лечче": "lecce",
    "венеция": "venezia", "монца": "monza",
    "комо": "como", "парма": "parma",
    # ===== Бундеслига (D1) =====
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
    "хoffenheim": "hoffenheim", "хoffенхайм": "hoffenheim",
    "унион берлин": "union berlin",
    "хольштайн киль": "holstein kiel",
    "санкт-паули": "st pauli",
    "хейденхайм": "heidenheim",
    "бохум": "bochum", "дармштадт": "darmstadt",
    "кёльн": "koln", "гамбург": "hamburger sv",
    "ганновер": "hannover",
    # ===== Лига 1 (F1) =====
    "псж": "psg", "пари сен-жермен": "psg",
    "марсель": "marseille", "лион": "lyon",
    "олимпик лион": "lyon", "монако": "monaco",
    "лилль": "lille", "рeн": "rennes", "рен": "rennes",
    "страсбур": "strasbourg", "ницца": "nice",
    "ланс": "lens", "тулуза": "toulouse",
    "монпелье": "montpellier", "нант": "nantes",
    "реймс": "reims", "брест": "brest",
    "гавр": "le havre", "метц": "metz",
    "лорьян": "lorient", "анже": "angers",
    "клермон": "clermont", "париж": "paris fc",
    # ===== РПЛ (R1) =====
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
    # ===== Эредивизи (N1) =====
    "аякс": "ajax", "псв": "psv", "фейеноорд": "feyenoord",
    "алмере сити": "almere city",
    "адо ден хааг": "ado den haag",
    "камбюр": "cambuur", "sc камбюр-леуварден": "cambuur",
    "sparta rotterdam": "sparta rotterdam",
    "heerenveen": "heerenveen", "хееренвен": "heerenveen",
    "willem ii tilburg": "willem ii",
    "фортуна ситтард": "fortuna sittard",
    "эксельсиор": "excelsior",
    # ===== Примейра (P1) =====
    "порту": "fc porto", "бенфика": "benfica",
    "спортинг": "sporting cp", "спортинг лиссабон": "sporting cp",
    "брага": "braga", "витория": "vitoria",
    "боавишта": "boavista", "маритиму": "maritimo",
    "насьонал": "nacional", "фейренсе": "feirense",
    "фамаликан": "famalicao", "жил vicente": "gil vicente",
    "жил виcente": "gil vicente",
    "риу аве": "rio ave", "арока": "arouca",
    "алверка": "alverca",
    # ===== Суперлига (T1) =====
    "галатасарай": "galatasaray",
    "фенербахче": "fenerbahce",
    "бешикташ": "besiktas",
    "башакшехир": "istanbul basaksehir",
    "истанбул башакшехир": "istanbul basaksehir",
    "тrabzonспор": "trabzonspor", "трабзонспор": "trabzonspor",
    # ===== ЛЧ/ЛЕ/ЛК =====
    "севилья": "sevilla", "псв эйндховен": "psv",
    "спортинг лиссабон": "sporting cp",
}


def _find_team_match(team_en: str, team_ru: str) -> bool:
    """Консервативное сопоставление без опасного совпадения по подстроке."""
    te = _norm(team_en)
    tr = _norm(team_ru)
    if not te or not tr:
        return False
    if te == tr:
        return True
    en_from_ru = RU_TO_EN_TEAM.get(tr, "")
    return bool(en_from_ru and te == _norm(en_from_ru))


def _event_date_matches(date_iso: str, date_str: str) -> bool:
    """Если дата известна в обоих источниках, требует совпадения дня."""
    if not date_iso:
        return True
    expected = str(date_iso).strip()[:10]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", expected):
        return True
    actual = str(date_str or "").strip()
    parsed = None
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d", "%d.%m.%y"):
        try:
            parsed = datetime.strptime(actual[:10], fmt).date()
            break
        except ValueError:
            continue
    if parsed is None:
        return False
    try:
        return parsed == datetime.strptime(expected, "%Y-%m-%d").date()
    except ValueError:
        return False


def marathon_odds(home_en: str, away_en: str, date_iso: str = "",
                  div_code: str = "") -> Optional[dict]:
    """
    Ищет кэфы Марафона для матча (прямой вызов, загружает лигу сам).
    Обычно используется `fetch_league_events` + локальный кэш.
    """
    if not home_en or not away_en:
        return None

    divs_to_try = [div_code] if div_code else list(DIV_TO_CATEGORY.keys())

    for dv in divs_to_try:
        events = fetch_league_events(dv)
        for ev in events:
            if (
                _find_team_match(home_en, ev["home"])
                and _find_team_match(away_en, ev["away"])
                and _event_date_matches(date_iso, ev.get("date_str", ""))
            ):
                _log(f"FOUND: {ev['home']} vs {ev['away']} = "
                     f"П1 {ev['odds'].get('П1')} / X {ev['odds'].get('X')} / "
                     f"П2 {ev['odds'].get('П2')}")
                return {
                    "П1": ev["odds"].get("П1"),
                    "X": ev["odds"].get("X"),
                    "П2": ev["odds"].get("П2"),
                    "source": "marathon",
                }

    _log(f"NOT FOUND: {home_en} vs {away_en}")
    return None
