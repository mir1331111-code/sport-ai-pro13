"""data/team_names.py — нормализация названий команд."""
from __future__ import annotations
import re
import unicodedata

# Словарь алиасов: любое написание → каноническое имя
ALIASES = {
    # ===== Англия =====
    "man utd": "man_united",
    "manchester utd": "man_united",
    "manchester united": "man_united",
    "манчестер юнайтед": "man_united",
    "man city": "man_city",
    "manchester city": "man_city",
    "манчестер сити": "man_city",
    "newcastle": "newcastle",
    "newcastle united": "newcastle",
    "нюкасл": "newcastle",
    "nottm forest": "nottm_forest",
    "nottingham forest": "nottm_forest",
    "ноттингем форест": "nottm_forest",
    "tottenham": "tottenham",
    "tottenham hotspur": "tottenham",
    "тоттенхэм": "tottenham",
    "west ham": "west_ham",
    "west ham united": "west_ham",
    "вест хэм": "west_ham",
    "brighton": "brighton",
    "brighton & hove albion": "brighton",
    "брайтон": "brighton",
    "wolves": "wolves",
    "wolverhampton": "wolves",
    "вулверхэмптон": "wolves",
    "leicester": "leicester",
    "leicester city": "leicester",
    "лестер": "leicester",
    "leeds": "leeds",
    "leeds united": "leeds",
    "лидс": "leeds",
    
    # ===== Испания =====
    "real madrid": "real_madrid",
    "реал мадрид": "real_madrid",
    "barcelona": "barcelona",
    "barça": "barcelona",
    "barca": "barcelona",
    "барселона": "barcelona",
    "atletico": "atletico",
    "atletico madrid": "atletico",
    "атлетико мадрид": "atletico",
    "athletic": "athletic",
    "athletic bilbao": "athletic",
    "athletic club": "athletic",
    "атлетик бильбао": "athletic",
    "real sociedad": "real_sociedad",
    "реал сосьедад": "real_sociedad",
    "real betis": "real_betis",
    "бетис": "real_betis",
    "sevilla": "sevilla",
    "севилья": "sevilla",
    "villarreal": "villarreal",
    "вильярреал": "villarreal",
    "valencia": "valencia",
    "валенсия": "valencia",
    "girona": "girona",
    "жирона": "girona",
    "levante": "levante",
    "levante ud": "levante",
    "леванте": "levante",
    "osasuna": "osasuna",
    "осасуна": "osasuna",
    "celta": "celta",
    "celta vigo": "celta",
    "сельта": "celta",
    "alaves": "alaves",
    "deportivo alaves": "alaves",
    "алавес": "alaves",
    "rayo": "rayo",
    "rayo vallecano": "rayo",
    "райо": "rayo",
    
    # ===== Италия =====
    "inter": "inter",
    "internazionale": "inter",
    "fc internazionale": "inter",
    "fc internazionale milano": "inter",
    "интер": "inter",
    "ac milan": "milan",
    "milan": "milan",
    "милан": "milan",
    "juventus": "juventus",
    "ювентус": "juventus",
    "napoli": "napoli",
    "наполи": "napoli",
    "roma": "roma",
    "as roma": "roma",
    "рома": "roma",
    "lazio": "lazio",
    "лацио": "lazio",
    "atalanta": "atalanta",
    "аталанта": "atalanta",
    "fiorentina": "fiorentina",
    "фиорентина": "fiorentina",
    "como": "como",
    "como 1907": "como",
    "комо": "como",
    "udinese": "udinese",
    "удинезе": "udinese",
    "torino": "torino",
    "торино": "torino",
    "bologna": "bologna",
    "болонья": "bologna",
    "cagliari": "cagliari",
    "кальяри": "cagliari",
    "venezia": "venezia",
    "венеция": "venezia",
    "lecce": "lecce",
    "лечче": "lecce",
    
    # ===== Германия =====
    "bayern": "bayern",
    "bayern munich": "bayern",
    "bayern munchen": "bayern",
    "fc bayern": "bayern",
    "fc bayern munchen": "bayern",
    "бавария": "bayern",
    "dortmund": "dortmund",
    "borussia dortmund": "dortmund",
    "боруссия дортмунд": "dortmund",
    "rb leipzig": "rb_leipzig",
    "лейпциг": "rb_leipzig",
    "bayer leverkusen": "leverkusen",
    "leverkusen": "leverkusen",
    "байер леверкузен": "leverkusen",
    "frankfurt": "frankfurt",
    "eintracht frankfurt": "frankfurt",
    "франкфурт": "frankfurt",
    "freiburg": "freiburg",
    "sc freiburg": "freiburg",
    "фрайбург": "freiburg",
    "stuttgart": "stuttgart",
    "vfb stuttgart": "stuttgart",
    "штутгарт": "stuttgart",
    
    # ===== Франция =====
    "psg": "psg",
    "paris": "psg",
    "paris sg": "psg",
    "paris saint germain": "psg",
    "псж": "psg",
    "marseille": "marseille",
    "марсель": "marseille",
    "lyon": "lyon",
    "olympique lyonnais": "lyon",
    "лион": "lyon",
    "monaco": "monaco",
    "монако": "monaco",
    "lille": "lille",
    "лилль": "lille",
    "lens": "lens",
    "rc lens": "lens",
    "ланс": "lens",
    "nice": "nice",
    "ницца": "nice",
    
    # ===== Россия =====
    "zenit": "zenit",
    "зенит": "zenit",
    "spartak": "spartak",
    "spartak moscow": "spartak",
    "спартак": "spartak",
    "cska": "cska",
    "cska moscow": "cska",
    "цска": "cska",
    "dynamo": "dynamo",
    "dynamo moscow": "dynamo",
    "динамо": "dynamo",
    "lokomotiv": "lokomotiv",
    "lokomotiv moscow": "lokomotiv",
    "локомотив": "lokomotiv",
    "krasnodar": "krasnodar",
    "краснодар": "krasnodar",
    "rostov": "rostov",
    "ростов": "rostov",
    
    # ===== Португалия =====
    "porto": "porto",
    "fc porto": "porto",
    "порту": "porto",
    "benfica": "benfica",
    "бенфика": "benfica",
    "sporting": "sporting",
    "sporting cp": "sporting",
    "sporting lisbon": "sporting",
    "спортинг": "sporting",
    
    # ===== Нидерланды =====
    "ajax": "ajax",
    "аякс": "ajax",
    "psv": "psv",
    "psv eindhoven": "psv",
    "псв": "psv",
    "feyenoord": "feyenoord",
    "фейеноорд": "feyenoord",
    
    # ===== Турция =====
    "galatasaray": "galatasaray",
    "галатасарай": "galatasaray",
    "fenerbahce": "fenerbahce",
    "фенербахче": "fenerbahce",
    
    # ===== Бельгия =====
    "club brugge": "club_brugge",
    "brugge": "club_brugge",
    "брюгге": "club_brugge",
    
    # ===== Украина =====
    "shakhtar": "shakhtar",
    "shakhtar donetsk": "shakhtar",
    "шахтер": "shakhtar",
}


def _alias_key(value: str) -> str:
    """Canonical exact-match key; punctuation/accents vary across providers."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-zа-я0-9]+", " ", text).strip()


_ALIAS_LOOKUP = {_alias_key(alias): canonical for alias, canonical in ALIASES.items()}


def normalize_team(name: str) -> str:
    """Normalize exact provider aliases without fuzzy-merging different teams."""
    if not name:
        return ""

    key = _alias_key(name)
    if not key:
        return ""
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]

    # Keep reserves, B teams, academies and age groups separate from senior teams.
    # In particular, never use substring aliases such as "Atletico Madrid" to
    # turn "Atletico Madrid B" into the senior club.
    reserve_marker = re.search(
        r"(?:^| )(?:b|b team|ii|iii|iv|u\\s*\\d{2}|under\\s*[- ]?\\s*\\d{2}|"
        r"youth|reserves?|academy|castilla|atletic)(?: |$)",
        key,
    )
    if reserve_marker:
        return re.sub(r"[^a-zа-я0-9]", "", key)

    # Only strip known organizational prefixes/suffixes, then retry exact aliases.
    parts = key.split()
    prefixes = {"fc", "afc", "cf", "sc", "ac", "as", "ss", "us", "rc", "sk"}
    suffixes = {"fc", "afc", "cf", "sc", "ac", "town", "rovers", "county", "club"}
    if parts and parts[0] in prefixes:
        parts = parts[1:]
    if parts and parts[-1] in suffixes:
        parts = parts[:-1]
    key = " ".join(parts)
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]

    # Unknown clubs are compared conservatively by their complete normalized name.
    return re.sub(r"[^a-zа-я0-9]", "", key)

def teams_match(name1: str, name2: str) -> bool:
    """Проверяет, что два названия — одна и та же команда."""
    return normalize_team(name1) == normalize_team(name2)
