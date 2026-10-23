"""data/team_names.py — нормализация названий команд."""
from __future__ import annotations
import re

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


def normalize_team(name: str) -> str:
    """Normalize common aliases without merging distinct clubs or age groups."""
    if not name:
        return ""
    s = re.sub(r"\s+", " ", str(name).strip().lower())

    # Exact aliases must be resolved before removing generic club prefixes:
    # "Manchester United" and "Manchester City" are different teams.
    if s in ALIASES:
        return ALIASES[s]

    for prefix in ("fc ", "afc ", "cf ", "sc ", "ac ", "as ", "ss ", "us ", "rc ", "sk "):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
            break
    if s in ALIASES:
        return ALIASES[s]

    # Preserve age-group/reserve markers: an U21/reserve fixture must not be
    # normalized to the senior team just because its name contains an alias.
    if re.search(r"\b(?:u\s*\d{2}|under\s*[- ]?\s*\d{2}|youth|reserves?|ii)\b", s):
        return re.sub(r"[^a-zа-я0-9]", "", s)

    # Only remove suffixes that are organizational labels, not identity-bearing
    # words such as "United", "City" or "Utd".
    for suffix in (" fc", " afc", " cf", " sc", " ac", " town", " rovers", " county", " club"):
        if s.endswith(suffix):
            s = s[:-len(suffix)].strip()
            break
    if s in ALIASES:
        return ALIASES[s]

    # Substring aliases are a fallback for common provider variants only.
    for alias, canonical in sorted(ALIASES.items(), key=lambda kv: -len(kv[0])):
        if len(alias) >= 5 and alias in s:
            return canonical

    return re.sub(r"[^a-zа-я0-9]", "", s)


def teams_match(name1: str, name2: str) -> bool:
    """Проверяет, что два названия — одна и та же команда."""
    return normalize_team(name1) == normalize_team(name2)
