"""config.py — все константы приложения."""
from __future__ import annotations
import os

APP_VERSION = "13.0.0"
DATA_VERSION = 17

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(_APP_DIR, "neuro.db")
LOCAL_FILE = os.path.join(_APP_DIR, "neuro_local.json")
DISK_CACHE_DIR = os.path.join(_APP_DIR, "neuro_cache")
os.makedirs(DISK_CACHE_DIR, exist_ok=True)

AUTO_SETTLE_LIMIT = 20
AUTO_SETTLE_THROTTLE_SEC = 21600
LLM_DAILY_LIMIT = 20
LLM_TOP_N = 8
ODDS_LIMIT_DAILY = 25

CACHE_TTL = {
    "seasonal": 86400 * 3,
    "tsdb_day": 1800,
    "tsdb_past": 86400 * 3,
    "tsdb_result": 86400,
    "odds": 7200,
    "llm": 86400 * 3,
    "engine": 86400 * 14,
}

LLM_PROVIDERS = {
    "Groq (бесплатно, быстро)": {
        "base": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_url": "https://console.groq.com/keys",
    },
    "Gemini (Google, бесплатно)": {
        "base": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "Grok (x.ai)": {
        "base": "https://api.x.ai/v1",
        "model": "grok-beta",
        "key_url": "https://console.x.ai",
    },
    "OpenRouter (Llama 3.3)": {
        "base": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "key_url": "https://openrouter.ai/keys",
    },
    "OpenAI (gpt-4o-mini)": {
        "base": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "DeepSeek (дёшево)": {
        "base": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "key_url": "https://platform.deepseek.com/api_keys",
    },
}

DIV_NAMES = {
    "E0": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 АПЛ", "E1": "🏴󠁧󠁢󠁥󠁮󠁧󠁿 Чемпионшип",
    "D1": "🇩🇪 Бундеслига", "D2": "🇩🇪 2.Бундеслига",
    "I1": "🇮🇹 Серия A", "I2": "🇮🇹 Серия B",
    "SP1": "🇪🇸 Ла Лига", "SP2": "🇪🇸 Сегунда",
    "F1": "🇫🇷 Лига 1", "F2": "🇫🇷 Лига 2",
    "N1": "🇳🇱 Эредивизи", "B1": "🇧🇪 Про-лига",
    "P1": "🇵🇹 Примейра", "T1": "🇹🇷 Суперлига",
    "G1": "🇬🇷 Греция", "R1": "🇷🇺 РПЛ",
    "C1": "🏆 Лига Чемпионов", "EL": "🏆 Лига Европы",
    "EC": "🏆 Лига Конференций",
}

DIV_TO_ODDS = {
    "E0": "soccer_epl", "E1": "soccer_efl_champ",
    "D1": "soccer_germany_bundesliga", "D2": "soccer_germany_bundesliga2",
    "I1": "soccer_italy_serie_a", "I2": "soccer_italy_serie_b",
    "SP1": "soccer_spain_la_liga", "SP2": "soccer_spain_segunda",
    "F1": "soccer_france_ligue_one", "F2": "soccer_france_ligue_two",
    "N1": "soccer_netherlands_eredivisie", "B1": "soccer_belgium_first_div",
    "P1": "soccer_portugal_primeira_liga", "T1": "soccer_turkey_super_league",
    "G1": "soccer_greece_super_league",
    "C1": "soccer_uefa_champs_league", "EL": "soccer_uefa_europa_league",
}

STADIUM_WALLS = {
    "E0": "linear-gradient(135deg, rgba(30,64,175,.55), rgba(15,23,42,.95))",
    "E1": "linear-gradient(135deg, rgba(37,99,235,.45), rgba(15,23,42,.95))",
    "SP1": "linear-gradient(135deg, rgba(220,38,38,.55), rgba(15,23,42,.95))",
    "SP2": "linear-gradient(135deg, rgba(239,68,68,.45), rgba(15,23,42,.95))",
    "I1": "linear-gradient(135deg, rgba(22,163,74,.55), rgba(15,23,42,.95))",
    "I2": "linear-gradient(135deg, rgba(34,197,94,.45), rgba(15,23,42,.95))",
    "D1": "linear-gradient(135deg, rgba(202,138,4,.55), rgba(15,23,42,.95))",
    "D2": "linear-gradient(135deg, rgba(234,179,8,.45), rgba(15,23,42,.95))",
    "F1": "linear-gradient(135deg, rgba(37,99,235,.55), rgba(30,58,138,.55))",
    "F2": "linear-gradient(135deg, rgba(59,130,246,.45), rgba(30,58,138,.45))",
    "N1": "linear-gradient(135deg, rgba(249,115,22,.55), rgba(15,23,42,.95))",
    "B1": "linear-gradient(135deg, rgba(202,138,4,.45), rgba(120,53,15,.55))",
    "P1": "linear-gradient(135deg, rgba(22,163,74,.55), rgba(220,38,38,.45))",
    "T1": "linear-gradient(135deg, rgba(220,38,38,.55), rgba(202,138,4,.45))",
    "G1": "linear-gradient(135deg, rgba(59,130,246,.55), rgba(15,23,42,.95))",
    "R1": "linear-gradient(135deg, rgba(220,38,38,.55), rgba(30,58,138,.55))",
    "C1": "linear-gradient(135deg, rgba(139,92,246,.55), rgba(15,23,42,.95))",
    "EL": "linear-gradient(135deg, rgba(249,115,22,.55), rgba(15,23,42,.95))",
    "EC": "linear-gradient(135deg, rgba(34,197,94,.55), rgba(15,23,42,.95))",
    "DEFAULT": "linear-gradient(135deg, rgba(71,85,105,.55), rgba(15,23,42,.95))",
}

TEAM_TRANSLATIONS = {
    "Manchester United": "Манчестер Юнайтед",
    "Manchester City": "Манчестер Сити",
    "Liverpool": "Ливерпуль", "Arsenal": "Арсенал", "Chelsea": "Челси",
    "Tottenham": "Тоттенхэм", "Newcastle": "Ньюкасл", "Aston Villa": "Астон Вилла",
    "Brighton": "Брайтон", "West Ham": "Вест Хэм", "Everton": "Эвертон",
    "Fulham": "Фулхэм", "Crystal Palace": "Кристал Пэлас", "Brentford": "Брентфорд",
    "Nottingham Forest": "Ноттингем Форест", "Wolverhampton": "Вулверхэмптон",
    "Wolves": "Вулверхэмптон", "Bournemouth": "Борнмут", "Leicester": "Лестер",
    "Southampton": "Саутгемптон", "Ipswich": "Ипсвич", "Leeds": "Лидс",
    "Burnley": "Бёрнли", "Watford": "Уотфорд", "Norwich": "Норвич",
    "Real Madrid": "Реал Мадрид", "Barcelona": "Барселона",
    "Atletico Madrid": "Атлетико Мадрид", "Sevilla": "Севилья",
    "Real Betis": "Бетис", "Real Sociedad": "Реал Сосьедад",
    "Athletic Bilbao": "Атлетик Бильбао", "Valencia": "Валенсия",
    "Villarreal": "Вильярреал", "Girona": "Жирона",
    "Inter": "Интер", "Inter Milan": "Интер", "AC Milan": "Милан",
    "Juventus": "Ювентус", "Napoli": "Наполи", "Roma": "Рома",
    "Lazio": "Лацио", "Atalanta": "Аталанта", "Fiorentina": "Фиорентина",
    "Bayern Munich": "Бавария", "Borussia Dortmund": "Боруссия Дортмунд",
    "RB Leipzig": "РБ Лейпциг", "Bayer Leverkusen": "Байер Леверкузен",
    "PSG": "ПСЖ", "Paris Saint-Germain": "ПСЖ", "Marseille": "Марсель",
    "Lyon": "Лион", "Monaco": "Монако", "Lille": "Лилль",
    "Zenit": "Зенит", "Spartak Moscow": "Спартак", "CSKA Moscow": "ЦСКА",
    "Lokomotiv Moscow": "Локомотив", "Dynamo Moscow": "Динамо",
    "Krasnodar": "Краснодар", "Rostov": "Ростов",
    "FC Porto": "Порту", "Benfica": "Бенфика", "Sporting CP": "Спортинг",
    "Ajax": "Аякс", "PSV": "ПСВ", "Feyenoord": "Фейеноорд",
}


# ============ FOOTBALL-DATA.ORG ============
FOOTBALL_DATA_ORG_HOST = "api.football-data.org"
FOOTBALL_DATA_ORG_DAILY_LIMIT = 100

DIV_TO_FDORG = {
    "E0": "PL", "E1": "ELC", "SP1": "PD",
    "I1": "SA", "D1": "BL1", "F1": "FL1",
    "N1": "DED", "P1": "PPL", "C1": "CL",
}
FDORG_TO_DIV = {v: k for k, v in DIV_TO_FDORG.items()}
