"""config.py — все константы приложения v13.3."""
from __future__ import annotations
import os

APP_VERSION = "13.3.0"
DATA_VERSION = 17

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(_APP_DIR, "neuro.db")
LOCAL_FILE = os.path.join(_APP_DIR, "neuro_local.json")
DISK_CACHE_DIR = os.path.join(_APP_DIR, "neuro_cache")
os.makedirs(DISK_CACHE_DIR, exist_ok=True)

AUTO_SETTLE_LIMIT = 999
AUTO_SETTLE_THROTTLE_SEC = 3600
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
        "model": "openai/gpt-oss-120b",
        "key_url": "https://console.groq.com/keys",
    },
    "Gemini (Google, бесплатно)": {
        "base": "https://generativelanguage.googleapis.com/v1beta",
        "model": "gemini-2.5-flash",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "Grok (x.ai)": {
        "base": "https://api.x.ai/v1",
        "model": "grok-beta",
        "key_url": "https://console.x.ai",
    },
    "OpenRouter (бесплатно)": {
        "base": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.1-8b-instruct:free",
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

# ============ МУЖСКИЕ КЛУБНЫЕ ЛИГИ (TheSportsDB ID) ============
DIV_TO_TSDB = {
    # Топ-лиги Европы
    "E0": "4328",   # English Premier League
    "E1": "4329",   # English Championship
    "D1": "4331",   # German Bundesliga
    "D2": "4389",   # German Bundesliga 2
    "I1": "4332",   # Italian Serie A
    "I2": "4388",   # Italian Serie B
    "SP1": "4335",  # Spanish La Liga
    "SP2": "4340",  # Spanish Segunda
    "F1": "4334",   # French Ligue 1
    "F2": "4387",   # French Ligue 2
    "N1": "4337",   # Dutch Eredivisie
    "B1": "4338",   # Belgian Pro League
    "P1": "4344",   # Portuguese Primeira Liga
    "T1": "4339",   # Turkish Super Lig
    "G1": "4336",   # Greek Super League
    "R1": "4357",   # Russian Premier League
    # Первые лиги мира
    "SC1": "4330",  # Scottish Premiership
    "MLS1": "4346", # MLS USA
    "BR1": "4351",  # Brazil Serie A
    "AR1": "4354",  # Argentina Primera
    "AT1": "4358",  # Austrian Bundesliga
    "CH1": "4359",  # Swiss Super League
    "NO1": "4360",  # Norway Eliteserien
    "SE1": "4361",  # Sweden Allsvenskan
    "DK1": "4362",  # Denmark Superligaen
    "PL1": "4363",  # Poland Ekstraklasa
    "JP1": "4365",  # Japan J1 League
    "KR1": "4366",  # Korea K League 1
    "MX1": "4367",  # Mexico Liga MX
    "CO1": "4368",  # Colombia Primera A
    # Европейские кубки
    "C1": "4480",   # UEFA Champions League
    "EL": "4481",   # UEFA Europa League
    "EC": "4482",   # UEFA Conference League
}

# ============ ЖЕНСКИЕ ЛИГИ ============
WOMEN_LEAGUE_IDS = {
    "4424": ("W_E0", "WSL"),
    "4483": ("W_SP1", "Liga F"),
    "4484": ("W_F1", "D1 Féminine"),
    "4485": ("W_D1", "Frauen-BL"),
    "4486": ("W_I1", "Serie A Fem"),
    "4487": ("W_C1", "UWCL"),
}

# ============ МАТЧИ СБОРНЫХ ============
NATIONAL_TEAM_LEAGUES = {
    "4502": ("NT_UNL", "🏆 Лига Наций УЕФА"),
    "4498": ("NT_WCE", "⚽ Отбор ЧМ Европа"),
    "4501": ("NT_WCC", "⚽ Отбор ЧМ Юж. Америка"),
    "4499": ("NT_WCA", "⚽ Отбор ЧМ Азия"),
    "4500": ("NT_WCF", "⚽ Отбор ЧМ Африка"),
    "4503": ("NT_WCN", "⚽ Отбор ЧМ Сев. Америка"),
    "4497": ("NT_EQ", "🇪🇺 Отбор Евро"),
    "4504": ("NT_CA", "🏆 Копа Америка"),
    "4505": ("NT_ACN", "🏆 Кубок Африки"),
    "4506": ("NT_ASC", "🏆 Кубок Азии"),
    "4507": ("NT_FR", "🤝 Товарищеские"),
    "4508": ("NT_EURO", "🏆 Чемпионат Европы"),
    "4509": ("NT_WC", "🏆 Чемпионат Мира"),
}

DIV_NAMES = {
    "E0": "🏴 АПЛ", "E1": "🏴󠁧󠁮󠁿 Чемпионшип",
    "D1": "🇩 Бундеслига", "D2": "🇩🇪 2.Бундеслига",
    "I1": "🇮 Серия A", "I2": "🇮 Серия B",
    "SP1": "🇪 Ла Лига", "SP2": "🇪🇸 Сегунда",
    "F1": "🇫🇷 Лига 1", "F2": "🇫🇷 Лига 2",
    "N1": "🇳 Эредивизи", "B1": "🇧 Про-лига",
    "P1": "🇵 Примейра", "T1": "🇹 Суперлига",
    "G1": "🇬 Греция", "R1": "🇷 РПЛ",
    "SC1": "🏴 Шотландия", "MLS1": "🇺 MLS",
    "BR1": "🇧 Бразилия", "AR1": "🇦 Аргентина",
    "AT1": "🇦 Австрия", "CH1": "🇨 Швейцария",
    "NO1": "🇳 Норвегия", "SE1": "🇸 Швеция",
    "DK1": "🇩 Дания", "PL1": "🇵 Польша",
    "JP1": "🇯 Япония", "KR1": "🇰 Корея",
    "MX1": "🇲 Мексика", "CO1": "🇨 Колумбия",
    "C1": "🏆 ЛЧ", "EL": "🏆 ЛЕ", "EC": "🏆 ЛК",
    "W_E0": "🏴 WSL (Ж)", "W_SP1": "🇪 Liga F (Ж)",
    "W_F1": "🇫🇷 D1 Féminine (Ж)", "W_D1": "🇩 Frauen-BL (Ж)",
    "W_I1": "🇮🇹 Serie A Fem (Ж)", "W_C1": "🏆 UWCL (Ж)",
    "NT_UNL": "🏆 Лига Наций", "NT_WCE": "⚽ Отбор ЧМ ЕС",
    "NT_WCC": "⚽ Отбор ЧМ ЮА", "NT_WCA": "⚽ Отбор ЧМ АЗ",
    "NT_WCF": "⚽ Отбор ЧМ АФ", "NT_WCN": "⚽ Отбор ЧМ СА",
    "NT_EQ": "🇪 Отбор Евро", "NT_CA": "🏆 Копа Америка",
    "NT_ACN": "🏆 Кубок Африки", "NT_ASC": "🏆 Кубок Азии",
    "NT_FR": "🤝 Товарищеские", "NT_EURO": "🏆 Евро",
    "NT_WC": "🏆 ЧМ",
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
    "SC1": "linear-gradient(135deg, rgba(0,102,204,.55), rgba(15,23,42,.95))",
    "MLS1": "linear-gradient(135deg, rgba(200,200,220,.15), rgba(15,23,42,.95))",
    "BR1": "linear-gradient(135deg, rgba(0,156,59,.55), rgba(15,23,42,.95))",
    "AR1": "linear-gradient(135deg, rgba(116,172,223,.55), rgba(15,23,42,.95))",
    "AT1": "linear-gradient(135deg, rgba(237,41,57,.55), rgba(15,23,42,.95))",
    "CH1": "linear-gradient(135deg, rgba(255,0,0,.45), rgba(15,23,42,.95))",
    "NO1": "linear-gradient(135deg, rgba(0,40,104,.55), rgba(15,23,42,.95))",
    "SE1": "linear-gradient(135deg, rgba(0,86,158,.55), rgba(15,23,42,.95))",
    "DK1": "linear-gradient(135deg, rgba(198,12,48,.55), rgba(15,23,42,.95))",
    "PL1": "linear-gradient(135deg, rgba(220,20,60,.55), rgba(15,23,42,.95))",
    "JP1": "linear-gradient(135deg, rgba(188,0,45,.55), rgba(15,23,42,.95))",
    "KR1": "linear-gradient(135deg, rgba(0,51,153,.55), rgba(15,23,42,.95))",
    "MX1": "linear-gradient(135deg, rgba(0,104,71,.55), rgba(15,23,42,.95))",
    "CO1": "linear-gradient(135deg, rgba(252,209,22,.55), rgba(15,23,42,.95))",
    "C1": "linear-gradient(135deg, rgba(139,92,246,.55), rgba(15,23,42,.95))",
    "EL": "linear-gradient(135deg, rgba(249,115,22,.55), rgba(15,23,42,.95))",
    "EC": "linear-gradient(135deg, rgba(34,197,94,.55), rgba(15,23,42,.95))",
    "W_E0": "linear-gradient(135deg, rgba(236,72,153,.55), rgba(15,23,42,.95))",
    "W_SP1": "linear-gradient(135deg, rgba(244,114,182,.55), rgba(15,23,42,.95))",
    "W_F1": "linear-gradient(135deg, rgba(217,70,239,.55), rgba(15,23,42,.95))",
    "W_D1": "linear-gradient(135deg, rgba(232,121,249,.55), rgba(15,23,42,.95))",
    "W_I1": "linear-gradient(135deg, rgba(192,132,252,.55), rgba(15,23,42,.95))",
    "W_C1": "linear-gradient(135deg, rgba(168,85,247,.55), rgba(15,23,42,.95))",
    "NT_UNL": "linear-gradient(135deg, rgba(251,191,36,.55), rgba(15,23,42,.95))",
    "NT_WCE": "linear-gradient(135deg, rgba(59,130,246,.55), rgba(15,23,42,.95))",
    "NT_WCC": "linear-gradient(135deg, rgba(34,197,94,.55), rgba(15,23,42,.95))",
    "NT_WCA": "linear-gradient(135deg, rgba(249,115,22,.55), rgba(15,23,42,.95))",
    "NT_WCF": "linear-gradient(135deg, rgba(220,38,38,.55), rgba(15,23,42,.95))",
    "NT_WCN": "linear-gradient(135deg, rgba(168,85,247,.55), rgba(15,23,42,.95))",
    "NT_EQ": "linear-gradient(135deg, rgba(14,165,233,.55), rgba(15,23,42,.95))",
    "NT_CA": "linear-gradient(135deg, rgba(236,72,153,.55), rgba(15,23,42,.95))",
    "NT_ACN": "linear-gradient(135deg, rgba(234,179,8,.55), rgba(15,23,42,.95))",
    "NT_ASC": "linear-gradient(135deg, rgba(20,184,166,.55), rgba(15,23,42,.95))",
    "NT_FR": "linear-gradient(135deg, rgba(148,163,184,.45), rgba(15,23,42,.95))",
    "NT_EURO": "linear-gradient(135deg, rgba(245,158,11,.55), rgba(15,23,42,.95))",
    "NT_WC": "linear-gradient(135deg, rgba(220,38,38,.55), rgba(59,130,246,.45))",
    "DEFAULT": "linear-gradient(135deg, rgba(71,85,105,.55), rgba(15,23,42,.95))",
}

TEAM_TRANSLATIONS = {
    "Manchester United": "Манчестер Юнайтед",
    "Manchester City": "Манчестер Сити",
    "Liverpool": "Ливерпуль", "Arsenal": "Арсенал", "Chelsea": "Челси",
    "Tottenham": "Тоттенхэм", "Newcastle": "Ньюкасл", "Aston Villa": "Астон Вилла",
    "Brighton": "Брайтон", "West Ham": "Вест Хэм", "Everton": "Эвертон",
    "Real Madrid": "Реал Мадрид", "Barcelona": "Барселона",
    "Atletico Madrid": "Атлетико Мадрид", "Sevilla": "Севилья",
    "Inter": "Интер", "AC Milan": "Милан", "Juventus": "Ювентус",
    "Napoli": "Наполи", "Roma": "Рома", "Lazio": "Лацио", "Atalanta": "Аталанта",
    "Bayern Munich": "Бавария", "Borussia Dortmund": "Боруссия Дортмунд",
    "RB Leipzig": "РБ Лейпциг", "Bayer Leverkusen": "Байер Леверкузен",
    "PSG": "ПСЖ", "Paris Saint-Germain": "ПСЖ", "Marseille": "Марсель",
    "Lyon": "Лион", "Monaco": "Монако", "Lille": "Лилль",
    "Zenit": "Зенит", "Spartak Moscow": "Спартак", "CSKA Moscow": "ЦСКА",
    "FC Porto": "Порту", "Benfica": "Бенфика", "Sporting CP": "Спортинг",
    "Ajax": "Аякс", "PSV": "ПСВ", "Feyenoord": "Фейеноорд",
    "Celtic": "Селтик", "Rangers": "Рейнджерс",
    "Flamengo": "Фламенго", "Palmeiras": "Палмейрас",
    "Boca Juniors": "Бока Хуниорс", "River Plate": "Ривер Плейт",
    "Galatasaray": "Галатасарай", "Fenerbahçe": "Фенербахче",
    # Сборные
    "England": "Англия", "France": "Франция", "Germany": "Германия",
    "Spain": "Испания", "Italy": "Италия", "Portugal": "Португалия",
    "Brazil": "Бразилия", "Argentina": "Аргентина", "Uruguay": "Уругвай",
    "Colombia": "Колумбия", "Chile": "Чили", "Mexico": "Мексика",
    "USA": "США", "Canada": "Канада", "Japan": "Япония",
    "South Korea": "Южная Корея", "Australia": "Австралия",
    "Morocco": "Марокко", "Senegal": "Сенегал", "Egypt": "Египет",
    "Nigeria": "Нигерия", "Cameroon": "Камерун",
    "Saudi Arabia": "Саудовская Аравия", "Iran": "Иран",
    "Russia": "Россия", "Ukraine": "Украина", "Poland": "Польша",
    "Czech Republic": "Чехия", "Sweden": "Швеция", "Norway": "Норвегия",
    "Denmark": "Дания", "Finland": "Финляндия", "Iceland": "Исландия",
    "Netherlands": "Голландия", "Belgium": "Бельгия", "Switzerland": "Швейцария",
    "Austria": "Австрия", "Croatia": "Хорватия", "Serbia": "Сербия",
    "Turkey": "Турция", "Greece": "Греция", "Romania": "Румыния",
    "Hungary": "Венгрия", "Wales": "Уэльс", "Scotland": "Шотландия",
    "Ireland": "Ирландия",
}

FOOTBALL_DATA_ORG_HOST = "api.football-data.org"
FOOTBALL_DATA_ORG_DAILY_LIMIT = 100

DIV_TO_FDORG = {
    "E0": "PL", "E1": "ELC", "SP1": "PD", "I1": "SA",
    "D1": "BL1", "F1": "FL1", "N1": "DED", "P1": "PPL", "C1": "CL",
}
FDORG_TO_DIV = {v: k for k, v in DIV_TO_FDORG.items()}
