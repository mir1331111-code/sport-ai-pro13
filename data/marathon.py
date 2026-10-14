"""data/marathon.py — парсинг кэфов Марафон/Прогнозист (fallback)."""
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
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
        "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
        "Referer": "https://www.prognozist.ru/",
    })
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


_sess = _session()


def _norm(s: str) -> str:
    return re.sub(r"[^a-zа-я0-9]", "", (s or "").lower())


def marathon_odds(home: str, away: str, date_iso: str) -> Optional[dict]:
    """
    Ищет кэфы Марафона через Прогнозист.ру.
    Возвращает {'П1': 2.1, 'X': 3.2, 'П2': 3.5} или None.

    ВНИМАНИЕ: это парсинг HTML — неофициальный способ.
    Используется как fallback, если The Odds API не дал кэфы.
    """
    if not home or not away or not date_iso:
        return None

    ck = f"marathon_{_norm(home)}_{_norm(away)}_{date_iso}"
    cached = cache_get(ck, 7200)  # 2 часа
    if cached is not None:
        return cached or None

    # Пытаемся найти страницу матча на prognozist.ru
    # Формат URL может быть разный, пробуем поиск
    try:
        search_url = "https://www.prognozist.ru/search"
        params = {"q": f"{home} {away}"}
        r = _sess.get(search_url, params=params, timeout=15, proxies=NO_PROXY)

        if r.status_code != 200:
            _log(f"search HTTP {r.status_code}")
            cache_put(ck, None)
            return None

        # Ищем ссылку на матч в результатах поиска
        # Это очень упрощённый парсер — реальная структура сайта может отличаться
        html = r.text

        # Ищем блоки с кэфами (примерный regex)
        # П1 — X — П2
        odds_pattern = re.search(
            r'(?:П1|1)[^0-9]{0,10}(\d+[.,]\d+).{0,50}'
            r'(?:X|Ничья)[^0-9]{0,10}(\d+[.,]\d+).{0,50}'
            r'(?:П2|2)[^0-9]{0,10}(\d+[.,]\d+)',
            html, re.IGNORECASE | re.DOTALL
        )

        if odds_pattern:
            p1 = float(odds_pattern.group(1).replace(",", "."))
            px = float(odds_pattern.group(2).replace(",", "."))
            p2 = float(odds_pattern.group(3).replace(",", "."))
            result = {"П1": p1, "X": px, "П2": p2, "source": "marathon"}
            _log(f"FOUND: {home} vs {away} — П1={p1}, X={px}, П2={p2}")
            cache_put(ck, result)
            return result

        _log(f"NOT FOUND: {home} vs {away}")
        cache_put(ck, None)
        return None

    except Exception as e:
        _log(f"ERROR: {type(e).__name__}: {str(e)[:100]}")
        return None
