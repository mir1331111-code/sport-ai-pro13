"""API-Football pre-match odds adapter.

Uses the API-Football v3 /odds?date=YYYY-MM-DD endpoint, which returns
fixture/team metadata together with bookmaker markets. Only explicit prices
from the API are returned; missing markets stay missing.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from datetime import datetime

import requests

from security import cache_get, cache_put

log = logging.getLogger(__name__)
BASE_URL = "https://v3.football.api-sports.io"


def _normalise(value: str) -> str:
    try:
        from data.team_names import normalize_team
        return normalize_team(value or "")
    except Exception:
        return re.sub(r"[^a-zа-я0-9]", "", (value or "").lower())


def _close_team(a: str, b: str) -> bool:
    a, b = _normalise(a), _normalise(b)
    return bool(a and b and (a == b or (len(a) >= 6 and a in b) or (len(b) >= 6 and b in a)))


def _request_page(date: str, page: int, api_key: str, logs=None):
    key = f"api_football_odds_v1_{date}_{page}"
    cached = cache_get(key, 900)
    if cached is not None:
        return cached
    try:
        response = requests.get(
            f"{BASE_URL}/odds",
            params={"date": date, "page": page},
            headers={"x-apisports-key": api_key},
            timeout=(5, 20),
        )
        if response.status_code != 200:
            detail = (response.text or "").replace("\n", " ")[:140]
            if logs is not None:
                logs.append(f"API-Football HTTP {response.status_code}: {detail}")
            return None
        payload = response.json()
        if not isinstance(payload, dict) or payload.get("errors"):
            if logs is not None:
                logs.append(f"API-Football API errors: {str((payload or {}).get('errors'))[:140]}")
            return None
        cache_put(key, payload)
        return payload
    except Exception as exc:
        if logs is not None:
            logs.append(f"API-Football request error: {type(exc).__name__}")
        log.debug("API-Football odds request failed: %s", type(exc).__name__)
        return None


def _parse_odds(event: dict) -> dict:
    acc = defaultdict(list)
    teams = event.get("teams") or {}
    home_name = (teams.get("home") or {}).get("name") or ""
    away_name = (teams.get("away") or {}).get("name") or ""
    for bookmaker in event.get("bookmakers") or []:
        for bet in bookmaker.get("bets") or []:
            market = re.sub(r"[^a-z0-9]+", " ", str(bet.get("name") or "").lower()).strip()
            for item in bet.get("values") or []:
                label = str(item.get("value") or "").strip().lower()
                try:
                    price = float(item.get("odd"))
                except (TypeError, ValueError):
                    continue
                if not 1.01 < price <= 100:
                    continue
                if market in ("match winner", "1x2", "fulltime result", "full time result"):
                    if label in ("home", "1"):
                        acc["П1"].append(price)
                    elif label in ("draw", "x"):
                        acc["X"].append(price)
                    elif label in ("away", "2"):
                        acc["П2"].append(price)
                elif "over under" in market or "goals over under" in market:
                    if re.search(r"over\s*2[.,]5", label):
                        acc["ТБ 2.5"].append(price)
                    elif re.search(r"under\s*2[.,]5", label):
                        acc["ТМ 2.5"].append(price)
                elif "both teams score" in market or market == "btts":
                    if label in ("yes", "both teams to score yes"):
                        acc["BTTS да"].append(price)
                    elif label in ("no", "both teams to score no"):
                        acc["BTTS нет"].append(price)
    # Average only the prices that are explicitly supplied by bookmakers.
    out = {market: sum(values) / len(values) for market, values in acc.items() if values}
    if not (all(k in out for k in ("П1", "X", "П2")) or
            all(k in out for k in ("ТБ 2.5", "ТМ 2.5")) or
            all(k in out for k in ("BTTS да", "BTTS нет"))):
        # Keep partial markets too, because scanner selects by model-market coverage.
        return out
    return out


def api_football_odds_fixture(home: str, away: str, match_date: str,
                              api_key: str, logs=None):
    """Find a match by exact-ish team names and return available real markets."""
    if not api_key or not home or not away or not match_date:
        return None
    try:
        parsed_date = datetime.strptime(str(match_date)[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        if logs is not None:
            logs.append("API-Football: дата матча не распознана")
        return None

    match_key = f"api_football_match_v1_{parsed_date}_{_normalise(home)}_{_normalise(away)}"
    cached = cache_get(match_key, 900)
    if cached is not None:
        if logs is not None:
            logs.append("API-Football: матч найден в кэше" if cached else "API-Football: пустой кэш матча")
        return cached or None

    max_pages = 5  # Avoid consuming the daily quota on a single scan.
    pages_checked = 0
    total_pages = 1
    for page in range(1, max_pages + 1):
        if page > total_pages:
            break
        payload = _request_page(parsed_date, page, api_key, logs)
        if payload is None:
            return None
        pages_checked += 1
        paging = payload.get("paging") or {}
        try:
            total_pages = max(1, int(paging.get("total") or 1))
        except (TypeError, ValueError):
            total_pages = 1
        for event in payload.get("response") or []:
            teams = event.get("teams") or {}
            h = (teams.get("home") or {}).get("name") or ""
            a = (teams.get("away") or {}).get("name") or ""
            if _close_team(home, h) and _close_team(away, a):
                odds = _parse_odds(event)
                cache_put(match_key, odds or None)
                if logs is not None:
                    if odds:
                        logs.append(f"API-Football: найден матч {h} — {a}, рынков {len(odds)}")
                    else:
                        logs.append(f"API-Football: матч {h} — {a} найден, но котировки отсутствуют")
                return odds or None
    if logs is not None:
        suffix = f"; просмотрено страниц {pages_checked}/{total_pages}" if total_pages > max_pages else ""
        logs.append(f"API-Football: матч не найден в выдаче за {parsed_date} — {home} / {away}{suffix}")
    # Do not cache a miss: another page may become available later.
    return None
