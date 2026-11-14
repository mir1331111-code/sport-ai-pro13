"""Shortlist selection and evidence-bounded batch analysis for top football matches."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from betting.ranking import model_favorite, rank_model_favorites
from config import CACHE_TTL, LLM_PROVIDERS
from betting.risk import MIN_BET_ODDS
from llm.analyst import (
    LLMRateLimitError,
    _extract_chat_completion,
    _retry_after_seconds,
    _log,
    _sess,
    has_sufficient_history,
)
from security import cache_get, cache_put
from storage import usage


MAX_TOP_MATCHES = 7
MIN_TOP_PROBABILITY = 0.65
MIN_TOP_TEAM_HISTORY = 5
DEEP_ANALYSIS_CACHE_TTL = 1800
NO_PROXY = {"http": None, "https": None}


def _number(value: object) -> float | None:
    try:
        number = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _complete_1x2_prices(card: dict) -> bool:
    """Require an SStats fixture ID and three real prices; this is not a Value gate."""
    if card.get("sstats_game_id") in (None, ""):
        return False
    prices = card.get("market_odds") or {}
    source = str(card.get("odds_source") or "").strip().casefold()
    if source in {"", "no_market", "model_only", "unknown"}:
        return False
    for key in ("П1", "X", "П2"):
        price = _number(prices.get(key))
        if price is None or not 1.01 < price <= 100:
            return False
    # The shortlist may be researched across markets, but it must contain at
    # least one real quote at the user's 1.70 floor before it can be promoted.
    quote_groups = (
        prices,
        card.get("total_market_odds") or {},
        card.get("btts_market_odds") or {},
        card.get("stat_market_odds") or {},
        card.get("asian_handicap_market_odds") or {},
    )
    return any(
        (price := _number(value)) is not None and MIN_BET_ODDS <= price <= 100
        for group in quote_groups
        if isinstance(group, dict)
        for value in group.values()
    )


def _candidate_identity(card: dict) -> str:
    fixture_id = str(card.get("fixture_id") or "").strip()
    if fixture_id:
        return fixture_id
    home, away = _split_teams(card)
    return "|".join((
        str(card.get("date_iso") or "")[:10],
        home.casefold(), away.casefold(),
    ))


def _candidate_evidence(card: dict) -> tuple:
    features = card.get("independent_features") or {}
    features = features if isinstance(features, dict) else {}
    home_n = int(_number(features.get("home_home_sample")) or 0)
    away_n = int(_number(features.get("away_away_sample")) or 0)
    markets = features.get("markets") or {}
    markets = markets if isinstance(markets, dict) else {}
    market_count = sum(
        1 for name in ("corners", "cards")
        if isinstance(markets.get(name), dict)
        and int(_number(markets[name].get("history_matches")) or 0) >= 5
    )
    xg_history = features.get("xg_history") or {}
    xg_samples = 0
    if isinstance(xg_history, dict):
        for source in ("provider", "proxy"):
            block = xg_history.get(source) or {}
            if isinstance(block, dict):
                xg_samples += sum(
                    int(_number(block.get(key)) or 0)
                    for key in (
                        "home_xg_samples", "home_xga_samples",
                        "away_xg_samples", "away_xga_samples",
                    )
                )
    context = card.get("deep_analysis_context") or card.get("api_context") or {}
    if not isinstance(context, dict):
        context = {}
    lineup = context.get("lineup") or {}
    lineup_count = sum(
        bool((lineup.get(side) or {}).get("confirmed"))
        for side in ("home", "away")
        if isinstance(lineup.get(side), dict)
    ) if isinstance(lineup, dict) else 0
    referee = context.get("referee") or {}
    referee_confirmed = int(
        isinstance(referee, dict) and bool(referee.get("is_confirmed") and referee.get("name"))
    )
    return (
        min(home_n, away_n),
        home_n + away_n,
        market_count,
        xg_samples,
        lineup_count + referee_confirmed,
    )


def top_match_shortlist_diagnostics(
    cards: object,
    *,
    min_team_history: int = MIN_TOP_TEAM_HISTORY,
) -> dict[str, int]:
    """Count evidence-ready fixtures independently of model probability or odds."""
    rows = [card for card in cards if isinstance(card, dict)] if isinstance(cards, list) else []
    result = {
        "fixtures": len(rows),
        "enough_venue_history": 0,
        "with_goal_market_history": 0,
        "with_corners_history": 0,
        "with_cards_history": 0,
        "with_xg_history": 0,
        "eligible": 0,
    }
    for card in rows:
        features = card.get("independent_features") or {}
        if not has_sufficient_history(features):
            continue
        home_n = int(_number(features.get("home_home_sample")) or 0)
        away_n = int(_number(features.get("away_away_sample")) or 0)
        if min(home_n, away_n) < int(min_team_history):
            continue
        result["enough_venue_history"] += 1
        markets = features.get("markets") or {}
        if isinstance(markets, dict):
            goal_rows = min(home_n, away_n)
            if goal_rows >= int(min_team_history):
                result["with_goal_market_history"] += 1
            for market in ("corners", "cards"):
                item = markets.get(market)
                if isinstance(item, dict) and int(_number(item.get("history_matches")) or 0) >= 5:
                    result[f"with_{market}_history"] += 1
        xg = features.get("xg_history") or {}
        if isinstance(xg, dict) and any(
            int(_number((xg.get(source) or {}).get(key)) or 0) >= 5
            for source in ("provider", "proxy")
            for key in ("home_xg_samples", "away_xg_samples")
            if isinstance(xg.get(source), dict)
        ):
            result["with_xg_history"] += 1
        result["eligible"] += 1
    return result


def select_top_matches(
    cards: object,
    *,
    limit: int = MAX_TOP_MATCHES,
    min_team_history: int = MIN_TOP_TEAM_HISTORY,
) -> list[dict]:
    """Select the best-documented matches, not just the statistical model's favourites.

    Selection uses venue-specific history, xG coverage and independently observed
    corners/cards data. It does not require a SStats odds line or a 65% model pick.
    """
    try:
        max_items = min(MAX_TOP_MATCHES, max(0, int(limit)))
    except (TypeError, ValueError):
        max_items = MAX_TOP_MATCHES
    if max_items <= 0 or not isinstance(cards, list):
        return []

    unique: dict[str, dict] = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        features = card.get("independent_features") or {}
        if not has_sufficient_history(features):
            continue
        try:
            home_n = int(features.get("home_home_sample") or 0)
            away_n = int(features.get("away_away_sample") or 0)
        except (TypeError, ValueError):
            continue
        if min(home_n, away_n) < int(min_team_history):
            continue
        identity = _candidate_identity(card)
        previous = unique.get(identity)
        if previous is None or _candidate_evidence(card) > _candidate_evidence(previous):
            unique[identity] = card

    ranked = sorted(
        unique.values(),
        key=lambda card: (
            _candidate_evidence(card),
            str(card.get("date_iso") or ""),
            _candidate_identity(card),
        ),
        reverse=True,
    )
    result = []
    for card in ranked[:max_items]:
        features = card.get("independent_features") or {}
        home_n = int(features.get("home_home_sample") or 0)
        away_n = int(features.get("away_away_sample") or 0)
        result.append({
            "card": card,
            "sample_games": min(home_n, away_n),
            "home_sample_games": home_n,
            "away_sample_games": away_n,
            "evidence_score": _candidate_evidence(card),
            "label": "максимум доступной истории",
        })
    return result


def _split_teams(card: dict) -> tuple[str, str]:
    home = str(card.get("home_team") or card.get("HomeTeam") or "").strip()
    away = str(card.get("away_team") or card.get("AwayTeam") or "").strip()
    if home and away:
        return home, away
    match = str(card.get("match") or "")
    if " vs " in match.casefold():
        parts = re.split(r"\s+vs\s+", match, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    match_ru = str(card.get("match_ru") or "")
    if " — " in match_ru:
        parts = match_ru.split(" — ", 1)
        return parts[0].strip(), parts[1].strip()
    return "Команда не указана", "Команда не указана"


def _player_names(values: object, limit: int = 11) -> list[str]:
    if not isinstance(values, list):
        return []
    names = []
    for item in values[:limit]:
        if isinstance(item, dict):
            name = item.get("name") or item.get("player_name") or item.get("player")
        else:
            name = item
        text = str(name or "").strip()
        if text and text not in names:
            names.append(text[:80])
    return names


def _injury_rows(values: object, limit: int = 8) -> list[dict]:
    result = []
    if not isinstance(values, list):
        return result
    for item in values[:limit]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("player_name") or item.get("name") or "").strip()
        if not name:
            continue
        result.append({
            "player": name[:80],
            "team": str(item.get("team_name") or "").strip()[:80] or "Команда не указана",
            "status": str(item.get("type") or "").strip()[:60],
            "reason": str(item.get("reason") or "").strip()[:100],
        })
    return result


def _market_history_block(item: object) -> dict:
    if not isinstance(item, dict):
        return {"available": False}
    fields = (
        "home_for_avg", "home_against_avg", "away_for_avg",
        "away_against_avg", "history_matches", "home_team_games",
        "away_team_games",
    )
    result: dict[str, Any] = {"available": bool(item)}
    for field in fields:
        value = _number(item.get(field))
        if value is not None:
            result[field] = round(value, 3)
    return result


def _xg_history_block(value: object) -> dict:
    if not isinstance(value, dict):
        return {"available": False}
    result = {"available": False}
    fields = (
        "home_xg_for_avg", "home_xga_avg", "away_xg_for_avg", "away_xga_avg",
        "home_xg_samples", "home_xga_samples", "away_xg_samples", "away_xga_samples",
    )
    for source in ("provider", "proxy"):
        raw = value.get(source)
        if not isinstance(raw, dict):
            result[source] = {"available": False}
            continue
        clean = {}
        for field in fields:
            number = _number(raw.get(field))
            if number is not None:
                clean[field] = int(number) if field.endswith("_samples") else round(number, 3)
        clean["available"] = bool(clean)
        result[source] = clean
        result["available"] = result["available"] or clean["available"]
    result["source_priority"] = "provider_then_proxy"
    return result


def build_match_dossier(card: dict, favorite: dict | None = None) -> dict:
    """Build a compact source-only dossier; model probabilities and odds are excluded."""
    if not isinstance(card, dict):
        raise TypeError("card must be a dictionary")
    home, away = _split_teams(card)
    date_iso = str(card.get("date_iso") or "")[:10]
    fixture_reference = str(
        card.get("fixture_id") or card.get("sstats_game_id") or ""
    ).strip()
    digest = hashlib.sha256(
        f"{date_iso}|{home.casefold()}|{away.casefold()}|{fixture_reference}".encode("utf-8")
    ).hexdigest()[:20]
    fixture_id = f"top-{digest}"

    features = card.get("independent_features") or {}
    features = features if isinstance(features, dict) else {}
    historical = card.get("historical_stats") or {}
    historical = historical if isinstance(historical, dict) else {}
    markets = features.get("markets") or {}
    markets = markets if isinstance(markets, dict) else {}

    context = card.get("deep_analysis_context")
    if not isinstance(context, dict):
        context = card.get("api_context") or {}
    if not isinstance(context, dict):
        context = {}

    lineup = context.get("lineup") or {}
    home_lineup = lineup.get("home") or {}
    away_lineup = lineup.get("away") or {}
    referee = context.get("referee") or {}
    referee_stats = ((historical.get("cards") or {}).get("referee") or {})
    injuries = _injury_rows(context.get("injuries"))

    home_names = _player_names(home_lineup.get("starting_xi"))
    away_names = _player_names(away_lineup.get("starting_xi"))
    home_confirmed = bool(home_lineup.get("confirmed"))
    away_confirmed = bool(away_lineup.get("confirmed"))
    referee_confirmed = bool(referee.get("is_confirmed") and referee.get("name"))
    injury_feed_confirmed = bool(context.get("injuries_is_confirmed"))

    home_sample = int(_number(features.get("home_home_sample")) or 0)
    away_sample = int(_number(features.get("away_away_sample")) or 0)
    context_complete = (
        home_confirmed and away_confirmed and referee_confirmed and injury_feed_confirmed
    )
    if context_complete and min(home_sample, away_sample) >= 8:
        quality_cap = "high"
    elif min(home_sample, away_sample) >= MIN_TOP_TEAM_HISTORY:
        quality_cap = "medium"
    else:
        quality_cap = "low"

    ref_stats_clean = {"available": bool(referee_stats.get("available"))}
    if referee_stats.get("available"):
        for field in ("n", "mean", "variance"):
            value = _number(referee_stats.get(field))
            if value is not None:
                ref_stats_clean[field] = round(value, 3)

    goal_history = {}
    for field in (
        "home_scored_at_home_avg", "home_conceded_at_home_avg",
        "away_scored_away_avg", "away_conceded_away_avg",
        "simple_baseline_lambda_home", "simple_baseline_lambda_away",
        "simple_baseline_lambda_total", "simple_baseline_over_25",
    ):
        value = _number(features.get(field))
        if value is not None:
            goal_history[field] = round(value, 3)
    goal_history["home_home_sample"] = home_sample
    goal_history["away_away_sample"] = away_sample

    return {
        "fixture_id": fixture_id,
        "match": {
            "home": home,
            "away": away,
            "league": str(card.get("league") or card.get("div") or "Лига не указана")[:100],
            "date": date_iso,
            "time": str(card.get("time") or "")[:16],
            "venue": str(card.get("venue") or "")[:100],
        },
        "recent_form": {
            "home": str(card.get("fh") or "нет данных")[:20],
            "away": str(card.get("fa") or "нет данных")[:20],
        },
        "history": {
            "goals": goal_history,
            "corners": _market_history_block(markets.get("corners")),
            "cards": _market_history_block(markets.get("cards")),
        },
        "xg": {
            "used_by_primary_model": bool(card.get("xg_used")),
            "primary_model_source": str(card.get("xg_source") or "not_used"),
            "primary_model_proxy": bool(card.get("xg_modelled_used")),
            "provider_history": _xg_history_block(features.get("xg_history")).get("provider"),
            "proxy_history": _xg_history_block(features.get("xg_history")).get("proxy"),
        },
        "sstats_context": {
            "provider": str(context.get("provider") or "sstats"),
            "sstats_game_id": str(card.get("sstats_game_id") or context.get("sstats_game_id") or ""),
            "event_id_resolved": bool(context.get("sstats_event_id")),
            "kickoff_at": str(context.get("kickoff_at") or card.get("kickoff_utc") or "")[:40],
            "observed_at": str(context.get("timestamp") or "")[:40],
            "lineup_observed_at": str(lineup.get("timestamp") or "")[:40],
            "injuries_observed_at": str(context.get("injuries_timestamp") or "")[:40],
            "referee_observed_at": str(referee.get("timestamp") or "")[:40],
            "lineup": {
                "home_confirmed": home_confirmed,
                "home_players": home_names,
                "away_confirmed": away_confirmed,
                "away_players": away_names,
            },
            "injury_feed_confirmed": injury_feed_confirmed,
            "injuries_listed": injuries,
            "referee": {
                "confirmed": referee_confirmed,
                "name": str(referee.get("name") or "")[:100] if referee_confirmed else None,
                "historical_cards": ref_stats_clean if referee_confirmed else {"available": False},
            },
        },
        "quality_cap": quality_cap,
        "source_status": {
            "history_minimum_met": min(home_sample, away_sample) >= MIN_TOP_TEAM_HISTORY,
            "starting_xi_both_teams_confirmed": home_confirmed and away_confirmed,
            "referee_confirmed": referee_confirmed,
            "injury_feed_confirmed": injury_feed_confirmed,
        },
    }


_DEEP_ANALYSIS_SYSTEM_PROMPT = """Ты — независимый предматчевый футбольный аналитик NEURO BET PRO.
Тебе передают до семи матчей и только фактическое досье: историю голов, venue-specific xG/xGA (если есть), историю угловых/карточек и подтверждённый контекст SStats.
Для каждого матча дай самостоятельную оценку 1X2 и отдельные прогнозы рынков:
- П1/X/П2: вероятности от 0 до 100, сумма ровно 100; independent_pick обязан совпадать с максимальной вероятностью.
- Голы: ожидаемые голы хозяев/гостей, ТБ 1.5/2.5/3.5/4.5 и обе забьют. Используй goal history и доступную provider xG. Provider xG имеет приоритет; proxy — только отдельно помеченная оценка по агрегированным ударам, не настоящий shot-level xG.
- Угловые: ожидаемые угловые хозяев/гостей и ТБ 8.5/9.5/10.5/11.5. Используй только переданную историю угловых.
- Карточки: ожидаемые карточки хозяев/гостей и ТБ 2.5/3.5/4.5/5.5. Используй только переданную историю карточек.
Если для рынка нет достаточной фактической выборки, верни null для всей группы рынка. Не переноси средние одного рынка на другой. Вероятности Over должны быть монотонными: чем выше линия, тем ниже или равна вероятность.
- составы: оценивай влияние только игроков из подтверждённого стартового XI; неподтверждённый состав обозначай как неизвестный;
- травмы/дисквалификации: используй только переданные записи. Пустой или неподтверждённый фид не доказывает отсутствие травм;
- судья: анализируй только подтверждённое назначение и переданную статистику;
- форма/общий разбор: выдели факторы за и против, неопределённость и то, что проверить перед матчем.
Строго запрещено придумывать фамилии, травмы, дисквалификации, имя судьи, тактику, мотивацию, погоду или новости. Не делай вывод, что травм нет, если источник просто не перечислил их.
Не получай и не угадывай букмекерские коэффициенты, EV, Edge, Kelly, модельные вероятности или выбор основной модели: их нет во входном досье. Не называй вывод ставкой и не обещай проходимость.
Верни только JSON с массивом analyses. Для каждого входного fixture_id верни один объект; не меняй ID. Текстовые оценки — краткие, до двух предложений. Если фактов нет, честно обозначь это."""


_DEEP_ANALYSIS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "top_match_independent_multimarket_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "analyses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "fixture_id": {"type": "string"},
                            "independent_pick": {"type": "string", "enum": ["П1", "X", "П2"]},
                            "probabilities": {
                                "type": "object",
                                "properties": {
                                    "П1": {"type": "number"},
                                    "X": {"type": "number"},
                                    "П2": {"type": "number"},
                                },
                                "required": ["П1", "X", "П2"],
                                "additionalProperties": False,
                            },
                            "goal_markets": {
                                "type": ["object", "null"],
                                "properties": {
                                    "expected_home_goals": {"type": "number"},
                                    "expected_away_goals": {"type": "number"},
                                    "over_1_5": {"type": "number"},
                                    "over_2_5": {"type": "number"},
                                    "over_3_5": {"type": "number"},
                                    "over_4_5": {"type": "number"},
                                    "btts_yes": {"type": "number"},
                                },
                                "required": ["expected_home_goals", "expected_away_goals", "over_1_5", "over_2_5", "over_3_5", "over_4_5", "btts_yes"],
                                "additionalProperties": False,
                            },
                            "corner_markets": {
                                "type": ["object", "null"],
                                "properties": {
                                    "expected_home_corners": {"type": "number"},
                                    "expected_away_corners": {"type": "number"},
                                    "over_8_5": {"type": "number"},
                                    "over_9_5": {"type": "number"},
                                    "over_10_5": {"type": "number"},
                                    "over_11_5": {"type": "number"},
                                },
                                "required": ["expected_home_corners", "expected_away_corners", "over_8_5", "over_9_5", "over_10_5", "over_11_5"],
                                "additionalProperties": False,
                            },
                            "card_markets": {
                                "type": ["object", "null"],
                                "properties": {
                                    "expected_home_cards": {"type": "number"},
                                    "expected_away_cards": {"type": "number"},
                                    "over_2_5": {"type": "number"},
                                    "over_3_5": {"type": "number"},
                                    "over_4_5": {"type": "number"},
                                    "over_5_5": {"type": "number"},
                                },
                                "required": ["expected_home_cards", "expected_away_cards", "over_2_5", "over_3_5", "over_4_5", "over_5_5"],
                                "additionalProperties": False,
                            },
                            "lineup_assessment": {"type": "string"},
                            "injury_assessment": {"type": "string"},
                            "referee_assessment": {"type": "string"},
                            "form_assessment": {"type": "string"},
                            "match_assessment": {"type": "string"},
                            "risks": {"type": "string"},
                            "data_quality": {"type": "string", "enum": ["low", "medium", "high"]},
                        },
                        "required": [
                            "fixture_id", "independent_pick", "probabilities",
                            "goal_markets", "corner_markets", "card_markets",
                            "lineup_assessment", "injury_assessment", "referee_assessment",
                            "form_assessment", "match_assessment", "risks", "data_quality",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["analyses"],
            "additionalProperties": False,
        },
    },
}


def merge_context_snapshots(existing: object, refreshed: object) -> dict:
    """Prefer newly confirmed SStats facts without erasing older confirmed facts."""
    old = existing if isinstance(existing, dict) else {}
    new = refreshed if isinstance(refreshed, dict) else {}
    merged = dict(old)
    merged.update({
        key: value for key, value in new.items()
        if key not in {"lineup", "referee", "injuries", "injuries_is_confirmed"}
        and value not in (None, "")
    })

    old_referee = old.get("referee") if isinstance(old.get("referee"), dict) else {}
    new_referee = new.get("referee") if isinstance(new.get("referee"), dict) else {}
    merged["referee"] = (
        dict(new_referee) if new_referee.get("is_confirmed") and new_referee.get("name")
        else dict(old_referee) if old_referee.get("is_confirmed") and old_referee.get("name")
        else dict(new_referee or old_referee)
    )

    old_lineup = old.get("lineup") if isinstance(old.get("lineup"), dict) else {}
    new_lineup = new.get("lineup") if isinstance(new.get("lineup"), dict) else {}
    merged_lineup = {}
    for side in ("home", "away"):
        previous = old_lineup.get(side) if isinstance(old_lineup.get(side), dict) else {}
        current = new_lineup.get(side) if isinstance(new_lineup.get(side), dict) else {}
        if current.get("confirmed") and current.get("starting_xi"):
            merged_lineup[side] = dict(current)
        elif previous.get("confirmed") and previous.get("starting_xi"):
            merged_lineup[side] = dict(previous)
        else:
            merged_lineup[side] = dict(current or previous)
    merged_lineup["is_confirmed"] = bool(
        merged_lineup["home"].get("confirmed")
        and merged_lineup["away"].get("confirmed")
    )
    merged_lineup["timestamp"] = (
        new_lineup.get("timestamp") or old_lineup.get("timestamp")
    )
    merged_lineup["source_reliability"] = (
        new_lineup.get("source_reliability") or old_lineup.get("source_reliability")
    )
    merged["lineup"] = merged_lineup

    if new.get("injuries_is_confirmed"):
        merged["injuries_is_confirmed"] = True
        merged["injuries"] = list(new.get("injuries") or [])
        merged["injuries_timestamp"] = new.get("injuries_timestamp")
    elif old.get("injuries_is_confirmed"):
        merged["injuries_is_confirmed"] = True
        merged["injuries"] = list(old.get("injuries") or [])
        merged["injuries_timestamp"] = old.get("injuries_timestamp")
    else:
        merged["injuries_is_confirmed"] = False
        merged["injuries"] = list(new.get("injuries") or old.get("injuries") or [])
        merged["injuries_timestamp"] = (
            new.get("injuries_timestamp") or old.get("injuries_timestamp")
        )
    merged["injuries_source_reliability"] = (
        new.get("injuries_source_reliability")
        or old.get("injuries_source_reliability")
    )
    merged["sstats_event_id"] = (
        new.get("sstats_event_id") or old.get("sstats_event_id")
    )
    merged["provider"] = new.get("provider") or old.get("provider") or "sstats"
    return merged


def _short_text(value: object, max_length: int = 360) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    return text[:max_length]



def _normalize_market_group(
    raw: object,
    *,
    expected_fields: tuple[str, ...],
    percentage_fields: tuple[str, ...],
    expected_max: float,
    monotonic_fields: tuple[str, ...],
) -> dict | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    clean = {}
    for field in expected_fields:
        value = _number(raw.get(field))
        if value is None:
            return None
        if field in percentage_fields:
            if not 0.0 <= value <= 100.0:
                return None
            clean[field] = round(value, 1)
        else:
            if not 0.0 <= value <= expected_max:
                return None
            clean[field] = round(value, 2)
    ordered = [clean[field] for field in monotonic_fields]
    if any(ordered[i] + 0.1 < ordered[i + 1] for i in range(len(ordered) - 1)):
        return None
    return clean


def _normalize_analysis(raw: object, known: dict[str, dict]) -> dict | None:
    if not isinstance(raw, dict):
        return None
    fixture_id = str(raw.get("fixture_id") or "").strip()
    dossier = known.get(fixture_id)
    if not fixture_id or dossier is None:
        return None
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, dict):
        return None
    values = {}
    for code in ("П1", "X", "П2"):
        number = _number(probabilities.get(code))
        if number is None or not 0 <= number <= 100:
            return None
        values[code] = number
    total = sum(values.values())
    if not 80 <= total <= 120:
        return None
    normalized = {key: round(value * 100.0 / total, 1) for key, value in values.items()}
    drift = round(100.0 - sum(normalized.values()), 1)
    largest = max(values, key=values.get)
    normalized[largest] = round(normalized[largest] + drift, 1)
    pick = max(("П1", "X", "П2"), key=lambda key: (normalized[key], -("П1", "X", "П2").index(key)))

    quality_rank = {"low": 0, "medium": 1, "high": 2}
    raw_quality = str(raw.get("data_quality") or "low").strip().casefold()
    requested = quality_rank.get(raw_quality, 0)
    cap = str(dossier.get("quality_cap") or "low")
    effective_rank = min(requested, quality_rank.get(cap, 0))
    quality = ("low", "medium", "high")[effective_rank]

    goal_markets = _normalize_market_group(
        raw.get("goal_markets"),
        expected_fields=(
            "expected_home_goals", "expected_away_goals",
            "over_1_5", "over_2_5", "over_3_5", "over_4_5", "btts_yes",
        ),
        percentage_fields=("over_1_5", "over_2_5", "over_3_5", "over_4_5", "btts_yes"),
        expected_max=8.0,
        monotonic_fields=("over_1_5", "over_2_5", "over_3_5", "over_4_5"),
    )
    corner_markets = _normalize_market_group(
        raw.get("corner_markets"),
        expected_fields=(
            "expected_home_corners", "expected_away_corners",
            "over_8_5", "over_9_5", "over_10_5", "over_11_5",
        ),
        percentage_fields=("over_8_5", "over_9_5", "over_10_5", "over_11_5"),
        expected_max=30.0,
        monotonic_fields=("over_8_5", "over_9_5", "over_10_5", "over_11_5"),
    )
    card_markets = _normalize_market_group(
        raw.get("card_markets"),
        expected_fields=(
            "expected_home_cards", "expected_away_cards",
            "over_2_5", "over_3_5", "over_4_5", "over_5_5",
        ),
        percentage_fields=("over_2_5", "over_3_5", "over_4_5", "over_5_5"),
        expected_max=20.0,
        monotonic_fields=("over_2_5", "over_3_5", "over_4_5", "over_5_5"),
    )

    return {
        "fixture_id": fixture_id,
        "independent_pick": pick,
        "probabilities": normalized,
        "goal_markets": goal_markets,
        "corner_markets": corner_markets,
        "card_markets": card_markets,
        "lineup_assessment": _short_text(raw.get("lineup_assessment")),
        "injury_assessment": _short_text(raw.get("injury_assessment")),
        "referee_assessment": _short_text(raw.get("referee_assessment")),
        "form_assessment": _short_text(raw.get("form_assessment")),
        "match_assessment": _short_text(raw.get("match_assessment"), 500),
        "risks": _short_text(raw.get("risks"), 500),
        "data_quality": quality,
        "quality_cap": cap,
    }


def analyze_top_matches(ctx: dict) -> dict[str, dict]:
    """Send one compact evidence-only batch request for up to seven shortlisted matches."""
    if not isinstance(ctx, dict):
        return {}
    api_key = str(ctx.get("api_key") or "").strip()
    candidates = ctx.get("candidates")
    if not api_key or not isinstance(candidates, list) or not candidates:
        return {}

    dossiers = [item for item in candidates[:MAX_TOP_MATCHES] if isinstance(item, dict)]
    if not dossiers:
        return {}
    known = {
        str(item.get("fixture_id") or "").strip(): item
        for item in dossiers
        if str(item.get("fixture_id") or "").strip()
    }
    if not known:
        return {}

    provider = ctx.get("provider") or "Groq (бесплатно, быстро)"
    if provider not in LLM_PROVIDERS:
        provider = "Groq (бесплатно, быстро)"
    cfg = LLM_PROVIDERS[provider]
    model = str(ctx.get("model") or cfg["model"])
    base = cfg["base"]

    cache_context = {
        "provider": provider,
        "model": model,
        "dossiers": dossiers,
    }
    digest = hashlib.sha256(
        json.dumps(cache_context, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    cache_key = f"independent_multimarket_batch_v2_{digest}"
    cached = cache_get(cache_key, DEEP_ANALYSIS_CACHE_TTL)
    if isinstance(cached, dict):
        return {
            key: value for key, value in cached.items()
            if key in known and isinstance(value, dict)
        }

    if usage.llm_remaining() <= 0:
        return {}

    user_message = (
        "Проанализируй каждый матч отдельно. Верни ровно один объект на каждый fixture_id. "
        "Не пропускай матч из-за отсутствующих сведений: явно обозначь пробелы.\n"
        "ДОСЬЕ (JSON):\n"
        + json.dumps({"matches": dossiers}, ensure_ascii=False, separators=(",", ":"))
    )
    is_groq = "groq" in str(base).casefold()
    is_xai = "api.x.ai" in str(base).casefold()
    is_gpt_oss = is_groq and model.casefold().startswith("openai/gpt-oss")
    if is_gpt_oss:
        messages = [{
            "role": "user",
            "content": f"{_DEEP_ANALYSIS_SYSTEM_PROMPT}\n\n{user_message}",
        }]
    else:
        messages = [
            {"role": "system", "content": _DEEP_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    if is_gpt_oss:
        body["max_completion_tokens"] = 6144
        body.update({
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
            "response_format": _DEEP_ANALYSIS_RESPONSE_FORMAT,
        })
    elif is_groq:
        body["max_completion_tokens"] = 6144
        body["response_format"] = {"type": "json_object"}
    elif is_xai:
        # xAI documents strict JSON Schema outputs on the OpenAI-compatible API.
        body["max_tokens"] = 6144
        body["response_format"] = _DEEP_ANALYSIS_RESPONSE_FORMAT
    else:
        body["max_tokens"] = 6144
        body["response_format"] = {"type": "json_object"}

    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = _sess.post(
            url,
            headers=headers,
            json=body,
            timeout=75,
            proxies=NO_PROXY,
        )
        usage.llm_increment(1)
        if response.status_code == 429:
            raise LLMRateLimitError(_retry_after_seconds(response))
        if response.status_code != 200:
            _log(
                f"TOP MATCHES ERROR: provider={provider}, HTTP {response.status_code}, "
                f"response={str(getattr(response, 'text', '') or '')[:180]}"
            )
            return {}
        content, diagnostics = _extract_chat_completion(response.json())
        if not content:
            _log(
                "TOP MATCHES ERROR: empty completion content; "
                + json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)
            )
            return {}
        try:
            payload = json.loads(content.strip())
        except (TypeError, ValueError):
            _log("TOP MATCHES ERROR: provider returned non-JSON content.")
            return {}
        rows = payload.get("analyses") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            _log("TOP MATCHES ERROR: response does not contain analyses array.")
            return {}

        normalized: dict[str, dict] = {}
        for raw in rows:
            item = _normalize_analysis(raw, known)
            if item is None:
                continue
            fixture_id = item["fixture_id"]
            if fixture_id in normalized:
                continue
            normalized[fixture_id] = item
        if normalized:
            cache_put(cache_key, normalized)
        return normalized
    except LLMRateLimitError:
        raise
    except requests.RequestException as exc:
        _log(f"TOP MATCHES ERROR: {type(exc).__name__}: {str(exc)[:180]}")
        return {}
    except Exception as exc:
        _log(f"TOP MATCHES ERROR: {type(exc).__name__}: {str(exc)[:180]}")
        return {}


def format_top_match_opinion(
    analysis: dict,
    model_pick_code: str = "",
) -> str:
    """Keep 1X2 parsing stable while retaining independent multi-market estimates."""
    probabilities = analysis.get("probabilities") or {}
    pick = str(analysis.get("independent_pick") or "X")
    labels = {"П1": "победа хозяев (П1)", "X": "ничья (X)", "П2": "победа гостей (П2)"}
    quality_labels = {"low": "низкое", "medium": "среднее", "high": "высокое"}
    result = (
        f"САМОСТОЯТЕЛЬНЫЙ ПРОГНОЗ ИИ: {labels.get(pick, 'ничья (X)')}. "
        f"Оценка ИИ: П1 {float(probabilities.get('П1') or 0):.1f}% · "
        f"X {float(probabilities.get('X') or 0):.1f}% · "
        f"П2 {float(probabilities.get('П2') or 0):.1f}%. "
        f"Качество исходных данных: {quality_labels.get(str(analysis.get('data_quality')), 'низкое')}. "
        f"{_short_text(analysis.get('match_assessment'), 500)}"
    )
    for section_key, title, fields in (
        ("goal_markets", "Голы", (
            ("expected_home_goals", "ожид. голы хозяев"),
            ("expected_away_goals", "ожид. голы гостей"),
            ("over_1_5", "ТБ 1.5"),
            ("over_2_5", "ТБ 2.5"),
            ("over_3_5", "ТБ 3.5"),
            ("over_4_5", "ТБ 4.5"),
            ("btts_yes", "ОЗ Да"),
        )),
        ("corner_markets", "Угловые", (
            ("expected_home_corners", "хозяева"),
            ("expected_away_corners", "гости"),
            ("over_8_5", "ТБ 8.5"),
            ("over_9_5", "ТБ 9.5"),
            ("over_10_5", "ТБ 10.5"),
            ("over_11_5", "ТБ 11.5"),
        )),
        ("card_markets", "Карточки", (
            ("expected_home_cards", "хозяева"),
            ("expected_away_cards", "гости"),
            ("over_2_5", "ТБ 2.5"),
            ("over_3_5", "ТБ 3.5"),
            ("over_4_5", "ТБ 4.5"),
            ("over_5_5", "ТБ 5.5"),
        )),
    ):
        group = analysis.get(section_key)
        if isinstance(group, dict):
            rendered = []
            for key, label in fields:
                value = group.get(key)
                if value is None:
                    continue
                suffix = "%" if key.startswith("over_") or key == "btts_yes" else ""
                rendered.append(f"{label} {float(value):.1f}{suffix}")
            if rendered:
                result += f" {title}: " + ", ".join(rendered) + "."
    risks = _short_text(analysis.get("risks"), 300)
    if risks:
        result += f" Риски: {risks}"
    if model_pick_code in {"П1", "X", "П2"}:
        if pick == model_pick_code:
            result += f" Совпадение: независимый ИИ и основная модель выбрали {pick}."
        else:
            result += (
                f" РАСХОЖДЕНИЕ: независимый ИИ — {pick}, "
                f"основная модель — {model_pick_code}."
            )
    return result
