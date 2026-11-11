"""Shortlist selection and evidence-bounded batch analysis for top football matches."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from betting.ranking import model_favorite, rank_model_favorites
from config import CACHE_TTL, LLM_PROVIDERS
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
    return True


def top_match_shortlist_diagnostics(
    cards: object,
    *,
    min_probability: float = MIN_TOP_PROBABILITY,
    min_team_history: int = MIN_TOP_TEAM_HISTORY,
) -> dict[str, int]:
    """Count why a fixture is or is not eligible for the top research shortlist."""
    rows = [card for card in cards if isinstance(card, dict)] if isinstance(cards, list) else []
    diagnostics = {
        "fixtures": len(rows),
        "strong_probability": 0,
        "enough_venue_history": 0,
        "complete_1x2_prices": 0,
        "eligible": 0,
    }
    for card in rows:
        favorite = model_favorite(card)
        if favorite is None or favorite["probability"] < float(min_probability):
            continue
        diagnostics["strong_probability"] += 1
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
        diagnostics["enough_venue_history"] += 1
        if not _complete_1x2_prices(card):
            continue
        diagnostics["complete_1x2_prices"] += 1
        diagnostics["eligible"] += 1
    return diagnostics


def select_top_matches(
    cards: object,
    *,
    limit: int = MAX_TOP_MATCHES,
    min_probability: float = MIN_TOP_PROBABILITY,
    min_team_history: int = MIN_TOP_TEAM_HISTORY,
) -> list[dict]:
    """Return up to seven unique, strong, priced 1X2 candidates for Groq review.

    Candidates must have a model favourite at/above the configured strong
    probability threshold, at least five same-venue historical matches for
    both teams, and a complete real 1X2 price set. This does not make a Value
    bet and never relaxes the portfolio's separate market/context gates.
    """
    try:
        max_items = min(MAX_TOP_MATCHES, max(0, int(limit)))
    except (TypeError, ValueError):
        max_items = MAX_TOP_MATCHES
    if max_items <= 0 or not isinstance(cards, list):
        return []

    # Venue-specific samples are checked below. Do not let the generic
    # aggregate card.games value reject a match whose home/away samples are valid.
    ranked = rank_model_favorites(cards, min_sample_games=0)
    selected = []
    for item in ranked:
        card = item.get("card")
        if not isinstance(card, dict):
            continue
        if float(item.get("probability") or 0.0) < float(min_probability):
            continue
        features = card.get("independent_features") or {}
        if not has_sufficient_history(features):
            continue
        try:
            if min(
                int(features.get("home_home_sample") or 0),
                int(features.get("away_away_sample") or 0),
            ) < int(min_team_history):
                continue
        except (TypeError, ValueError):
            continue
        if not _complete_1x2_prices(card):
            continue
        selected.append(item)
        if len(selected) >= max_items:
            break
    return selected


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
Тебе передают до семи матчей и только фактическое досье источника SStats и исторические выборки.
Для каждого матча дай самостоятельную оценку 1X2 и короткий разбор:
- стартовые составы: анализируй влияние только игроков, перечисленных в подтверждённом стартовом XI; если XI не подтверждён, прямо напиши, что состав не опубликован/не подтверждён;
- травмы/дисквалификации: упоминай только переданные записи; пустой или неподтверждённый фид НЕ доказывает отсутствие травм;
- судья: анализируй только назначенного и подтверждённого арбитра; если его нет, прямо укажи, что назначение не подтверждено;
- форма, история голов, угловые и карточки: используй только переданные выборки и их размеры; не подменяй отсутствующую историю средними другой лиги;
- общий разбор: выдели факторы за/против, неопределённость и то, что ещё следует проверить.
Строго запрещено придумывать фамилии игроков, травмы, дисквалификации, имя судьи, тактику, мотивацию, погоду или события. Не делай вывод, что травм нет, если источник просто не перечислил их.
Не получай и не угадывай bookmaker odds, EV, Edge, Kelly, модельные вероятности или выбор основной модели: их нет во входе. Не называй вывод ставкой и не обещай проходимость.
Верни только JSON с массивом analyses. Для каждого входного fixture_id верни один объект; не меняй ID. Проценты 1X2 должны быть числами от 0 до 100 и в сумме давать около 100. Все поля текстовых оценок — краткие, до двух предложений. Если фактов нет, честно укажи это в соответствующем поле."""


_DEEP_ANALYSIS_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "top_match_deep_analysis",
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
                            "lineup_assessment": {"type": "string"},
                            "injury_assessment": {"type": "string"},
                            "referee_assessment": {"type": "string"},
                            "form_assessment": {"type": "string"},
                            "match_assessment": {"type": "string"},
                            "risks": {"type": "string"},
                            "data_quality": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                        "required": [
                            "fixture_id", "independent_pick", "probabilities",
                            "lineup_assessment", "injury_assessment",
                            "referee_assessment", "form_assessment",
                            "match_assessment", "risks", "data_quality",
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

    return {
        "fixture_id": fixture_id,
        "independent_pick": pick,
        "probabilities": normalized,
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
    cache_key = f"groq_top_match_dossier_v1_{digest}"
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
    if is_groq:
        body["max_completion_tokens"] = 4096
    else:
        body["max_tokens"] = 3072
    if is_gpt_oss:
        body.update({
            "reasoning_effort": "low",
            "reasoning_format": "hidden",
            "response_format": _DEEP_ANALYSIS_RESPONSE_FORMAT,
        })
    else:
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
    """Keep the existing immutable forecast ledger compatible with batch Groq output."""
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
