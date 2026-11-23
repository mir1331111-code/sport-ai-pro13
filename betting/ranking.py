"""betting/ranking.py — Signal Ranking.

Ранжирование найденных сигналов по композитному score.
НЕ влияет на создание ставок (это делает scanner через P+Edge+EV+Kelly+Risk).
Только порядок отображения и приоритет в UI.

score = w_conf * confidence
      + w_ev   * sigmoid(EV / scale)
      + w_edge * sigmoid(edge / scale)
      + w_kelly* sigmoid(kelly / scale)
      - penalty * risk_penalty
"""
from __future__ import annotations

import math
import hashlib
import json
import re
from datetime import datetime
from data.fixture_time import app_timezone, fixture_kickoff_local
from data.team_names import normalize_team
from typing import Optional


DEFAULT_WEIGHTS = {
    "confidence": 0.55,   # основная вероятность/уверенность
    "ev": 0.20,
    "edge": 0.15,
    "kelly": 0.10,
}

# Шкалы для нормализации (чтобы привести разные метрики к [0;1])
SCALE_EV = 0.10       # EV 10% = очень хорошо
SCALE_EDGE = 0.08     # Edge 8% = отлично
SCALE_KELLY = 0.05    # Kelly 5% = максимальный риск


def balanced_value_score(probability, ev=None, edge=None) -> float:
    """Balanced display/selection score; it never decides Value eligibility.

    Event probability has 60% weight. Executable EV and no-vig edge contribute
    25% and 15% only when both were produced by an already validated market
    candidate. With no validated price, ranking falls back to probability.
    """
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(probability):
        return 0.0
    probability = max(0.0, min(1.0, probability))
    if ev is None and edge is None:
        return probability
    try:
        ev_value = float(ev)
        edge_value = float(edge)
    except (TypeError, ValueError):
        return probability
    if not math.isfinite(ev_value) or not math.isfinite(edge_value):
        return probability
    ev_score = max(0.0, min(1.0, ev_value / SCALE_EV))
    edge_score = max(0.0, min(1.0, edge_value / SCALE_EDGE))
    return 0.60 * probability + 0.25 * ev_score + 0.15 * edge_score


def _sigmoid(x: float, scale: float) -> float:
    """Гладкая нормализация в [0;1]. x=0 → 0.5, x=scale → ~0.73, x=2*scale → ~0.88."""
    if scale <= 0:
        return 0.5
    try:
        return 1.0 / (1.0 + math.exp(-x / scale))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _confidence_norm(card: dict) -> float:
    """Достаёт Confidence из карточки. Если нет — fallback на prob."""
    v = card.get("verdict", {}) or {}

    # Приоритет 1: явный confidence (0..1)
    c = v.get("confidence_score")
    if c is None:
        c = card.get("confidence_score")
    if c is not None:
        try:
            c = float(c)
            if c > 1.0:
                c = c / 100.0
            return max(0.0, min(1.0, c))
        except (TypeError, ValueError):
            pass

    # Приоритет 2: текстовая confidence → число
    txt = (v.get("confidence") or "").lower()
    map_txt = {
        "очень высокая": 0.95,
        "высокая": 0.85,
        "хорошая": 0.70,
        "средняя": 0.55,
        "низкая": 0.35,
    }
    for key, val in map_txt.items():
        if key in txt:
            return val

    # Приоритет 3: prob как прокси
    try:
        return max(0.0, min(1.0, float(v.get("prob", 0.5))))
    except (TypeError, ValueError):
        return 0.5


def _risk_penalty(card: dict) -> float:
    """Штраф за риск-флаги: дерби, усталость, плохой судья, national без данных и т.п."""
    ctx = card.get("context") or {}
    penalty = 0.0
    for flag in ctx.get("flags", []) or []:
        ftype = (flag or {}).get("type", "")
        if ftype == "schedule":       # back-to-back / усталость
            penalty += 0.15
        elif ftype == "referee":      # строгий судья — неоднозначно
            penalty += 0.05
        elif ftype == "derby":        # дерби — больше волатильности
            penalty += 0.10
    return min(0.5, penalty)          # ограничиваем


def signal_score(card: dict) -> float:
    """Сила модельного сигнала, без зависимости от цены букмекера."""
    conf = _confidence_norm(card)
    try:
        prob = max(0.0, min(1.0, float((card.get("verdict") or {}).get("prob", 0.0))))
    except (TypeError, ValueError):
        prob = 0.0
    score = 0.65 * conf + 0.35 * prob
    score -= _risk_penalty(card)
    return max(0.0, min(1.0, score))


def value_score(card: dict) -> float:
    """Привлекательность текущей цены. Без рынка score = 0."""
    v = card.get("verdict", {}) or {}

    def metric(name):
        try:
            raw = v.get(name)
            return None if raw is None else float(raw)
        except (TypeError, ValueError):
            return None

    ev = metric("ev")
    edge = metric("edge")
    kelly_ = metric("kelly_pct")
    if ev is None and edge is None and kelly_ is None:
        return 0.0

    ev_s = max(0.0, min(1.0, ev / SCALE_EV)) if ev is not None else 0.0
    edge_s = max(0.0, min(1.0, edge / SCALE_EDGE)) if edge is not None else 0.0
    kelly_s = max(0.0, min(1.0, kelly_ / SCALE_KELLY)) if kelly_ is not None else 0.0
    return 0.50 * ev_s + 0.35 * edge_s + 0.15 * kelly_s

def rank_score(card: dict,
               weights: Optional[dict] = None) -> float:
    """Возвращает score в [0;1] — чем выше, тем сильнее сигнал."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    v = card.get("verdict", {}) or {}

    conf = _confidence_norm(card)

    def _metric(name: str):
        raw = v.get(name)
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    ev = _metric("ev")
    edge = _metric("edge")
    kelly_ = _metric("kelly_pct")

    # Отсутствие рынка/EV/Edge не должно давать искусственные 0.5.
    # Для model-only кандидата эти компоненты нейтрально равны 0.
    s_ev = _sigmoid(ev, SCALE_EV) if ev is not None else 0.0
    s_edge = _sigmoid(edge, SCALE_EDGE) if edge is not None else 0.0
    s_kelly = _sigmoid(kelly_, SCALE_KELLY) if kelly_ is not None else 0.0

    penalty = _risk_penalty(card)

    score = (
        w["confidence"] * conf +
        w["ev"] * s_ev +
        w["edge"] * s_edge +
        w["kelly"] * s_kelly
    ) - penalty

    return max(0.0, min(1.0, score))


def rank_label(score: float) -> str:
    """Текстовая метка для UI."""
    if score >= 0.80:
        return "🔥 TOP"
    if score >= 0.65:
        return "⭐ STRONG"
    if score >= 0.50:
        return "✅ OK"
    if score >= 0.35:
        return "🟡 WEAK"
    return "⚪ LOW"


def rank_color(score: float) -> str:
    if score >= 0.80:
        return "#f87171"     # красный — топ
    if score >= 0.65:
        return "#fbbf24"     # жёлтый
    if score >= 0.50:
        return "#34d399"     # зелёный
    return "#8b93a7"          # серый


def sort_cards(cards: list,
               mode: str = "rank") -> list:
    """Сортирует карточки по разным режимам.

    mode: "rank" (композит) | "prob" | "ev" | "date"
    """
    if not isinstance(cards, list):
        return []

    if mode == "prob":
        return sorted(cards,
                      key=lambda c: -(c.get("verdict", {}).get("prob") or 0))
    if mode == "ev":
        return sorted(cards,
                      key=lambda c: -(c.get("verdict", {}).get("ev") or 0))

    if mode == "date":
        from ui.cards import parse_card_date
        return sorted(cards, key=lambda c: parse_card_date(c))

    # Default: rank
    return sorted(cards, key=lambda c: -rank_score(c))


# Probability-first unique fixture ordering, shared by the scanner UI.
def _number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _team_key(value) -> str:
    """Use exact team aliases for provider-independent fixture deduplication."""
    return normalize_team(str(value or ""))


def _teams(card: dict) -> tuple[str, str]:
    home = str(card.get("home") or card.get("HomeTeam") or "").strip()
    away = str(card.get("away") or card.get("AwayTeam") or "").strip()
    if home and away:
        return home, away
    match_ru = str(card.get("match_ru") or "")
    if " — " in match_ru:
        return tuple(part.strip() for part in match_ru.split(" — ", 1))
    match = str(card.get("match") or "")
    if " vs " in match.casefold():
        parts = re.split(r"\s+vs\s+", match, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
    return home, away


def _fixture_key(card: dict, index: int) -> tuple:
    """Provider-independent event identity; source order is never used to pick a winner."""
    home, away = _teams(card)
    date = str(card.get("date_iso") or card.get("date") or "")[:10]
    home_key, away_key = _team_key(home), _team_key(away)
    if date and home_key and away_key:
        return ("fixture", date, home_key, away_key)
    fixture_id = str(card.get("fixture_id") or card.get("sstats_game_id") or "").strip()
    if fixture_id:
        return ("id", fixture_id)
    # An unidentifiable row is retained instead of being accidentally collapsed.
    return ("row", index)


def _duplicate_quality(card: dict) -> tuple:
    """Resolve duplicate provider rows by forecast quality and data coverage."""
    favorite = model_favorite(card)
    odds = sum(
        len(card.get(field) or {})
        for field in (
            "market_odds", "total_market_odds", "btts_market_odds",
            "stat_market_odds", "asian_handicap_market_odds",
        )
    )
    return (
        favorite["probability"] if favorite else 0.0,
        favorite["probability_margin"] if favorite else 0.0,
        favorite["sample_games"] if favorite else 0,
        int(bool(card.get("xg_used"))),
        odds,
        str(card.get("fixture_id") or ""),
        str(card.get("league") or ""),
        str(card.get("match") or ""),
    )


def model_favorite(card: object) -> dict | None:
    """Return the most likely 1X2 outcome plus a reliability-aware ranking score."""
    if not isinstance(card, dict):
        return None

    probabilities = []
    for key in ("p1", "px", "p2"):
        value = _number(card.get(key))
        if value is None or not 0.0 <= value <= 1.0:
            return None
        probabilities.append(value)

    total = sum(probabilities)
    # Reject percentages, partial distributions, and malformed model output.
    if total <= 0.0 or not 0.90 <= total <= 1.10:
        return None

    order = sorted(range(3), key=lambda idx: probabilities[idx], reverse=True)
    best_idx, second_idx = order[0], order[1]
    probability = probabilities[best_idx]
    margin = max(0.0, probability - probabilities[second_idx])

    features = card.get("independent_features") or {}
    if not isinstance(features, dict):
        features = {}
    # Scanner cards carry true venue-specific support. Do not replace zero or
    # sparse home/away samples with the broader all-venue Engine.games count.
    if "home_home_sample" in features or "away_away_sample" in features:
        home_sample = max(0.0, _number(features.get("home_home_sample"), 0.0) or 0.0)
        away_sample = max(0.0, _number(features.get("away_away_sample"), 0.0) or 0.0)
    else:
        # Backward compatibility for older persisted cards and simple callers.
        legacy_sample = max(0.0, _number(card.get("games"), 0.0) or 0.0)
        home_sample = legacy_sample
        away_sample = legacy_sample
    sample_games = min(home_sample, away_sample)
    # Probability is the strict primary key. Separation and sample size are
    # used only as tie-breakers; a lower-probability event never jumps above a
    # higher-probability event because it has more historical matches.
    sample_support = min(1.0, sample_games / 12.0)
    rank_score = probability

    home, away = _teams(card)
    outcomes = (
        ("П1", f"Победа {home}" if home else "Победа хозяев"),
        ("X", "Ничья"),
        ("П2", f"Победа {away}" if away else "Победа гостей"),
    )
    pick, label = outcomes[best_idx]
    if probability >= 0.70:
        strength = "Сильный фаворит"
    elif probability >= 0.60:
        strength = "Фаворит"
    elif probability >= 0.52:
        strength = "Небольшое преимущество"
    else:
        strength = "Равный матч"

    return {
        "pick": pick,
        "label": label,
        "probability": probability,
        "probability_margin": margin,
        "sample_games": int(sample_games),
        "home_sample": int(home_sample),
        "away_sample": int(away_sample),
        "sample_support": sample_support,
        "rank_score": rank_score,
        "strength": strength,
        # Legacy cards without calibration metadata are not treated as validated.
        "calibration_ready": bool(card.get("probability_calibrated", False)),
        "calibration_matches": max(0, int(_number(card.get("calibration_matches"), 0.0) or 0)),
    }


def _daily_event_quality(card: dict, prediction: dict) -> dict:
    """Return a transparent, forecast-only quality score (not a win probability).

    Components are normalized to 0..1 and weighted:
    favourite probability 40%, separation from the second outcome 30%,
    venue-specific history 20%, genuine provider xG 5%, complete 1X2 odds 5%.
    """
    probability = min(1.0, max(0.0, float(prediction.get("probability") or 0.0)))
    margin = min(1.0, max(0.0, float(prediction.get("probability_margin") or 0.0)))
    probability_component = min(1.0, max(0.0, (probability - (1.0 / 3.0)) / (2.0 / 3.0)))
    margin_component = min(1.0, max(0.0, margin / (2.0 / 3.0)))

    home_sample = max(0, int(prediction.get("home_sample") or 0))
    away_sample = max(0, int(prediction.get("away_sample") or 0))
    history_component = min(1.0, min(home_sample, away_sample) / 12.0)

    genuine_provider_xg = (
        bool(card.get("xg_used"))
        and not bool(card.get("xg_modelled_used"))
        and str(card.get("xg_source") or "").casefold() == "provider"
    )
    odds = card.get("market_odds") if isinstance(card.get("market_odds"), dict) else {}
    complete_odds = all(
        _number(odds.get(key)) is not None and float(_number(odds.get(key))) > 1.0
        for key in ("П1", "X", "П2")
    )

    components = {
        "probability": probability_component,
        "separation": margin_component,
        "venue_history": history_component,
        "provider_xg": 1.0 if genuine_provider_xg else 0.0,
        "complete_market_line": 1.0 if complete_odds else 0.0,
    }
    score = 100.0 * (
        0.40 * components["probability"]
        + 0.30 * components["separation"]
        + 0.20 * components["venue_history"]
        + 0.05 * components["provider_xg"]
        + 0.05 * components["complete_market_line"]
    )
    return {
        "quality_score": round(score, 2),
        "quality_components": components,
        "has_provider_xg": genuine_provider_xg,
        "has_complete_market_line": complete_odds,
    }


def rank_best_fixtures(
    cards: object,
    *,
    now: datetime | None = None,
    period_end: datetime | None = None,
    limit: int | None = 5,
    min_sample_games: int = 5,
    min_probability: float = 0.52,
    include_limited_history: bool = False,
    require_calibration: bool = False,
    diagnostics: dict | None = None,
) -> list[dict]:
    """Rank upcoming fixtures, optionally keeping provisional forecasts visible.

    Strict candidates have a valid 1X2 distribution, at least the configured
    number of dated home-at-home observations for the home team and away-at-away
    observations for the away team, and a favourite probability at or above
    the configured probability floor.

    When include_limited_history is enabled, otherwise valid future model
    forecasts are also returned as clearly labelled display-only candidates.
    This fallback never changes Value/EV/Edge/Kelly or staking eligibility.
    """
    counters = {
        "input_cards": len(cards) if isinstance(cards, list) else 0,
        "invalid_kickoff": 0,
        "already_started": 0,
        "outside_window": 0,
        "invalid_forecast": 0,
        "unsupported_model": 0,
        "uncalibrated_candidates": 0,
        "missing_venue_history": 0,
        "insufficient_home_history": 0,
        "insufficient_away_history": 0,
        "below_probability": 0,
        "strict_candidates": 0,
        "provisional_candidates": 0,
    }
    if not isinstance(cards, list):
        if isinstance(diagnostics, dict):
            diagnostics.clear()
            diagnostics.update(counters)
        return []

    tz = app_timezone()
    local_now = datetime.now(tz) if now is None else (
        now.replace(tzinfo=tz) if now.tzinfo is None else now.astimezone(tz)
    )
    local_end = None
    if period_end is not None:
        local_end = (
            period_end.replace(tzinfo=tz)
            if period_end.tzinfo is None
            else period_end.astimezone(tz)
        )
        if local_end <= local_now:
            if isinstance(diagnostics, dict):
                diagnostics.clear()
                diagnostics.update(counters)
            return []

    try:
        history_floor = max(0, int(min_sample_games))
    except (TypeError, ValueError, OverflowError):
        history_floor = 5
    try:
        probability_floor = float(min_probability)
    except (TypeError, ValueError, OverflowError):
        probability_floor = 0.52
    if not math.isfinite(probability_floor):
        probability_floor = 0.52
    probability_floor = min(1.0, max(0.0, probability_floor))

    timezone_name = getattr(tz, "key", None)
    unique: dict[tuple, dict] = {}
    for index, card in enumerate(cards):
        if not isinstance(card, dict):
            continue
        support_status = str(card.get("model_support_status") or "").strip().casefold()
        if support_status and support_status != "team_supported":
            counters["unsupported_model"] += 1
            continue
        kickoff = fixture_kickoff_local(card, timezone_name=timezone_name)
        if kickoff is None:
            counters["invalid_kickoff"] += 1
            continue
        if kickoff <= local_now:
            counters["already_started"] += 1
            continue
        if local_end is not None and kickoff >= local_end:
            counters["outside_window"] += 1
            continue

        prediction = model_favorite(card)
        if prediction is None:
            counters["invalid_forecast"] += 1
            continue

        features = card.get("independent_features")
        has_venue_history = (
            isinstance(features, dict)
            and "home_home_sample" in features
            and "away_away_sample" in features
        )
        if has_venue_history:
            home_sample = max(0, int(prediction.get("home_sample") or 0))
            away_sample = max(0, int(prediction.get("away_sample") or 0))
        else:
            home_sample = 0
            away_sample = 0
            counters["missing_venue_history"] += 1

        history_ready = (
            has_venue_history
            and home_sample >= history_floor
            and away_sample >= history_floor
        )
        if has_venue_history and home_sample < history_floor:
            counters["insufficient_home_history"] += 1
        if has_venue_history and away_sample < history_floor:
            counters["insufficient_away_history"] += 1

        probability_ready = prediction["probability"] >= probability_floor
        if not probability_ready:
            counters["below_probability"] += 1
        calibration_ready = bool(prediction.get("calibration_ready"))
        if history_ready and probability_ready and not calibration_ready:
            counters["uncalibrated_candidates"] += 1
        calibration_passed = calibration_ready or not require_calibration

        strict_eligible = history_ready and probability_ready and calibration_passed
        if strict_eligible:
            counters["strict_candidates"] += 1
        else:
            counters["provisional_candidates"] += 1
            if not include_limited_history:
                continue

        prediction = dict(prediction)
        prediction["home_sample"] = home_sample
        prediction["away_sample"] = away_sample
        prediction["sample_games"] = min(home_sample, away_sample)
        prediction["sample_support"] = min(1.0, prediction["sample_games"] / 12.0)

        note_parts = []
        if not history_ready:
            if not has_venue_history:
                note_parts.append("нет раздельной домашней/выездной истории")
            else:
                if home_sample < history_floor:
                    note_parts.append(f"дома {home_sample}/{history_floor}")
                if away_sample < history_floor:
                    note_parts.append(f"в гостях {away_sample}/{history_floor}")
        if not probability_ready:
            note_parts.append(
                f"нет выраженного фаворита ({prediction['probability']:.0%} < "
                f"{probability_floor:.0%})"
            )
        if require_calibration and not calibration_ready:
            note_parts.append(
                f"нет калибровки вероятности (выборка калибратора: "
                f"{prediction.get('calibration_matches', 0)})"
            )
        selection_note = (
            "Подтверждённая история, фаворит и калибровка"
            if strict_eligible and require_calibration
            else "Подтверждённая история и фаворит"
            if strict_eligible
            else "Ознакомительный прогноз · " + "; ".join(note_parts)
        )

        quality = _daily_event_quality(card, prediction)
        key = _fixture_key(card, index)
        candidate = {
            "card": card,
            **prediction,
            **quality,
            "kickoff": kickoff,
            "history_ready": history_ready,
            "probability_ready": probability_ready,
            "strict_eligible": strict_eligible,
            "selection_note": selection_note,
        }
        current = unique.get(key)
        candidate_quality = (
            int(strict_eligible),
            quality["quality_score"],
            prediction["probability_margin"],
            prediction["sample_games"],
            prediction["probability"],
            str(card.get("league") or ""),
            str(card.get("match") or card.get("match_ru") or ""),
        )
        if current is None:
            unique[key] = candidate
        else:
            current_quality = (
                int(current["strict_eligible"]),
                current["quality_score"],
                current["probability_margin"],
                current["sample_games"],
                current["probability"],
                str(current["card"].get("league") or ""),
                str(current["card"].get("match") or current["card"].get("match_ru") or ""),
            )
            if candidate_quality > current_quality:
                unique[key] = candidate

    ranked = list(unique.values())
    # Kickoff time is deliberately not a ranking key. It is display metadata,
    # not evidence that one event is a better forecast than another.
    ranked.sort(key=lambda item: (
        -int(item["strict_eligible"]),
        -item["quality_score"],
        -item["probability_margin"],
        -item["sample_games"],
        -item["probability"],
        str(item["card"].get("league") or "").casefold(),
        str(item["card"].get("match") or item["card"].get("match_ru") or "").casefold(),
    ))
    if limit is not None:
        try:
            maximum = max(0, int(limit))
            if maximum == 0:
                ranked = []
            elif len(ranked) > maximum:
                cutoff = ranked[maximum - 1]
                cutoff_tier = bool(cutoff["strict_eligible"])
                cutoff_score = round(float(cutoff["quality_score"]), 2)
                # Keep every event tied at the display boundary instead of
                # arbitrarily preferring the earlier kickoff.
                ranked = [
                    item for item in ranked
                    if (
                        bool(item["strict_eligible"]) > cutoff_tier
                        or (
                            bool(item["strict_eligible"]) == cutoff_tier
                            and round(float(item["quality_score"]), 2) >= cutoff_score
                        )
                    )
                ]
        except (TypeError, ValueError, OverflowError):
            pass

    if isinstance(diagnostics, dict):
        counters["unique_candidates"] = len(unique)
        counters["returned_candidates"] = len(ranked)
        counters["strict_unique"] = sum(bool(item["strict_eligible"]) for item in unique.values())
        counters["provisional_unique"] = sum(not item["strict_eligible"] for item in unique.values())
        diagnostics.clear()
        diagnostics.update(counters)
    return ranked


def rank_model_favorites_with_minimum(
    cards: object, *, minimum: int = 7, limit: int = 10,
    min_sample_games: int = 5,
) -> list[dict]:
    """Return a probability-ranked forecast list with an explicit history flag.

    All valid model forecasts remain eligible for this display-only list.
    Historical support is surfaced explicitly rather than silently dropping
    fixtures. The function never changes Value eligibility, EV/Edge, Kelly or
    stake sizing.
    """
    try:
        required = max(0, int(minimum))
    except (TypeError, ValueError):
        required = 7
    try:
        maximum = max(0, int(limit))
    except (TypeError, ValueError):
        maximum = 10
    display_limit = max(required, maximum)
    try:
        history_floor = max(0, int(min_sample_games))
    except (TypeError, ValueError):
        history_floor = 5

    ranked = rank_model_favorites(
        cards, limit=display_limit, min_sample_games=0,
    )
    return [
        {
            **item,
            "history_sufficient": int(item.get("sample_games") or 0) >= history_floor,
            "history_floor": history_floor,
        }
        for item in ranked
    ]


def favorite_display(item: object) -> dict:
    """Only call a pick strong when venue history and its league calibrator exist."""
    if not isinstance(item, dict):
        return {
            "pick": None,
            "label": "НЕДОСТАТОЧНО ДАННЫХ · НЕТ ВЫБОРА",
            "probability": None,
            "history_sufficient": False,
            "calibrated": False,
        }

    sample = max(0, int(item.get("sample_games") or 0))
    floor = max(1, int(item.get("history_floor") or 10))
    history_sufficient = bool(item.get("history_sufficient", False))
    calibration_ready = bool(item.get("calibration_ready", False))
    calibration_matches = max(0, int(item.get("calibration_matches") or 0))

    if sample == 0:
        label = "НЕДОСТАТОЧНО ИСТОРИИ · PRIOR-ONLY"
    elif not history_sufficient:
        label = f"ОГРАНИЧЕННАЯ ИСТОРИЯ · {sample}/{floor} МАТЧЕЙ"
    elif not calibration_ready:
        label = (
            "НЕ КАЛИБРОВАНО · СЫРАЯ ВЕРОЯТНОСТЬ "
            f"(калибратор лиги: {calibration_matches} матчей)"
        )
    else:
        return {
            "pick": item.get("pick"),
            "label": str(item.get("label") or "Наиболее вероятный исход"),
            "probability": item.get("probability"),
            "history_sufficient": True,
            "calibrated": True,
        }

    return {
        "pick": None,
        "label": label,
        "probability": None,
        "history_sufficient": history_sufficient,
        "calibrated": False,
    }


def rank_model_favorites(
    cards: object, limit: int | None = None, *, min_sample_games: int = 0,
) -> list[dict]:
    """Return unique fixtures ranked by probability, decisiveness and data support.

    Duplicate fixtures are resolved by the strongest forecast/data row, not by
    whichever provider or row happens to appear first in the input list.
    """
    if not isinstance(cards, list):
        return []

    try:
        minimum_history = max(0, int(min_sample_games))
    except (TypeError, ValueError):
        minimum_history = 0

    unique = {}
    for index, card in enumerate(cards):
        # A zero-history fallback vector is a generic prior, not a fixture
        # forecast. Keep it out of probability-ranked model favorites.
        if isinstance(card, dict) and card.get("model_support_status") in {"prior_only", "league_only"}:
            continue
        favorite = model_favorite(card)
        if favorite is None or favorite["sample_games"] < minimum_history:
            continue
        key = _fixture_key(card, index)
        candidate = {"card": card, **favorite}
        current = unique.get(key)
        if current is None:
            unique[key] = candidate
            continue
        if _duplicate_quality(card) > _duplicate_quality(current["card"]):
            unique[key] = candidate

    ranked = list(unique.values())
    ranked.sort(key=lambda item: (
        -item["probability"],
        -item["probability_margin"],
        -item["sample_games"],
        str(item["card"].get("date_iso") or item["card"].get("date") or ""),
        str(item["card"].get("match") or item["card"].get("match_ru") or ""),
        hashlib.sha256(
            json.dumps(item["card"], ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    ))
    if limit is not None:
        try:
            ranked = ranked[:max(0, int(limit))]
        except (TypeError, ValueError):
            pass
    return ranked

