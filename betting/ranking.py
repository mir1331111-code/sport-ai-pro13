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

    sample_games = _number(card.get("games"), 0.0) or 0.0
    if sample_games <= 0:
        features = card.get("independent_features") or {}
        if not isinstance(features, dict):
            features = {}
        home_sample = _number(features.get("home_home_sample"), 0.0) or 0.0
        away_sample = _number(features.get("away_away_sample"), 0.0) or 0.0
        sample_games = min(home_sample, away_sample)
    sample_games = max(0.0, sample_games)
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
        "sample_support": sample_support,
        "rank_score": rank_score,
        "strength": strength,
    }


def rank_model_favorites(cards: object, limit: int | None = None) -> list[dict]:
    """Return unique fixtures ranked by probability, decisiveness and data support.

    Duplicate fixtures are resolved by the strongest forecast/data row, not by
    whichever provider or row happens to appear first in the input list.
    """
    if not isinstance(cards, list):
        return []

    unique = {}
    for index, card in enumerate(cards):
        favorite = model_favorite(card)
        if favorite is None:
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

