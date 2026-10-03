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
from typing import Optional


DEFAULT_WEIGHTS = {
    "confidence": 0.40,   # основной вес
    "ev": 0.25,
    "edge": 0.20,
    "kelly": 0.15,
}

# Шкалы для нормализации (чтобы привести разные метрики к [0;1])
SCALE_EV = 0.10       # EV 10% = очень хорошо
SCALE_EDGE = 0.08     # Edge 8% = отлично
SCALE_KELLY = 0.05    # Kelly 5% = максимальный риск


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


def rank_score(card: dict,
               weights: Optional[dict] = None) -> float:
    """Возвращает score в [0;1] — чем выше, тем сильнее сигнал."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    v = card.get("verdict", {}) or {}

    conf = _confidence_norm(card)

    try:
        ev = float(v.get("ev") or 0.0)
    except (TypeError, ValueError):
        ev = 0.0
    try:
        edge = float(v.get("edge") or v.get("ev") or 0.0)
    except (TypeError, ValueError):
        edge = 0.0
    try:
        kelly_ = float(v.get("stake") or 0.0)
        if kelly_ <= 0:
            kelly_ = 0.0
    except (TypeError, ValueError):
        kelly_ = 0.0

    s_ev = _sigmoid(ev, SCALE_EV)
    s_edge = _sigmoid(edge, SCALE_EDGE)
    s_kelly = _sigmoid(kelly_, SCALE_KELLY)

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
