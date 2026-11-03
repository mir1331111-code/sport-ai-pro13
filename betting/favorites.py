"""Rank fixtures by the model's most likely 1X2 outcome.

This is a forecast-only ordering helper. It deliberately does not use bookmaker
prices, EV, or value gates and must never be treated as bet authorization.
"""
from __future__ import annotations

import math


def model_favorite(card: object) -> dict | None:
    """Return the most likely 1X2 outcome for a fixture, or None for bad inputs."""
    if not isinstance(card, dict):
        return None

    probabilities = []
    for key in ("p1", "px", "p2"):
        try:
            value = float(card.get(key))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        probabilities.append(value)

    total = sum(probabilities)
    # Reject percentages, partial probability sets, and malformed model output.
    if total <= 0.0 or not 0.90 <= total <= 1.10:
        return None

    home = str(card.get("home") or card.get("HomeTeam") or "").strip()
    away = str(card.get("away") or card.get("AwayTeam") or "").strip()
    match_ru = str(card.get("match_ru") or "").strip()
    if (not home or not away) and " — " in match_ru:
        home, away = (part.strip() for part in match_ru.split(" — ", 1))
    if (not home or not away) and " vs " in str(card.get("match") or ""):
        home, away = (
            part.strip() for part in str(card.get("match") or "").split(" vs ", 1)
        )

    outcomes = (
        ("П1", f"Победа {home}" if home else "Победа хозяев", probabilities[0]),
        ("X", "Ничья", probabilities[1]),
        ("П2", f"Победа {away}" if away else "Победа гостей", probabilities[2]),
    )
    pick, label, probability = max(outcomes, key=lambda item: item[2])
    return {
        "pick": pick,
        "label": label,
        "probability": probability,
        "strength": (
            "Сильный фаворит" if probability >= 0.70 else
            "Фаворит" if probability >= 0.60 else
            "Небольшое преимущество" if probability >= 0.52 else
            "Равный матч"
        ),
    }


def rank_model_favorites(cards: object) -> list[dict]:
    """Return forecastable fixtures, strongest 1X2 probability first."""
    if not isinstance(cards, list):
        return []

    ranked = []
    for card in cards:
        favorite = model_favorite(card)
        if favorite is None:
            continue
        ranked.append({"card": card, **favorite})

    ranked.sort(key=lambda item: (
        -item["probability"],
        str(item["card"].get("date_iso") or item["card"].get("date") or ""),
        str(item["card"].get("match") or ""),
    ))
    return ranked
