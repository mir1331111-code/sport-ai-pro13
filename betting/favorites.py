"""Select unique fixtures by strongest, better-supported 1X2 forecasts.

This ranking controls display/analysis priority only. It is not a betting signal:
bookmaker prices, EV, edge, Kelly and market validation remain separate gates.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata


def _number(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _team_key(value) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"\b(football club|fc|afc|cf|sc|ac)\b", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text)


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
