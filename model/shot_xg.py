"""Leakage-safe aggregate-shot expected-goals proxy.

football-data.co.uk has match-level shot totals and shots on target, not
individual shot locations, body parts or situations. This estimates an
aggregate-shot proxy, not genuine shot-level xG. Fits are chronological and
competition-specific, and estimates are kept separate from provider xG.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

SHOT_XG_VERSION = "shot_aggregate_ridge_v1"
MIN_TRAINING_SIDES = 60
_PRIOR = (0.04, 0.24, 0.025)
_RIDGE = (24.0, 5.0, 8.0)


def _number(value, *, maximum: int) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < 0 or number > maximum or not number.is_integer():
        return None
    return int(number)


def _match_date(value) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%d/%m/%y",
    ):
        try:
            candidate = raw[:19] if ("T" in fmt or "%H" in fmt) else raw
            return datetime.strptime(candidate, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _xg_value(value) -> float | None:
    try:
        number = float(str(value).strip().replace(",", "."))
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 8.0 else None


def _has_xg_pair(row: dict) -> bool:
    home_keys = ("HxG", "home_xg", "homeXG", "xG_home", "xg_home",
                 "HomeXG", "home_expected_goals", "xG", "xg")
    away_keys = ("AxG", "away_xg", "awayXG", "xG_away", "xg_away",
                 "AwayXG", "away_expected_goals", "xGA", "xga")
    home = next((_xg_value(row.get(k)) for k in home_keys
                 if _xg_value(row.get(k)) is not None), None)
    away = next((_xg_value(row.get(k)) for k in away_keys
                 if _xg_value(row.get(k)) is not None), None)
    return home is not None and away is not None


def _side_features(total_shots, shots_on_target, goals):
    total = _number(total_shots, maximum=45)
    on_target = _number(shots_on_target, maximum=30)
    score = _number(goals, maximum=15)
    if total is None or on_target is None or score is None:
        return None
    if on_target > total or score > on_target:
        return None
    return (float(on_target), float(total - on_target), float(score))


def _solve_3x3(matrix, vector):
    augmented = [list(matrix[i]) + [vector[i]] for i in range(3)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(augmented[r][col]))
        if abs(augmented[pivot][col]) < 1e-10:
            return None
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        divisor = augmented[col][col]
        augmented[col] = [v / divisor for v in augmented[col]]
        for row in range(3):
            if row == col:
                continue
            factor = augmented[row][col]
            augmented[row] = [
                augmented[row][idx] - factor * augmented[col][idx]
                for idx in range(4)
            ]
    return tuple(augmented[i][3] for i in range(3))


def fit_shot_xg_coefficients(samples):
    """Fit goals ~ shots-on-target + other shots with a regularized linear model."""
    if len(samples) < MIN_TRAINING_SIDES:
        return None
    matrix = [[0.0] * 3 for _ in range(3)]
    vector = [0.0] * 3
    for on_target, other_shots, goals in samples:
        features = (1.0, float(on_target), float(other_shots))
        for i in range(3):
            vector[i] += features[i] * float(goals)
            for j in range(3):
                matrix[i][j] += features[i] * features[j]
    for i, penalty in enumerate(_RIDGE):
        matrix[i][i] += penalty
        vector[i] += penalty * _PRIOR[i]
    fitted = _solve_3x3(matrix, vector)
    if fitted is None:
        return None
    intercept, on_target_weight, other_weight = fitted
    return (
        min(0.25, max(0.0, intercept)),
        min(0.55, max(0.08, on_target_weight)),
        min(0.10, max(0.0, other_weight)),
    )


def _estimate(features, coefficients):
    on_target, other_shots, _goals = features
    intercept, on_target_weight, other_weight = coefficients
    value = intercept + on_target * on_target_weight + other_shots * other_weight
    return round(min(4.5, max(0.05, value)), 3)


def enrich_history_with_shot_xg(history: dict, divisions, logs=None) -> dict:
    """Generate same-day estimates using only earlier-date data per competition."""
    totals = {
        "generated": 0, "provider_preserved": 0, "eligible": 0,
        "missing_shot_data": 0, "insufficient_history": 0,
        "training_sides": 0, "divisions": 0,
    }
    for division in divisions or []:
        by_day = defaultdict(list)
        for season_rows in (history or {}).get(division, []) or []:
            for row in season_rows or []:
                if isinstance(row, dict):
                    day = _match_date(row.get("Date") or row.get("date"))
                    if day:
                        by_day[day].append(row)
        if not by_day:
            continue
        totals["divisions"] += 1
        samples = []
        for day in sorted(by_day):
            day_rows = by_day[day]
            coefficients = fit_shot_xg_coefficients(samples)
            for row in day_rows:
                home = _side_features(row.get("HS"), row.get("HST"), row.get("FTHG"))
                away = _side_features(row.get("AS"), row.get("AST"), row.get("FTAG"))
                if home is None or away is None:
                    totals["missing_shot_data"] += 1
                    continue
                totals["eligible"] += 1
                if _has_xg_pair(row):
                    totals["provider_preserved"] += 1
                    continue
                if coefficients is None:
                    totals["insufficient_history"] += 1
                    continue
                row["HxG"] = _estimate(home, coefficients)
                row["AxG"] = _estimate(away, coefficients)
                row["xg_source"] = SHOT_XG_VERSION
                row["xg_kind"] = "modelled_proxy"
                row["xg_model_version"] = SHOT_XG_VERSION
                row["xg_training_sides"] = len(samples)
                totals["generated"] += 1
            # No fixture from this date can train another fixture from this date.
            for row in day_rows:
                home = _side_features(row.get("HS"), row.get("HST"), row.get("FTHG"))
                away = _side_features(row.get("AS"), row.get("AST"), row.get("FTAG"))
                if home is not None and away is not None:
                    samples.extend((home, away))
        totals["training_sides"] += len(samples)
    if logs is not None:
        logs.append(
            "📐 Собственный xG proxy (модель агрегированных ударов, не shot-level xG): "
            f"рассчитано={totals['generated']}, настоящих xG сохранено={totals['provider_preserved']}, "
            f"подходящих строк={totals['eligible']}, без корректных HS/HST/AS/AST={totals['missing_shot_data']}, "
            f"недостаточно предыдущей истории={totals['insufficient_history']}, "
            f"лиг={totals['divisions']}."
        )
    return totals
