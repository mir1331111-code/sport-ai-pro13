"""Leakage-safe context blending, confidence gates and paired ablation metrics."""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any


OUTCOMES = ("H", "D", "A")
CONTEXT_FEATURES = (
    "lineup_strength_home",
    "lineup_strength_away",
    "key_players_missing_home",
    "key_players_missing_away",
    "source_reliability",
)


def _valid_probability_map(values: dict) -> dict[str, float]:
    if not isinstance(values, dict):
        raise ValueError("probabilities must be a mapping")
    probabilities = {key: float(values[key]) for key in OUTCOMES}
    if any(not math.isfinite(value) or value < 0 for value in probabilities.values()):
        raise ValueError("probabilities must be finite and non-negative")
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("probabilities must sum to a positive value")
    return {key: value / total for key, value in probabilities.items()}


def assess_context_quality(context: dict | None) -> dict:
    context = context if isinstance(context, dict) else {}
    lineup = context.get("lineup") or {}
    referee = context.get("referee") or {}
    def _reliability_ok(source):
        raw = (source or {}).get("source_reliability")
        if raw is None:
            return True, False
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return False, True
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return False, True
        return value >= 0.60, True

    lineup_reliability_ok, lineup_reliability_known = _reliability_ok(lineup)
    referee_reliability_ok, referee_reliability_known = _reliability_ok(referee)
    lineup_confirmed = bool(
        lineup.get("is_confirmed")
        or (
            (lineup.get("home") or {}).get("confirmed")
            and (lineup.get("away") or {}).get("confirmed")
        )
    ) and lineup_reliability_ok
    referee_confirmed = bool(
        referee.get("is_confirmed") and str(referee.get("name") or "").strip()
    ) and referee_reliability_ok
    injuries_confirmed = bool(context.get("injuries_is_confirmed"))
    injuries_reliability_ok, injuries_reliability_known = _reliability_ok({
        "source_reliability": context.get("injuries_source_reliability")
    })
    injuries_confirmed = injuries_confirmed and injuries_reliability_ok
    core_complete = lineup_confirmed and referee_confirmed
    reliability_calibrated = (
        lineup_reliability_known and referee_reliability_known
        and (not context.get("injuries_is_confirmed") or injuries_reliability_known)
    )
    if core_complete and injuries_confirmed and reliability_calibrated:
        level = "High"
    elif core_complete or lineup_confirmed or referee_confirmed or injuries_confirmed:
        level = "Medium"
    else:
        level = "Low"
    missing = []
    if not lineup_confirmed:
        missing.append("confirmed_starting_XI")
    if not referee_confirmed:
        missing.append("referee")
    if not injuries_confirmed:
        missing.append("injuries_feed")
    if (
        (lineup_reliability_known and not lineup_reliability_ok)
        or (referee_reliability_known and not referee_reliability_ok)
        or (context.get("injuries_is_confirmed") and injuries_reliability_known
            and not injuries_reliability_ok)
    ):
        missing.append("low_source_reliability")
    return {
        "level": level,
        "lineup_confirmed": lineup_confirmed,
        "referee_confirmed": referee_confirmed,
        "injuries_confirmed": injuries_confirmed,
        "strong_value_eligible": core_complete,
        "source_reliability_calibrated": reliability_calibrated,
        "missing": missing,
        "reason": (
            "контекст XI и судья подтверждён"
            if core_complete else
            "нет подтверждённого стартового XI и/или судьи; Strong/Value заблокированы"
        ),
    }


def _historical_context_eligible(row: dict) -> bool:
    if not bool(row.get("is_confirmed")):
        return False
    if not bool(row.get("lineup_is_confirmed", row.get("is_confirmed"))):
        return False
    if not bool(row.get("referee_is_confirmed", row.get("is_confirmed"))):
        return False
    reliability_values = [
        row.get("lineup_source_reliability"),
        row.get("referee_source_reliability"),
        row.get("source_reliability"),
    ]
    known = [float(value) for value in reliability_values if value is not None]
    if any(not math.isfinite(value) or value < 0.60 or value > 1.0 for value in known):
        return False
    return bool(known)


def _feature_vector(context: dict) -> list[float]:
    values = []
    for name in CONTEXT_FEATURES:
        raw = context.get(name)
        if raw is None:
            raise ValueError(f"missing context feature: {name}")
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError(f"invalid context feature: {name}")
        values.append(value)
    return values


def context_probabilities(
    base_probabilities: dict,
    context: dict | None,
    model: dict | None = None,
) -> dict:
    """Return paired Base/Context probabilities.

    Context stays equal to Base unless a leakage-safe, trained blend model is
    supplied. No heuristic player or referee effect is fabricated.
    """
    base = _valid_probability_map(base_probabilities)
    quality = assess_context_quality(context)
    adjusted = dict(base)
    applied = False
    if (isinstance(model, dict) and model.get("trained") is True
            and model.get("walk_forward_validated") is True):
        try:
            vector = _feature_vector(context or {})
            means = list(model["feature_means"])
            scales = list(model["feature_scales"])
            weights = model["weights"]
            bias = model["bias"]
            if not (len(vector) == len(means) == len(scales)
                    and all(len(weights[outcome]) == len(vector) for outcome in OUTCOMES)):
                raise ValueError("context model feature shape mismatch")
            logits = []
            for outcome in OUTCOMES:
                value = math.log(max(base[outcome], 1e-12)) + float(bias[outcome])
                for index, feature in enumerate(vector):
                    scale = float(scales[index]) or 1.0
                    value += float(weights[outcome][index]) * (
                        (feature - float(means[index])) / scale
                    )
                logits.append(value)
            maximum = max(logits)
            exp_values = [math.exp(value - maximum) for value in logits]
            total = sum(exp_values)
            adjusted = {
                outcome: exp_values[index] / total
                for index, outcome in enumerate(OUTCOMES)
            }
            applied = True
        except (KeyError, TypeError, ValueError, OverflowError):
            adjusted = dict(base)
            applied = False
    return {
        "Base": dict(base),
        "Context": adjusted,
        "delta": {key: adjusted[key] - base[key] for key in OUTCOMES},
        "adjustment_applied": applied,
        "quality": quality,
    }


def fit_context_blender(
    rows: list[dict],
    *,
    min_samples: int = 200,
    epochs: int = 600,
    learning_rate: float = 0.025,
    l2: float = 0.02,
) -> dict:
    """Fit a multinomial residual blender on confirmed pre-kickoff context.

    Rows must contain fixture_id, kickoff_at, observed_at, result, confirmed
    context features and Base H/D/A probabilities. A model is not emitted below
    min_samples; caller must validate it walk-forward before production use.
    """
    prepared = []
    seen = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        fixture_id = str(row.get("fixture_id") or "")
        if not fixture_id or fixture_id in seen:
            continue
        try:
            kickoff = datetime.fromisoformat(str(row["kickoff_at"]).replace("Z", "+00:00"))
            observed = datetime.fromisoformat(str(row["observed_at"]).replace("Z", "+00:00"))
            if kickoff.tzinfo is None or observed.tzinfo is None or observed >= kickoff:
                continue
            if not _historical_context_eligible(row):
                continue
            x = _feature_vector(row)
            base = _valid_probability_map({
                "H": row.get("base_p_home"),
                "D": row.get("base_p_draw"),
                "A": row.get("base_p_away"),
            })
            outcome = str(row.get("result") or "").upper()
            if outcome not in OUTCOMES:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        seen.add(fixture_id)
        prepared.append((fixture_id, x, base, outcome))
    if len(prepared) < int(min_samples) or len({r[3] for r in prepared}) < 2:
        return {
            "trained": False, "n": len(prepared),
            "reason": f"need at least {int(min_samples)} unique confirmed pre-kickoff fixtures",
        }

    n_features = len(CONTEXT_FEATURES)
    means = [
        sum(item[1][j] for item in prepared) / len(prepared)
        for j in range(n_features)
    ]
    scales = []
    for j in range(n_features):
        variance = sum((item[1][j] - means[j]) ** 2 for item in prepared) / len(prepared)
        scales.append(math.sqrt(variance) or 1.0)
    x_values = [
        [(item[1][j] - means[j]) / scales[j] for j in range(n_features)]
        for item in prepared
    ]
    weights = {outcome: [0.0] * n_features for outcome in OUTCOMES}
    bias = {outcome: 0.0 for outcome in OUTCOMES}
    rate = float(learning_rate)
    for _ in range(max(1, int(epochs))):
        grad_w = {outcome: [0.0] * n_features for outcome in OUTCOMES}
        grad_b = {outcome: 0.0 for outcome in OUTCOMES}
        for index, (fixture_id, _, base, actual) in enumerate(prepared):
            logits = [
                math.log(max(base[outcome], 1e-12)) + bias[outcome]
                + sum(weights[outcome][j] * x_values[index][j] for j in range(n_features))
                for outcome in OUTCOMES
            ]
            maximum = max(logits)
            exp_values = [math.exp(value - maximum) for value in logits]
            total = sum(exp_values)
            probs = [value / total for value in exp_values]
            for class_index, outcome in enumerate(OUTCOMES):
                error = probs[class_index] - float(actual == outcome)
                grad_b[outcome] += error / len(prepared)
                for j in range(n_features):
                    grad_w[outcome][j] += (
                        error * x_values[index][j] / len(prepared)
                    )
        for outcome in OUTCOMES:
            bias[outcome] -= rate * grad_b[outcome]
            for j in range(n_features):
                weights[outcome][j] -= rate * (
                    grad_w[outcome][j] + float(l2) * weights[outcome][j]
                )

    return {
        "trained": True,
        "n": len(prepared),
        "feature_names": list(CONTEXT_FEATURES),
        "feature_means": means,
        "feature_scales": scales,
        "weights": weights,
        "bias": bias,
        "training_method": "multinomial_logit_residual_blender",
        "validation_required": True,
    }



def walk_forward_context_ablation(
    rows: list[dict],
    *,
    min_train: int = 200,
    min_validation: int = 500,
    epochs: int = 600,
) -> dict:
    """Run chronological, same-day-batched Base vs Context ablation.

    A day's results are never used to predict other fixtures on that same day.
    Production model is promoted only when the out-of-time paired sample reaches
    min_validation and Context improves both Brier and Log Loss.
    """
    latest_snapshots = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        fixture_id = str(row.get("fixture_id") or "")
        if not fixture_id:
            continue
        try:
            kickoff = datetime.fromisoformat(str(row["kickoff_at"]).replace("Z", "+00:00"))
            observed = datetime.fromisoformat(str(row["observed_at"]).replace("Z", "+00:00"))
            if (
                kickoff.tzinfo is None or observed.tzinfo is None
                or observed >= kickoff
            ):
                continue
            previous = latest_snapshots.get(fixture_id)
            if previous is None:
                latest_snapshots[fixture_id] = row
            else:
                previous_observed = datetime.fromisoformat(
                    str(previous["observed_at"]).replace("Z", "+00:00")
                )
                if observed > previous_observed:
                    latest_snapshots[fixture_id] = row
        except (KeyError, TypeError, ValueError):
            continue

    prepared = []
    for row in latest_snapshots.values():
        try:
            kickoff = datetime.fromisoformat(str(row["kickoff_at"]).replace("Z", "+00:00"))
            _feature_vector(row)
            base = _valid_probability_map({
                "H": row.get("base_p_home"),
                "D": row.get("base_p_draw"),
                "A": row.get("base_p_away"),
            })
            outcome = str(row.get("result") or "").upper()
            if not _historical_context_eligible(row) or outcome not in OUTCOMES:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        prepared.append((kickoff, row, base, outcome))
    prepared.sort(key=lambda item: (item[0], str(item[1].get("fixture_id") or "")))

    train_rows = []
    base_rows, context_rows = [], []
    cursor = 0
    while cursor < len(prepared):
        day = prepared[cursor][0].date()
        group = []
        while cursor < len(prepared) and prepared[cursor][0].date() == day:
            group.append(prepared[cursor])
            cursor += 1

        model = fit_context_blender(
            train_rows, min_samples=min_train, epochs=epochs
        )
        # In-sample promotion is not allowed. This temporary flag is used only
        # to score the next unseen date block during the walk-forward run.
        if model.get("trained"):
            model = dict(model, walk_forward_validated=True)
        for _, row, base, outcome in group:
            predicted = context_probabilities(base, row, model)
            fixture_id = str(row["fixture_id"])
            base_rows.append({
                "fixture_id": fixture_id,
                "result": outcome,
                "probabilities": dict(base),
            })
            context_rows.append({
                "fixture_id": fixture_id,
                "result": outcome,
                "probabilities": dict(predicted["Context"]),
            })
        # Append labels only after every fixture in the date block is forecast.
        train_rows.extend(row for _, row, _, _ in group)

    report = evaluate_ablation(base_rows, context_rows)
    report.update({
        "walk_forward": True,
        "training_floor": int(min_train),
        "validation_floor": int(min_validation),
        "training_rows": len(train_rows),
    })
    deltas = report.get("delta_context_minus_base") or {}
    improved = (
        report["n"] >= int(min_validation)
        and deltas.get("brier") is not None
        and deltas.get("log_loss") is not None
        and deltas["brier"] < 0
        and deltas["log_loss"] < 0
    )
    final_model = fit_context_blender(
        train_rows, min_samples=min_train, epochs=epochs
    )
    if improved and final_model.get("trained"):
        final_model["walk_forward_validated"] = True
        final_model["validation"] = {
            "n": report["n"],
            "brier_delta": deltas["brier"],
            "log_loss_delta": deltas["log_loss"],
        }
    else:
        final_model = {
            "trained": False,
            "walk_forward_validated": False,
            "reason": (
                "context not promoted: require at least "
                f"{int(min_validation)} paired fixtures and improvement in both "
                "Brier and Log Loss"
            ),
        }
    report["promoted_model"] = final_model
    return report

def _metrics(rows: list[tuple[dict, str]]) -> dict:
    if not rows:
        return {"n": 0, "brier": None, "log_loss": None}
    brier, log_loss = 0.0, 0.0
    for probabilities, actual in rows:
        normalized = _valid_probability_map(probabilities)
        brier += sum(
            (normalized[key] - float(actual == key)) ** 2 for key in OUTCOMES
        )
        log_loss -= math.log(max(normalized[actual], 1e-15))
    n = len(rows)
    return {"n": n, "brier": brier / n, "log_loss": log_loss / n}


def evaluate_ablation(base_rows: list[dict], context_rows: list[dict]) -> dict:
    """Compare Base and Context on identical, unique settled fixture IDs."""
    base_map, context_map = {}, {}
    for collection, target in ((base_rows or [], base_map), (context_rows or [], context_map)):
        for row in collection:
            if not isinstance(row, dict):
                continue
            fixture_id = str(row.get("fixture_id") or "")
            outcome = str(row.get("result") or "").upper()
            if fixture_id and outcome in OUTCOMES and fixture_id not in target:
                try:
                    target[fixture_id] = (_valid_probability_map(row.get("probabilities") or {}), outcome)
                except (TypeError, ValueError, KeyError):
                    continue
    shared = sorted(set(base_map) & set(context_map))
    base = [(base_map[key][0], base_map[key][1]) for key in shared]
    context = [(context_map[key][0], context_map[key][1]) for key in shared]
    base_metrics, context_metrics = _metrics(base), _metrics(context)
    return {
        "n": len(shared),
        "fixture_ids": shared,
        "base": base_metrics,
        "context": context_metrics,
        "delta_context_minus_base": {
            name: (
                context_metrics[name] - base_metrics[name]
                if base_metrics[name] is not None and context_metrics[name] is not None
                else None
            )
            for name in ("brier", "log_loss")
        },
        "same_sample": True,
        "interpretation": "negative delta favours Context; positive delta favours Base",
    }
