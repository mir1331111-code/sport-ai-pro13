"""Leakage-safe chronological model and xG ablation evaluation."""
from __future__ import annotations

import math
from datetime import date, datetime

from model.engine import Engine, _row_xg_pair


_XG_KEYS = {
    "hxg", "axg", "home_xg", "away_xg", "homexg", "awayxg",
    "xg_home", "xg_away", "xg", "xga", "home_expected_goals",
    "away_expected_goals",
}


def _get(row, *names):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _parse_date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        raw = str(value).strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw).date()
        except ValueError:
            return date.fromisoformat(raw[:10])
    except (TypeError, ValueError):
        return None


def _without_xg(row):
    """Copy a historical row without xG fields for a matched ablation arm."""
    if not isinstance(row, dict):
        return {}
    return {key: value for key, value in row.items()
            if str(key).strip().casefold() not in _XG_KEYS}


def _metrics(records):
    if not records:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    brier, losses, correct = [], [], 0
    for probabilities, outcome in records:
        brier.append(sum((probabilities[k] - float(k == outcome)) ** 2 for k in ("H", "D", "A")))
        losses.append(-math.log(max(probabilities[outcome], 1e-15)))
        correct += max(("H", "D", "A"), key=lambda k: probabilities[k]) == outcome
    n = len(records)
    return {"n": n, "brier": sum(brier) / n, "log_loss": sum(losses) / n, "accuracy": correct / n}


def _probs(prediction):
    return {"H": prediction["p1"], "D": prediction["x"], "A": prediction["p2"]}


def walk_forward_compare(rows, *, matrix_n: int = 12) -> dict:
    """Compare goals-only and xG engines on identical chronological fixtures.

    All fixtures on the same calendar date are predicted before any result from
    that date is added. The xG ablation pair uses identical temporal weighting
    and rho settings; only historical xG ingestion differs. The overall
    baseline-vs-candidate comparison remains available for backwards compatibility.
    """
    prepared, skipped = [], 0
    for row in rows:
        try:
            stamp = _parse_date(_get(row, "Date", "date", "match_date", "kickoff_date"))
            home = _get(row, "HomeTeam", "home_team", "home")
            away = _get(row, "AwayTeam", "away_team", "away")
            hg = int(_get(row, "FTHG", "home_goals", "hg"))
            ag = int(_get(row, "FTAG", "away_goals", "ag"))
            league = str(_get(row, "Div", "league", "competition") or "G")
            if stamp is None or not home or not away or hg < 0 or ag < 0:
                skipped += 1
                continue
            prepared.append((stamp, str(league), str(home), str(away), hg, ag, row))
        except (TypeError, ValueError):
            skipped += 1
    prepared.sort(key=lambda item: item[0])

    baseline = Engine(matrix_n=matrix_n, temporal_weighting=False, league_rho_fitting=False)
    candidate = Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True)
    temporal_goals = Engine(matrix_n=matrix_n, temporal_weighting=True, league_rho_fitting=True)
    records = {"baseline": [], "candidate": [], "temporal_goals_only": []}
    xg_forecasts = xg_eligible_rows = 0
    index = 0
    cursor = 0
    while cursor < len(prepared):
        stamp = prepared[cursor][0]
        group = []
        while cursor < len(prepared) and prepared[cursor][0] == stamp:
            group.append(prepared[cursor])
            cursor += 1
        pending = []
        for _, league, home, away, hg, ag, row in group:
            row_copy = dict(row) if isinstance(row, dict) else {}
            row_copy["Div"] = league
            no_xg_row = _without_xg(row_copy)
            if _row_xg_pair(row_copy)[0] is not None:
                xg_eligible_rows += 1

            pred_base = baseline.predict(home, away, league, match_date=stamp.isoformat())
            pred_candidate = candidate.predict(home, away, league, match_date=stamp.isoformat())
            pred_temporal_goals = temporal_goals.predict(home, away, league, match_date=stamp.isoformat())
            if pred_candidate.get("xg_used"):
                xg_forecasts += 1
            outcome = "H" if hg > ag else "A" if hg < ag else "D"
            records["baseline"].append((_probs(pred_base), outcome))
            records["candidate"].append((_probs(pred_candidate), outcome))
            records["temporal_goals_only"].append((_probs(pred_temporal_goals), outcome))
            pending.append((home, away, hg, ag, row_copy, no_xg_row, stamp))

        # Ingest outcomes only after every forecast for this date has been made.
        for home, away, hg, ag, row_copy, no_xg_row, match_stamp in pending:
            baseline.add(home, away, hg, ag, no_xg_row, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            candidate.add(home, away, hg, ag, row_copy, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            temporal_goals.add(home, away, hg, ag, no_xg_row, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            index += 1

    metrics = {name: _metrics(value) for name, value in records.items()}
    base_metrics, candidate_metrics = metrics["baseline"], metrics["candidate"]
    temporal_metrics = metrics["temporal_goals_only"]
    delta = lambda left, right, key: (
        left[key] - right[key] if left[key] is not None and right[key] is not None else None
    )
    return {
        "n": len(prepared),
        "skipped": skipped,
        "xg_rows_available": xg_eligible_rows,
        "xg_forecasts_used": xg_forecasts,
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "temporal_goals_only": temporal_metrics,
        "delta_candidate_minus_baseline": {
            key: delta(candidate_metrics, base_metrics, key)
            for key in ("brier", "log_loss", "accuracy")
        },
        "xg_delta_vs_same_temporal_model": {
            key: delta(candidate_metrics, temporal_metrics, key)
            for key in ("brier", "log_loss", "accuracy")
        },
        "interpretation": (
            "xG effect is isolated by comparing candidate against temporal_goals_only on identical fixtures. "
            "Lower Brier/log_loss and higher accuracy are better. Do not claim a gain unless it persists "
            "on a separate untouched chronological holdout with adequate xG coverage."
        ),
    }
