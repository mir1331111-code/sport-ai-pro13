"""Chronological out-of-sample comparison of the legacy and temporal engines."""
from __future__ import annotations

import math
from datetime import date, datetime

from model.engine import Engine


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


def walk_forward_compare(rows, *, matrix_n: int = 12) -> dict:
    """Compare equal-weight baseline against temporal weighting + fitted league rho.

    Rows are sorted by kickoff date. Each match is predicted before its score is
    added to either engine, so the match cannot train its own forecast. This is
    an evaluation utility, not a guarantee that the candidate will win.
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
    baseline_records, candidate_records = [], []
    index = 0
    cursor = 0
    while cursor < len(prepared):
        # If historical data only has calendar dates, predict every fixture on
        # the same date before adding any result from that date.
        stamp = prepared[cursor][0]
        group = []
        while cursor < len(prepared) and prepared[cursor][0] == stamp:
            group.append(prepared[cursor])
            cursor += 1
        pending = []
        for _, league, home, away, hg, ag, row in group:
            baseline_prediction = baseline.predict(home, away, league, match_date=stamp.isoformat())
            candidate_prediction = candidate.predict(home, away, league, match_date=stamp.isoformat())
            outcome = "H" if hg > ag else "A" if hg < ag else "D"
            baseline_records.append(({"H": baseline_prediction["p1"], "D": baseline_prediction["x"], "A": baseline_prediction["p2"]}, outcome))
            candidate_records.append(({"H": candidate_prediction["p1"], "D": candidate_prediction["x"], "A": candidate_prediction["p2"]}, outcome))
            row_copy = dict(row) if isinstance(row, dict) else {}
            row_copy["Div"] = league
            pending.append((home, away, hg, ag, row_copy, stamp))
        for home, away, hg, ag, row_copy, match_stamp in pending:
            baseline.add(home, away, hg, ag, row_copy, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            candidate.add(home, away, hg, ag, row_copy, match_num=index, total=len(prepared), match_date=match_stamp.isoformat())
            index += 1
    base_metrics, candidate_metrics = _metrics(baseline_records), _metrics(candidate_records)
    return {
        "n": len(prepared),
        "skipped": skipped,
        "baseline": base_metrics,
        "candidate": candidate_metrics,
        "delta_candidate_minus_baseline": {
            key: (candidate_metrics[key] - base_metrics[key])
            if candidate_metrics[key] is not None and base_metrics[key] is not None else None
            for key in ("brier", "log_loss", "accuracy")
        },
        "interpretation": "Lower Brier/log_loss is better; higher accuracy is better. Promote the candidate only after a separate untouched chronological holdout confirms the gain.",
    }
