"""Leakage-aware historical model for corners and disciplinary cards.

Uses only observed football-data.co.uk match statistics (HC/AC, HY/AY, HR/AR).
Predictions are Poisson approximations from pre-match team/league rates; no odds
or fabricated missing values are used. This is a baseline to evaluate, not a
claim of validated betting edge.
"""
from __future__ import annotations

import math
from collections import defaultdict


_FIELDS = {
    "corners": ("HC", "AC"),
    "cards": ("HY", "AY"),
}


def _num(value):
    try:
        x = float(str(value).strip().replace(",", "."))
        return x if math.isfinite(x) and 0 <= x <= 50 else None
    except (TypeError, ValueError):
        return None


def _date_key(row):
    return str(row.get("Date", ""))


class HistoricalMatchStats:
    """Online, chronological rates; call predict before learn_step for each match."""
    def __init__(self, min_team_games=4):
        self.min_team_games = int(min_team_games)
        self.league = defaultdict(lambda: defaultdict(list))
        self.teams = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.trained_matches = defaultdict(int)

    def learn_step(self, row, league="G"):
        if not isinstance(row, dict):
            return False
        values = {}
        for market, (home_key, away_key) in _FIELDS.items():
            h, a = _num(row.get(home_key)), _num(row.get(away_key))
            if h is None or a is None:
                continue
            values[market] = (h, a)
        if not values:
            return False
        home, away = str(row.get("HomeTeam") or "").strip(), str(row.get("AwayTeam") or "").strip()
        if not home or not away:
            return False
        for market, (h, a) in values.items():
            self.league[league][market].append((h, a))
            self.teams[league][home][market]["home_for"].append(h)
            self.teams[league][home][market]["home_against"].append(a)
            self.teams[league][away][market]["away_for"].append(a)
            self.teams[league][away][market]["away_against"].append(h)
        self.trained_matches[league] += 1
        return True

    def predict(self, home, away, league="G"):
        result = {}
        for market in _FIELDS:
            league_rows = self.league[league][market]
            if not league_rows:
                continue
            lh = sum(x[0] for x in league_rows) / len(league_rows)
            la = sum(x[1] for x in league_rows) / len(league_rows)
            h = self.teams[league][home][market]
            a = self.teams[league][away][market]
            h_for = h["home_for"]
            h_against = h["home_against"]
            a_for = a["away_for"]
            a_against = a["away_against"]
            exp_h = lh
            exp_a = la
            if len(h_for) >= self.min_team_games:
                exp_h = 0.5 * exp_h + 0.25 * (sum(h_for) / len(h_for)) + 0.25 * (sum(a_against) / len(a_against) if a_against else la)
            if len(a_for) >= self.min_team_games:
                exp_a = 0.5 * exp_a + 0.25 * (sum(a_for) / len(a_for)) + 0.25 * (sum(h_against) / len(h_against) if h_against else lh)
            result[market] = {
                "home": max(0.1, exp_h),
                "away": max(0.1, exp_a),
                "total": max(0.2, exp_h + exp_a),
                "history_matches": len(league_rows),
                "home_team_games": len(h_for),
                "away_team_games": len(a_for),
            }
        return result


def poisson_over_probability(mean, line):
    """P(X > line), supporting integer/half lines; returns None for invalid input."""
    try:
        mean, line = float(mean), float(line)
        if mean <= 0 or not math.isfinite(mean) or not math.isfinite(line):
            return None
        # For half-lines, over 9.5 means X >= 10; for integer 9.0, pushes are
        # possible, so this returns P(X >= 10) and should be treated as an estimate.
        cutoff = math.floor(line) + 1
        p = math.exp(-mean)
        cdf = p
        for k in range(1, cutoff):
            p *= mean / k
            cdf += p
        return min(1.0, max(0.0, 1.0 - cdf))
    except (TypeError, ValueError, OverflowError):
        return None
