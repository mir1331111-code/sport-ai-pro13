"""Leakage-aware historical model for corners and disciplinary cards.

Uses only observed football-data.co.uk match statistics (HC/AC, HY/AY, HR/AR).
Predictions are Poisson approximations from pre-match team/league rates; no odds
or fabricated missing values are used. This is a baseline to evaluate, not a
claim of validated betting edge.
"""
from __future__ import annotations

import math
from collections import defaultdict

from model.market_forecasts import count_over_probability


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
        self.teams = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )
        self.trained_matches = defaultdict(int)
        self.referees = defaultdict(list)
        self.corners_halftime_state = defaultdict(
            lambda: defaultdict(list)
        )

    @staticmethod
    def _plain(value):
        """Convert nested defaultdict containers to ordinary pickle-safe values."""
        if isinstance(value, dict):
            return {
                key: HistoricalMatchStats._plain(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [HistoricalMatchStats._plain(item) for item in value]
        if isinstance(value, tuple):
            return tuple(HistoricalMatchStats._plain(item) for item in value)
        return value

    def __getstate__(self):
        # The live defaultdicts use lambdas as factories. Those lambdas cannot
        # be pickled, so persist plain dictionaries and rebuild factories on load.
        return {
            "min_team_games": self.min_team_games,
            "league": self._plain(self.league),
            "teams": self._plain(self.teams),
            "trained_matches": self._plain(self.trained_matches),
            "referees": self._plain(self.referees),
            "corners_halftime_state": self._plain(self.corners_halftime_state),
        }

    def __setstate__(self, state):
        self.__init__(state.get("min_team_games", 4))
        for league, markets in (state.get("league") or {}).items():
            self.league[league] = defaultdict(
                list,
                {market: list(values) for market, values in markets.items()},
            )
        for league, teams in (state.get("teams") or {}).items():
            league_teams = defaultdict(
                lambda: defaultdict(lambda: defaultdict(list))
            )
            for team, markets in teams.items():
                team_markets = defaultdict(lambda: defaultdict(list))
                for market, dimensions in markets.items():
                    team_markets[market] = defaultdict(
                        list,
                        {dimension: list(values) for dimension, values in dimensions.items()},
                    )
                league_teams[team] = team_markets
            self.teams[league] = league_teams
        self.trained_matches.update({
            league: int(count)
            for league, count in (state.get("trained_matches") or {}).items()
        })
        self.referees.update({
            str(name): list(values)
            for name, values in (state.get("referees") or {}).items()
        })
        for league, states in (state.get("corners_halftime_state") or {}).items():
            self.corners_halftime_state[league] = defaultdict(
                list,
                {str(name): list(values) for name, values in states.items()},
            )

    def learn_step(self, row, league="G"):
        if not isinstance(row, dict):
            return False
        values = {}
        for market, (home_key, away_key) in _FIELDS.items():
            h, a = _num(row.get(home_key)), _num(row.get(away_key))
            if market == "cards":
                # Total cards include yellow + red. If either red-card field is
                # missing, this match is not a valid cards observation.
                hr, ar = _num(row.get("HR")), _num(row.get("AR"))
                if h is None or a is None or hr is None or ar is None:
                    continue
                if not all(value.is_integer() for value in (h, a, hr, ar)):
                    continue
                h += hr
                a += ar
            else:
                if h is None or a is None or not h.is_integer() or not a.is_integer():
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

            if market == "cards":
                referee = str(
                    row.get("Referee") or row.get("referee_id") or row.get("referee") or ""
                ).strip().casefold()
                if referee:
                    self.referees[referee].append(h + a)

            if market == "corners":
                half_home = _num(row.get("HTHG"))
                half_away = _num(row.get("HTAG"))
                if (
                    half_home is not None and half_away is not None
                    and half_home.is_integer() and half_away.is_integer()
                ):
                    state = (
                        "home_trailing" if half_home < half_away
                        else "away_trailing" if half_home > half_away
                        else "level"
                    )
                    self.corners_halftime_state[league][state].append(h + a)

        self.trained_matches[league] += 1
        return True

    def predict(self, home, away, league="G", referee_id=None):
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
            home_for_avg = sum(h_for) / len(h_for) if h_for else None
            home_against_avg = sum(h_against) / len(h_against) if h_against else None
            away_for_avg = sum(a_for) / len(a_for) if a_for else None
            away_against_avg = sum(a_against) / len(a_against) if a_against else None

            exp_h = lh
            exp_a = la
            if len(h_for) >= self.min_team_games:
                exp_h = (
                    0.50 * lh
                    + 0.25 * home_for_avg
                    + 0.25 * (away_against_avg if away_against_avg is not None else la)
                )
            if len(a_for) >= self.min_team_games:
                exp_a = (
                    0.50 * la
                    + 0.25 * away_for_avg
                    + 0.25 * (home_against_avg if home_against_avg is not None else lh)
                )

            league_totals = [float(x[0]) + float(x[1]) for x in league_rows]
            league_total_mean = sum(league_totals) / len(league_totals)
            league_total_variance = (
                sum((value - league_total_mean) ** 2 for value in league_totals)
                / max(1, len(league_totals) - 1)
                if len(league_totals) > 1 else league_total_mean
            )
            total_mean = max(0.2, exp_h + exp_a)
            total_variance = max(0.1, league_total_variance)
            referee_report = {
                "available": False, "n": 0, "mean": None, "variance": None,
                "blend_weight": 0.0,
            }
            if market == "cards" and referee_id:
                referee_key = str(referee_id).strip().casefold()
                referee_values = self.referees.get(referee_key, [])
                referee_n = len(referee_values)
                if referee_n >= 5:
                    referee_mean = sum(referee_values) / referee_n
                    referee_variance = (
                        sum((value - referee_mean) ** 2 for value in referee_values)
                        / max(1, referee_n - 1)
                    )
                    # Conservative empirical-Bayes shrinkage to the team/league
                    # estimate. The market remains forecast-only pending OOS tests.
                    referee_weight = min(0.35, referee_n / (referee_n + 20.0))
                    blended_mean = (
                        (1.0 - referee_weight) * total_mean
                        + referee_weight * referee_mean
                    )
                    blended_variance = (
                        (1.0 - referee_weight) * total_variance
                        + referee_weight * referee_variance
                        + referee_weight * (1.0 - referee_weight)
                        * (referee_mean - total_mean) ** 2
                    )
                    scale = blended_mean / total_mean if total_mean > 0 else 1.0
                    exp_h *= scale
                    exp_a *= scale
                    total_mean = blended_mean
                    total_variance = max(0.1, blended_variance)
                    referee_report = {
                        "available": True,
                        "n": referee_n,
                        "mean": referee_mean,
                        "variance": referee_variance,
                        "blend_weight": referee_weight,
                    }

            result[market] = {
                "home": max(0.1, exp_h),
                "away": max(0.1, exp_a),
                "total": total_mean,
                "total_variance": total_variance,
                "referee": referee_report,
                "league_home_avg": lh,
                "league_away_avg": la,
                "home_for_avg": home_for_avg,
                "home_against_avg": home_against_avg,
                "away_for_avg": away_for_avg,
                "away_against_avg": away_against_avg,
                "history_matches": len(league_rows),
                "home_team_games": len(h_for),
                "away_team_games": len(a_for),
            }
            if market == "corners":
                state_splits = {}
                for state_name, observations in self.corners_halftime_state[league].items():
                    n_state = len(observations)
                    if not n_state:
                        continue
                    mean_state = sum(observations) / n_state
                    variance_state = (
                        sum((value - mean_state) ** 2 for value in observations)
                        / max(1, n_state - 1)
                        if n_state > 1 else mean_state
                    )
                    state_splits[state_name] = {
                        "n": n_state,
                        "mean": mean_state,
                        "variance": variance_state,
                    }
                result[market]["game_state_splits"] = state_splits
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




def walk_forward_market_validation(
    rows, *, min_history: int = 30, min_n: int = 500,
    bootstrap_samples: int = 250,
) -> dict:
    """Chronological corners/cards probability tests versus prior league-rate baselines.

    Same-day fixtures are forecast before any same-day result is ingested. Only
    observed HC/AC and HY/AY statistics are used. These tests validate forecast
    quality, not bookmaker edge; current prices still need a coherent same-book line.
    """
    from model.walk_forward import (
        _parse_date, _binary_metrics, _binary_paired_differences,
        totals_market_is_validated,
    )

    prepared = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        stamp = _parse_date(row.get("Date") or row.get("date"))
        home = str(row.get("HomeTeam") or row.get("home_team") or "").strip()
        away = str(row.get("AwayTeam") or row.get("away_team") or "").strip()
        league = str(row.get("Div") or row.get("league") or "G")
        if stamp and home and away:
            prepared.append((stamp, league, home, away, row))
    prepared.sort(key=lambda item: item[0])

    lines = {
        "corners": (7.5, 8.5, 9.5, 10.5, 11.5),
        "cards": (2.5, 3.5, 4.5, 5.5, 6.5),
    }
    model = HistoricalMatchStats()
    candidate_records = {
        market: {line: [] for line in market_lines}
        for market, market_lines in lines.items()
    }
    baseline_records = {
        market: {line: [] for line in market_lines}
        for market, market_lines in lines.items()
    }

    cursor = 0
    while cursor < len(prepared):
        stamp = prepared[cursor][0]
        group = []
        while cursor < len(prepared) and prepared[cursor][0] == stamp:
            group.append(prepared[cursor])
            cursor += 1

        pending_rows = []
        for _, league, home, away, row in group:
            forecasts = model.predict(home, away, league)
            for market, market_lines in lines.items():
                actual = None
                if market == "corners":
                    hc, ac = _num(row.get("HC")), _num(row.get("AC"))
                    if hc is not None and ac is not None:
                        actual = hc + ac
                else:
                    hy, ay = _num(row.get("HY")), _num(row.get("AY"))
                    hr, ar = _num(row.get("HR")), _num(row.get("AR"))
                    if all(value is not None for value in (hy, ay, hr, ar)):
                        actual = hy + ay + hr + ar
                if actual is not None and not float(actual).is_integer():
                    actual = None
                forecast = forecasts.get(market) or {}
                league_rows = model.league[league][market]
                if (
                    actual is None
                    or len(league_rows) < min_history
                    or forecast.get("history_matches", 0) < min_history
                    or forecast.get("home_team_games", 0) < model.min_team_games
                    or forecast.get("away_team_games", 0) < model.min_team_games
                ):
                    continue
                baseline_mean = sum(home_total + away_total for home_total, away_total in league_rows) / len(league_rows)
                candidate_mean = float(forecast.get("total") or 0.0)
                for line in market_lines:
                    candidate_over = count_over_probability(
                        candidate_mean, line, forecast.get("total_variance")
                    )
                    baseline_variance = (
                        sum(
                            (float(x[0]) + float(x[1]) - baseline_mean) ** 2
                            for x in league_rows
                        ) / max(1, len(league_rows) - 1)
                        if len(league_rows) > 1 else baseline_mean
                    )
                    baseline_over = count_over_probability(
                        baseline_mean, line, baseline_variance
                    )
                    if candidate_over is None or baseline_over is None:
                        continue
                    outcome = "over" if actual > line else "under"
                    candidate_records[market][line].append(
                        ({"over": candidate_over, "under": 1.0 - candidate_over}, outcome)
                    )
                    baseline_records[market][line].append(
                        ({"over": baseline_over, "under": 1.0 - baseline_over}, outcome)
                    )
            pending_rows.append((row, league))
        # Update only after every fixture for the same date has been forecast.
        for row, league in pending_rows:
            model.learn_step(row, league)

    report = {}
    for market, market_lines in lines.items():
        report[market] = {}
        for line in market_lines:
            candidate = candidate_records[market][line]
            baseline = baseline_records[market][line]
            paired = _binary_paired_differences(
                candidate, baseline,
                bootstrap_samples=bootstrap_samples,
                seed=50100 + int(line * 10) + (0 if market == "corners" else 1000),
            )
            report[market][str(line)] = {
                "line": line,
                "n": len(candidate),
                "candidate": _binary_metrics(
                    candidate, bootstrap_samples=bootstrap_samples,
                    seed=51100 + int(line * 10),
                ),
                "baseline": _binary_metrics(
                    baseline, bootstrap_samples=bootstrap_samples,
                    seed=52100 + int(line * 10),
                ),
                "candidate_minus_baseline": paired,
                "validation_passed": totals_market_is_validated(paired, min_n=min_n),
            }
    return {
        "n_rows": len(prepared),
        "min_history": int(min_history),
        "min_n": int(min_n),
        "markets": report,
        "interpretation": (
            "Validation compares team-specific Poisson forecasts against a prior "
            "league-average Poisson baseline on identical chronological fixtures. "
            "It does not establish positive betting EV; a current exact-market price "
            "from one bookmaker is still required."
        ),
    }


def walk_forward_evaluate(rows, league="G", min_history=30):
    """Evaluate totals chronologically; each forecast sees only earlier rows."""
    model = HistoricalMatchStats()
    metrics = {
        market: {"n": 0, "absolute_error_sum": 0.0, "mae": None}
        for market in _FIELDS
    }
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        home = str(row.get("HomeTeam") or "").strip()
        away = str(row.get("AwayTeam") or "").strip()
        if home and away:
            forecasts = model.predict(home, away, league)
            actuals = {}
            hc, ac = _num(row.get("HC")), _num(row.get("AC"))
            if (
                hc is not None and ac is not None
                and hc.is_integer() and ac.is_integer()
            ):
                actuals["corners"] = hc + ac
            hy, ay = _num(row.get("HY")), _num(row.get("AY"))
            hr, ar = _num(row.get("HR")), _num(row.get("AR"))
            if (
                all(value is not None for value in (hy, ay, hr, ar))
                and all(value.is_integer() for value in (hy, ay, hr, ar))
            ):
                actuals["cards"] = hy + ay + hr + ar
            for market, actual in actuals.items():
                forecast = forecasts.get(market)
                if forecast and forecast.get("history_matches", 0) >= min_history:
                    metrics[market]["n"] += 1
                    metrics[market]["absolute_error_sum"] += abs(
                        forecast["total"] - actual
                    )
        model.learn_step(row, league)
    for result in metrics.values():
        n = result["n"]
        result["mae"] = result["absolute_error_sum"] / n if n else None
    return metrics
