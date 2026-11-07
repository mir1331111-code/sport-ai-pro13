"""Monte Carlo bankroll diagnostics from settled market-ledger picks.

The simulation resamples historical, settled fixture recommendations with
replacement. It is a sensitivity diagnostic, not a forecast of future profit
and not the user's realized portfolio history.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Iterable

VALID_STATUSES = {"won", "lost", "push", "void"}
VALID_MODES = {"flat_fraction", "fractional_kelly"}


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return number if math.isfinite(number) else default


def _date_key(row: dict) -> str:
    raw = str(row.get("kickoff_at") or row.get("forecast_at") or "")
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except (TypeError, ValueError):
        return raw[:10] or "unknown"


def _prepare_samples(
    rows: Iterable[dict],
    *,
    market: str | None = "1X2",
    model_version_prefix: str = "main-model-",
) -> tuple[list[dict], dict[str, int]]:
    """Keep settled recommendations and at most one selection per fixture."""
    counts = {
        "ledger_rows": 0,
        "settled_rows": 0,
        "eligible_rows": 0,
        "invalid_rows": 0,
        "duplicate_fixture_rows": 0,
    }
    by_fixture: dict[str, list[dict]] = defaultdict(list)
    market_key = str(market or "").upper().strip() or None

    for row in rows or []:
        if not isinstance(row, dict):
            continue
        counts["ledger_rows"] += 1
        if str(row.get("model_version") or "").startswith(model_version_prefix) is False:
            continue
        status = str(row.get("result_status") or "").lower()
        if status not in VALID_STATUSES:
            continue
        counts["settled_rows"] += 1
        if market_key is not None and str(row.get("market") or "").upper() != market_key:
            continue
        probability = _finite(row.get("probability"))
        odds = _finite(row.get("entry_odds"))
        stake = _finite(row.get("unit_stake"))
        if (
            probability is None or not 0.0 < probability < 1.0
            or odds is None or odds <= 1.0
            or stake is None or stake <= 0.0
        ):
            counts["invalid_rows"] += 1
            continue
        fixture_id = str(row.get("fixture_id") or "").strip()
        if not fixture_id:
            counts["invalid_rows"] += 1
            continue

        counts["eligible_rows"] += 1
        item = dict(row)
        item.update({
            "probability": probability,
            "entry_odds": odds,
            "unit_stake": stake,
            "result_status": status,
            "net_return_per_stake": (
                odds - 1.0 if status == "won" else
                -1.0 if status == "lost" else
                0.0
            ),
            "expected_value": probability * odds - 1.0,
            "date": _date_key(row),
        })
        by_fixture[fixture_id].append(item)

    samples = []
    for fixture_id, candidates in by_fixture.items():
        # The ledger may contain several approved market families / lines for a
        # fixture. A single-bankroll path must not stake all of them as if they
        # were independent. Pick the highest pre-match EV within the selected
        # family; this uses only information recorded before kickoff.
        candidates.sort(key=lambda x: (
            float(x["expected_value"]),
            float(x["forecast_at"] and 1 or 0),
            str(x.get("forecast_at") or ""),
        ), reverse=True)
        chosen = candidates[0]
        chosen["fixture_id"] = fixture_id
        samples.append(chosen)
        counts["duplicate_fixture_rows"] += max(0, len(candidates) - 1)

    samples.sort(key=lambda item: (
        str(item.get("kickoff_at") or ""),
        str(item.get("fixture_id") or ""),
    ))
    return samples, counts


def _stake_fraction(
    sample: dict,
    *,
    mode: str,
    flat_fraction: float,
    kelly_fraction: float,
    kelly_cap: float,
) -> float:
    if mode == "flat_fraction":
        return flat_fraction
    probability = float(sample["probability"])
    odds = float(sample["entry_odds"])
    net_odds = odds - 1.0
    full_kelly = (odds * probability - 1.0) / net_odds
    if full_kelly <= 0.0:
        return 0.0
    return min(kelly_cap, full_kelly * kelly_fraction)


def simulate_bankroll(
    rows: Iterable[dict],
    *,
    initial_bank: float = 10_000.0,
    simulations: int = 5_000,
    bets_per_path: int | None = None,
    market: str | None = "1X2",
    stake_mode: str = "flat_fraction",
    flat_fraction: float = 0.01,
    kelly_fraction: float = 0.25,
    kelly_cap: float = 0.02,
    model_version_prefix: str = "main-model-",
    seed: int = 20261107,
) -> dict[str, Any]:
    """Bootstrap settled fixture returns and simulate fractional-bankroll paths.

    Each path resamples one settled recommendation per fixture with replacement.
    Flat stake is a fixed fraction of current bankroll. Fractional Kelly sizes
    each bet using its archived pre-match probability/odds with a hard cap.
    Push/void records return zero profit and consume one observation.
    """
    initial_bank = _finite(initial_bank)
    if initial_bank is None or initial_bank <= 0:
        raise ValueError("initial_bank must be finite and > 0")
    if isinstance(simulations, bool) or not isinstance(simulations, int) or not 100 <= simulations <= 100_000:
        raise ValueError("simulations must be an integer between 100 and 100000")
    if stake_mode not in VALID_MODES:
        raise ValueError("stake_mode must be flat_fraction or fractional_kelly")
    for name, value, upper in (
        ("flat_fraction", flat_fraction, 0.05),
        ("kelly_fraction", kelly_fraction, 1.0),
        ("kelly_cap", kelly_cap, 0.05),
    ):
        if not math.isfinite(float(value)) or float(value) <= 0 or float(value) > upper:
            raise ValueError(f"{name} must be in (0, {upper}]")

    samples, counts = _prepare_samples(
        rows, market=market, model_version_prefix=model_version_prefix,
    )
    sample_n = len(samples)
    if sample_n < 10:
        return {
            "available": False,
            "reason": "Для симуляции нужно не менее 10 уникальных завершённых рекомендаций выбранного рынка.",
            "sample_n": sample_n,
            "counts": counts,
            "market": market or "BEST_PER_FIXTURE",
        }

    horizon = sample_n if bets_per_path is None else bets_per_path
    if isinstance(horizon, bool) or not isinstance(horizon, int) or not 1 <= horizon <= 100_000:
        raise ValueError("bets_per_path must be an integer between 1 and 100000")

    rng = random.Random(int(seed))
    final_banks: list[float] = []
    max_drawdowns: list[float] = []
    ruined = below_start = 0
    for _ in range(simulations):
        bank = initial_bank
        peak = initial_bank
        max_dd = 0.0
        for _bet in range(horizon):
            sample = samples[rng.randrange(sample_n)]
            fraction = _stake_fraction(
                sample,
                mode=stake_mode,
                flat_fraction=float(flat_fraction),
                kelly_fraction=float(kelly_fraction),
                kelly_cap=float(kelly_cap),
            )
            if fraction > 0.0 and bank > 0.0:
                net = float(sample["net_return_per_stake"])
                bank *= max(0.0, 1.0 + fraction * net)
            if bank > peak:
                peak = bank
            if peak > 0:
                max_dd = max(max_dd, (peak - bank) / peak)
            if bank <= 0.0:
                bank = 0.0
                max_dd = 1.0
                break
        final_banks.append(bank)
        max_drawdowns.append(max_dd)
        below_start += int(bank < initial_bank)
        ruined += int(bank <= initial_bank * 0.5)

    final_sorted = sorted(final_banks)
    dd_sorted = sorted(max_drawdowns)

    def percentile(values: list[float], q: float) -> float:
        if not values:
            return 0.0
        position = (len(values) - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    sample_net_returns = [float(row["net_return_per_stake"]) for row in samples]
    wins = sum(row["result_status"] == "won" for row in samples)
    losses = sum(row["result_status"] == "lost" for row in samples)
    pushes = sum(row["result_status"] == "push" for row in samples)
    voids = sum(row["result_status"] == "void" for row in samples)
    return {
        "available": True,
        "market": market or "BEST_PER_FIXTURE",
        "sample_n": sample_n,
        "bets_per_path": horizon,
        "simulations": simulations,
        "stake_mode": stake_mode,
        "initial_bank": float(initial_bank),
        "stake_parameters": {
            "flat_fraction": float(flat_fraction),
            "kelly_fraction": float(kelly_fraction),
            "kelly_cap": float(kelly_cap),
        },
        "empirical_sample": {
            "wins": wins, "losses": losses, "pushes": pushes, "voids": voids,
            "win_rate_decisive": wins / (wins + losses) if wins + losses else None,
            "mean_net_return_per_unit": (
                sum(sample_net_returns) / sample_n if sample_n else None
            ),
        },
        "final_bank": {
            "p05": percentile(final_sorted, 0.05),
            "p25": percentile(final_sorted, 0.25),
            "median": percentile(final_sorted, 0.50),
            "p75": percentile(final_sorted, 0.75),
            "p95": percentile(final_sorted, 0.95),
            "mean": sum(final_sorted) / len(final_sorted),
        },
        "max_drawdown": {
            "p50": percentile(dd_sorted, 0.50),
            "p90": percentile(dd_sorted, 0.90),
            "p95": percentile(dd_sorted, 0.95),
        },
        "probability_below_start": below_start / simulations,
        "probability_50pct_drawdown": ruined / simulations,
        "counts": counts,
        "assumptions": [
            "Исторические закрытые рекомендации пересэмплируются с возвращением.",
            "Из каждого матча берётся не более одной рекомендации выбранного рынка; для дублей выбирается максимальный pre-match EV.",
            "Распределение результатов считается стабильным на горизонте симуляции; смена формы, лиг, сезона и коэффициентов отдельно не моделируется.",
            "Расчёт не учитывает корреляцию между разными матчами, комиссии, лимиты букмекера и проскальзывание.",
            "Это виртуальная симуляция рекомендаций, а не фактическая история банка или гарантия будущего дохода.",
        ],
    }
