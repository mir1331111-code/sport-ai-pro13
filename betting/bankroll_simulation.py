"""Monte Carlo bankroll risk analysis for fixed fractional stakes."""
from __future__ import annotations

import math
import random


def simulate_bankroll(returns, initial_bankroll=1000.0, stake_fraction=0.01,
                      n_simulations=5000, seed=2026):
    """Bootstrap historical per-bet net returns with fixed fractional staking.

    returns are net returns per unit stake (e.g. +0.8 for a winning 1.8-odds bet,
    -1.0 for a loss, 0 for a push). This is a risk scenario, not a forecast.
    """
    values = [float(x) for x in returns]
    if not values or any(not math.isfinite(x) or x < -1 for x in values):
        raise ValueError("returns must be finite net returns >= -1")
    if not math.isfinite(initial_bankroll) or initial_bankroll <= 0:
        raise ValueError("initial bankroll must be positive")
    if not 0 < stake_fraction <= 0.05:
        raise ValueError("stake_fraction must be in (0, 0.05]")
    if n_simulations < 100:
        raise ValueError("n_simulations must be >= 100")
    rng = random.Random(seed)
    final_values, max_drawdowns, ruin = [], [], 0
    horizon = len(values)
    for _ in range(n_simulations):
        bank = float(initial_bankroll)
        peak = bank
        max_dd = 0.0
        ruined = False
        for _ in range(horizon):
            net = values[rng.randrange(horizon)]
            bank *= max(0.0, 1.0 + stake_fraction * net)
            peak = max(peak, bank)
            if peak > 0:
                max_dd = max(max_dd, (peak - bank) / peak)
            if bank <= initial_bankroll * 0.01:
                ruined = True
                break
        final_values.append(bank)
        max_drawdowns.append(max_dd)
        ruin += int(ruined)
    final_values.sort()
    max_drawdowns.sort()
    q05 = final_values[max(0, int(0.05 * n_simulations)-1)]
    median = final_values[n_simulations // 2]
    dd95 = max_drawdowns[min(n_simulations-1, int(0.95*n_simulations))]
    return {"simulations": n_simulations, "bets_per_path": horizon,
            "initial_bankroll": float(initial_bankroll), "stake_fraction": stake_fraction,
            "final_bankroll_p05": q05, "final_bankroll_median": median,
            "max_drawdown_p95": dd95, "ruin_probability": ruin / n_simulations,
            "warning": "Historical-return bootstrap assumes exchangeability and does not predict future profit."}
