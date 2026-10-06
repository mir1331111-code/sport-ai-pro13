"""betting/adaptive.py — Adaptive priority engine.

Только приоритизация кандидатов. Не меняет строгие BET/WATCH/SKIP-фильтры.
Использует только уже завершённые priced decision snapshots и только данные,
которые предшествуют дате предстоящего матча.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from datetime import datetime


MIN_ACTION_N = 20
PRIOR_MIN = 0.90
PRIOR_MAX = 1.10
SHRINK_N = 20
DECAY_HALF_LIFE_DAYS = 60.0
# Минимальная effective sample size после time-decay.
MIN_EFFECTIVE_N = 8.0


def _dt(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def build_profiles(snapshots):
    """Индексирует completed priced snapshots по (market, league) и fallback-сегментам."""
    groups = defaultdict(list)
    for row in snapshots or []:
        if not isinstance(row, dict):
            continue
        if str(row.get("result_status") or "").lower() not in ("won", "lost"):
            continue
        try:
            odd = float(row.get("market_odd") or 0.0)
            if odd <= 1.01:
                continue
            pnl = odd - 1.0 if str(row.get("result_status")).lower() == "won" else -1.0
        except (TypeError, ValueError):
            continue
        dt = _dt(row.get("date_iso"))
        if dt is None:
            continue
        market = str(row.get("market") or "").strip() or "—"
        league = str(row.get("league") or "").strip() or "—"
        groups[("market", market)].append((dt, pnl))
        groups[("league", league)].append((dt, pnl))
        groups[("market_league", market, league)].append((dt, pnl))

    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def _segment_stats(rows, before_dt, decay_days=None, shrink_n=None):
    if not rows or before_dt is None:
        return None
    dates = [x[0] for x in rows]
    cut = bisect_left(dates, before_dt)
    if cut < MIN_ACTION_N:
        return None
    usable = rows[:cut]
    n = len(usable)
    decay_days = float(decay_days or DECAY_HALF_LIFE_DAYS)
    shrink_n = float(shrink_n or SHRINK_N)
    # Exponential decay: half-life задаётся параметром. Старые исходы не исчезают,
    # но их влияние постепенно уменьшается.
    import math
    weighted = []
    for dt, pnl_i in usable:
        age_days = max(0.0, (before_dt - dt).total_seconds() / 86400.0)
        w = math.exp(-math.log(2.0) * age_days / decay_days)
        weighted.append((w, pnl_i))
    weight_sum = sum(w for w, _ in weighted)
    pnl = sum(w * p for w, p in weighted)
    roi = pnl / weight_sum if weight_sum > 0 else 0.0
    # Небольшое сглаживание к нейтральному ROI=0, чтобы decay не создавал
    # агрессивный factor на малой эффективной выборке.
    prior = shrink_n
    adj_roi = (pnl + 0.0 * prior) / (weight_sum + prior)
    recent = usable[max(0, n // 2):]
    recent_weight = []
    for dt, pnl_i in recent:
        age_days = max(0.0, (before_dt - dt).total_seconds() / 86400.0)
        w = math.exp(-math.log(2.0) * age_days / decay_days)
        recent_weight.append((w, pnl_i))
    rw = sum(w for w, _ in recent_weight)
    recent_roi = sum(w * p for w, p in recent_weight) / rw if rw > 0 else 0.0
    return {"n": n, "effective_n": weight_sum, "roi": roi, "adj_roi": adj_roi, "recent_roi": recent_roi}


REGIME_MIN_N = 30
REGIME_RECENT_N = 20


def _factor_from_stats(stats, wf_ok, regime):
    """Единый расчёт factor для scanner/monitor/attribution."""
    if not stats or not wf_ok:
        return 1.0
    if float(stats.get("effective_n") or 0.0) < MIN_EFFECTIVE_N:
        return 1.0
    adj = float(stats.get("adj_roi") or 0.0)
    recent = float(stats.get("recent_roi") or 0.0)
    factor = 1.0
    if adj >= 0.05 and recent >= 0.0:
        factor = 1.10
    elif adj >= 0.025 and recent >= 0.0:
        factor = 1.05
    elif adj <= -0.05 and recent <= 0.0:
        factor = 0.90
    elif adj <= -0.025 and recent <= 0.0:
        factor = 0.95
    if regime == "COOLING" and factor > 1.0:
        factor = 1.0
    elif regime == "COOLING" and factor == 1.0:
        factor = 0.95
    elif regime == "RECOVERY" and factor < 1.0:
        factor = 1.0
    return max(PRIOR_MIN, min(PRIOR_MAX, factor))


def _regime(rows, before_dt, recent_n=None):
    """Сравнивает свежий режим с историческим baseline до даты матча."""
    if not rows or before_dt is None:
        return "UNKNOWN", 0.0, "Regime: нет данных"
    dates = [x[0] for x in rows]
    cut = bisect_left(dates, before_dt)
    usable = rows[:cut]
    if len(usable) < REGIME_MIN_N:
        return "UNKNOWN", 0.0, f"Regime: N={len(usable)} < {REGIME_MIN_N}"
    recent_n = int(recent_n or REGIME_RECENT_N)
    recent = usable[-recent_n:]
    baseline = usable[:-recent_n]
    recent_roi = sum(x[1] for x in recent) / len(recent)
    base_roi = sum(x[1] for x in baseline) / len(baseline)
    delta = recent_roi - base_roi
    if recent_roi <= -0.05 and delta <= -0.05:
        return "COOLING", delta, f"Regime COOLING · recent {recent_roi:.1%} vs base {base_roi:.1%}"
    if recent_roi >= 0.05 and delta >= 0.05:
        return "RECOVERY", delta, f"Regime RECOVERY · recent {recent_roi:.1%} vs base {base_roi:.1%}"
    return "NORMAL", delta, f"Regime NORMAL · recent {recent_roi:.1%} vs base {base_roi:.1%}"

def priority(profile, card):
    """Возвращает (factor, reason). Factor ограничен ±10% и включается при N>=20."""
    if not profile or not isinstance(card, dict):
        return 1.0, "Adaptive: недостаточно истории"
    before = _dt(card.get("date_iso"))
    if before is None:
        return 1.0, "Adaptive: нет даты"
    verdict = card.get("verdict") or {}
    market = str(verdict.get("market") or verdict.get("market_type") or "").strip() or "—"
    league = str(card.get("league") or card.get("div") or "").strip() or "—"
    choices = [
        (("market_league", market, league), "рынок+лига"),
        (("market", market), "рынок"),
        (("league", league), "лига"),
    ]
    chosen = None
    for key, label in choices:
        rows = profile.get(key, [])
        stats = _segment_stats(rows, before)
        if stats is not None:
            wf_ok, wf_reason = _walk_forward_gate(rows, before)
            chosen = (stats, label, rows, wf_ok, wf_reason)
            break
    if chosen is None:
        return 1.0, "Adaptive: N<20 — нейтрально"
    stats, label, rows, wf_ok, wf_reason = chosen
    if not wf_ok:
        return 1.0, f"Adaptive: {label} · {wf_reason} · нейтрально"
    regime, _, regime_reason = _regime(rows, before)
    factor = _factor_from_stats(stats, wf_ok, regime)
    if float(stats.get("effective_n") or 0.0) < MIN_EFFECTIVE_N:
        return 1.0, (
            f"Adaptive: {label} · effective N={stats.get('effective_n', 0):.1f} "
            f"< {MIN_EFFECTIVE_N:.0f} · нейтрально"
        )
    sign = "↑" if factor > 1 else "↓" if factor < 1 else "→"
    return factor, (
        f"Adaptive {sign} {factor:.2f} · {label} · N={stats['n']} · "
        f"eff.N={stats['effective_n']:.1f} · Adj ROI {stats['adj_roi']:.1%} · "
        f"recent {stats['recent_roi']:.1%} · decay {DECAY_HALF_LIFE_DAYS:.0f}d · "
        f"{regime_reason} · {wf_reason}"
    )



# Walk-forward gate: минимум 12 исторических наблюдений в OOS-половине
# и положительный/не ухудшившийся результат относительно train.
WF_MIN_OOS = 12
WF_MIN_TRAIN = 12


def _walk_forward_gate(rows, before_dt):
    if not rows or before_dt is None:
        return False, "WF: нет истории"
    dates = [x[0] for x in rows]
    cut = bisect_left(dates, before_dt)
    usable = rows[:cut]
    n = len(usable)
    if n < WF_MIN_TRAIN + WF_MIN_OOS:
        return False, f"WF: N={n} < {WF_MIN_TRAIN + WF_MIN_OOS}"
    split = max(WF_MIN_TRAIN, n * 7 // 10)
    train = usable[:split]
    test = usable[split:]
    if len(test) < WF_MIN_OOS:
        return False, "WF: мало OOS"
    train_roi = sum(x[1] for x in train) / len(train)
    test_roi = sum(x[1] for x in test) / len(test)
    # Не требуем высокой доходности: OOS должен хотя бы не разрушать train.
    stable = test_roi >= min(0.0, train_roi - 0.05)
    return stable, f"WF train {train_roi:.1%} · OOS {test_roi:.1%} · N={len(test)}"


def _selection_lift(items, factor_fn, top_n=3):
    """Top-N daily lift versus neutral on one candidate pool."""
    by_day = defaultdict(list)
    for item in items:
        by_day[item["date"]].append(item)
    daily = []
    for day_items in by_day.values():
        if len(day_items) < top_n:
            continue
        adaptive = sorted(
            day_items,
            key=lambda x: (factor_fn(x), x["neutral"]),
            reverse=True,
        )[:top_n]
        neutral = sorted(
            day_items,
            key=lambda x: (x["neutral"], factor_fn(x)),
            reverse=True,
        )[:top_n]
        ar = sum(x["pnl"] for x in adaptive) / top_n
        nr = sum(x["pnl"] for x in neutral) / top_n
        daily.append(ar - nr)
    return (sum(daily) / len(daily) if daily else 0.0), len(daily)


def _selection_lift_stats(items, factor_fn, top_n=3):
    """Daily paired lift with a bootstrap-ready distribution."""
    by_day = defaultdict(list)
    for item in items:
        by_day[item["date"]].append(item)
    daily = []
    for day_items in by_day.values():
        if len(day_items) < top_n:
            continue
        adaptive = sorted(
            day_items,
            key=lambda x: (factor_fn(x), x["neutral"]),
            reverse=True,
        )[:top_n]
        neutral = sorted(
            day_items,
            key=lambda x: (x["neutral"], factor_fn(x)),
            reverse=True,
        )[:top_n]
        ar = sum(x["pnl"] for x in adaptive) / top_n
        nr = sum(x["pnl"] for x in neutral) / top_n
        daily.append(ar - nr)
    return (sum(daily) / len(daily) if daily else 0.0), daily


def _bootstrap_ci(values, reps=3000, seed=2026):
    """Research-only percentile bootstrap over independent daily paired lifts."""
    if not values:
        return 0.0, 0.0, 0.0
    if len(values) < 5:
        mean = sum(values) / len(values)
        return mean, mean, 0.0
    import random
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(reps):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    mean = sum(values) / n
    p_positive = sum(1 for x in means if x > 0) / len(means)
    return lo, hi, p_positive


def _sign_flip_test(values, reps=5000, seed=2027):
    """Research-only paired sign-flip test for mean daily lift."""
    if len(values) < 5:
        return {"p_value": 1.0, "n_days": len(values)}
    import random
    rng = random.Random(seed)
    observed = sum(values) / len(values)
    extreme = 0
    n = len(values)
    for _ in range(reps):
        total = 0.0
        for value in values:
            total += value if rng.random() >= 0.5 else -value
        if total / n >= observed:
            extreme += 1
    return {
        "p_value": (extreme + 1) / (reps + 1),
        "n_days": n,
    }


def _candidate_from_row(r):
    try:
        odd = float(r.get("market_odd") or 0.0)
        model_prob = float(r.get("model_prob") or 0.0)
        edge = float(r.get("edge") or 0.0)
        ev = float(r.get("ev") or 0.0)
    except (TypeError, ValueError):
        return None
    dt = _dt(r.get("date_iso"))
    status = str(r.get("result_status") or "").lower()
    if dt is None or odd <= 1.01 or status not in ("won", "lost"):
        return None
    if model_prob <= 0 or (edge <= 0 and ev <= 0):
        return None
    return {
        "dt": dt,
        "date": dt.date().isoformat(),
        "market": str(r.get("market") or "—").strip() or "—",
        "league": str(r.get("league") or "—").strip() or "—",
        "neutral": (
            0.40 * model_prob
            + 0.25 * max(0.0, ev)
            + 0.20 * max(0.0, edge)
        ),
        "pnl": odd - 1.0 if status == "won" else -1.0,
    }


def _priority_with_params(profile, card, decay_days, shrink_n, regime_recent):
    """Research-only parameterized Adaptive priority."""
    if not profile:
        return 1.0
    before = card.get("dt")
    market = str(card.get("market") or "—")
    league = str(card.get("league") or "—")
    choices = [
        ("market_league", market, league),
        ("market", market),
        ("league", league),
    ]
    for key in choices:
        rows = profile.get(key, [])
        stats = _segment_stats(rows, before, decay_days, shrink_n)
        if stats is None:
            continue
        wf_ok, _ = _walk_forward_gate(rows, before)
        if not wf_ok:
            continue
        regime, _, _ = _regime(rows, before, regime_recent)
        return _factor_from_stats(stats, wf_ok, regime)
    return 1.0


def walk_forward_tune(snapshots, min_train=30, min_test=12):
    """Research-only selection-aware rolling tuner; never changes live constants."""
    import itertools

    candidates = []
    for r in snapshots or []:
        item = _candidate_from_row(r)
        if item is not None:
            candidates.append(item)
    candidates.sort(key=lambda x: x["dt"])
    if len(candidates) < min_train + min_test:
        return {"status": "INSUFFICIENT", "n": len(candidates)}

    configs = list(
        itertools.product((45.0, 60.0, 90.0), (15, 20, 30), (10, 20, 30))
    )
    fold_size = max(min_test, len(candidates) // 5)
    folds = []
    cursor = min_train

    while cursor + min_test <= len(candidates) and len(folds) < 3:
        train = candidates[:cursor]
        test = candidates[cursor:min(len(candidates), cursor + fold_size)]
        if len(train) < min_train or len(test) < min_test:
            break

        split = max(20, int(len(train) * 0.70))
        if len(train) - split < 6:
            break
        inner_train = train[:split]
        validation = train[split:]
        profile = defaultdict(list)
        for x in inner_train:
            profile[("market", x["market"])].append((x["dt"], x["pnl"]))
            profile[("league", x["league"])].append((x["dt"], x["pnl"]))
            profile[("market_league", x["market"], x["league"])].append(
                (x["dt"], x["pnl"])
            )
        for key in profile:
            profile[key].sort(key=lambda z: z[0])

        scored = []
        for decay, shrink_n, regime_recent in configs:
            def factor_fn(x, d=decay, s=shrink_n, rr=regime_recent):
                return _priority_with_params(profile, x, d, s, rr)
            validation_lifts = {}
            validation_days = {}
            for top_n in (1, 3, 5):
                lift, daily = _selection_lift_stats(validation, factor_fn, top_n=top_n)
                validation_lifts[top_n] = lift
                validation_days[top_n] = len(daily)
            robust_score = sum(validation_lifts.values()) / 3.0
            scored.append((
                robust_score,
                validation_lifts.get(3, 0.0),
                validation_days.get(3, 0),
                decay,
                shrink_n,
                regime_recent,
                validation_lifts,
                validation_days,
            ))

        scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        (
            best_lift,
            best_top3,
            val_days,
            decay,
            shrink_n,
            regime_recent,
            validation_lifts,
            validation_days,
        ) = scored[0]

        full_profile = defaultdict(list)
        for x in train:
            full_profile[("market", x["market"])].append((x["dt"], x["pnl"]))
            full_profile[("league", x["league"])].append((x["dt"], x["pnl"]))
            full_profile[("market_league", x["market"], x["league"])].append(
                (x["dt"], x["pnl"])
            )
        for key in full_profile:
            full_profile[key].sort(key=lambda z: z[0])

        def oos_factor(x, d=decay, s=shrink_n, rr=regime_recent):
            return _priority_with_params(full_profile, x, d, s, rr)

        oos_lifts = {}
        oos_daily = {}
        for top_n in (1, 3, 5):
            lift, daily = _selection_lift_stats(test, oos_factor, top_n=top_n)
            oos_lifts[top_n] = lift
            oos_daily[top_n] = daily
        oos_lift = oos_lifts[3]
        oos_days = len(oos_daily[3])
        folds.append({
            "train_n": len(train),
            "oos_n": len(test),
            "validation_days": val_days,
            "validation_lifts": validation_lifts,
            "validation_days_by_topn": validation_days,
            "oos_days": oos_days,
            "oos_lifts": oos_lifts,
            "oos_days_by_topn": {k: len(v) for k, v in oos_daily.items()},
            "decay": decay,
            "shrink_n": shrink_n,
            "regime_recent": regime_recent,
            "train_selection_lift": best_lift,
            "train_top3_lift": best_top3,
            "oos_selection_lift": oos_lift,
            "robust_oos_lift": sum(oos_lifts.values()) / 3.0,
            "oos_roi": sum(x["pnl"] for x in test) / len(test) if test else 0.0,
            "baseline_roi": sum(x["pnl"] for x in train) / len(train) if train else 0.0,
            "oos_delta": oos_lift,
            "oos_daily_lifts": oos_daily,
        })
        cursor += fold_size

    if not folds:
        return {"status": "INSUFFICIENT", "n": len(candidates)}

    from collections import Counter
    cfg_counts = Counter(
        (x["decay"], x["shrink_n"], x["regime_recent"]) for x in folds
    )
    best_cfg = cfg_counts.most_common(1)[0][0]
    neighborhood = [
        x["oos_delta"] for x in folds
        if abs(x["decay"] - best_cfg[0]) <= 15.0
        and abs(x["shrink_n"] - best_cfg[1]) <= 5
        and abs(x["regime_recent"] - best_cfg[2]) <= 10
    ]
    neighborhood_mean = sum(neighborhood) / len(neighborhood) if neighborhood else 0.0
    neighborhood_positive = (
        sum(1 for x in neighborhood if x > 0) / len(neighborhood)
        if neighborhood else 0.0
    )
    stable_config, stable_count = cfg_counts.most_common(1)[0]
    mean_delta = sum(x["oos_delta"] for x in folds) / len(folds)
    mean_topn = {
        top_n: sum(x["oos_lifts"][top_n] for x in folds) / len(folds)
        for top_n in (1, 3, 5)
    }
    robust_mean = sum(mean_topn.values()) / 3.0
    pooled_daily = {
        top_n: [
            lift
            for fold in folds
            for lift in fold["oos_daily_lifts"].get(top_n, [])
        ]
        for top_n in (1, 3, 5)
    }
    ci_by_topn = {}
    for top_n in (1, 3, 5):
        lo, hi, p_pos = _bootstrap_ci(
            pooled_daily[top_n],
            reps=3000,
            seed=2026 + top_n,
        )
        ci_by_topn[top_n] = {
            "low": lo,
            "high": hi,
            "p_positive": p_pos,
            "n_days": len(pooled_daily[top_n]),
        }
    permutation_by_topn = {
        top_n: _sign_flip_test(
            pooled_daily[top_n],
            reps=5000,
            seed=2027 + top_n,
        )
        for top_n in (1, 3, 5)
    }
    robust_ci_low = sum(ci_by_topn[k]["low"] for k in (1, 3, 5)) / 3.0
    positive_folds = sum(1 for x in folds if x["oos_delta"] > 0)
    positive_topn = sum(1 for k in (1, 3, 5) if mean_topn[k] > 0)
    stable_gate = bool(
        len(folds) >= 3
        and positive_folds >= 2
        and positive_topn >= 2
        and robust_mean > 0.0
        and robust_ci_low >= 0.0
        and permutation_by_topn.get(3, {}).get("p_value", 1.0) <= 0.10
        and stable_count / len(folds) >= 0.67
        and neighborhood_mean >= 0.0
        and neighborhood_positive >= 0.50
    )
    return {
        "status": "OK",
        "n": len(candidates),
        "folds": folds,
        "fold_n": len(folds),
        "decay": stable_config[0],
        "shrink_n": stable_config[1],
        "regime_recent": stable_config[2],
        "config_stability": stable_count / len(folds),
        "mean_oos_delta": mean_delta,
        "mean_topn": mean_topn,
        "robust_mean_oos_lift": robust_mean,
        "ci_by_topn": ci_by_topn,
        "robust_ci_low": robust_ci_low,
        "permutation_by_topn": permutation_by_topn,
        "positive_topn": positive_topn,
        "positive_folds": positive_folds,
        "stability_gate": stable_gate,
        "neighborhood_mean": neighborhood_mean,
        "neighborhood_positive": neighborhood_positive,
        "neighborhood_n": len(neighborhood),
        "selection_aware": True,
    }
