"""ui/tabs/stats.py — Статистика по СВОИМ ставкам."""
from __future__ import annotations
from collections import defaultdict

import streamlit as st

from storage import sqlite_store as db
from data.sources import tsdb_match_result, espn_match_result, fdorg_match_result


def _sharpe(returns: list, periods_per_year: int = 252) -> float:
    if len(returns) < 3:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / len(returns)
    sigma = var ** 0.5
    if sigma <= 1e-12:
        return 0.0
    return (mu / sigma) * (periods_per_year ** 0.5)


def _closed_bets(D):
    return [
        b for b in D.get("bets", [])
        if isinstance(b, dict)
        and b.get("status") in ("won", "lost")
    ]


def _bet_pnl(b):
    stake = float(b.get("stake") or 0)
    odds = float(b.get("odds") or 1)
    if b.get("status") == "won":
        return stake * (odds - 1)
    if b.get("status") == "lost":
        return -stake
    return 0.0


def _render_model(D):
    bets = _closed_bets(D)
    calibration = []
    for b in bets:
        try:
            p = float(b.get("prob"))
        except (TypeError, ValueError):
            continue
        if not 0 <= p <= 1:
            continue
        calibration.append((p, 1.0 if b.get("status") == "won" else 0.0))

    if not calibration:
        st.info("Недостаточно закрытых ставок с вероятностью модели для Calibration.")
        return

    import pandas as pd

    rows = []
    for lo in [i / 20 for i in range(20)]:
        hi = lo + 0.05
        bucket = [
            (p, y) for p, y in calibration
            if (lo <= p < hi) or (hi >= 1.0 and lo <= p <= 1.0)
        ]
        if not bucket:
            continue
        n = len(bucket)
        avg_p = sum(p for p, _ in bucket) / n
        actual = sum(y for _, y in bucket) / n
        rows.append({
            "Диапазон P": f"{lo*100:.0f}–{hi*100:.0f}%",
            "N": n,
            "Model P": avg_p,
            "Факт": actual,
            "Ошибка": actual - avg_p,
        })

    brier = sum((p - y) ** 2 for p, y in calibration) / len(calibration)
    abs_cal_error = (
        sum(abs(r["Ошибка"]) * r["N"] for r in rows)
        / len(calibration)
        if rows else 0.0
    )

    st.subheader("🎯 Calibration модели")
    c1, c2, c3 = st.columns(3)
    c1.metric("Brier Score", f"{brier:.3f}")
    c2.metric("Средняя ошибка", f"{abs_cal_error*100:.1f} п.п.")
    c3.metric("Ставок в выборке", len(calibration))

    df = pd.DataFrame(rows)
    if not df.empty:
        chart = df.set_index("Диапазон P")[["Model P", "Факт"]] * 100
        st.line_chart(chart, height=280)
        st.dataframe(
            df.assign(
                **{
                    "Model P": df["Model P"].map(lambda x: f"{x*100:.1f}%"),
                    "Факт": df["Факт"].map(lambda x: f"{x*100:.1f}%"),
                    "Ошибка": df["Ошибка"].map(lambda x: f"{x*100:+.1f} п.п."),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Идеальная калибровка: Model P и Факт близки. "
            "Brier Score ниже — лучше; сравнивать его корректно на одинаковой выборке."
        )


    _render_snapshot_calibration(D)

    _render_walk_forward_thresholds()

    _render_kelly_sensitivity()

    _render_strategy_diagnostics()

    _render_rolling_performance()
    _render_system_scorecard(D)
    _render_adaptive_segments()
    _render_drift_monitor()

    # ==================== QUALITY BUCKETS ====================
    def _num(v):
        try:
            x = float(v)
            return x if x == x else None
        except (TypeError, ValueError):
            return None

    def _edge_value(b):
        edge = _num(b.get("edge"))
        if edge is not None:
            return edge
        p = _num(b.get("prob"))
        o = _num(b.get("odds"))
        if p is not None and o is not None and o > 0:
            return p - (1.0 / o)
        return None

    def _confidence_value(b):
        c = _num(b.get("confidence"))
        if c is None:
            c = _num(b.get("model_confidence"))
        if c is None:
            c = _num(b.get("prob"))
        if c is None:
            return None
        return c / 100.0 if c > 1 else c

    def _bucket_stats(items, label, ranges):
        rows = []
        for lo, hi in ranges:
            bucket = [
                b for b in items
                if _num(b.get("_bucket_value")) is not None
                and lo <= _num(b.get("_bucket_value")) < hi
            ]
            if not bucket:
                continue
            pnl = sum(_bet_pnl(b) for b in bucket)
            turnover = sum(float(b.get("stake") or 0) for b in bucket)
            won = sum(1 for b in bucket if b.get("status") == "won")
            roi = pnl / turnover * 100 if turnover else 0.0
            rows.append({
                label: f"{lo*100:.0f}–{hi*100:.0f}%",
                "N": len(bucket),
                "Win Rate": won / len(bucket) * 100,
                "PnL": pnl,
                "ROI": roi,
            })
        return rows

    edge_items = []
    conf_items = []
    for b in bets:
        edge = _edge_value(b)
        if edge is not None:
            x = dict(b)
            x["_bucket_value"] = edge
            edge_items.append(x)
        conf = _confidence_value(b)
        if conf is not None:
            x = dict(b)
            x["_bucket_value"] = conf
            conf_items.append(x)

    st.subheader("📊 Где модель реально зарабатывает")
    q1, q2 = st.columns(2)
    with q1:
        st.markdown("**ROI по Edge**")
        edge_rows = _bucket_stats(
            edge_items, "Edge",
            [(0.00, 0.03), (0.03, 0.05), (0.05, 0.08), (0.08, 1.01)],
        )
        if edge_rows:
            df_edge = pd.DataFrame(edge_rows)
            st.dataframe(
                df_edge.assign(
                    **{
                        "Win Rate": df_edge["Win Rate"].map(lambda x: f"{x:.1f}%"),
                        "PnL": df_edge["PnL"].map(lambda x: f"{x:+.2f}"),
                        "ROI": df_edge["ROI"].map(lambda x: f"{x:+.1f}%"),
                    }
                ), use_container_width=True, hide_index=True,
            )
        else:
            st.caption("Пока нет закрытых ставок с Edge.")

    with q2:
        st.markdown("**ROI по Confidence**")
        conf_rows = _bucket_stats(
            conf_items, "Confidence",
            [(0.00, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 1.01)],
        )
        if conf_rows:
            df_conf = pd.DataFrame(conf_rows)
            st.dataframe(
                df_conf.assign(
                    **{
                        "Win Rate": df_conf["Win Rate"].map(lambda x: f"{x:.1f}%"),
                        "PnL": df_conf["PnL"].map(lambda x: f"{x:+.2f}"),
                        "ROI": df_conf["ROI"].map(lambda x: f"{x:+.1f}%"),
                    }
                ), use_container_width=True, hide_index=True,
            )


    # ==================== MARKET × EDGE ====================
    market_edge_rows = []
    edge_ranges = [
        (0.00, 0.03, "0–3%"),
        (0.03, 0.05, "3–5%"),
        (0.05, 0.08, "5–8%"),
        (0.08, 1.01, "8%+"),
    ]
    market_edge_groups = defaultdict(list)
    for b in bets:
        edge = _edge_value(b)
        market = str(b.get("market") or "OTHER").strip() or "OTHER"
        if edge is None or edge < 0:
            continue
        for lo, hi, label in edge_ranges:
            if lo <= edge < hi:
                market_edge_groups[(market, label)].append(b)
                break

    for (market, edge_label), items in market_edge_groups.items():
        n = len(items)
        stake = sum(float(b.get("stake") or 0) for b in items)
        pnl = sum(_bet_pnl(b) for b in items)
        won = sum(1 for b in items if b.get("status") == "won")
        market_edge_rows.append({
            "Рынок": market,
            "Edge": edge_label,
            "N": n,
            "Win Rate": won / n * 100 if n else 0.0,
            "PnL": pnl,
            "ROI": pnl / stake * 100 if stake else 0.0,
        })

    st.subheader("🔬 Рынок × Edge")
    st.caption(
        "Помогает понять, в каких конкретных рынках высокий Edge действительно окупается. "
        "Это исследовательская сегментация; малые выборки не являются доказательством преимущества."
    )
    if market_edge_rows:
        market_edge_rows.sort(key=lambda r: (r["N"] >= 10, r["ROI"], r["N"]), reverse=True)
        df_me = pd.DataFrame(market_edge_rows)
        st.dataframe(
            df_me.assign(
                **{
                    "Win Rate": df_me["Win Rate"].map(lambda x: f"{x:.1f}%"),
                    "PnL": df_me["PnL"].map(lambda x: f"{x:+.2f}"),
                    "ROI": df_me["ROI"].map(lambda x: f"{x:+.1f}%"),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        reliable_me = [r for r in market_edge_rows if r["N"] >= 10]
        if reliable_me:
            best_me = max(reliable_me, key=lambda r: r["ROI"])
            st.success(
                f"Лучший сегмент при N≥10: {best_me['Рынок']} · Edge {best_me['Edge']} · "
                f"ROI {best_me['ROI']:+.1f}% · N={best_me['N']}"
            )
        else:
            st.caption("Надёжного сегмента пока нет: нужно минимум 10 закрытых ставок в группе.")
    else:
        st.info("Пока недостаточно закрытых ставок с Edge для сегментации.")


    # ==================== MODEL VERDICT ====================
    def _best_reliable(rows):
        reliable = [r for r in rows if r["N"] >= 10]
        if not reliable:
            return None
        return max(reliable, key=lambda r: r["ROI"])

    best_edge = _best_reliable(edge_rows)
    best_conf = _best_reliable(conf_rows)

    st.subheader("🧠 Где модель сильнее")
    vc1, vc2 = st.columns(2)
    with vc1:
        if best_edge:
            st.metric(
                "Лучший диапазон Edge",
                best_edge["Edge"],
                f"ROI {best_edge['ROI']:+.1f}% · N={best_edge['N']}",
            )
        else:
            st.caption("Для надёжного вывода по Edge нужно ≥10 ставок в диапазоне.")
    with vc2:
        if best_conf:
            st.metric(
                "Лучший диапазон Confidence",
                best_conf["Confidence"],
                f"ROI {best_conf['ROI']:+.1f}% · N={best_conf['N']}",
            )
        else:
            st.caption("Для надёжного вывода по Confidence нужно ≥10 ставок в диапазоне.")

    if edge_items or conf_items:
        st.caption(
            "Edge = P модели − 1/кэф. Для старых ставок без отдельного Confidence "
            "используется P модели. Малые выборки не являются доказательством преимущества."
        )

    ordered = sorted(
        bets,
        key=lambda b: (
            b.get("settled_at")
            or b.get("date_time")
            or b.get("date_iso")
            or ""
        ),
    )
    initial = float(D.get("meta", {}).get("initial_bank", 10000.0) or 10000.0)
    equity = initial
    peak = initial
    max_dd = 0.0
    curve = [{"Шаг": 0, "Банк": equity, "Drawdown": 0.0}]
    for i, b in enumerate(ordered, 1):
        equity += _bet_pnl(b)
        peak = max(peak, equity)
        dd = ((peak - equity) / peak) if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
        curve.append({"Шаг": i, "Банк": equity, "Drawdown": dd * 100})

    st.subheader("📈 Equity Curve")
    ec = pd.DataFrame(curve).set_index("Шаг")
    st.line_chart(ec[["Банк"]], height=280)
    st.caption(
        f"Старт: {initial:.0f} · текущая расчётная equity: {equity:.0f} · "
        f"Max Drawdown: -{max_dd*100:.1f}% · закрытых ставок: {len(ordered)}"
    )

    dd_positive = ec[["Drawdown"]]
    st.subheader("📉 Drawdown")
    st.line_chart(dd_positive, height=220)


def _render_walk_forward_thresholds():
    """Простая walk-forward проверка Edge/EV: train -> следующий test-период."""
    if not db.SQLITE_BOOT_OK:
        return

    snapshots = db.fetch_decision_snapshots(limit=100000)
    samples = []
    for r in snapshots:
        if str(r.get("result_status") or "").lower() not in ("won", "lost"):
            continue
        try:
            date = str(r.get("date_iso") or "")[:10]
            edge = float(r.get("edge"))
            ev = float(r.get("ev"))
            pnl = float(r.get("virtual_pnl"))
            if not date or not all(x == x for x in (edge, ev, pnl)):
                continue
        except (TypeError, ValueError):
            continue
        samples.append({"date": date, "edge": edge, "ev": ev, "pnl": pnl,
                        "won": str(r.get("result_status")).lower() == "won"})

    if len(samples) < 20:
        return

    samples.sort(key=lambda x: x["date"])
    dates = sorted({x["date"] for x in samples})
    if len(dates) < 4:
        return

    thresholds = [0.00, 0.01, 0.02, 0.03, 0.05]
    split = max(2, int(len(dates) * 0.70))
    if split >= len(dates):
        return

    train_dates = set(dates[:split])
    test_dates = set(dates[split:])
    train = [x for x in samples if x["date"] in train_dates]
    test = [x for x in samples if x["date"] in test_dates]

    def score(items, metric, threshold):
        chosen = [x for x in items if x[metric] >= threshold]
        if not chosen:
            return {"N": 0, "WR": 0.0, "PnL": 0.0, "ROI": 0.0}
        pnl = sum(x["pnl"] for x in chosen)
        return {
            "N": len(chosen),
            "WR": sum(x["won"] for x in chosen) / len(chosen) * 100,
            "PnL": pnl,
            "ROI": pnl / len(chosen) * 100,
        }

    st.subheader("🧪 Walk-forward Threshold Test")
    st.caption(
        "70% первых дат используются только для выбора порога; последние 30% — "
        "отдельный test. В текущие правила BET результат не подмешивается."
    )

    import pandas as pd
    rows = []
    for metric, label in (("edge", "Edge"), ("ev", "EV")):
        train_scores = {t: score(train, metric, t) for t in thresholds}
        reliable = [t for t in thresholds if train_scores[t]["N"] >= 10]
        if not reliable:
            continue
        best_t = max(reliable, key=lambda t: train_scores[t]["ROI"])
        for t in thresholds:
            tr, te = train_scores[t], score(test, metric, t)
            rows.append({
                "Метрика": label,
                "Threshold": f"{t*100:.0f}%",
                "Train N": tr["N"],
                "Train ROI": tr["ROI"],
                "Test N": te["N"],
                "Test Win Rate": te["WR"],
                "Test PnL": te["PnL"],
                "Test ROI": te["ROI"],
                "Выбран": "←" if t == best_t else "",
            })

    if not rows:
        st.info("Недостаточно данных для выбора threshold на train-периоде.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(
        df.assign(
            **{
                "Train ROI": df["Train ROI"].map(lambda x: f"{x:+.1f}%"),
                "Test Win Rate": df["Test Win Rate"].map(lambda x: f"{x:.1f}%"),
                "Test PnL": df["Test PnL"].map(lambda x: f"{x:+.2f}"),
                "Test ROI": df["Test ROI"].map(lambda x: f"{x:+.1f}%"),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    for label in ("Edge", "EV"):
        part = [r for r in rows if r["Метрика"] == label and r["Выбран"] == "←"]
        if not part:
            continue
        r = part[0]
        st.metric(
            f"Out-of-sample {label}",
            r["Threshold"],
            f"Test ROI {r['Test ROI']:+.1f}% · N={r['Test N']}",
        )

    st.caption(
        f"Train: {dates[0]} → {dates[split-1]} · Test: {dates[split]} → {dates[-1]}. "
        "Это один временной split, а не доказательство устойчивого преимущества."
    )


def _render_system_scorecard(D):
    """Единый health-check системы: модель, value, форма и риск."""
    if not db.SQLITE_BOOT_OK:
        return

    snapshots = db.fetch_decision_snapshots(limit=100000)
    closed = [
        r for r in snapshots
        if str(r.get("result_status") or "").lower() in ("won", "lost")
    ]

    def n(v):
        try:
            x = float(v)
            return x if x == x else None
        except (TypeError, ValueError):
            return None

    def score_status(ok, warn):
        return "🟢 OK" if ok else ("🟡 MONITOR" if warn else "🔴 ATTENTION")

    # Model: use model-only accuracy when available, otherwise closed snapshots.
    model_rows = [r for r in closed if n(r.get("model_prob")) is not None]
    model_only_rows = [
        r for r in model_rows
        if str(r.get("decision") or "").upper() == "MODEL_ONLY"
    ]
    calibration_rows = model_only_rows or model_rows
    model_brier = None
    if calibration_rows:
        model_brier = sum(
            (n(r.get("model_prob")) - (1.0 if str(r.get("result_status")).lower() == "won" else 0.0)) ** 2
            for r in calibration_rows
            if n(r.get("model_prob")) is not None
        ) / len([r for r in calibration_rows if n(r.get("model_prob")) is not None])

    # Value: only signals with an actual market price.
    priced = []
    for r in closed:
        odd = n(r.get("market_odd"))
        if odd is None or odd <= 1.01:
            continue
        pnl = odd - 1.0 if str(r.get("result_status")).lower() == "won" else -1.0
        priced.append((r, pnl))
    value_roi = sum(p for _, p in priced) / len(priced) * 100 if priced else None
    avg_edge = sum(n(r.get("edge")) for r, _ in priced if n(r.get("edge")) is not None) / len(
        [r for r, _ in priced if n(r.get("edge")) is not None]
    ) * 100 if any(n(r.get("edge")) is not None for r, _ in priced) else None

    # Current form: last 20 priced decisions.
    priced.sort(key=lambda x: str(x[0].get("date_iso") or ""))
    recent = priced[-20:]
    recent_roi = sum(p for _, p in recent) / len(recent) * 100 if recent else None

    # Risk: actual bankroll drawdown.
    try:
        ib = float(D.get("meta", {}).get("initial_bank", 10000.0))
    except Exception:
        ib = 10000.0
    hist = db.bank_history(limit=100000)
    banks = [ib] + [float(h.get("bank") or 0) for h in hist]
    max_dd = 0.0
    if banks:
        peak = banks[0]
        for bank in banks:
            peak = max(peak, bank)
            if peak > 0:
                max_dd = max(max_dd, (peak - bank) / peak)

    model_ok = model_brier is not None and model_brier <= 0.25
    model_warn = model_brier is not None and model_brier <= 0.30
    value_ok = value_roi is not None and value_roi >= 0
    value_warn = value_roi is not None and value_roi >= -5
    form_ok = recent_roi is not None and recent_roi >= 0
    form_warn = recent_roi is not None and recent_roi >= -5
    risk_ok = max_dd <= 0.10
    risk_warn = max_dd <= 0.20

    st.subheader("🧭 SYSTEM SCORECARD")
    st.caption(
        "Диагностический health-check. Он не меняет BET/WATCH/SKIP и не является "
        "научно валидированным рейтингом стратегии."
    )
    cols = st.columns(4)
    items = [
        ("MODEL", score_status(model_ok, model_warn), f"Brier {model_brier:.3f}" if model_brier is not None else "нет данных"),
        ("VALUE", score_status(value_ok, value_warn), f"ROI {value_roi:+.1f}%" if value_roi is not None else "нет цены"),
        ("FORM", score_status(form_ok, form_warn), f"Last 20 ROI {recent_roi:+.1f}%" if recent_roi is not None else "нет данных"),
        ("RISK", score_status(risk_ok, risk_warn), f"Max DD {max_dd:.1%}"),
    ]
    for col, (label, status, detail) in zip(cols, items):
        col.metric(label, status)
        col.caption(detail)

    details = pd.DataFrame([
        {
            "Axis": "MODEL",
            "Metric": "Brier",
            "Value": f"{model_brier:.3f}" if model_brier is not None else "—",
            "Interpretation": "≤0.25 OK · ≤0.30 monitor",
        },
        {
            "Axis": "VALUE",
            "Metric": "ROI",
            "Value": f"{value_roi:+.1f}%" if value_roi is not None else "—",
            "Interpretation": "≥0% OK · ≥−5% monitor",
        },
        {
            "Axis": "FORM",
            "Metric": "Last 20 ROI",
            "Value": f"{recent_roi:+.1f}%" if recent_roi is not None else "—",
            "Interpretation": "≥0% OK · ≥−5% monitor",
        },
        {
            "Axis": "RISK",
            "Metric": "Max Drawdown",
            "Value": f"{max_dd:.1%}",
            "Interpretation": "≤10% OK · ≤20% monitor",
        },
    ])
    st.dataframe(details, use_container_width=True, hide_index=True)
    st.caption(
        f"Sample: {len(closed)} завершённых snapshots · priced: {len(priced)} · "
        f"MODEL ONLY: {len(model_only_rows)}. Пороговые значения — эвристические."
    )

def _render_adaptive_segments():
    """Рейтинг рынков и лиг по фактическому результату с защитой от малых выборок."""
    if not db.SQLITE_BOOT_OK:
        return

    snapshots = db.fetch_decision_snapshots(limit=100000)
    rows = []
    for r in snapshots:
        status = str(r.get("result_status") or "").lower()
        if status not in ("won", "lost"):
            continue
        try:
            odd = float(r.get("market_odd"))
        except (TypeError, ValueError):
            continue
        if odd <= 1.01:
            continue
        pnl = odd - 1.0 if status == "won" else -1.0
        rows.append({
            "market": str(r.get("market") or "Unknown"),
            "league": str(r.get("league") or "Unknown"),
            "pnl": pnl,
            "won": 1 if status == "won" else 0,
            "edge": r.get("edge"),
            "ev": r.get("ev"),
        })

    st.subheader("🧩 ADAPTIVE MARKET / LEAGUE SCORE")
    st.caption(
        "Рейтинг сегментов по завершённым сигналам с реальным кэфом. "
        "Минимум N=10 для рейтинга; N<20 помечается как ранняя выборка. "
        "Рейтинг диагностический и ничего автоматически не отключает."
    )
    if not rows:
        st.info("Пока нет завершённых priced-сигналов для сегментного рейтинга.")
        return

    import pandas as pd
    from collections import defaultdict

    def aggregate(key):
        groups = defaultdict(list)
        for x in rows:
            groups[x[key]].append(x)
        out = []
        for name, items in groups.items():
            n = len(items)
            if n < 10:
                continue
            pnl = sum(x["pnl"] for x in items)
            edges = [float(x["edge"]) for x in items if x["edge"] is not None]
            evs = [float(x["ev"]) for x in items if x["ev"] is not None]
            wr = sum(x["won"] for x in items) / n
            roi = pnl / n * 100
            # Shrinkage: небольшие сегменты подтягиваем к общему ROI.
            prior_n = 20.0
            global_roi = sum(x["pnl"] for x in rows) / len(rows) * 100
            adj_roi = (pnl + global_roi / 100 * prior_n) / (n + prior_n) * 100
            out.append({
                "Segment": name,
                "N": n,
                "Win Rate": wr * 100,
                "ROI": roi,
                "Adj ROI": adj_roi,
                "Avg Edge": sum(edges) / len(edges) * 100 if edges else None,
                "Avg EV": sum(evs) / len(evs) * 100 if evs else None,
                "Status": "CONFIRMED" if n >= 20 else "EARLY",
            })
        return sorted(out, key=lambda x: (-x["Adj ROI"], -x["N"]))

    for title, key in (("По рынкам", "market"), ("По лигам", "league")):
        data = aggregate(key)
        st.markdown(f"**{title}**")
        if not data:
            st.info("Нужно минимум 10 завершённых сигналов в сегменте.")
            continue
        df = pd.DataFrame(data)
        st.dataframe(
            df.assign(
                **{
                    "Win Rate": df["Win Rate"].map(lambda x: f"{x:.1f}%"),
                    "ROI": df["ROI"].map(lambda x: f"{x:+.1f}%"),
                    "Adj ROI": df["Adj ROI"].map(lambda x: f"{x:+.1f}%"),
                    "Avg Edge": df["Avg Edge"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"),
                    "Avg EV": df["Avg EV"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Adj ROI использует shrinkage к общей истории, чтобы N=10 не выглядело "
        "столь же надёжно, как N=100. Это статистическая стабилизация, не гарантия будущего результата."
    )

def _render_rolling_performance():
    """Rolling view: последние N завершённых сигналов без смешивания всей истории."""
    if not db.SQLITE_BOOT_OK:
        return

    snapshots = db.fetch_decision_snapshots(limit=100000)
    completed = []
    model_only = []
    for r in snapshots:
        status = str(r.get("result_status") or "").lower()
        if status not in ("won", "lost"):
            continue
        if str(r.get("decision") or "").upper() == "MODEL_ONLY":
            model_only.append(r)
            continue
        try:
            odd = float(r.get("market_odd"))
            if odd <= 1.01:
                continue
        except (TypeError, ValueError):
            continue
        completed.append(r)

    import pandas as pd
    from datetime import datetime

    def dt(r):
        try:
            return datetime.fromisoformat(str(r.get("date_iso") or "")[:19])
        except Exception:
            try:
                return datetime.strptime(str(r.get("date_iso") or "")[:10], "%Y-%m-%d")
            except Exception:
                return datetime.min

    completed.sort(key=dt)
    st.subheader("📉 Rolling Performance")
    st.caption(
        "Последние 20 / 50 / 100 завершённых ценовых сигналов. "
        "Это монитор текущей формы, а не замена полной истории."
    )

    rows = []
    for window in (20, 50, 100):
        sample = completed[-window:]
        if not sample:
            continue
        wins = sum(1 for r in sample if str(r.get("result_status")).lower() == "won")
        pnl = 0.0
        edges = []
        evs = []
        for r in sample:
            try:
                odd = float(r.get("market_odd"))
                pnl += odd - 1.0 if str(r.get("result_status")).lower() == "won" else -1.0
            except (TypeError, ValueError):
                pass
            for key, target in (("edge", edges), ("ev", evs)):
                try:
                    target.append(float(r.get(key)))
                except (TypeError, ValueError):
                    pass
        roi = pnl / len(sample) * 100
        rows.append({
            "Window": f"Last {window}",
            "N": len(sample),
            "Win Rate": wins / len(sample) * 100,
            "ROI": roi,
            "Avg Edge": sum(edges) / len(edges) * 100 if edges else None,
            "Avg EV": sum(evs) / len(evs) * 100 if evs else None,
        })

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df.assign(
                **{
                    "Win Rate": df["Win Rate"].map(lambda x: f"{x:.1f}%"),
                    "ROI": df["ROI"].map(lambda x: f"{x:+.1f}%"),
                    "Avg Edge": df["Avg Edge"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"),
                    "Avg EV": df["Avg EV"].map(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—"),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Пока недостаточно завершённых ценовых сигналов для rolling-аналитики.")

    if model_only:
        model_only.sort(key=dt)
        st.markdown("**🧠 MODEL ONLY — rolling accuracy**")
        mrows = []
        for window in (20, 50, 100):
            sample = model_only[-window:]
            if not sample:
                continue
            wins = sum(1 for r in sample if str(r.get("result_status")).lower() == "won")
            probs = []
            brier_terms = []
            for r in sample:
                try:
                    p = float(r.get("model_prob"))
                    y = 1.0 if str(r.get("result_status")).lower() == "won" else 0.0
                    probs.append(p)
                    brier_terms.append((p - y) ** 2)
                except (TypeError, ValueError):
                    pass
            mrows.append({
                "Window": f"Last {window}",
                "N": len(sample),
                "Win Rate": wins / len(sample) * 100,
                "Brier": sum(brier_terms) / len(brier_terms) if brier_terms else None,
            })
        if mrows:
            dfm = pd.DataFrame(mrows)
            st.dataframe(
                dfm.assign(
                    **{
                        "Win Rate": dfm["Win Rate"].map(lambda x: f"{x:.1f}%"),
                        "Brier": dfm["Brier"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—"),
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        st.caption("MODEL ONLY не имеет цены входа, поэтому здесь показываем только accuracy/Brier, без ROI.")

def _render_drift_monitor():
    """Сравнивает свежие завершённые snapshots с предыдущим периодом."""
    if not db.SQLITE_BOOT_OK:
        return

    snapshots = db.fetch_decision_snapshots(limit=100000)
    rows = []
    for r in snapshots:
        if str(r.get("result_status") or "").lower() not in ("won", "lost"):
            continue
        try:
            odd = float(r.get("market_odd"))
            prob = float(r.get("model_prob"))
            edge = float(r.get("edge"))
            ev = float(r.get("ev"))
            date = str(r.get("date_iso") or "")[:10]
            if odd <= 1.01 or not date or not (0.0 <= prob <= 1.0):
                continue
            won = 1.0 if str(r.get("result_status")).lower() == "won" else 0.0
            pnl = (odd - 1.0) if won else -1.0
        except (TypeError, ValueError):
            continue
        rows.append({
            "date": date,
            "won": won,
            "pnl": pnl,
            "edge": edge,
            "ev": ev,
            "brier": (prob - won) ** 2,
        })

    if len(rows) < 40:
        return

    rows.sort(key=lambda x: x["date"])
    split = len(rows) // 2
    old = rows[:split]
    new = rows[split:]

    def metrics(data):
        n = len(data)
        return {
            "N": n,
            "Win Rate": sum(x["won"] for x in data) / n,
            "ROI": sum(x["pnl"] for x in data) / n * 100.0,
            "Avg Edge": sum(x["edge"] for x in data) / n,
            "Avg EV": sum(x["ev"] for x in data) / n,
            "Brier": sum(x["brier"] for x in data) / n,
        }

    a = metrics(old)
    b = metrics(new)

    comparisons = [
        ("Win Rate", a["Win Rate"], b["Win Rate"], "pp"),
        ("ROI", a["ROI"], b["ROI"], "pp"),
        ("Avg Edge", a["Avg Edge"], b["Avg Edge"], "pp"),
        ("Avg EV", a["Avg EV"], b["Avg EV"], "pp"),
        ("Brier", a["Brier"], b["Brier"], "raw"),
    ]

    # Для ROI/Win Rate/Edge/EV падение негативно; для Brier рост негативен.
    deterioration = []
    for name, old_v, new_v, unit in comparisons:
        delta = new_v - old_v
        bad = delta < -0.03 if unit == "pp" and name != "Brier" else (
            delta > 0.03 if unit == "pp" and name == "Brier" else (
                delta > 0.02 if name == "Brier" else False
            )
        )
        deterioration.append((name, delta, bad))

    bad_count = sum(x[2] for x in deterioration)
    if bad_count >= 3:
        status = "🔴 DEGRADING"
    elif bad_count >= 1:
        status = "🟡 DRIFT"
    else:
        status = "🟢 STABLE"

    st.subheader("⚠️ Drift Monitor")
    st.metric("Model status", status)
    st.caption(
        f"Сравнение двух последовательных половин завершённых snapshots: "
        f"{old[0]['date']} → {old[-1]['date']} vs {new[0]['date']} → {new[-1]['date']}. "
        "Это диагностический индикатор, не статистический тест."
    )

    table = pd.DataFrame([
        {
            "Metric": name,
            "Previous": old_v,
            "Recent": new_v,
            "Delta": new_v - old_v,
        }
        for name, old_v, new_v, _ in comparisons
    ])
    st.dataframe(
        table.assign(
            Previous=table.apply(
                lambda r: f"{r['Previous']:.1%}" if r["Metric"] != "Brier"
                else f"{r['Previous']:.3f}", axis=1
            ),
            Recent=table.apply(
                lambda r: f"{r['Recent']:.1%}" if r["Metric"] != "Brier"
                else f"{r['Recent']:.3f}", axis=1
            ),
            Delta=table.apply(
                lambda r: (
                    f"{r['Delta']:+.1%}" if r["Metric"] != "Brier"
                    else f"{r['Delta']:+.3f}"
                ), axis=1
            ),
        ),
        use_container_width=True,
        hide_index=True,
    )

    if bad_count:
        names = ", ".join(x[0] for x in deterioration if x[2])
        st.warning(
            f"Негативная динамика: {names}. "
            "Проверь рынок, источник коэффициентов и последние сигналы перед повышением риска."
        )
    else:
        st.success("Существенного ухудшения по выбранным метрикам не обнаружено.")

    st.caption(
        "Пороговые значения эвристические. При малой истории или смене состава лиг "
        "Drift Monitor может давать ложные предупреждения."
    )


def _render_strategy_diagnostics():
    """Агрегированная диагностика завершившихся decision snapshots."""
    if not db.SQLITE_BOOT_OK:
        return

    snapshots = db.fetch_decision_snapshots(limit=100000)
    rows = []
    for r in snapshots:
        if str(r.get("result_status") or "").lower() not in ("won", "lost"):
            continue
        try:
            odd = float(r.get("market_odd"))
            pnl = (odd - 1.0) if str(r.get("result_status")).lower() == "won" else -1.0
            edge = float(r.get("edge"))
            ev = float(r.get("ev"))
            conf = float(r.get("confidence"))
            league = str(r.get("league") or "Unknown")
            market = str(r.get("market") or "Unknown")
            decision = str(r.get("decision") or "Unknown").upper()
        except (TypeError, ValueError):
            continue

        if odd <= 1.01:
            continue

        rows.append({
            "league": league,
            "market": market,
            "edge": edge,
            "ev": ev,
            "confidence": conf,
            "decision": decision,
            "pnl": pnl,
            "won": 1 if pnl > 0 else 0,
        })

    if len(rows) < 20:
        return

    import pandas as pd

    df = pd.DataFrame(rows)

    def bucket(x, cuts, labels):
        return labels[max(0, min(len(labels) - 1, sum(x >= c for c in cuts)))]

    df["Edge bucket"] = df["edge"].map(
        lambda x: bucket(x, [0.01, 0.03, 0.05], ["<1%", "1–3%", "3–5%", "5%+"])
    )
    df["Confidence bucket"] = df["confidence"].map(
        lambda x: bucket(x, [0.60, 0.70, 0.80], ["<60%", "60–70%", "70–80%", "80%+"])
    )

    def aggregate(frame, dimension):
        out = (
            frame.groupby(dimension, dropna=False)
            .agg(
                N=("pnl", "size"),
                WinRate=("won", "mean"),
                PnL=("pnl", "sum"),
                AvgEdge=("edge", "mean"),
                AvgEV=("ev", "mean"),
            )
            .reset_index()
        )
        out["ROI"] = out["PnL"] / out["N"] * 100.0
        return out.sort_values(["N", "ROI"], ascending=[False, False])

    st.subheader("🧠 Strategy Diagnostics")
    st.caption(
        "Историческая диагностика завершившихся snapshots. ROI нормирован на 1u "
        "виртуальной ставки; реальные BET/WATCH правила не меняются."
    )

    min_n = 10

    for dimension, title in [
        ("market", "🎯 По рынкам"),
        ("league", "🏆 По лигам"),
        ("Edge bucket", "📐 По Edge"),
        ("Confidence bucket", "🎯 По Confidence"),
        ("decision", "⚖️ BET / WATCH / SKIP"),
    ]:
        agg = aggregate(df, dimension)
        agg = agg[agg["N"] >= min_n].copy()
        if agg.empty:
            continue

        display = agg.rename(columns={
            dimension: "Segment",
            "N": "N",
            "WinRate": "Win Rate",
            "PnL": "PnL",
            "ROI": "ROI",
            "AvgEdge": "Avg Edge",
            "AvgEV": "Avg EV",
        })[
            ["Segment", "N", "Win Rate", "ROI", "PnL", "Avg Edge", "Avg EV"]
        ]
        st.markdown(f"**{title}**")
        st.dataframe(
            display.assign(
                **{
                    "Win Rate": display["Win Rate"].map(lambda x: f"{x:.1%}"),
                    "ROI": display["ROI"].map(lambda x: f"{x:+.1f}%"),
                    "PnL": display["PnL"].map(lambda x: f"{x:+.2f}"),
                    "Avg Edge": display["Avg Edge"].map(lambda x: f"{x:.1%}"),
                    "Avg EV": display["Avg EV"].map(lambda x: f"{x:.1%}"),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    reliable = df.groupby(["market", "league"], dropna=False).agg(
        N=("pnl", "size"),
        WinRate=("won", "mean"),
        PnL=("pnl", "sum"),
        AvgEdge=("edge", "mean"),
        AvgEV=("ev", "mean"),
    ).reset_index()
    reliable["ROI"] = reliable["PnL"] / reliable["N"] * 100.0
    reliable = reliable[reliable["N"] >= 20].copy()

    if not reliable.empty:
        best = reliable.sort_values(["ROI", "N"], ascending=[False, False]).head(3)
        weak = reliable.sort_values(["ROI", "N"], ascending=[True, False]).head(3)

        st.markdown("**🔥 BEST ZONES**")
        for _, r in best.iterrows():
            st.write(
                f"🟢 **{r['market']} · {r['league']}** — "
                f"N={int(r['N'])} · Win {r['WinRate']:.1%} · ROI {r['ROI']:+.1f}% · "
                f"Edge {r['AvgEdge']:.1%} · EV {r['AvgEV']:.1%}"
            )

        st.markdown("**⚠️ WEAK ZONES**")
        for _, r in weak.iterrows():
            st.write(
                f"🔴 **{r['market']} · {r['league']}** — "
                f"N={int(r['N'])} · Win {r['WinRate']:.1%} · ROI {r['ROI']:+.1f}% · "
                f"Edge {r['AvgEdge']:.1%} · EV {r['AvgEV']:.1%}"
            )

    st.caption(
        "BEST/WEAK — исследовательские сигналы, не автоматическое изменение фильтров. "
        "Для зон требуется минимум 20 завершённых observations."
    )


def _render_kelly_sensitivity():
    """Исследовательский тест Kelly fractions на завершившихся snapshots."""
    if not db.SQLITE_BOOT_OK:
        return

    snapshots = db.fetch_decision_snapshots(limit=100000)
    samples = []
    for r in snapshots:
        if str(r.get("result_status") or "").lower() not in ("won", "lost"):
            continue
        try:
            odd = float(r.get("market_odd"))
            kelly = float(r.get("kelly_pct"))
            if odd <= 1.01 or kelly <= 0:
                continue
            base_pnl = (odd - 1.0) if str(r.get("result_status")).lower() == "won" else -1.0
            date = str(r.get("date_iso") or "")[:10]
            if not date:
                continue
        except (TypeError, ValueError):
            continue
        samples.append({"date": date, "kelly": kelly, "pnl": base_pnl})

    if len(samples) < 20:
        return

    fractions = [0.25, 0.50, 0.75, 1.00]
    import pandas as pd

    rows = []
    for frac in fractions:
        equity = 100.0
        peak = equity
        max_dd = 0.0
        pnl = 0.0
        for s in sorted(samples, key=lambda x: x["date"]):
            # kelly_pct хранится как доля банка, поэтому ограничиваем риск
            # консервативно 5% банка на один виртуальный сигнал.
            stake_pct = min(max(s["kelly"], 0.0) * frac, 0.05)
            trade_pnl = equity * stake_pct * s["pnl"]
            equity += trade_pnl
            pnl += trade_pnl
            peak = max(peak, equity)
            dd = (peak - equity) / peak if peak > 0 else 0.0
            max_dd = max(max_dd, dd)

        rows.append({
            "Kelly": f"{frac:.2f}×",
            "Signals": len(samples),
            "Final Equity": equity,
            "PnL": pnl,
            "ROI": (equity / 100.0 - 1.0) * 100,
            "Max Drawdown": max_dd * 100,
        })

    st.subheader("💰 Kelly Sensitivity")
    st.caption(
        "Виртуальная симуляция на завершившихся signals. Kelly fraction меняется, "
        "реальный банк не затрагивается. Для сравнения применяется одинаковый cap 5% "
        "виртуального банка на один сигнал."
    )
    df = pd.DataFrame(rows)
    st.dataframe(
        df.assign(
            **{
                "Final Equity": df["Final Equity"].map(lambda x: f"{x:.2f}"),
                "PnL": df["PnL"].map(lambda x: f"{x:+.2f}"),
                "ROI": df["ROI"].map(lambda x: f"{x:+.1f}%"),
                "Max Drawdown": df["Max Drawdown"].map(lambda x: f"-{x:.1f}%"),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    best = max(rows, key=lambda r: r["ROI"])
    low_dd = min(rows, key=lambda r: r["Max Drawdown"])
    st.caption(
        f"Лучший исторический ROI: {best['Kelly']} · {best['ROI']:+.1f}%. "
        f"Минимальная просадка: {low_dd['Kelly']} · -{low_dd['Max Drawdown']:.1f}%. "
        "Это backtest, а не автоматическая рекомендация менять Kelly."
    )


def _render_snapshot_calibration(D):
    """Калибровка модели по завершившимся decision snapshots."""
    if not db.SQLITE_BOOT_OK:
        return

    rows = db.fetch_decision_snapshots(limit=100000)
    samples = []
    for r in rows:
        outcome = str(r.get("result_status") or "").lower()
        if outcome not in ("won", "lost"):
            continue
        try:
            p = float(r.get("model_prob"))
        except (TypeError, ValueError):
            continue
        if not 0.0 <= p <= 1.0:
            continue
        samples.append((p, 1.0 if outcome == "won" else 0.0))

    if len(samples) < 5:
        return

    import pandas as pd

    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    rows_out = []
    for lo, hi in bins:
        bucket = [
            (p, y) for p, y in samples
            if (lo <= p < hi) or (hi >= 1.0 and lo <= p <= 1.0)
        ]
        if not bucket:
            continue
        n = len(bucket)
        avg_p = sum(p for p, _ in bucket) / n
        actual = sum(y for _, y in bucket) / n
        rows_out.append({
            "P модели": f"{lo*100:.0f}–{hi*100:.0f}%",
            "N": n,
            "Model P": avg_p,
            "Факт": actual,
            "Ошибка": actual - avg_p,
        })

    brier = sum((p - y) ** 2 for p, y in samples) / len(samples)
    mae = sum(abs(p - y) for p, y in samples) / len(samples)

    st.subheader("🧪 Signal Calibration")
    st.caption(
        "Калибровка по завершившимся сохранённым сигналам, включая WATCH. "
        "Это исследовательская выборка и не влияет на банк или правила BET."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Signals", len(samples))
    c2.metric("Brier", f"{brier:.3f}")
    c3.metric("Средняя абсолютная ошибка", f"{mae*100:.1f} п.п.")

    df = pd.DataFrame(rows_out)
    if not df.empty:
        chart = df.set_index("P модели")[["Model P", "Факт"]] * 100
        st.line_chart(chart, height=260)
        st.dataframe(
            df.assign(
                **{
                    "Model P": df["Model P"].map(lambda x: f"{x*100:.1f}%"),
                    "Факт": df["Факт"].map(lambda x: f"{x*100:.1f}%"),
                    "Ошибка": df["Ошибка"].map(lambda x: f"{x*100:+.1f} п.п."),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        reliable = [r for r in rows_out if r["N"] >= 10]
        if reliable:
            worst = max(reliable, key=lambda r: abs(r["Ошибка"]))
            direction = "недооценивает" if worst["Ошибка"] > 0 else "переоценивает"
            st.warning(
                f"Наиболее заметное отклонение при N≥10: {worst['P модели']} — "
                f"модель {direction} фактический результат на "
                f"{abs(worst['Ошибка'])*100:.1f} п.п."
            )
    st.caption(
        "Важно: snapshots разных решений могут быть зависимыми между сканами; "
        "это мониторинг калибровки, а не независимый backtest."
    )


def _render_overview(D):
    bets = [b for b in D.get("bets", []) if isinstance(b, dict)]
    if not bets:
        st.info("Ставок пока нет. Сделай ⚡ СКАН.")
        return
    total = len(bets)
    pending = [b for b in bets if b.get("status") == "pending"]
    won = [b for b in bets if b.get("status") == "won"]
    lost = [b for b in bets if b.get("status") == "lost"]
    push = [b for b in bets if b.get("status") == "push"]
    void = [b for b in bets if b.get("status") == "void"]
    closed = len(won) + len(lost)
    total_stake = sum(float(b.get("stake") or 0) for b in bets)
    total_pnl = 0.0
    for b in bets:
        s_ = b.get("status")
        stake = float(b.get("stake") or 0)
        odds = float(b.get("odds") or 1)
        if s_ == "won":
            total_pnl += stake * (odds - 1)
        elif s_ == "lost":
            total_pnl -= stake
    winrate = (len(won) / closed * 100) if closed else 0.0
    roi = (total_pnl / total_stake * 100) if total_stake else 0.0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Всего", total)
    k2.metric("В работе", len(pending))
    k3.metric("WinRate", f"{winrate:.1f}%")
    k4.metric("PnL", f"{total_pnl:+.2f}")
    k5.metric("ROI", f"{roi:+.2f}%")

    bank = float(D.get("bank", 10000))
    initial = float(D.get("meta", {}).get("initial_bank", 10000))
    growth = ((bank - initial) / initial * 100) if initial else 0
    k1, k2, k3 = st.columns(3)
    k1.metric("Банкролл", f"{bank:.0f}", f"{growth:+.1f}%")
    k2.metric("Заморожено", f"{sum(float(b.get('stake') or 0) for b in pending):.0f}")
    k3.metric("В обороте всего", f"{total_stake:.0f}")

    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🟢 Выиграно", len(won))
    s2.metric("🔴 Проиграно", len(lost))
    s3.metric("⚪ Возврат", len(push))
    s4.metric("⬜ Void", len(void))

    try:
        history = db.bank_history(limit=5000)
        if len(history) > 1:
            import pandas as pd
            df = pd.DataFrame(history)
            st.subheader("📈 Динамика банка")
            st.line_chart(df.set_index("ts")["bank"])
    except Exception:
        pass


def _render_by_league(D):
    bets = [b for b in D.get("bets", []) if isinstance(b, dict)
            and b.get("status") in ("won", "lost")]
    if not bets:
        st.info("Нет закрытых ставок.")
        return
    agg = defaultdict(lambda: {"won": 0, "lost": 0, "pnl": 0.0,
                                 "stake": 0.0, "total": 0})
    for b in bets:
        lg = b.get("league") or b.get("div") or "—"
        stake = float(b.get("stake") or 0)
        odds = float(b.get("odds") or 1)
        agg[lg]["total"] += 1
        agg[lg]["stake"] += stake
        if b.get("status") == "won":
            agg[lg]["won"] += 1
            agg[lg]["pnl"] += stake * (odds - 1)
        else:
            agg[lg]["lost"] += 1
            agg[lg]["pnl"] -= stake
    rows = []
    for lg, d in sorted(agg.items(), key=lambda kv: -kv[1]["pnl"]):
        closed = d["won"] + d["lost"]
        wr = (d["won"] / closed * 100) if closed else 0
        roi = (d["pnl"] / d["stake"] * 100) if d["stake"] else 0
        rows.append({"Лига": lg, "Ставок": d["total"],
                     "Won": d["won"], "Lost": d["lost"],
                     "WinRate": f"{wr:.0f}%",
                     "PnL": f"{d['pnl']:+.2f}",
                     "ROI": f"{roi:+.1f}%"})
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_by_day(D):
    bets = [b for b in D.get("bets", []) if isinstance(b, dict)
            and b.get("status") in ("won", "lost")]
    if not bets:
        st.info("Нет закрытых ставок.")
        return
    agg = defaultdict(lambda: {"won": 0, "lost": 0, "pnl": 0.0, "total": 0})
    for b in bets:
        d_iso = (b.get("date_iso") or "")[:10] or "—"
        stake = float(b.get("stake") or 0)
        odds = float(b.get("odds") or 1)
        agg[d_iso]["total"] += 1
        if b.get("status") == "won":
            agg[d_iso]["won"] += 1
            agg[d_iso]["pnl"] += stake * (odds - 1)
        else:
            agg[d_iso]["lost"] += 1
            agg[d_iso]["pnl"] -= stake
    rows = []
    for d in sorted(agg.keys())[-30:]:
        v = agg[d]
        closed = v["won"] + v["lost"]
        wr = (v["won"] / closed * 100) if closed else 0
        rows.append({"Дата": d, "Ставок": v["total"],
                     "Won": v["won"], "Lost": v["lost"],
                     "WinRate": f"{wr:.0f}%",
                     "PnL": f"{v['pnl']:+.2f}"})
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_by_market(D):
    bets = [b for b in D.get("bets", []) if isinstance(b, dict)
            and b.get("status") in ("won", "lost")]
    if not bets:
        st.info("Нет закрытых ставок.")
        return
    agg = defaultdict(lambda: {"won": 0, "lost": 0, "pnl": 0.0,
                                 "stake": 0.0, "total": 0})
    for b in bets:
        mkt = b.get("market") or "OTHER"
        stake = float(b.get("stake") or 0)
        odds = float(b.get("odds") or 1)
        agg[mkt]["total"] += 1
        agg[mkt]["stake"] += stake
        if b.get("status") == "won":
            agg[mkt]["won"] += 1
            agg[mkt]["pnl"] += stake * (odds - 1)
        else:
            agg[mkt]["lost"] += 1
            agg[mkt]["pnl"] -= stake
    rows = []
    for mkt, d in sorted(agg.items(), key=lambda kv: -kv[1]["pnl"]):
        closed = d["won"] + d["lost"]
        wr = (d["won"] / closed * 100) if closed else 0
        roi = (d["pnl"] / d["stake"] * 100) if d["stake"] else 0
        rows.append({"Рынок": mkt, "Ставок": d["total"],
                     "Won": d["won"], "Lost": d["lost"],
                     "WinRate": f"{wr:.0f}%",
                     "PnL": f"{d['pnl']:+.2f}",
                     "ROI": f"{roi:+.1f}%"})
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _watch_pick_result(pick, home, away):
    """Возвращает win/loss/push для сохранённого рынка по финальному счёту."""
    try:
        h, a = int(home), int(away)
    except (TypeError, ValueError):
        return None
    p = str(pick or "").strip()
    if p == "П1":
        return "won" if h > a else "lost"
    if p == "X":
        return "won" if h == a else "lost"
    if p == "П2":
        return "won" if a > h else "lost"
    if p == "ТБ 2.5":
        return "won" if h + a >= 3 else "lost"
    if p == "ТМ 2.5":
        return "won" if h + a <= 2 else "lost"
    if p == "BTTS да":
        return "won" if h > 0 and a > 0 else "lost"
    if p == "BTTS нет":
        return "won" if h == 0 or a == 0 else "lost"
    return None


def _render_watch_lab(D):
    """Исследование исторических WATCH без фиктивного учёта их как ставок."""
    if not db.SQLITE_BOOT_OK:
        st.info("WATCH LAB доступен после инициализации SQLite.")
        return

    rows = db.fetch_decision_snapshots(limit=100000)
    watch = [r for r in rows if str(r.get("decision", "")).upper() == "WATCH"]
    model_only = [r for r in rows if str(r.get("decision", "")).upper() == "MODEL_ONLY"]
    if not watch and not model_only:
        st.info(
            "Пока нет сохранённых WATCH / MODEL ONLY снимков. Запусти несколько сканов."
        )
        return

    # Обновляем только завершившиеся WATCH. Результат кешируется в snapshot,
    # поэтому повторные открытия вкладки не создают новую историю.
    import pandas as pd
    from datetime import datetime, timedelta

    fd_token = str((D.get("meta") or {}).get("fdorg_token") or "")
    now = datetime.now()
    evaluated = []
    model_evaluated = []
    pending_eval = 0

    def evaluate_snapshot(r, need_odd=False):
        if r.get("result_status") in ("won", "lost", "push"):
            return dict(r), False
        date_iso = str(r.get("date_iso") or "")[:10]
        try:
            match_date = datetime.strptime(date_iso, "%Y-%m-%d")
        except Exception:
            return None, False
        if match_date > now - timedelta(hours=2):
            return None, True

        fid = str(r.get("fixture_id") or "")
        result = None
        if fid.startswith("espn:"):
            result = espn_match_result(fid, date_iso)
        elif fd_token and fid:
            result = fdorg_match_result(fid, fd_token)
        if result is None and fid:
            result = tsdb_match_result(fid)
        if not result:
            return None, True

        home_score, away_score = result.get("home"), result.get("away")
        outcome = _watch_pick_result(r.get("pick"), home_score, away_score)
        if not outcome:
            return None, True

        update = {
            "result_status": outcome,
            "result_score": f"{int(home_score)}:{int(away_score)}",
            "evaluated_at": datetime.now().isoformat(),
        }
        if need_odd:
            try:
                num_odd = float(r.get("market_odd"))
            except (TypeError, ValueError):
                num_odd = 0.0
            if num_odd <= 1.01:
                return None, True
            update["virtual_pnl"] = (
                num_odd - 1.0 if outcome == "won"
                else (-1.0 if outcome == "lost" else 0.0)
            )

        db.update_decision_snapshot(int(r["id"]), **update)
        rr = dict(r)
        rr.update(update)
        return rr, False

    for r in watch[:1000]:
        rr, pending = evaluate_snapshot(r, need_odd=True)
        if rr:
            evaluated.append(rr)
        elif pending:
            pending_eval += 1

    for r in model_only[:1000]:
        rr, pending = evaluate_snapshot(r, need_odd=False)
        if rr:
            model_evaluated.append(rr)
        elif pending:
            pending_eval += 1

    model_settled = [
        r for r in model_evaluated
        if r.get("result_status") in ("won", "lost")
    ]

    st.subheader("🧠 MODEL ONLY")
    st.caption(
        "Матчи, которые модель рекомендовала, но реального кэфа не было. "
        "Это оценка точности прогноза, не ставки и не PnL."
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("MODEL ONLY", len(model_only))
    m2.metric("Результат найден", len(model_settled))
    if model_settled:
        wins = sum(1 for x in model_settled if x.get("result_status") == "won")
        m3.metric("Win Rate", f"{wins / len(model_settled):.1%}")
    else:
        m3.metric("Win Rate", "—")

    if model_settled:
        probs = [num(x.get("model_prob")) for x in model_settled]
        probs = [x for x in probs if x is not None]
        brier = None
        if probs and len(probs) == len(model_settled):
            brier = sum(
                (p - (1.0 if x.get("result_status") == "won" else 0.0)) ** 2
                for p, x in zip(probs, model_settled)
            ) / len(probs)
        st.write(
            f"Завершено: **{len(model_settled)}** · "
            f"Brier: **{brier:.3f}**" if brier is not None
            else f"Завершено: **{len(model_settled)}**"
        )
        st.dataframe(
            pd.DataFrame([
                {
                    "Match": x.get("match_ru") or x.get("match"),
                    "Pick": x.get("pick"),
                    "Model Prob": f"{float(x.get('model_prob')):.1%}" if x.get("model_prob") is not None else "—",
                    "Result": x.get("result_score"),
                    "Status": x.get("result_status"),
                }
                for x in model_settled[-20:]
            ]),
            use_container_width=True,
            hide_index=True,
        )

    # WATCH — наблюдение, но теперь для завершённых матчей можно честно считать
    # виртуальный результат по тому кэфу, который был сохранён в момент сигнала.
    settled_watch = [
        r for r in evaluated
        if r.get("result_status") in ("won", "lost", "push")
        and r.get("virtual_pnl") is not None
    ]

    # Сначала показываем распределение причин и качества цены.

    def num(v):
        try:
            x = float(v)
            return x if x == x else None
        except (TypeError, ValueError):
            return None

    def bucket(x):
        if x is None:
            return "нет данных"
        if x < 0:
            return "<0%"
        if x < 0.03:
            return "0–3%"
        if x < 0.05:
            return "3–5%"
        if x < 0.08:
            return "5–8%"
        return "8%+"

    n = len(watch)
    edges = [num(r.get("edge")) for r in watch if num(r.get("edge")) is not None]
    evs = [num(r.get("ev")) for r in watch if num(r.get("ev")) is not None]
    kells = [num(r.get("kelly_pct")) for r in watch if num(r.get("kelly_pct")) is not None]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("WATCH snapshots", n)
    c2.metric("Результат найден", len(settled_watch))
    c3.metric("Средний Edge", f"{sum(edges)/len(edges)*100:+.1f}%" if edges else "—")
    c4.metric("Средний EV", f"{sum(evs)/len(evs)*100:+.1f}%" if evs else "—")

    if settled_watch:
        v_pnl = sum(float(x.get("virtual_pnl") or 0) for x in settled_watch)
        v_turnover = sum(1.0 for _ in settled_watch)
        v_roi = v_pnl / v_turnover * 100 if v_turnover else 0.0
        w1, w2, w3 = st.columns(3)
        w1.metric("WATCH virtual PnL", f"{v_pnl:+.2f}u")
        w2.metric("WATCH ROI", f"{v_roi:+.1f}%")
        w3.metric("Незавершено", pending_eval)
        st.caption(
            "1u = условная единица равная одной ставке. Это исследовательский PnL, "
            "он не влияет на банкролл и реальные ставки."
        )
    else:
        st.caption(
            f"Результаты ещё не сопоставлены. Ожидаем: {pending_eval}. "
            "Виртуальный PnL не влияет на банкролл."
        )

    reason_counts = defaultdict(int)
    for r in watch:
        reason_counts[r.get("decision_reason") or "Причина не сохранена"] += 1
    reason_rows = [
        {"Причина": k, "N": v, "%": v / n * 100}
        for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1])
    ]
    st.markdown("**Почему WATCH**")
    df_reason = pd.DataFrame(reason_rows)
    st.dataframe(
        df_reason.assign(
            **{"%": df_reason["%"].map(lambda x: f"{x:.1f}%")}
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Разрез по Edge/EV показывает, где находится WATCH относительно порога.
    grouped = defaultdict(list)
    for r in watch:
        grouped[bucket(num(r.get("edge")))].append(r)

    bucket_rows = []
    for label, items in grouped.items():
        vals_edge = [num(x.get("edge")) for x in items if num(x.get("edge")) is not None]
        vals_ev = [num(x.get("ev")) for x in items if num(x.get("ev")) is not None]
        bucket_rows.append({
            "Edge": label,
            "N": len(items),
            "Avg Edge": sum(vals_edge) / len(vals_edge) * 100 if vals_edge else 0.0,
            "Avg EV": sum(vals_ev) / len(vals_ev) * 100 if vals_ev else 0.0,
            "Kelly > 0": (
                sum(1 for x in items if (num(x.get("kelly_pct")) or 0) > 0)
                / len(items) * 100
            ),
        })
    bucket_rows.sort(key=lambda x: ["<0%", "0–3%", "3–5%", "5–8%", "8%+", "нет данных"].index(x["Edge"])
                         if x["Edge"] in ["<0%", "0–3%", "3–5%", "5–8%", "8%+", "нет данных"] else 99)
    if bucket_rows:
        st.markdown("**WATCH по диапазону Edge**")
        df_b = pd.DataFrame(bucket_rows)
        st.dataframe(
            df_b.assign(
                **{
                    "Avg Edge": df_b["Avg Edge"].map(lambda x: f"{x:+.1f}%"),
                    "Avg EV": df_b["Avg EV"].map(lambda x: f"{x:+.1f}%"),
                    "Kelly > 0": df_b["Kelly > 0"].map(lambda x: f"{x:.1f}%"),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Последние WATCH**")
    latest = sorted(
        watch,
        key=lambda x: x.get("created_at") or "",
        reverse=True,
    )[:20]
    latest_rows = []
    for r in latest:
        latest_rows.append({
            "Дата": r.get("date_iso") or "—",
            "Матч": r.get("match_ru") or r.get("match") or "—",
            "Рынок": r.get("market") or "—",
            "Исход": r.get("pick") or "—",
            "Кэф": f"{num(r.get('market_odd')):.2f}" if num(r.get("market_odd")) else "—",
            "Edge": f"{num(r.get('edge')):+.1%}" if num(r.get("edge")) is not None else "—",
            "EV": f"{num(r.get('ev')):+.1%}" if num(r.get("ev")) is not None else "—",
            "Причина": r.get("decision_reason") or "—",
        })
    st.dataframe(latest_rows, use_container_width=True, hide_index=True)

    # Sensitivity: какой совместный Edge/EV threshold дал бы лучший
    # виртуальный результат на уже завершённых WATCH.
    if settled_watch:
        thresholds = [0.00, 0.01, 0.02, 0.03, 0.05]
        sens = []
        for t in thresholds:
            selected = [
                x for x in settled_watch
                if (float(x.get("edge") or 0) >= t)
                and (float(x.get("ev") or 0) >= t)
            ]
            pnl = sum(float(x.get("virtual_pnl") or 0) for x in selected)
            roi = pnl / len(selected) * 100 if selected else 0.0
            wins = sum(1 for x in selected if x.get("result_status") == "won")
            sens.append({
                "Edge + EV порог": f"{t:.0%}",
                "N": len(selected),
                "Win Rate": wins / len(selected) * 100 if selected else 0.0,
                "Virtual PnL": pnl,
                "ROI": roi,
            })
        st.markdown("**🧪 Чувствительность порога Edge + EV**")
        df_s = pd.DataFrame(sens)
        st.dataframe(
            df_s.assign(
                **{
                    "Win Rate": df_s["Win Rate"].map(lambda x: f"{x:.1f}%"),
                    "Virtual PnL": df_s["Virtual PnL"].map(lambda x: f"{x:+.2f}u"),
                    "ROI": df_s["ROI"].map(lambda x: f"{x:+.1f}%"),
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        reliable = [x for x in sens if x["N"] >= 10]
        if reliable:
            best = max(reliable, key=lambda x: x["ROI"])
            st.success(
                f"Лучший совместный порог при N≥10: {best['Edge + EV порог']} · "
                f"ROI {best['ROI']:+.1f}% · N={best['N']}"
            )
        else:
            st.caption("Для сравнения порогов пока нужна выборка минимум 10 завершённых WATCH.")

        # Независимая диагностика: Edge и EV по отдельности.
        # Это позволяет не делать ложный вывод, что оба метрика одинаково полезны.
        st.markdown("**📐 Edge отдельно vs EV отдельно**")
        edge_rows, ev_rows = [], []
        for t in thresholds:
            for label, key, target in (
                ("Edge", "edge", edge_rows),
                ("EV", "ev", ev_rows),
            ):
                selected = [
                    x for x in settled_watch
                    if float(x.get(key) or 0) >= t
                ]
                pnl = sum(float(x.get("virtual_pnl") or 0) for x in selected)
                roi = pnl / len(selected) * 100 if selected else 0.0
                wins = sum(1 for x in selected if x.get("result_status") == "won")
                target.append({
                    "Порог": f"{t:.0%}",
                    "N": len(selected),
                    "Win Rate": wins / len(selected) * 100 if selected else 0.0,
                    "Virtual PnL": pnl,
                    "ROI": roi,
                })

        col_e, col_v = st.columns(2)
        with col_e:
            st.caption("EDGE")
            df_e = pd.DataFrame(edge_rows)
            st.dataframe(
                df_e.assign(
                    **{
                        "Win Rate": df_e["Win Rate"].map(lambda x: f"{x:.1f}%"),
                        "Virtual PnL": df_e["Virtual PnL"].map(lambda x: f"{x:+.2f}u"),
                        "ROI": df_e["ROI"].map(lambda x: f"{x:+.1f}%"),
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        with col_v:
            st.caption("EV")
            df_v = pd.DataFrame(ev_rows)
            st.dataframe(
                df_v.assign(
                    **{
                        "Win Rate": df_v["Win Rate"].map(lambda x: f"{x:.1f}%"),
                        "Virtual PnL": df_v["Virtual PnL"].map(lambda x: f"{x:+.2f}u"),
                        "ROI": df_v["ROI"].map(lambda x: f"{x:+.1f}%"),
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

        edge_reliable = [x for x in edge_rows if x["N"] >= 10]
        ev_reliable = [x for x in ev_rows if x["N"] >= 10]
        if edge_reliable or ev_reliable:
            parts = []
            if edge_reliable:
                b = max(edge_reliable, key=lambda x: x["ROI"])
                parts.append(f"Edge {b['Порог']} → {b['ROI']:+.1f}% ROI (N={b['N']})")
            if ev_reliable:
                b = max(ev_reliable, key=lambda x: x["ROI"])
                parts.append(f"EV {b['Порог']} → {b['ROI']:+.1f}% ROI (N={b['N']})")
            st.info(" | ".join(parts))

        # Пересечение: высокий Edge при слабом EV и наоборот.
        cross = []
        for t in thresholds[1:]:
            e_only = [
                x for x in settled_watch
                if float(x.get("edge") or 0) >= t
                and float(x.get("ev") or 0) < t
            ]
            v_only = [
                x for x in settled_watch
                if float(x.get("ev") or 0) >= t
                and float(x.get("edge") or 0) < t
            ]
            cross.append({
                "Порог": f"{t:.0%}",
                "Edge≥T / EV<T": len(e_only),
                "EV≥T / Edge<T": len(v_only),
            })
        if cross:
            st.caption("Где метрики расходятся")
            st.dataframe(pd.DataFrame(cross), use_container_width=True, hide_index=True)


def _render_adaptive_monitor():
    """Мониторит, какие сегменты реально допускаются Adaptive WF-gate."""
    if not db.SQLITE_BOOT_OK:
        return
    try:
        from betting.adaptive import build_profiles, _segment_stats, _walk_forward_gate, _regime
        snapshots = db.fetch_decision_snapshots(limit=100000)
        profile = build_profiles(snapshots)
    except Exception:
        return

    rows = []
    for key, values in profile.items():
        if not key or key[0] != "market_league" or len(values) < 20:
            continue
        before = max(v[0] for v in values) if values else None
        stats = _segment_stats(values, before)
        wf_ok, wf_reason = _walk_forward_gate(values, before)
        regime, _, _ = _regime(values, before)
        if not stats:
            continue
        factor = 1.0
        if wf_ok:
            if stats["adj_roi"] >= 0.05 and stats["recent_roi"] >= 0: factor = 1.10
            elif stats["adj_roi"] >= 0.025 and stats["recent_roi"] >= 0: factor = 1.05
            elif stats["adj_roi"] <= -0.05 and stats["recent_roi"] <= 0: factor = 0.90
            elif stats["adj_roi"] <= -0.025 and stats["recent_roi"] <= 0: factor = 0.95
        rows.append({"Рынок": key[1], "Лига": key[2], "N": stats["n"],
                     "Adj ROI": stats["adj_roi"], "Recent ROI": stats["recent_roi"],
                     "WF": "PASS" if wf_ok else "HOLD", "Regime": regime,
                     "Factor": factor, "WF detail": wf_reason})

    st.subheader("🧠 Adaptive Monitor")
    if not rows:
        st.info("Нужно минимум 20 завершённых priced-сигналов в сегменте.")
        return
    df = pd.DataFrame(rows).sort_values(["Factor", "N"], ascending=[False, False])
    c1, c2, c3 = st.columns(3)
    c1.metric("Сегментов", len(df))
    c2.metric("WF PASS", int((df["WF"] == "PASS").sum()))
    c3.metric("Активных boost/penalty", int((df["Factor"] != 1.0).sum()))
    view = df.copy()
    view["Adj ROI"] = view["Adj ROI"].map(lambda x: f"{x:+.1%}")
    view["Recent ROI"] = view["Recent ROI"].map(lambda x: f"{x:+.1%}")
    view["Factor"] = view["Factor"].map(lambda x: f"x{x:.2f}")
    st.dataframe(view[["Рынок", "Лига", "N", "Adj ROI", "Recent ROI", "WF", "Regime", "Factor"]],
                 use_container_width=True, hide_index=True)
    st.caption("WF PASS разрешает только приоритизацию; BET/WATCH/SKIP остаются без изменений.")
def _render_adaptive_attribution():
    """Разделяет исторический результат на MODEL / VALUE / ADAPTIVE / REGIME."""
    if not db.SQLITE_BOOT_OK:
        return
    try:
        from betting.adaptive import build_profiles, priority
        snapshots = db.fetch_decision_snapshots(limit=100000)
        profile = build_profiles(snapshots)
    except Exception:
        return

    rows = []
    for r in snapshots:
        if str(r.get("result_status") or "").lower() not in ("won", "lost"):
            continue
        odd = float(r.get("market_odd") or 0.0)
        if odd <= 1.01:
            continue
        pnl = odd - 1.0 if str(r.get("result_status")).lower() == "won" else -1.0
        base = {
            "date_iso": r.get("date_iso"),
            "league": r.get("league"),
            "div": r.get("league"),
            "verdict": {"market": r.get("market")},
        }
        factor, reason = priority(profile, base)
        rows.append({
            "model_prob": float(r.get("model_prob") or 0.0),
            "edge": float(r.get("edge") or 0.0),
            "ev": float(r.get("ev") or 0.0),
            "pnl": pnl,
            "factor": factor,
            "reason": reason,
        })
    st.subheader("🧩 Adaptive Attribution")
    if not rows:
        st.info("Нужны закрытые priced decision snapshots.")
        return

    def _roi(items):
        return sum(x["pnl"] for x in items) / len(items) if items else 0.0

    base = rows
    boosted = [x for x in rows if x["factor"] > 1.0]
    penalized = [x for x in rows if x["factor"] < 1.0]
    neutral = [x for x in rows if x["factor"] == 1.0]
    high_value = [x for x in rows if x["edge"] >= 0.03 and x["ev"] >= 0.03]
    model_strong = [x for x in rows if x["model_prob"] >= 0.60]

    all_roi = _roi(base)
    neutral_roi = _roi(neutral)
    summary = [
        {"Слой": "ALL priced", "N": len(base), "ROI": all_roi, "Δ vs neutral": 0.0},
        {"Слой": "MODEL P≥60%", "N": len(model_strong), "ROI": _roi(model_strong), "Δ vs neutral": _roi(model_strong) - neutral_roi},
        {"Слой": "VALUE Edge≥3% + EV≥3%", "N": len(high_value), "ROI": _roi(high_value), "Δ vs neutral": _roi(high_value) - neutral_roi},
        {"Слой": "ADAPTIVE boost", "N": len(boosted), "ROI": _roi(boosted), "Δ vs neutral": _roi(boosted) - neutral_roi},
        {"Слой": "ADAPTIVE penalty", "N": len(penalized), "ROI": _roi(penalized), "Δ vs neutral": _roi(penalized) - neutral_roi},
        {"Слой": "ADAPTIVE neutral", "N": len(neutral), "ROI": neutral_roi, "Δ vs neutral": 0.0},
    ]
    df = pd.DataFrame(summary)
    df["ROI"] = df["ROI"].map(lambda x: f"{x:+.1%}")
    df["Δ vs neutral"] = df["Δ vs neutral"].map(lambda x: f"{x:+.1%}")
    st.dataframe(df, use_container_width=True, hide_index=True)

    if boosted or penalized:
        st.caption(
            "Attribution — диагностический анализ: Adaptive не меняет результат ставки задним числом; "
            "группы показывают, в каких исторических сегментах система применяла приоритет."
        )

def _render_decision_log(D):
    """Показывает сохранённый снимок решения для закрытых ставок."""
    bets = _closed_bets(D)
    rows = []
    for b in bets:
        snap = b.get("decision_snapshot") or {}
        if not snap and b.get("decision"):
            snap = {"model_prob": b.get("prob"), "market_odd": b.get("odds"), "ev": b.get("ev"), "market": b.get("market")}
        rows.append({
            "Дата": b.get("date") or "—",
            "Матч": b.get("match_ru") or b.get("match") or "—",
            "Рынок": snap.get("market") or b.get("market") or "—",
            "Решение": b.get("decision") or "BET",
            "P": f"{float(snap.get('model_prob') or b.get('prob') or 0):.1%}",
            "Fair": f"{float(snap.get('fair_odd') or 0):.2f}" if snap.get("fair_odd") else "—",
            "Кэф": f"{float(snap.get('market_odd') or b.get('odds') or 0):.2f}",
            "Edge": f"{float(snap.get('edge') or 0):+.1%}" if snap.get("edge") is not None else "—",
            "EV": f"{float(snap.get('ev') or b.get('ev') or 0):+.1%}" if (snap.get("ev") is not None or b.get("ev") is not None) else "—",
            "Kelly": f"{float(snap.get('kelly_pct') or 0):.1%}",
            "Conf": f"{float(snap.get('confidence') or 0):.1%}" if snap.get("confidence") is not None else "—",
            "Value": f"{float(snap.get('value_score') or 0):.2f}",
            "Результат": b.get("status", "—").upper(),
        })
    st.subheader("🧾 Decision Log")
    if not rows:
        st.info("Decision Log появится после первых закрытых ставок. Новые ставки сохраняют снимок решения BET на момент входа.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption("Снимок фиксируется в момент ставки и не пересчитывается задним числом. Старые ставки без snapshot показываются с доступными полями.")

def render():
    D = st.session_state.data
    _render_decision_log(D)
    _render_adaptive_attribution()
    _render_adaptive_monitor()
    st.header("📈 Статистика")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 Обзор", "🧪 Модель", "🏆 По лигам", "📅 По дням", "🎯 По рынкам", "🟡 WATCH LAB"])
    with tab1:
        _render_overview(D)
    with tab2:
        _render_model(D)
    with tab3:
        _render_by_league(D)
    with tab4:
        _render_by_day(D)
    with tab5:
        _render_by_market(D)
    with tab6:
        _render_watch_lab(D)

    if db.SQLITE_BOOT_OK:
        st.divider()
        st.subheader("🗄 CLV / Drawdown / Sharpe")
        try:
            clv = db.clv_summary()
            ib = float(D.get("meta", {}).get("initial_bank", 10000.0))
            hist = db.bank_history(limit=100000)
            banks = [ib] + [float(h.get("bank") or 0)
                            for h in hist if h.get("bank") is not None]
            max_dd = 0.0
            if len(banks) >= 2:
                peak = banks[0]
                for b in banks:
                    if b > peak:
                        peak = b
                    if peak > 0:
                        max_dd = max(max_dd, (peak - b) / peak)
            rets = []
            for i in range(1, len(banks)):
                if banks[i - 1] > 0:
                    rets.append((banks[i] - banks[i - 1]) / banks[i - 1])
            sharpe = _sharpe(rets)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("CLV avg", f"{clv.get('avg_clv', 0)*100:+.2f}%")
            c2.metric("CLV N", clv.get("n", 0))
            c3.metric("CLV +", f"{clv.get('positive_share', 0)*100:.1f}%")
            c4.metric("Max DD", f"-{max_dd*100:.1f}%")
            c5.metric("Sharpe", f"{sharpe:.2f}")

            st.caption(
                "CLV показывает изменение цены между входом и закрытием рынка; "
                "положительная доля — процент ставок с CLV > 0."
            )
            try:
                import pandas as pd
                market_rows = db.clv_breakdown("market")
                league_rows = db.clv_breakdown("league")
                if market_rows or league_rows:
                    st.subheader("🔎 CLV по рынкам и лигам")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Рынки**")
                        df_m = pd.DataFrame(market_rows)
                        if not df_m.empty:
                            df_m = df_m.rename(columns={
                                "group": "Рынок", "n": "N",
                                "avg_clv": "CLV avg",
                                "positive_share": "CLV +"
                            })
                            df_m["CLV avg"] = df_m["CLV avg"].map(lambda x: f"{x*100:+.2f}%")
                            df_m["CLV +"] = df_m["CLV +"].map(lambda x: f"{x*100:.1f}%")
                            st.dataframe(df_m, use_container_width=True, hide_index=True)
                    with c2:
                        st.markdown("**Лиги**")
                        df_l = pd.DataFrame(league_rows)
                        if not df_l.empty:
                            df_l = df_l.rename(columns={
                                "group": "Лига", "n": "N",
                                "avg_clv": "CLV avg",
                                "positive_share": "CLV +"
                            })
                            df_l["CLV avg"] = df_l["CLV avg"].map(lambda x: f"{x*100:+.2f}%")
                            df_l["CLV +"] = df_l["CLV +"].map(lambda x: f"{x*100:.1f}%")
                            st.dataframe(df_l, use_container_width=True, hide_index=True)
            except Exception:
                pass
        except Exception:
            pass
