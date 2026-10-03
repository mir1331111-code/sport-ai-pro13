"""ui/tabs/stats.py — Статистика по СВОИМ ставкам."""
from __future__ import annotations
from collections import defaultdict

import streamlit as st

from storage import sqlite_store as db


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


def render():
    D = st.session_state.data
    st.header("📈 Статистика")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Обзор", "🧪 Модель", "🏆 По лигам", "📅 По дням", "🎯 По рынкам"])
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
