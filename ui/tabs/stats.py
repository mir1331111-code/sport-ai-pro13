"""ui/tabs/stats.py — вкладка Статистика."""
from __future__ import annotations
import streamlit as st

from storage import sqlite_store as db


def _drawdown(initial_bank, history):
    banks = [initial_bank] + [h["bank"] for h in history
                              if h.get("bank") is not None]
    if len(banks) < 2:
        return {"peak": initial_bank, "max_dd_pct": 0.0, "current_dd_pct": 0.0}
    peak = banks[0]
    max_dd = 0.0
    for b in banks:
        if b > peak:
            peak = b
        if peak > 0:
            max_dd = max(max_dd, (peak - b) / peak)
    cur_peak = max(banks)
    cur_dd = (cur_peak - banks[-1]) / cur_peak if cur_peak > 0 else 0.0
    return {"peak": cur_peak, "max_dd_pct": max_dd, "current_dd_pct": cur_dd}


def _sharpe(initial_bank, history, periods_per_year=252):
    banks = [initial_bank] + [h["bank"] for h in history
                              if h.get("bank") is not None]
    if len(banks) < 3:
        return 0.0
    rets = []
    for i in range(1, len(banks)):
        prev = banks[i - 1]
        if prev > 0:
            rets.append((banks[i] - prev) / prev)
    if not rets:
        return 0.0
    mu = sum(rets) / len(rets)
    var = sum((r - mu) ** 2 for r in rets) / len(rets)
    sigma = var ** 0.5
    if sigma <= 1e-12:
        return 0.0
    return (mu / sigma) * (periods_per_year ** 0.5)


def render():
    D = st.session_state.data
    st.header("Статистика")
    s = D["stats"]
    tot = s["won"] + s["lost"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Банк", f"{D['bank']:.2f}")
    m2.metric("Ставок всего", len(D["bets"]))
    m3.metric("WinRate", f"{(s['won']/tot*100) if tot else 0:.1f}%")
    m4.metric("Profit", f"{s['profit']:+.2f}")

    if db.SQLITE_BOOT_OK:
        st.divider()
        st.subheader("SQLite + CLV + Drawdown")
        bets = db.fetch_bets(limit=100000)
        if bets:
            st.caption(f"SQLite: {len(bets)} ставок")

        real_bets = [b for b in bets if b.get("mode") == "real"]
        paper_bets = [b for b in bets if b.get("mode") != "real"]
        c1, c2 = st.columns(2)
        c1.metric("Real-ставок", len(real_bets))
        c2.metric("Paper-ставок", len(paper_bets))

        clv = db.clv_summary()
        ib = float(D.get("meta", {}).get("initial_bank", 10000.0))
        hist = db.bank_history(limit=100000)
        dd = _drawdown(ib, hist)
        shp = _sharpe(ib, hist)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CLV avg", f"{clv['avg_clv']*100:+.2f}%")
        c2.metric("CLV N", clv["n"])
        c3.metric("Max DD", f"-{dd['max_dd_pct']*100:.1f}%")
        c4.metric("Sharpe", f"{shp:.2f}")

        hist_chart = db.bank_history(limit=5000)
        if len(hist_chart) > 1:
            try:
                import pandas as pd
                df = pd.DataFrame(hist_chart)
                st.line_chart(df.set_index("ts")["bank"])
            except Exception:
                pass
