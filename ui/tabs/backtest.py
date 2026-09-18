"""ui/tabs/backtest.py — walk-forward бэктест."""
from __future__ import annotations
import streamlit as st

from config import DIV_NAMES
from data.sources import load_seasonal, parse_date
from model.engine import Engine
from betting.kelly import kelly


def _f(v):
    try:
        return float(v)
    except Exception:
        return None


def render(matrix_n):
    st.header("Бэктест (walk-forward)")
    st.caption("Источник: football-data.co.uk. EV против Pinnacle/B365.")
    b1, b2, b3, b4 = st.columns(4)
    bt_div = b1.selectbox("Лига", list(DIV_NAMES.keys()),
                          format_func=lambda k: DIV_NAMES[k])
    bt_season = b2.selectbox("Сезон", ["2526", "2425", "2324"], index=1)
    bt_edge = b3.slider("Мин. edge", 0.00, 0.15, 0.03, 0.01)
    bt_mode = b4.selectbox("Стейк", ["Flat", "Kelly"])

    if not st.button("Прогнать", type="primary"):
        return
    rows_all = load_seasonal(bt_div, bt_season)
    rows_all = [r for r in rows_all
                if r.get("FTHG") not in (None, "")
                and parse_date(r.get("Date", ""))]
    rows_all.sort(key=lambda r: parse_date(r.get("Date", "")))
    if len(rows_all) < 150:
        st.error("Мало матчей для бэктеста.")
        return

    engine = Engine(matrix_n=matrix_n)
    log = []
    bank = 10000.0
    prog = st.progress(0.0)
    total = len(rows_all)
    for j, r in enumerate(rows_all):
        h = (r.get("HomeTeam") or "").strip()
        a = (r.get("AwayTeam") or "").strip()
        hg, ag = float(r["FTHG"]), float(r["FTAG"])
        md = parse_date(r.get("Date", ""))
        if j >= 120:
            P = engine.predict(h, a, bt_div, match_date=md,
                               cup=bt_div in ("C1", "EL", "EC"))
            for pick, prob, odd in (
                ("П1", P["p1"], _f(r.get("PSH")) or _f(r.get("B365H"))),
                ("X", P["x"], _f(r.get("PSD")) or _f(r.get("B365D"))),
                ("П2", P["p2"], _f(r.get("PSA")) or _f(r.get("B365A"))),
            ):
                if not odd or odd <= 1.01:
                    continue
                edge = prob - 1.0 / odd
                if edge < bt_edge:
                    continue
                stake = (kelly(prob, odd, bank, 0.25) if bt_mode == "Kelly"
                         else round(bank * 0.01, 2))
                if stake <= 0:
                    continue
                won = ((pick == "П1" and hg > ag) or
                       (pick == "X" and hg == ag) or
                       (pick == "П2" and hg < ag))
                pnl = stake * (odd - 1) if won else -stake
                bank += pnl
                log.append({"pick": pick, "prob": round(prob, 3),
                            "odd": odd, "edge": round(edge, 3),
                            "stake": stake, "won": won, "pnl": round(pnl, 2)})
        engine.learn_step(h, a, hg, ag, r, lg=bt_div,
                          match_num=j, total=total, match_date=md)
        if j % 25 == 0:
            prog.progress(j / total)
    prog.progress(1.0)

    if not log:
        st.warning("Сигналов нет (edge выше порога не встретился).")
        return
    n = len(log)
    wins = sum(1 for x in log if x["won"])
    profit = sum(x["pnl"] for x in log)
    staked = sum(x["stake"] for x in log)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ставок", n)
    k2.metric("WinRate", f"{wins/n*100:.1f}%")
    k3.metric("PnL", f"{profit:+.1f}")
    k4.metric("ROI", f"{profit/staked*100:+.2f}%" if staked else "0.00%")
    st.dataframe(log[-30:], use_container_width=True, hide_index=True)
