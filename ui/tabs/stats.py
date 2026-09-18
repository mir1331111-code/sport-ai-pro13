def render():
    D = st.session_state.data
    st.header("Статистика")
    s = D.get("stats", {})
    won = int(s.get("won", 0))
    lost = int(s.get("lost", 0))
    tot = won + lost
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Банк", f"{D.get('bank', 0):.2f}")
    m2.metric("Ставок всего", len(D.get("bets", [])))
    m3.metric("WinRate", f"{(won/tot*100) if tot else 0:.1f}%")
    m4.metric("Profit", f"{float(s.get('profit', 0)):+.2f}")

    if not db.SQLITE_BOOT_OK:
        return
    st.divider()
    st.subheader("SQLite + CLV + Drawdown")

    try:
        bets = db.fetch_bets(limit=100000)
    except Exception:
        bets = []
    if bets:
        st.caption(f"SQLite: {len(bets)} ставок")

    real_bets = [b for b in bets if b.get("mode") == "real"]
    paper_bets = [b for b in bets if b.get("mode") != "real"]
    c1, c2 = st.columns(2)
    c1.metric("Real-ставок", len(real_bets))
    c2.metric("Paper-ставок", len(paper_bets))

    try:
        clv = db.clv_summary()
    except Exception:
        clv = {"n": 0, "avg_clv": 0.0, "positive_share": 0.0}
    ib = float(D.get("meta", {}).get("initial_bank", 10000.0))
    try:
        hist = db.bank_history(limit=100000)
    except Exception:
        hist = []

    # Drawdown
    banks = [ib] + [float(h.get("bank") or 0) for h in hist if h.get("bank") is not None]
    max_dd_pct = 0.0
    if len(banks) >= 2:
        peak = banks[0]
        for b in banks:
            if b > peak:
                peak = b
            if peak > 0:
                max_dd_pct = max(max_dd_pct, (peak - b) / peak)
        cur_peak = max(banks)
        cur_dd_pct = (cur_peak - banks[-1]) / cur_peak if cur_peak > 0 else 0.0
    else:
        cur_dd_pct = 0.0

    # Sharpe
    sharpe = 0.0
    if len(banks) >= 3:
        rets = []
        for i in range(1, len(banks)):
            if banks[i - 1] > 0:
                rets.append((banks[i] - banks[i - 1]) / banks[i - 1])
        if rets:
            mu = sum(rets) / len(rets)
            var = sum((r - mu) ** 2 for r in rets) / len(rets)
            sigma = var ** 0.5
            if sigma > 1e-12:
                sharpe = (mu / sigma) * (252 ** 0.5)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CLV avg", f"{clv.get('avg_clv', 0)*100:+.2f}%")
    c2.metric("CLV N", clv.get("n", 0))
    c3.metric("Max DD", f"-{max_dd_pct*100:.1f}%")
    c4.metric("Sharpe", f"{sharpe:.2f}")

    if len(hist) > 1:
        try:
            import pandas as pd
            df = pd.DataFrame(hist)
            st.line_chart(df.set_index("ts")["bank"])
        except Exception:
            pass
            
