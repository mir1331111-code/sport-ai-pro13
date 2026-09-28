"""ui/tabs/backtest.py — walk-forward бэктест."""
from __future__ import annotations

import streamlit as st

from config import DIV_NAMES
from data.sources import load_seasonal, parse_date
from model.engine import Engine
from betting.kelly import edge, expected_value, kelly


def _f(value):
    try:
        if value in (None, ""):
            return None

        value = float(value)

        if value <= 0:
            return None

        return value

    except (TypeError, ValueError):
        return None


def _result(pick: str, home_goals: float, away_goals: float) -> bool:
    if pick == "П1":
        return home_goals > away_goals

    if pick == "X":
        return home_goals == away_goals

    if pick == "П2":
        return home_goals < away_goals

    return False


def _max_drawdown(equity_curve):
    if not equity_curve:
        return 0.0

    peak = equity_curve[0]
    max_dd = 0.0

    for value in equity_curve:
        if value > peak:
            peak = value

        if peak > 0:
            dd = (peak - value) / peak

            if dd > max_dd:
                max_dd = dd

    return max_dd


def _pick_historical_odds(row):
    """
    Берём реальные исторические букмекерские odds.

    Приоритет:
    Pinnacle -> Bet365.

    Возвращаем:
        {
            "П1": ...,
            "X": ...,
            "П2": ...
        }
    """

    p1 = (
        _f(row.get("PSH"))
        or _f(row.get("B365H"))
    )

    draw = (
        _f(row.get("PSD"))
        or _f(row.get("B365D"))
    )

    p2 = (
        _f(row.get("PSA"))
        or _f(row.get("B365A"))
    )

    return {
        "П1": p1,
        "X": draw,
        "П2": p2,
    }


def _best_signal(P, odds, min_edge, min_ev):
    """
    На один матч разрешаем только ОДИН сигнал.

    Выбирается сигнал с максимальным EV,
    при условии:
        edge >= min_edge
        EV >= min_ev
        odds существуют.
    """

    candidates = []

    markets = (
        ("П1", P.get("p1")),
        ("X", P.get("x")),
        ("П2", P.get("p2")),
    )

    for pick, prob in markets:
        odd = odds.get(pick)

        if prob is None or odd is None:
            continue

        try:
            prob = float(prob)
            odd = float(odd)
        except (TypeError, ValueError):
            continue

        if not (0.0 < prob < 1.0):
            continue

        if odd <= 1.01:
            continue

        market_edge = edge(prob, odd)
        ev = expected_value(prob, odd)

        if market_edge < min_edge:
            continue

        if ev < min_ev:
            continue

        candidates.append(
            {
                "pick": pick,
                "prob": prob,
                "odd": odd,
                "edge": market_edge,
                "ev": ev,
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda x: (
            x["ev"],
            x["edge"],
            x["prob"],
        ),
        reverse=True,
    )

    return candidates[0]


def render(matrix_n):
    st.header("Бэктест (walk-forward)")

    st.caption(
        "Исторические букмекерские коэффициенты. "
        "Модель не видит результат матча до формирования сигнала."
    )

    b1, b2, b3, b4 = st.columns(4)

    bt_div = b1.selectbox(
        "Лига",
        list(DIV_NAMES.keys()),
        format_func=lambda k: DIV_NAMES[k],
    )

    bt_season = b2.selectbox(
        "Сезон",
        [
            "2526",
            "2425",
            "2324",
        ],
        index=1,
    )

    bt_edge = b3.slider(
        "Мин. edge",
        0.00,
        0.15,
        0.03,
        0.01,
    )

    bt_ev = b4.slider(
        "Мин. EV",
        0.00,
        0.20,
        0.03,
        0.01,
    )

    b5, b6, b7 = st.columns(3)

    bt_mode = b5.selectbox(
        "Стейк",
        [
            "Flat",
            "Kelly",
        ],
    )

    flat_pct = b6.slider(
        "Flat % банка",
        0.25,
        3.00,
        1.00,
        0.25,
        disabled=bt_mode != "Flat",
    )

    kelly_frac = b7.slider(
        "Kelly fraction",
        0.05,
        1.00,
        0.25,
        0.05,
        disabled=bt_mode != "Kelly",
    )

    if not st.button(
        "Прогнать",
        type="primary",
    ):
        return

    # ================================================================
    # Загрузка истории
    # ================================================================

    rows_all = load_seasonal(
        bt_div,
        bt_season,
    )

    rows_all = [
        r
        for r in rows_all
        if r.get("FTHG") not in (None, "")
        and r.get("FTAG") not in (None, "")
        and parse_date(
            r.get("Date", "")
        )
    ]

    rows_all.sort(
        key=lambda r: parse_date(
            r.get("Date", "")
        )
    )

    if len(rows_all) < 150:
        st.error(
            "Мало матчей для бэктеста. "
            "Нужно минимум 150."
        )
        return

    # ================================================================
    # Параметры
    # ================================================================

    INITIAL_BANK = 10000.0
    MIN_TRAIN_MATCHES = 120

    engine = Engine(
        matrix_n=matrix_n
    )

    bank = INITIAL_BANK

    log = []
    equity_curve = [
        bank
    ]

    peak_bank = bank

    progress = st.progress(
        0.0
    )

    total = len(rows_all)

    # ================================================================
    # WALK-FORWARD
    # ================================================================

    for j, r in enumerate(rows_all):
        h = (
            r.get("HomeTeam")
            or ""
        ).strip()

        a = (
            r.get("AwayTeam")
            or ""
        ).strip()

        if not h or not a:
            continue

        try:
            hg = float(
                r["FTHG"]
            )

            ag = float(
                r["FTAG"]
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        md = parse_date(
            r.get("Date", "")
        )

        if not md:
            continue

        # ------------------------------------------------------------
        # После начального training window начинаем тестировать.
        #
        # ВАЖНО:
        # predict вызывается ДО learn_step текущего матча.
        # Значит результат текущего матча не попадает в модель.
        # ------------------------------------------------------------

        if j >= MIN_TRAIN_MATCHES:
            try:
                P = engine.predict(
                    h,
                    a,
                    bt_div,
                    match_date=md,
                    cup=bt_div in (
                        "C1",
                        "EL",
                        "EC",
                    ),
                )

            except Exception:
                P = None

            if P:
                historical_odds = (
                    _pick_historical_odds(r)
                )

                signal = _best_signal(
                    P,
                    historical_odds,
                    bt_edge,
                    bt_ev,
                )

                if signal is not None:
                    pick = signal["pick"]
                    prob = signal["prob"]
                    odd = signal["odd"]
                    market_edge = signal["edge"]
                    ev = signal["ev"]

                    # ------------------------------------------------
                    # Размер ставки
                    # ------------------------------------------------

                    if bt_mode == "Kelly":
                        stake = kelly(
                            prob=prob,
                            odds=odd,
                            bank=bank,
                            frac=kelly_frac,
                            cap=0.05,
                        )

                    else:
                        stake = round(
                            bank
                            * (
                                flat_pct
                                / 100.0
                            ),
                            2,
                        )

                    # Без отрицательных/нулевых ставок.
                    stake = max(
                        0.0,
                        float(stake),
                    )

                    # Абсолютный safety cap:
                    # максимум 5% банка.
                    stake = round(
                        min(
                            stake,
                            bank * 0.05,
                        ),
                        2,
                    )

                    if stake > 0:
                        won = _result(
                            pick,
                            hg,
                            ag,
                        )

                        if won:
                            pnl = (
                                stake
                                * (odd - 1.0)
                            )

                        else:
                            pnl = -stake

                        bank += pnl

                        equity_curve.append(
                            bank
                        )

                        peak_bank = max(
                            peak_bank,
                            bank,
                        )

                        log.append(
                            {
                                "date": md.strftime(
                                    "%Y-%m-%d"
                                ),
                                "home": h,
                                "away": a,
                                "pick": pick,
                                "prob": round(
                                    prob,
                                    4,
                                ),
                                "odd": round(
                                    odd,
                                    3,
                                ),
                                "edge": round(
                                    market_edge,
                                    4,
                                ),
                                "ev": round(
                                    ev,
                                    4,
                                ),
                                "stake": round(
                                    stake,
                                    2,
                                ),
                                "won": won,
                                "pnl": round(
                                    pnl,
                                    2,
                                ),
                                "bank": round(
                                    bank,
                                    2,
                                ),
                            }
                        )

        # ------------------------------------------------------------
        # После прогноза текущий матч добавляется в модель.
        # ------------------------------------------------------------

        try:
            engine.learn_step(
                h,
                a,
                hg,
                ag,
                r,
                lg=bt_div,
                match_num=j,
                total=total,
                match_date=md,
            )

        except Exception:
            pass

        if j % 25 == 0:
            progress.progress(
                min(
                    1.0,
                    j / max(
                        1,
                        total,
                    ),
                )
            )

    progress.progress(1.0)

    # ================================================================
    # Результаты
    # ================================================================

    if not log:
        st.warning(
            "Сигналов нет. "
            "Попробуй уменьшить min edge / min EV."
        )
        return

    n = len(log)

    wins = sum(
        1
        for x in log
        if x["won"]
    )

    losses = n - wins

    profit = sum(
        x["pnl"]
        for x in log
    )

    staked = sum(
        x["stake"]
        for x in log
    )

    roi = (
        profit / staked * 100.0
        if staked > 0
        else 0.0
    )

    win_rate = (
        wins / n * 100.0
        if n > 0
        else 0.0
    )

    max_dd = (
        _max_drawdown(
            equity_curve
        )
        * 100.0
    )

    avg_ev = (
        sum(
            x["ev"]
            for x in log
        )
        / n
        if n
        else 0.0
    )

    avg_edge = (
        sum(
            x["edge"]
            for x in log
        )
        / n
        if n
        else 0.0
    )

    # ================================================================
    # Верхние метрики
    # ================================================================

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Ставок",
        n,
    )

    k2.metric(
        "WinRate",
        f"{win_rate:.1f}%",
    )

    k3.metric(
        "PnL",
        f"{profit:+.2f}",
    )

    k4.metric(
        "ROI",
        f"{roi:+.2f}%",
    )

    k5, k6, k7, k8 = st.columns(4)

    k5.metric(
        "Банк",
        f"{bank:.2f}",
    )

    k6.metric(
        "Max Drawdown",
        f"{max_dd:.2f}%",
    )

    k7.metric(
        "Средний Edge",
        f"{avg_edge * 100:.2f}%",
    )

    k8.metric(
        "Средний EV",
        f"{avg_ev * 100:.2f}%",
    )

    # ================================================================
    # Дополнительная статистика
    # ================================================================

    st.caption(
        f"Побед: {wins} · "
        f"Поражений: {losses} · "
        f"Оборот: {staked:.2f} · "
        f"Начальный банк: {INITIAL_BANK:.2f} · "
        f"Финальный банк: {bank:.2f}"
    )

    # ================================================================
    # Таблица
    # ================================================================

    st.subheader(
        "Последние ставки"
    )

    st.dataframe(
        log[-50:],
        use_container_width=True,
        hide_index=True,
    )

    # ================================================================
    # Кривая банка
    # ================================================================

    st.subheader(
        "Кривая банка"
    )

    st.line_chart(
        equity_curve,
        height=300,
    )
