"""ui/tabs/backtest.py — leakage-safe, risk-capped walk-forward backtest."""
from __future__ import annotations

import math
from streamlit import runtime
import streamlit as st

from config import DIV_NAMES
from data.sources import load_seasonal
from model.engine import Engine
from model.training import (
    iter_calendar_day_batches,
    prepare_completed_training_rows,
)
from betting.kelly import edge, expected_value, kelly
from betting.risk import (
    KELLY_FRACTION,
    MAX_DAILY_EXPOSURE_PCT,
    MAX_MATCH_EXPOSURE_PCT,
    MAX_MARKET_STAKE_PCT,
    size_bet,
)


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
        MAX_MARKET_STAKE_PCT * 100.0,
        min(1.00, MAX_MARKET_STAKE_PCT * 100.0),
        0.25,
        disabled=bt_mode != "Flat",
    )

    kelly_frac = b7.slider(
        "Kelly fraction",
        0.05,
        KELLY_FRACTION,
        KELLY_FRACTION,
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

    raw_rows = load_seasonal(
        bt_div,
        bt_season,
    )
    prepared_rows, skipped_rows = prepare_completed_training_rows(
        {bt_div: [raw_rows]},
        (bt_div,),
    )

    if len(prepared_rows) < 150:
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

    total = len(prepared_rows)
    processed = 0

    # ================================================================
    # WALK-FORWARD — all fixtures from one calendar day are forecast
    # from the same pre-day model state before any result is ingested.
    # ================================================================

    for day_batch in iter_calendar_day_batches(prepared_rows):
        day_start_bank = bank
        day_date = day_batch[0]["match_date"].date()
        day_stamp = datetime.combine(day_date, datetime.min.time())
        candidates = []

        # Phase 1: generate every same-day forecast without learning any
        # result from that day. CSV row order is not a reliable kickoff order.
        for offset, item in enumerate(day_batch):
            index = processed + offset
            h = item["home"]
            a = item["away"]
            hg = item["home_goals"]
            ag = item["away_goals"]
            md = item["match_date"]
            row = item["row"]

            if index < MIN_TRAIN_MATCHES:
                continue

            try:
                prediction = engine.predict(
                    h,
                    a,
                    bt_div,
                    match_date=md,
                    cup=bt_div in ("C1", "EL", "EC"),
                )
            except Exception:
                prediction = None

            if not prediction:
                continue

            signal = _best_signal(
                prediction,
                _pick_historical_odds(row),
                bt_edge,
                bt_ev,
            )
            if signal is None:
                continue

            fixture_id = (
                f"backtest:{day_date.isoformat()}:{bt_div}:"
                f"{h.casefold()}:{a.casefold()}"
            )
            candidates.append({
                "date": day_date.isoformat(),
                "home": h,
                "away": a,
                "home_goals": hg,
                "away_goals": ag,
                "fixture_id": fixture_id,
                "match": f"{h} vs {a}",
                "pick": signal["pick"],
                "prob": signal["prob"],
                "odd": signal["odd"],
                "edge": signal["edge"],
                "ev": signal["ev"],
            })

        # Phase 2: size positions against one opening-of-day bankroll.
        # The central size_bet() policy applies quarter Kelly, 1% market,
        # 2% match and 5% daily exposure limits to this simulated portfolio.
        # Strongest same-day EV candidates receive the limited daily room first;
        # same-day outcomes cannot change another fixture's stake.
        virtual_day_bets = []
        day_free_bank = day_start_bank
        allocated = []
        for candidate in sorted(
            candidates,
            key=lambda item: (
                item["ev"], item["edge"], item["prob"],
                item["home"].casefold(), item["away"].casefold(),
            ),
            reverse=True,
        ):
            _, risk = size_bet(
                probability=candidate["prob"],
                odds=candidate["odd"],
                bank=day_free_bank,
                bets=virtual_day_bets,
                fixture_id=candidate["fixture_id"],
                match=candidate["match"],
                date_iso=candidate["date"],
                market="1X2",
                pick=candidate["pick"],
                division=bt_div,
                now=day_stamp,
            )
            if risk.get("reason") not in ("approved", "no_positive_quarter_kelly"):
                continue

            if bt_mode == "Kelly":
                requested_stake = kelly(
                    prob=candidate["prob"],
                    odds=candidate["odd"],
                    bank=day_start_bank,
                    frac=min(float(kelly_frac), KELLY_FRACTION),
                    cap=MAX_MARKET_STAKE_PCT,
                )
            else:
                requested_stake = day_start_bank * (float(flat_pct) / 100.0)

            room = min(
                max(0.0, float(risk.get("market_limit", 0.0)) - float(risk.get("market_used", 0.0))),
                max(0.0, float(risk.get("match_limit", 0.0)) - float(risk.get("match_used", 0.0))),
                max(0.0, float(risk.get("daily_limit", 0.0)) - float(risk.get("daily_used", 0.0))),
                max(0.0, day_free_bank),
            )
            stake = math.floor(max(0.0, min(requested_stake, room)) * 100.0 + 1e-9) / 100.0
            if stake <= 0.0:
                continue

            virtual_day_bets.append({
                "fixture_id": candidate["fixture_id"],
                "match": candidate["match"],
                "date_iso": candidate["date"],
                "div": bt_div,
                "market": "1X2",
                "pick": candidate["pick"],
                "stake": stake,
                "status": "pending",
                "placed_at": day_stamp.isoformat(),
            })
            day_free_bank = max(0.0, day_free_bank - stake)
            allocated.append({**candidate, "stake": stake})

        # Phase 3: settle the day's simulated positions together. Equity is
        # marked once per day, avoiding arbitrary same-day drawdown ordering.
        day_pnl = 0.0
        settled_day = []
        for candidate in allocated:
            won = _result(
                candidate["pick"],
                candidate["home_goals"],
                candidate["away_goals"],
            )
            pnl = (
                candidate["stake"] * (candidate["odd"] - 1.0)
                if won else -candidate["stake"]
            )
            day_pnl += pnl
            settled_day.append({**candidate, "won": won, "pnl": pnl})

        bank = day_start_bank + day_pnl
        if settled_day:
            equity_curve.append(bank)
            peak_bank = max(peak_bank, bank)
            for candidate in settled_day:
                log.append({
                    "date": candidate["date"],
                    "home": candidate["home"],
                    "away": candidate["away"],
                    "pick": candidate["pick"],
                    "prob": round(candidate["prob"], 4),
                    "odd": round(candidate["odd"], 3),
                    "edge": round(candidate["edge"], 4),
                    "ev": round(candidate["ev"], 4),
                    "stake": round(candidate["stake"], 2),
                    "won": candidate["won"],
                    "pnl": round(candidate["pnl"], 2),
                    "bank": round(bank, 2),
                })

        # Phase 4: only now let any score from this day enter the model.
        try:
            engine.learn_batch(day_batch)
        except Exception as exc:
            st.error(
                f"Обучение остановлено после {day_date.isoformat()}: "
                f"{type(exc).__name__}. Неполная хронология не продолжена."
            )
            return

        processed += len(day_batch)
        if processed == total or processed - (processed // 25 * 25) < len(day_batch):
            progress.progress(min(1.0, processed / max(1, total)))

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
        width="stretch",
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
