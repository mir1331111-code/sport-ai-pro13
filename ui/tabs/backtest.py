"""ui/tabs/backtest.py — walk-forward бэктест."""
from __future__ import annotations

from datetime import date
import math

import streamlit as st

from config import DIV_NAMES
from data.sources import load_seasonal, parse_date
from model.engine import Engine
from betting.kelly import edge, expected_value, kelly
from model.training_data import completed_training_row
from model.evaluation import reliability_table, expected_calibration_error
from betting.historical_odds import pick_historical_odds as _pick_historical_odds


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
    st.warning(
        "Ограничение данных: в исторических колонках нет подтверждённого времени "
        "фиксации коэффициента. Бэктест не доказывает, что цена была доступна "
        "в момент ставки; виртуальный PnL следует считать оценкой, а не "
        "гарантируемым исполнимым результатом."
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

    # Use the same strict validation as the main training path: missing,
    # fractional, non-finite, impossible, and future scores are not results.
    rows_all = [
        r
        for r in rows_all
        if completed_training_row(
            r,
            parse_date(r.get("Date", "")),
            date.today(),
        ) is not None
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

    # Do not let one fixture's result influence another fixture on the
    # same calendar day when the source has no kickoff timestamps.
    pending_day = None
    pending_rows = []

    def _learn_pending(day_rows):
        for pending_index, pending, pending_prediction in day_rows:
            ph = (pending.get("HomeTeam") or "").strip()
            pa = (pending.get("AwayTeam") or "").strip()
            pdate = parse_date(pending.get("Date", ""))
            try:
                phg, pag = float(pending["FTHG"]), float(pending["FTAG"])
                engine.learn_step(
                    ph, pa, phg, pag, pending, lg=bt_div,
                    match_num=pending_index, total=total,
                    match_date=pdate, prediction=pending_prediction,
                )
            except (TypeError, ValueError, KeyError):
                continue

    scored_predictions = []

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

        if pending_day is None:
            pending_day = md
        elif md != pending_day:
            _learn_pending(pending_rows)
            pending_rows = []
            pending_day = md

        # ------------------------------------------------------------
        # После начального training window начинаем тестировать.
        #
        # ВАЖНО:
        # predict вызывается ДО learn_step текущего матча.
        # Значит результат текущего матча не попадает в модель.
        # ------------------------------------------------------------

        # Freeze the prediction before any result from this date is learned.
        try:
            P = engine.predict(
                h, a, bt_div, match_date=md,
                cup=bt_div in ("C1", "EL", "EC"),
            )
        except Exception:
            P = None

        if j >= MIN_TRAIN_MATCHES and P:
                outcome_index = 0 if hg > ag else (1 if hg == ag else 2)
                historical_odds = _pick_historical_odds(r)
                market_probs = None
                if all(historical_odds.get(k) is not None for k in ("П1", "X", "П2")):
                    implied = [1.0 / historical_odds[k] for k in ("П1", "X", "П2")]
                    implied_total = sum(implied)
                    if implied_total > 0:
                        market_probs = tuple(p / implied_total for p in implied)
                scored_predictions.append((
                    P["p1"], P["x"], P["p2"], outcome_index, market_probs,
                    P.get("p1_raw"), P.get("x_raw"), P.get("p2_raw"),
                ))

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

        # If prediction failed, skip this row rather than recomputing it later
        # after same-day results have been applied.
        if P is not None:
            pending_rows.append((j, r, P))

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

    if pending_rows:
        _learn_pending(pending_rows)

    progress.progress(1.0)

    # ================================================================
    # Результаты
    # ================================================================

    # Probability metrics over the held-out walk-forward matches, not just
    # selected bets. Multiclass Brier and log loss use pre-result predictions.
    if scored_predictions:
        eps = 1e-12
        brier = 0.0
        logloss = 0.0
        market_brier = 0.0
        market_logloss = 0.0
        market_count = 0
        model_market_brier = 0.0
        model_market_logloss = 0.0
        raw_brier = 0.0
        raw_logloss = 0.0
        raw_scored_count = 0
        for prediction_row in scored_predictions:
            p1, px, p2, actual, market_probs = prediction_row[:5]
            probs = [max(eps, min(1.0, p1)), max(eps, min(1.0, px)), max(eps, min(1.0, p2))]
            total_p = sum(probs) or 1.0
            probs = [p / total_p for p in probs]
            row_brier = sum((p - (1.0 if i == actual else 0.0)) ** 2 for i, p in enumerate(probs))
            row_logloss = -math.log(probs[actual])
            brier += row_brier
            logloss += row_logloss
            if len(prediction_row) >= 8 and all(v is not None for v in prediction_row[5:8]):
                raw_probs = [max(eps, min(1.0, float(v))) for v in prediction_row[5:8]]
                raw_total = sum(raw_probs) or 1.0
                raw_probs = [v / raw_total for v in raw_probs]
                raw_brier += sum((p - (1.0 if i == actual else 0.0)) ** 2 for i, p in enumerate(raw_probs))
                raw_logloss += -math.log(raw_probs[actual])
                raw_scored_count += 1
            if market_probs is not None:
                market_count += 1
                market_brier += sum((p - (1.0 if i == actual else 0.0)) ** 2 for i, p in enumerate(market_probs))
                market_logloss += -math.log(max(eps, market_probs[actual]))
                model_market_brier += row_brier
                model_market_logloss += row_logloss
        brier /= len(scored_predictions)
        logloss /= len(scored_predictions)
        st.caption(
            f"Out-of-sample matches: {len(scored_predictions)} · "
            f"Calibrated Brier: {brier:.4f} · Calibrated log loss: {logloss:.4f}"
        )
        if raw_scored_count:
            st.caption(
                f"Сравнение на тех же {raw_scored_count} матчах — "
                f"до калибровки: Brier {raw_brier / raw_scored_count:.4f}, "
                f"log loss {raw_logloss / raw_scored_count:.4f}; "
                f"после текущей калибровки: Brier {brier:.4f}, log loss {logloss:.4f}. "
                "Меньше — лучше. Это диагностическое сравнение, калибратор автоматически не переключается."
            )
        # Reliability is measured only on pre-result walk-forward predictions.
        # It is a diagnostic, not a guarantee that future probabilities calibrate.
        reliability = reliability_table(scored_predictions, n_bins=10)
        ece = expected_calibration_error(reliability)
        st.markdown("#### Проверка калибровки вероятностей")
        q1, q2 = st.columns(2)
        q1.metric("Expected Calibration Error", f"{ece:.4f}" if ece is not None else "—")
        q2.metric("Матчей вне обучения", len(scored_predictions))
        st.caption(
            "Таблица сравнивает среднюю заявленную вероятность с фактической частотой "
            "для П1, X и П2 по диапазонам вероятности. Чем меньше разрыв, тем лучше; "
            "редкие диапазоны с малым N ненадёжны. ECE зависит от разбиения на интервалы."
        )
        if reliability:
            display_reliability = []
            for item in reliability:
                display_reliability.append({
                    **item,
                    "Средняя P": f"{item['Средняя P']:.1%}",
                    "Фактическая частота": f"{item['Фактическая частота']:.1%}",
                })
            st.dataframe(display_reliability, use_container_width=True, hide_index=True)
        else:
            st.info("Недостаточно корректных out-of-sample прогнозов для таблицы калибровки.")
        if market_count:
            st.caption(
                f"На {market_count} матчах с полными коэффициентами "
                f"(сравнение на одной и той же выборке): "
                f"модель Brier {model_market_brier / market_count:.4f} "
                f"vs рынок {market_brier / market_count:.4f}; "
                f"модель log loss {model_market_logloss / market_count:.4f} "
                f"vs рынок {market_logloss / market_count:.4f}. "
                "Для рынка использованы нормализованные обратные коэффициенты."
            )


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
