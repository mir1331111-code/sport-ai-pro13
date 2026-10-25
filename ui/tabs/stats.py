"""ui/tabs/stats.py — Статистика по СВОИМ ставкам."""
from __future__ import annotations
from collections import defaultdict

import streamlit as st

from storage import sqlite_store as db
from config import DB_FILE
from model.independent_forecasts import comparison_report, settle_due_forecasts


def _sharpe(returns: list, periods_per_year: int = 252) -> float:
    if len(returns) < 3:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / len(returns)
    sigma = var ** 0.5
    if sigma <= 1e-12:
        return 0.0
    return (mu / sigma) * (periods_per_year ** 0.5)


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
    st.dataframe(rows, width="stretch", hide_index=True)


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
    st.dataframe(rows, width="stretch", hide_index=True)


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
    st.dataframe(rows, width="stretch", hide_index=True)


def _render_independent_ai():
    st.subheader("🤖 Независимый прогноз ИИ против основной модели")
    st.caption(
        "Сравниваются вероятности П1/X/П2 на одних и тех же матчах. "
        "В журнал попадают только прогнозы, записанные до начала матча; "
        "результат учитывается после подтверждённого статуса FT."
    )

    import time
    now_ts = time.time()
    last_check = float(st.session_state.get("_independent_ai_last_check", 0) or 0)
    if now_ts - last_check >= 600:
        try:
            with st.spinner("Сверяю уже завершившиеся матчи с источниками результатов…"):
                settle_status = settle_due_forecasts(db_path=DB_FILE, max_fixtures=8)
            st.session_state["_independent_ai_settlement_status"] = settle_status
            st.session_state["_independent_ai_last_check"] = now_ts
        except Exception as exc:
            st.session_state["_independent_ai_settlement_status"] = {
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"
            }
            st.session_state["_independent_ai_last_check"] = now_ts

    if st.button("🔄 Проверить результаты завершённых матчей ИИ", use_container_width=False):
        try:
            with st.spinner("Запрашиваю подтверждённые результаты…"):
                settle_status = settle_due_forecasts(db_path=DB_FILE, max_fixtures=30)
            st.session_state["_independent_ai_settlement_status"] = settle_status
            st.session_state["_independent_ai_last_check"] = time.time()
        except Exception as exc:
            st.error(f"Не удалось обновить результаты: {type(exc).__name__}: {str(exc)[:160]}")

    settle_status = st.session_state.get("_independent_ai_settlement_status") or {}
    if settle_status.get("error"):
        st.warning("Автоматическая проверка результатов временно недоступна: " + str(settle_status["error"]))
    elif settle_status:
        st.caption(
            f"Последняя сверка: проверено матчей {settle_status.get('fixtures_checked', 0)}, "
            f"добавлено завершённых прогнозов {settle_status.get('forecasts_settled', 0)}, "
            f"ошибок источников {settle_status.get('errors', 0)}."
        )

    try:
        report = comparison_report(DB_FILE)
    except Exception as exc:
        st.error(f"Журнал независимых прогнозов недоступен: {type(exc).__name__}: {str(exc)[:180]}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Прогнозы основной модели", report["main_forecasts"])
    c2.metric("Прогнозы ИИ", report["ai_forecasts"])
    c3.metric("Сверено основной модели", report["main_settled"])
    c4.metric("Сверено ИИ", report["ai_settled"])

    best = report.get("best_paired")
    if not best or not best.get("n"):
        st.info(
            "Пока недостаточно завершённых матчей с прогнозами ОБЕИХ систем. "
            "После сканов журнал будет накапливать прогнозы до начала игры; "
            "после подтверждённого FT результат попадёт в сравнение. "
            "Отсутствие данных не считается ни победой, ни поражением."
        )
        if report.get("unsettled_due"):
            st.caption(f"Ожидают проверки результата: {report['unsettled_due']} записей.")
        return

    st.markdown(f"**Сравнение на одинаковых матчах:** {best['n']} завершённых матчей")
    main_metrics, ai_metrics = best["main"], best["ai"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Точность основной модели", f"{main_metrics['accuracy']:.1%}")
    k2.metric("Точность ИИ", f"{ai_metrics['accuracy']:.1%}",
              delta=f"{best['accuracy_delta']:+.1%} к модели")
    k3.metric("Brier · основная модель", f"{main_metrics['brier']:.4f}")
    k4.metric("Brier · ИИ", f"{ai_metrics['brier']:.4f}",
              delta=f"{best['brier_delta']:+.4f} к модели", delta_color="inverse")

    l1, l2 = st.columns(2)
    l1.metric("Log loss · основная модель", f"{main_metrics['log_loss']:.4f}")
    l2.metric("Log loss · ИИ", f"{ai_metrics['log_loss']:.4f}",
              delta=f"{best['log_loss_delta']:+.4f} к модели", delta_color="inverse")
    st.caption(
        "Для Brier и log loss меньше — лучше; для точности больше — лучше. "
        "Это качество вероятностного прогноза, а не доходность ставок и не гарантия будущего результата."
    )
    st.caption("Сравниваемый вариант ИИ: " + str(best.get("model_version") or "не определён"))

    if report.get("models"):
        model_rows = []
        for item in report["models"]:
            model_rows.append({
                "Версия ИИ": item["model_version"],
                "Общих завершённых матчей": item["n"],
                "Точность основной модели": (
                    f"{item['main']['accuracy']:.1%}" if item["main"]["accuracy"] is not None else "—"
                ),
                "Точность ИИ": (
                    f"{item['ai']['accuracy']:.1%}" if item["ai"]["accuracy"] is not None else "—"
                ),
                "Brier Δ (ИИ − модель)": (
                    f"{item['brier_delta']:+.4f}" if item["brier_delta"] is not None else "—"
                ),
                "Log loss Δ (ИИ − модель)": (
                    f"{item['log_loss_delta']:+.4f}" if item["log_loss_delta"] is not None else "—"
                ),
            })
        st.dataframe(model_rows, width="stretch", hide_index=True)

    picks = {"H": "П1", "D": "X", "A": "П2"}
    recent = []
    for row in best.get("recent_pairs") or []:
        mp, ap = row["main_probs"], row["ai_probs"]
        recent.append({
            "Дата": row["date"],
            "Матч": f"{row['home']} — {row['away']}",
            "Факт": picks.get(row["result"], row["result"]),
            "Основная модель": picks.get(row["main_pick"], row["main_pick"]),
            "ИИ": picks.get(row["ai_pick"], row["ai_pick"]),
            "Модель верна": "Да" if row["main_correct"] else "Нет",
            "ИИ верен": "Да" if row["ai_correct"] else "Нет",
            "Вероятности модели П1/X/П2": " / ".join(f"{mp[k]*100:.0f}%" for k in ("H", "D", "A")),
            "Вероятности ИИ П1/X/П2": " / ".join(f"{ap[k]*100:.0f}%" for k in ("H", "D", "A")),
        })
    if recent:
        st.subheader("Последние парные прогнозы")
        st.dataframe(recent, width="stretch", hide_index=True)

    if report.get("unsettled_due"):
        st.caption(
            f"Прогнозов ожидают подтверждённого результата: {report['unsettled_due']}. "
            "Если источник не подтверждает матч как FT или названия не совпали, запись не закрывается автоматически."
        )


def render():
    D = st.session_state.data
    st.header("📈 Статистика")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📊 Обзор", "🏆 По лигам", "📅 По дням", "🎯 По рынкам", "🤖 ИИ против модели"])
    with tab1:
        _render_overview(D)
    with tab2:
        _render_by_league(D)
    with tab3:
        _render_by_day(D)
    with tab4:
        _render_by_market(D)
    with tab5:
        _render_independent_ai()

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
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CLV avg", f"{clv.get('avg_clv', 0)*100:+.2f}%")
            c2.metric("CLV N", clv.get("n", 0))
            c3.metric("Max DD", f"-{max_dd*100:.1f}%")
            c4.metric("Sharpe", f"{sharpe:.2f}")
        except Exception:
            pass
