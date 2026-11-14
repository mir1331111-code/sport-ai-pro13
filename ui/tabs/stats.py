"""ui/tabs/stats.py — Статистика по СВОИМ ставкам."""
from __future__ import annotations
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone

import streamlit as st

from data.scores import completed_matches_for_day
from data.fixture_time import app_timezone
from storage import sqlite_store as db
from config import DB_FILE
from model.independent_forecasts import (
    comparison_report, settle_due_forecasts, parse_llm_probabilities,
)
from model.forecast_ledger import ForecastLedger
from model.forecast_ledger import MarketForecastLedger
from model.audit_export import build_audit_export
from model.independent_forecasts import compare_forecast_rows


def _sharpe(returns: list, periods_per_year: int = 252) -> float:
    if len(returns) < 3:
        return 0.0
    mu = sum(returns) / len(returns)
    var = sum((r - mu) ** 2 for r in returns) / len(returns)
    sigma = var ** 0.5
    if sigma <= 1e-12:
        return 0.0
    return (mu / sigma) * (periods_per_year ** 0.5)


def _scan_summary(cards: object) -> dict[str, int]:
    """Summarize current-scan fixtures without treating forecasts as bets."""
    valid_cards = [card for card in cards or [] if isinstance(card, dict)]
    forecasted = ai_forecasted = quoted_1x2 = quoted_any = value_signals = ai_opinions = 0
    for card in valid_cards:
        try:
            probabilities = [float(card.get(key)) for key in ("p1", "px", "p2")]
            if all(0 <= p <= 1 for p in probabilities) and 0.90 <= sum(probabilities) <= 1.10:
                forecasted += 1
        except (TypeError, ValueError):
            pass
        market_odds = card.get("market_odds") or {}
        if isinstance(market_odds, dict) and market_odds:
            quoted_1x2 += 1
        if any(
            isinstance(card.get(field), dict) and bool(card.get(field))
            for field in (
                "market_odds", "total_market_odds", "btts_market_odds",
                "stat_market_odds", "asian_handicap_market_odds",
            )
        ):
            quoted_any += 1
        verdict = card.get("verdict") or {}
        best = card.get("best")
        if (
            isinstance(best, (tuple, list)) and len(best) >= 6
            and verdict.get("real_odds") and verdict.get("value_passed")
        ):
            value_signals += 1
        opinion = str(card.get("llm_opinion") or "").strip()
        if opinion:
            ai_opinions += 1
        if parse_llm_probabilities(opinion) is not None:
            ai_forecasted += 1
    return {
        "matches": len(valid_cards),
        "forecasted": forecasted,
        "ai_forecasted": ai_forecasted,
        "quoted_1x2": quoted_1x2,
        "quoted_any": quoted_any,
        "value_signals": value_signals,
        "ai_opinions": ai_opinions,
    }


def _render_forecast_journal(D):
    summary = _scan_summary(D.get("cards") or [])
    st.subheader("🔎 Результаты последнего сканирования")
    if not summary["matches"]:
        st.info("Пока нет сохранённого скана. Запусти сканер — здесь появятся прогнозы и охват котировок.")
    else:
        columns = st.columns(5)
        columns[0].metric("Матчей проанализировано", summary["matches"])
        columns[1].metric("Прогнозов 1X2", summary["forecasted"])
        columns[2].metric("Прогнозов ИИ", summary["ai_forecasted"])
        columns[3].metric("Матчей с кэфами 1X2", summary["quoted_1x2"])
        columns[4].metric("Value-проходов", summary["value_signals"])
        if summary["ai_opinions"] > summary["ai_forecasted"]:
            st.caption(
                f"Текстовых мнений ИИ: {summary['ai_opinions']}; "
                f"валидных прогнозов с вероятностями: {summary['ai_forecasted']}."
            )
        st.caption(
            f"Котировки хотя бы одного рынка есть у {summary['quoted_any']} матчей. "
            "Прогнозы и котировки не равны ставкам в портфеле."
        )

    funnel = D.get("funnel") or {}
    if funnel:
        st.caption(
            "Сводка последнего скана: "
            f"источники расписания — {funnel.get('src', 0)}; "
            f"найдено Value-сигналов — {funnel.get('found', 0)}; "
            f"добавлено в портфель — {funnel.get('added', 0)}; "
            f"LLM-мнений — {funnel.get('llm', 0)}."
        )

    try:
        rows = ForecastLedger(DB_FILE).rows()
        main_rows = [
            row for row in rows
            if str(row.get("model_version") or "").startswith("main-model-")
        ]
        ai_rows = [
            row for row in rows
            if str(row.get("model_version") or "").startswith("independent-llm-")
        ]
        settled_main = sum(row.get("result") in {"H", "D", "A"} for row in main_rows)
        settled_ai = sum(row.get("result") in {"H", "D", "A"} for row in ai_rows)
        pending_main = len(main_rows) - settled_main
        pending_ai = len(ai_rows) - settled_ai
        journal_columns = st.columns(6)
        journal_columns[0].metric("Журнал · модель", len(main_rows))
        journal_columns[1].metric("Модель · FT подтверждены", settled_main)
        journal_columns[2].metric("Модель · ждут FT", pending_main)
        journal_columns[3].metric("Журнал · ИИ", len(ai_rows))
        journal_columns[4].metric("ИИ · FT подтверждены", settled_ai)
        journal_columns[5].metric("ИИ · ждут FT", pending_ai)
        xg_tracked = [row for row in main_rows if row.get("xg_used") is not None]
        xg_used_n = sum(bool(row.get("xg_used")) for row in xg_tracked)
        st.caption(
            "Журнал сохраняет только предматчевые прогнозы. Результат попадает в него "
            "после подтверждённого FT; не найденный счёт остаётся в ожидании, а не "
            "считается проигрышем. "
            f"xG учтён у {xg_used_n}/{len(xg_tracked)} размеченных прогнозов; "
            f"старых записей без флага xG: {len(main_rows) - len(xg_tracked)}."
        )
        market_rows = MarketForecastLedger(DB_FILE).rows()
        market_settled = sum(row.get("result_status") is not None for row in market_rows)
        market_pending = len(market_rows) - market_settled
        stat_market_pending = sum(
            row.get("result_status") is None and str(row.get("market") or "").upper() == "STAT"
            for row in market_rows
        )
        st.caption(
            f"Аудит Value-рынков: {len(market_rows)} рекомендаций · "
            f"{market_settled} закрыто · {market_pending} ожидают. "
            f"Из них статистические рынки (угловые/карточки) ожидают SStats: {stat_market_pending}."
        )
            calibration = ForecastLedger(DB_FILE).reliability_bins(
                model_version_prefix="main-model-", bins=10,
            )
            if calibration.get("rows"):
                import pandas as pd
                labels = {"H": "П1", "D": "X", "A": "П2"}
                table = [{
                    "Исход": labels.get(item["outcome"], item["outcome"]),
                    "Интервал P": f"{item['bin_from']:.0%}–{item['bin_to']:.0%}",
                    "N": item["n"],
                    "Средняя P": item["mean_probability"],
                    "Фактическая частота": item["observed_frequency"],
                    "Факт − прогноз": item["observed_minus_predicted"],
                } for item in calibration["rows"]]
                with st.expander("Калибровка основной модели по бинам (только FT)"):
                    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
                    ece = calibration.get("expected_calibration_error") or {}
                    st.caption(
                        "ECE по исходам: "
                        + " · ".join(
                            f"{labels.get(key, key)} {value:.3f}"
                            for key, value in ece.items() if value is not None
                        )
                        + ". Ни один бин не оценивается без фактического результата."
                    )
            else:
                st.caption("Калибровочные бины появятся после накопления подтверждённых результатов прогнозов основной модели.")

            # Export only audit records and derived metrics. API credentials,
            # Streamlit secrets and provider session state are never included.
            try:
                market_ledger = MarketForecastLedger(DB_FILE)
                market_rows = market_ledger.rows()
                market_report = market_ledger.metrics()
                settled_main_rows = [
                    row for row in main_rows
                    if row.get("result") in {"H", "D", "A"}
                ]
                xg_usage = {
                    "total_main_forecasts": len(main_rows),
                    "tracked": len(xg_tracked),
                    "xg_used": xg_used_n,
                    "share_among_tracked": (
                        xg_used_n / len(xg_tracked) if xg_tracked else None
                    ),
                    "untracked_legacy": len(main_rows) - len(xg_tracked),
                }
                audit_payload = {
                    "schema_version": 1,
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_database": "application forecast ledger (sensitive credentials excluded)",
                    "summary": {
                        "main_forecasts": len(main_rows),
                        "main_settled": len(settled_main_rows),
                        "llm_forecasts": len(ai_rows),
                        "llm_settled": sum(
                            row.get("result") in {"H", "D", "A"} for row in ai_rows
                        ),
                        "market_forecasts": len(market_rows),
                        "xg_usage": xg_usage,
                    },
                    "main_reliability_bins": calibration,
                    "main_settled_metrics": ForecastLedger(DB_FILE).metrics(
                        model_version=(
                            sorted({str(row.get("model_version")) for row in main_rows})[-1]
                            if main_rows else None
                        )
                    ) if main_rows else {},
                    "main_vs_llm_paired_comparison": compare_forecast_rows(rows),
                    "market_ledger_metrics": market_report,
                    "forecasts": rows,
                    "market_forecasts": market_rows,
                    "paper_bets": db.fetch_bets(limit=100000),
                    "bank_history": db.bank_history(limit=100000),
                    "notes": {
                        "h2h_rest_days_ablation": (
                            "Not stored as live ledger variants. Historical walk-forward "
                            "ablation exists separately in model.walk_forward."
                        ),
                        "ensemble": (
                            "The LLM is evaluated as a separate forecaster; no weighted "
                            "ensemble is served by the application."
                        ),
                        "xg_usage": (
                            "xg_used is authoritative only for tracked rows. Legacy NULL "
                            "values are not inferred or backfilled."
                        ),
                    },
                }
                st.download_button(
                    "⬇️ Скачать архив ledger для аудита",
                    data=build_audit_export(audit_payload),
                    file_name="neuro_bet_forecast_audit.zip",
                    mime="application/zip",
                    key="download_forecast_audit_archive",
                    help=(
                        "Включает прогнозы, подтверждённые результаты, калибровку, "
                        "xG-флаги, корреляцию ошибок LLM/модели, рыночный ledger "
                        "и paper-портфель. API-ключи не включаются."
                    ),
                )
            except Exception as export_exc:
                st.warning(
                    "Не удалось подготовить архив аудита: "
                    f"{type(export_exc).__name__}: {str(export_exc)[:120]}"
                )
    except Exception as exc:
        st.warning(f"Журнал прогнозов пока недоступен: {type(exc).__name__}: {str(exc)[:120]}")


def _render_overview(D):
    _render_forecast_journal(D)
    st.divider()
    st.subheader("💼 Статистика виртуального портфеля")
    bets = [b for b in D.get("bets", []) if isinstance(b, dict)]
    if not bets:
        st.info(
            "Ставок в портфеле пока нет. Это не означает, что прогнозов нет: "
            "они показаны выше. Win Rate, PnL и ROI появятся после добавления и закрытия ставок."
        )
        bank = float(D.get("bank", 10000))
        initial = float(D.get("meta", {}).get("initial_bank", 10000))
        growth = ((bank - initial) / initial * 100) if initial else 0
        b1, b2 = st.columns(2)
        b1.metric("Текущий банкролл", f"{bank:.0f}")
        b2.metric("Изменение к начальному банку", f"{growth:+.1f}%")
        return

    total = len(bets)
    pending = [b for b in bets if b.get("status") == "pending"]
    won = [b for b in bets if b.get("status") == "won"]
    lost = [b for b in bets if b.get("status") == "lost"]
    push = [b for b in bets if b.get("status") == "push"]
    void = [b for b in bets if b.get("status") == "void"]
    closed = len(won) + len(lost)
    total_stake = sum(float(b.get("stake") or 0) for b in bets)
    closed_stake = sum(
        float(b.get("stake") or 0) for b in bets
        if b.get("status") in {"won", "lost"}
    )
    total_pnl = 0.0
    for b in bets:
        status = b.get("status")
        stake = float(b.get("stake") or 0)
        odds = float(b.get("odds") or 1)
        if status == "won":
            total_pnl += stake * (odds - 1)
        elif status == "lost":
            total_pnl -= stake
    winrate = (len(won) / closed * 100) if closed else None
    roi = (total_pnl / closed_stake * 100) if closed_stake else None

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Всего ставок", total)
    k2.metric("В работе", len(pending))
    k3.metric("WinRate", "—" if winrate is None else f"{winrate:.1f}%")
    k4.metric("PnL", f"{total_pnl:+.2f}" if closed else "—")
    k5.metric("ROI закрытых ставок", "—" if roi is None else f"{roi:+.2f}%")

    bank = float(D.get("bank", 10000))
    initial = float(D.get("meta", {}).get("initial_bank", 10000))
    growth = ((bank - initial) / initial * 100) if initial else 0
    k1, k2, k3 = st.columns(3)
    k1.metric("Банкролл", f"{bank:.0f}", f"{growth:+.1f}%")
    k2.metric("Заморожено", f"{sum(float(b.get('stake') or 0) for b in pending):.0f}")
    k3.metric("Сумма стейков", f"{total_stake:.0f}")

    st.divider()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("🟢 Выиграно", len(won))
    s2.metric("🔴 Проиграно", len(lost))
    s3.metric("⚪ Возврат", len(push))
    s4.metric("⬜ Void", len(void))

    if not closed:
        st.info("Закрытых ставок пока нет. Прибыльность стратегии ещё нельзя оценить.")

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

    # The app-level synchronizer handles automatic checks for both portfolio bets
    # and forecast ledgers. Keep this tab's button as a ledger-only manual retry.
    if st.button("🔄 Дополнительно проверить журнал прогнозов", use_container_width=False):
        try:
            import time
            with st.spinner("Запрашиваю подтверждённые результаты…"):
                settle_status = settle_due_forecasts(db_path=DB_FILE, max_fixtures=30)
            settle_status["checked_at"] = datetime.now(timezone.utc).isoformat()
            st.session_state["_forecast_ledger_sync_status"] = settle_status
            st.session_state["_last_forecast_ledger_sync_ts"] = time.time()
        except Exception as exc:
            st.session_state["_forecast_ledger_sync_status"] = {
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"
            }

    settle_status = st.session_state.get("_forecast_ledger_sync_status") or {}
    if settle_status.get("error"):
        st.warning("Автоматическая сверка временно недоступна: " + str(settle_status["error"]))
    elif settle_status:
        st.caption(
            f"Последняя общая сверка: проверено матчей {settle_status.get('fixtures_checked', 0)}, "
            f"обновлено прогнозов модели/ИИ {settle_status.get('forecasts_settled', 0)}, "
            f"закрыто рыночных рекомендаций {settle_status.get('market_forecasts_settled', 0)}, "
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
    correlation = best.get("error_correlation") or {}
    corr1, corr2, corr3 = st.columns(3)
    corr1.metric("Парных ошибок для корреляции", correlation.get("n", 0))
    corr2.metric(
        "Корреляция Log loss ошибок",
        "—" if correlation.get("log_loss") is None else f"{correlation['log_loss']:+.3f}",
    )
    corr3.metric(
        "Корреляция Brier ошибок",
        "—" if correlation.get("brier") is None else f"{correlation['brier']:+.3f}",
    )
    st.caption(
        "Положительная корреляция означает, что системы чаще ошибаются на одних и тех же матчах. "
        "Это проверка разнообразия ошибок, не готовый ансамбль: веса прогнозов не смешиваются."
    )

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


def _render_market_audit():
    st.subheader("📚 Аудит рекомендаций по рынкам")
    st.caption(
        "Одна лучшая pre-match рекомендация на матч. ROI рассчитан на фиксированную "
        "единичную ставку и не является доходностью фактического банка."
    )
    report = MarketForecastLedger(DB_FILE).metrics()
    overall = report["overall"]
    if not overall["n"]:
        st.info("Пока нет рекомендаций, прошедших проверку цены и рынка.")
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Рекомендаций", overall["n"])
    c2.metric("Закрыто", overall["settled"])
    c3.metric("ROI · 1 ед.", "—" if overall["roi"] is None else f"{overall['roi']*100:+.2f}%")
    c4.metric("CLV", "—" if overall["mean_clv"] is None else f"{overall['mean_clv']*100:+.2f}%")
    rows = []
    for item in report["by_market"]:
        rows.append({
            "Рынок": item["market"], "Всего": item["n"], "Закрыто": item["settled"],
            "Победы / поражения": f"{item['wins']} / {item['losses']}",
            "Win rate": "—" if item["win_rate"] is None else f"{item['win_rate']*100:.1f}%",
            "Средний кэф": "—" if item["avg_odds"] is None else f"{item['avg_odds']:.2f}",
            "ROI · 1 ед.": "—" if item["roi"] is None else f"{item['roi']*100:+.2f}%",
            "CLV": "—" if item["mean_clv"] is None else f"{item['mean_clv']*100:+.2f}%",
            "Brier": "—" if item["brier"] is None else f"{item['brier']:.4f}",
        })
    st.dataframe(rows, width="stretch", hide_index=True)
    if overall["pending"]:
        st.caption(
            f"Ожидают результата: {overall['pending']}. Угловые и карточки закрываются "
            "только после ввода фактической статистики в портфеле."
        )


def render():
    D = st.session_state.data
    st.header("📈 Статистика")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        ["📊 Обзор", "🏆 По лигам", "📅 По дням", "🎯 По рынкам", "🤖 ИИ против модели", "⚽ Результаты"])
    with tab1:
        _render_overview(D)
    with tab2:
        _render_by_league(D)
    with tab3:
        _render_by_day(D)
    with tab4:
        _render_by_market(D)
        _render_market_audit()
    with tab5:
        _render_independent_ai()
    with tab6:
        _render_results_archive()

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


def _render_results_archive():
    """Daily football results are separate from betting and forecast performance."""
    st.subheader("⚽ Результаты завершённых матчей")
    st.caption(
        "Архив фактических счетов по выбранной дате. Эти результаты не означают, "
        "что на каждый матч у модели был прогноз или что на него ставили."
    )

    today = datetime.now(app_timezone()).date()
    default_day = today - timedelta(days=1)
    selected_day = st.date_input(
        "Дата матчей",
        value=default_day,
        min_value=today - timedelta(days=365),
        max_value=today,
        key="stats_results_selected_day",
    )
    date_key = selected_day.strftime("%Y-%m-%d")
    result_state_key = f"_completed_results_{date_key}"

    if st.button(
        f"Загрузить результаты за {selected_day.strftime('%d.%m.%Y')}",
        key="load_completed_results",
        type="primary",
    ):
        try:
            with st.spinner("Получаю завершённые матчи из источников…"):
                result_data = completed_matches_for_day(
                    date_key,
                    token=str(st.session_state.get("data", {}).get("meta", {}).get("fdorg_token") or ""),
                )
            st.session_state[result_state_key] = result_data
        except Exception as exc:
            st.session_state[result_state_key] = {
                "date": date_key,
                "matches": [],
                "source_counts": {},
                "source_errors": {"archive": f"{type(exc).__name__}: {str(exc)[:180]}"},
                "total": 0,
                "goals": 0,
                "home_wins": 0,
                "draws": 0,
                "away_wins": 0,
                "score_conflicts": 0,
            }

    result_data = st.session_state.get(result_state_key)
    if not isinstance(result_data, dict):
        st.info("Выбери дату и нажми «Загрузить результаты». По умолчанию выбрано вчера.")
        return

    matches = result_data.get("matches") or []
    metrics = st.columns(5)
    metrics[0].metric("Матчей найдено", int(result_data.get("total") or len(matches)))
    metrics[1].metric("Победы хозяев", int(result_data.get("home_wins") or 0))
    metrics[2].metric("Ничьи", int(result_data.get("draws") or 0))
    metrics[3].metric("Победы гостей", int(result_data.get("away_wins") or 0))
    metrics[4].metric("Голов", int(result_data.get("goals") or 0))

    if matches:
        import pandas as pd

        table = []
        for item in matches:
            conflict = bool(item.get("score_conflict"))
            alternate = "; ".join(
                f"{x.get('score')} ({x.get('source')})"
                for x in item.get("alternate_scores") or []
            )
            table.append({
                "Лига": item.get("league") or "—",
                "Матч": f"{item.get('home')} — {item.get('away')}",
                "Счёт": item.get("score") or "—",
                "Статус": (
                    "Пенальти" if item.get("duration") == "PENALTY_SHOOTOUT"
                    else "Доп. время" if item.get("status") == "AET"
                    else "Завершён"
                ),
                "Источник": ", ".join(item.get("sources") or []),
                "Проверка": (
                    f"Расхождение: {alternate}" if conflict else "OK"
                ),
            })
        st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    else:
        st.warning(
            "Завершённые матчи за эту дату не найдены в подключённых источниках. "
            "Это не доказывает, что матчей не было: покрытие лиг у источников ограничено."
        )

    source_counts = result_data.get("source_counts") or {}
    st.caption(
        "Получено до удаления дублей: "
        + " · ".join(f"{name}: {int(count or 0)}" for name, count in source_counts.items())
    )
    if result_data.get("score_conflicts"):
        st.warning(
            f"У {int(result_data['score_conflicts'])} матчей источники вернули разные счета. "
            "Расхождения помечены в таблице; для расчёта точности прогнозов они не используются."
        )
    source_errors = result_data.get("source_errors") or {}
    if source_errors.get("archive"):
        st.error(source_errors["archive"])
    elif source_errors:
        disabled = [
            name for name, error in source_errors.items()
            if error == "API token not configured"
        ]
        failures = [
            f"{name}: {error}" for name, error in source_errors.items()
            if error != "API token not configured"
        ]
        if disabled:
            st.caption("football-data.org пропущен: API-токен не подключён.")
        if failures:
            st.warning("Ошибки источников: " + " · ".join(failures))

    st.caption(
        "Источники: ESPN (широкая футбольная лента), football-data.org "
        "(настроенный список соревнований при наличии API-ключа), OpenLigaDB "
        "(Bundesliga, 2. Bundesliga, 3. Liga). Это расширяемое, но не исчерпывающее "
        "покрытие мирового футбола."
    )
