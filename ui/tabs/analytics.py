"""Analytics terminal: data quality, learning audit, model lab and portfolio risk."""
from __future__ import annotations

from datetime import datetime, timezone
import streamlit as st

from config import DB_FILE
from model.analytics_quality import card_quality, portfolio_risk
from model.forecast_ledger import ForecastLedger, MarketForecastLedger


STATE = {
    "confirmed": ("Подтверждено", "🟢"),
    "empty": ("Источник ответил / данных нет", "🟡"),
    "unavailable": ("Источник недоступен", "🔴"),
    "stale": ("Данные устарели", "🟠"),
    "conflict": ("Источники противоречат", "🔴"),
    "not_configured": ("Не настроен", "⚪"),
}


def render_data_quality():
    st.header("🧭 DATA QUALITY")
    st.caption("Проверяем наличие и состояние фактов. Полнота данных не равна вероятности выигрыша.")
    cards = [c for c in (st.session_state.get("data") or {}).get("cards", []) if isinstance(c, dict)]
    if not cards:
        st.info("Сначала выполни сканирование: здесь появится аудит источников по каждому матчу.")
        return
    reports = [(c, card_quality(c)) for c in cards]
    good = sum(sum(x["state"] == "confirmed" for x in q["checks"]) for _, q in reports)
    total = sum(len(q["checks"]) for _, q in reports)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Матчей", len(reports))
    c2.metric("Подтверждённые поля", f"{good}/{total}")
    scores = [q["completeness"] for _, q in reports if q["completeness"] is not None]
    c3.metric("Средняя полнота", f"{sum(scores)/len(scores):.0f}/100" if scores else "—")
    c4.metric("Неизвестный контекст", sum(q["context_level"] == "Unknown" for _, q in reports))
    st.warning("«Источник ответил, данных нет» не означает, что событие отсутствует. Неизвестные поля не считаются подтверждёнными.")
    search = st.text_input("Фильтр по матчу или турниру", key="dq_filter")
    for card, report in reports:
        if search and search.casefold() not in (report["match"] + " " + report["league"]).casefold():
            continue
        with st.expander(f"{report['match']} · {report['league']} · полнота {report['completeness'] if report['completeness'] is not None else '—'}/100", expanded=False):
            left, right = st.columns(2)
            for i, item in enumerate(report["checks"]):
                label, icon = STATE.get(item["state"], ("Неизвестно", "⚪"))
                col = left if i % 2 == 0 else right
                col.markdown(f"**{icon} {item['name']}**  \n{label} · {item['detail']}")
            st.caption(f"Контекст: {report['context_level']} · {report['context_reason']}")


def render_learning_center():
    st.header("🧠 AI LEARNING CENTER")
    st.caption("Журнал ошибок и качества прогнозов. Этот экран анализирует результаты, но не меняет веса модели автоматически.")
    try:
        ledger = ForecastLedger(DB_FILE)
        metrics = ledger.metrics()
        rows = ledger.rows(settled_only=True)
        market = MarketForecastLedger(DB_FILE).metrics()
    except Exception as exc:
        st.error(f"Не удалось прочитать журнал: {type(exc).__name__}")
        return
    a, b, c, d = st.columns(4)
    a.metric("Закрытых прогнозов 1X2", metrics.get("n", 0))
    b.metric("Brier score", f"{metrics['brier']:.4f}" if metrics.get("brier") is not None else "—")
    c.metric("Log loss", f"{metrics['log_loss']:.4f}" if metrics.get("log_loss") is not None else "—")
    d.metric("Средний CLV", f"{metrics['mean_clv']:+.2%}" if metrics.get("mean_clv") is not None else "—")
    st.subheader("Разбор последних завершённых прогнозов")
    if rows:
        display = []
        for row in rows[-30:][::-1]:
            probs = row.get("probabilities") or {}
            actual = row.get("result")
            predicted = max(probs, key=probs.get) if probs else "—"
            display.append({"Матч": f"{row.get('home_team','')} — {row.get('away_team','')}",
                            "Лига": row.get("league") or "—", "Прогноз": predicted,
                            "Факт": actual, "P факта": f"{probs.get(actual, 0):.1%}" if actual in probs else "—",
                            "Ошибка Brier": round(sum((probs.get(k,0)-float(k==actual))**2 for k in ("H","D","A")),4) if actual else None,
                            "Версия": row.get("model_version") or "—"})
        st.dataframe(display, use_container_width=True, hide_index=True)
    else:
        st.info("Пока нет подтверждённых завершённых прогнозов в ledger. Данные не подменяются статистикой портфеля.")
    st.subheader("Результаты по рынкам")
    market_rows = market.get("by_market") or []
    if market_rows:
        st.dataframe([{"Рынок": x.get("market"), "Прогнозов": x.get("n"),
                       "Brier": x.get("brier"), "Log loss": x.get("log_loss"),
                       "ROI": x.get("roi")} for x in market_rows],
                     use_container_width=True, hide_index=True)
    else:
        st.info("Рыночный ledger пока не содержит закрытых прогнозов.")
    st.info("Рекомендация системы: сначала накопить репрезентативную выборку и проверить улучшение walk-forward. Автоматическое переобучение и изменение коэффициентов отключено.")


def render_model_lab():
    st.header("🧪 MODEL LAB")
    st.caption("Калибровка, надёжность вероятностей и независимый статус каждого рынка.")
    try:
        ledger = ForecastLedger(DB_FILE)
        metrics = ledger.metrics()
        reliability = ledger.reliability_bins(bins=10)
        market_metrics = MarketForecastLedger(DB_FILE).metrics()
    except Exception as exc:
        st.error(f"Ошибка чтения модельного аудита: {type(exc).__name__}")
        return
    n = metrics.get("n", 0)
    xg = metrics.get("xg_usage") or {}
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("OOS/settled sample", n)
    c2.metric("Brier", f"{metrics['brier']:.4f}" if metrics.get("brier") is not None else "—")
    c3.metric("Log loss", f"{metrics['log_loss']:.4f}" if metrics.get("log_loss") is not None else "—")
    c4.metric("xG coverage", f"{xg.get('share'):.1%}" if xg.get("share") is not None else "—")
    st.subheader("Reliability / калибровка")
    if reliability.get("rows"):
        st.dataframe([{"Исход": x["outcome"], "Диапазон P": f"{x['bin_from']:.0%}–{x['bin_to']:.0%}",
                       "N": x["n"], "Средняя P": round(x["mean_probability"],3),
                       "Факт": round(x["observed_frequency"],3),
                       "Разница": round(x["observed_minus_predicted"],3)}
                      for x in reliability["rows"]], use_container_width=True, hide_index=True)
    else:
        st.info("Недостаточно закрытых прогнозов для оценки калибровки.")
    st.subheader("Независимая проверка рынков")
    statuses = {
        "1X2": ("Основной ledger", "Проверяй Brier, log loss, ECE и CLV на закрытых прогнозах."),
        "Тоталы голов": ("Отдельный ledger", "Только подтверждённые линии и OOS результаты."),
        "BTTS": ("Отдельный ledger", "Не наследует статус Value от 1X2."),
        "Угловые": ("Отдельный ledger", "Нужны завершённые матчи и подтверждённая статистика."),
        "Карточки": ("Отдельный ledger", "Нужны подтверждённый судья и фактические карточки."),
        "Asian Handicap": ("Отдельный ledger", "Нужны quarter-line settlement и OOS тест."),
    }
    for name,(ledger_name,rule) in statuses.items():
        with st.container(border=True):
            st.markdown(f"**{name}** · {ledger_name}")
            st.caption(rule)
            st.warning("Не допущен к Value/автопортфелю без независимого подтверждения.")
    if market_metrics.get("markets"):
        st.caption("Данные закрытых рыночных прогнозов:")
        st.dataframe(market_metrics["markets"], use_container_width=True, hide_index=True)


def render_portfolio_risk():
    st.header("🛡️ PORTFOLIO RISK")
    st.caption("Концентрация, дневная нагрузка, риск одного матча и историческая просадка. Аналитика не размещает ставки.")
    data = st.session_state.get("data") or {}
    report = portfolio_risk(data.get("bets") or [], data.get("bank") or 0)
    a,b,c,d = st.columns(4)
    a.metric("Открытых позиций", report["open_count"])
    b.metric("Открытая экспозиция", f"{report['exposure']:.2f}")
    c.metric("Экспозиция / банк", f"{report['exposure_pct']:.1%}" if report["exposure_pct"] is not None else "—")
    d.metric("Макс. просадка", f"{report['max_drawdown']:.1%}")
    if report["warning"]:
        st.error("Экспозиция достигла 20% банка. Рассмотри сокращение новых позиций.")
    st.subheader("Концентрация на одном матче")
    st.dataframe([{"Матч / ID": k, "Открытый риск": round(v,2)} for k,v in report["fixture_top"]],
                 use_container_width=True, hide_index=True) if report["fixture_top"] else st.info("Открытых позиций пока нет.")
    st.subheader("Риск по лигам")
    st.dataframe([{"Лига": k, "Открытый риск": round(v,2)} for k,v in report["league_top"]],
                 use_container_width=True, hide_index=True) if report["league_top"] else st.info("Нет открытых позиций.")
    st.subheader("Риск по дням")
    st.dataframe([{"Дата": k, "Открытый риск": round(v,2)} for k,v in report["day_top"]],
                 use_container_width=True, hide_index=True) if report["day_top"] else st.info("Нет открытых позиций.")
    st.caption("Корреляция исходов одного матча пока не оценивается численно: без исторической совместной матрицы она не должна изображаться точной.")
