"""Analytics terminal: data quality, learning audit, model lab and portfolio risk."""
from __future__ import annotations

from datetime import datetime, timezone
import os
import streamlit as st

from config import DB_FILE
from model.analytics_quality import card_quality, portfolio_risk
from model.forecast_ledger import ForecastLedger, MarketForecastLedger


STATE = {
    "confirmed": ("Подтверждено", "🟢"),
    "empty": ("Нет данных / подтверждение отсутствует", "🟡"),
    "unavailable": ("Источник недоступен", "🔴"),
    "stale": ("Данные устарели", "🟠"),
    "conflict": ("Источники противоречат", "🔴"),
    "not_configured": ("Не настроен", "⚪"),
}


def render_data_quality():
    cards = [c for c in (st.session_state.get("data") or {}).get("cards", []) if isinstance(c, dict)]
    if not cards:
        st.info("Сначала выполни сканирование: здесь появится аудит источников по каждому матчу.")
        return
    api_football_configured = bool(
        st.session_state.get("api_football_key_input")
        or os.environ.get("API_FOOTBALL_KEY")
    )
    if not api_football_configured:
        try:
            api_football_configured = bool(st.secrets.get("API_FOOTBALL_KEY", ""))
        except Exception:
            api_football_configured = False
    reports = [
        (c, card_quality(c, api_football_configured=api_football_configured))
        for c in cards
    ]
    good = sum(sum(x["state"] == "confirmed" for x in q["checks"]) for _, q in reports)
    total = sum(len(q["checks"]) for _, q in reports)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Матчей", len(reports))
    c2.metric("Подтверждённые поля", f"{good}/{total}")
    scores = [q["completeness"] for _, q in reports if q["completeness"] is not None]
    c3.metric("Средняя полнота", f"{sum(scores)/len(scores):.0f}/100" if scores else "—")
    c4.metric("Матчей с Low-контекстом", sum(q["context_level"] == "Low" for _, q in reports))
    st.warning("Жёлтый статус означает, что подтверждённых данных нет. Он не доказывает отсутствие события; неизвестные поля не считаются подтверждёнными.")
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
    try:
        ledger = ForecastLedger(DB_FILE)
        metrics = ledger.metrics()
        rows = ledger.rows(settled_only=True)
        market = MarketForecastLedger(DB_FILE).metrics()
    except Exception as exc:
        st.error(f"Не удалось прочитать журнал: {type(exc).__name__}")
        return
    total_forecasts = len(ledger.rows())
    settled_forecasts = int(metrics.get("n", 0) or 0)
    pending_forecasts = max(0, total_forecasts - settled_forecasts)
    a, b, c, d = st.columns(4)
    a.metric("Всего записано прогнозов", total_forecasts)
    b.metric("Ожидают подтверждённого FT", pending_forecasts)
    c.metric("Закрыто", settled_forecasts)
    d.metric("Средний CLV", f"{metrics['mean_clv']:+.2%}" if metrics.get("mean_clv") is not None else "—")
    e, f = st.columns(2)
    e.metric("Brier score · только FT", f"{metrics['brier']:.4f}" if metrics.get("brier") is not None else "—")
    f.metric("Log loss · только FT", f"{metrics['log_loss']:.4f}" if metrics.get("log_loss") is not None else "—")
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
    try:
        ledger = ForecastLedger(DB_FILE)
        metrics = ledger.metrics()
        reliability = ledger.reliability_bins(bins=10)
        market_metrics = MarketForecastLedger(DB_FILE).metrics()
    except Exception as exc:
        st.error(f"Ошибка чтения модельного аудита: {type(exc).__name__}")
        return
    all_forecasts = ledger.rows()
    n = int(metrics.get("n", 0) or 0)
    total_forecasts = len(all_forecasts)
    pending_forecasts = max(0, total_forecasts - n)
    xg = metrics.get("xg_usage") or {}
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Всего pre-match прогнозов", total_forecasts)
    c2.metric("Ожидают результата", pending_forecasts)
    c3.metric("Закрыто · FT", n)
    c4.metric("xG coverage", f"{xg.get('share'):.1%}" if xg.get("share") is not None else "—")
    c5,c6 = st.columns(2)
    c5.metric("Brier · только FT", f"{metrics['brier']:.4f}" if metrics.get("brier") is not None else "—")
    c6.metric("Log loss · только FT", f"{metrics['log_loss']:.4f}" if metrics.get("log_loss") is not None else "—")
    st.caption("Исторические обучающие матчи — не прогнозы. В журнале учитываются только записанные до начала матча прогнозы; метрики качества рассчитываются после подтверждённого FT.")
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
        "1X2": ("Основная модель", "Действуют существующие проверки модели, Edge, EV, цены и ограничения сканера. MODEL LAB не меняет этот допуск."),
        "Тоталы голов": ("Отдельная проверка", "Нужны подтверждённые линии и самостоятельные out-of-sample результаты."),
        "BTTS": ("Forecast-only", "Не наследует статус Value от 1X2; нужен отдельный OOS benchmark."),
        "Угловые": ("Forecast-only", "Нужны завершённые матчи и подтверждённая статистика угловых."),
        "Карточки": ("Forecast-only", "Нужны подтверждённый судья и фактические карточки."),
        "Asian Handicap": ("Forecast-only", "Нужны четвертные линии, корректный settlement и OOS benchmark."),
    }
    for name,(ledger_name,rule) in statuses.items():
        with st.container(border=True):
            st.markdown(f"**{name}** · {ledger_name}")
            st.caption(rule)
            if name == "1X2":
                st.info("Используется действующий gate основного рынка. Этот экран не выдаёт новый допуск и не блокирует существующий.")
            else:
                st.warning("Не допущен к Value/автопортфелю без независимого подтверждения.")
    market_rows = market_metrics.get("by_market") or []
    if market_rows:
        st.caption("Фактические записи рыночного ledger (включая ожидающие расчёта):")
        st.dataframe([{
            "Рынок": item.get("market"), "Записано": item.get("n"),
            "Закрыто": item.get("settled"), "Ожидает": item.get("pending"),
            "Brier": item.get("brier"), "Log loss": item.get("log_loss"),
            "ROI": item.get("roi"),
        } for item in market_rows], use_container_width=True, hide_index=True)
    else:
        st.info("В рыночном ledger пока нет записанных рекомендаций. Это не отменяет отдельный основной gate 1X2.")


def render_portfolio_risk():
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

def render_api_center():
    """Show configured integrations and persisted daily counters without exposing credentials."""
    import json
    import os
    from pathlib import Path
    from config import DISK_CACHE_DIR, LLM_DAILY_LIMIT, ODDS_LIMIT_DAILY, FOOTBALL_DATA_ORG_DAILY_LIMIT

    providers = [
        ("API-Football", "api_football_key_input", "API_FOOTBALL_KEY", "Составы, травмы, судьи"),
        ("SStats.net", "sstats_api_key_input", "SSTATS_API_KEY", "Расписание и коэффициенты"),
        ("Highlightly", "highlightly_api_key_input", "HIGHLIGHTLY_API_KEY", "Коэффициенты при доступном тарифе"),
        ("football-data.org", "fdorg_token_input", "FDORG_TOKEN", "Результаты и архив"),
        ("Football API xG", "footballapi_key_input", "FOOTBALLAPI_KEY", "xG в пределах покрытия провайдера"),
        ("LLM", "llm_api_key_input", "LLM_API_KEY", "Независимый текстовый анализ"),
    ]
    for start in range(0, len(providers), 2):
        cols = st.columns(2)
        for col, item in zip(cols, providers[start:start+2]):
            name, session_key, env_key, purpose = item
            configured = bool(st.session_state.get(session_key) or os.environ.get(env_key))
            with col.container(border=True):
                st.markdown(f"**{name}**")
                st.markdown(("🟢 Ключ задан" if configured else "⚪ Ключ не задан"))
                st.caption(purpose)
    st.subheader("Дневные лимиты и использование")
    try:
        from storage import usage
        rows = [
            {"Сервис": "LLM", "Остаток": usage.llm_remaining(), "Лимит": LLM_DAILY_LIMIT},
            {"Сервис": "Odds API", "Остаток": usage.odds_remaining(), "Лимит": ODDS_LIMIT_DAILY},
            {"Сервис": "football-data.org", "Остаток": usage.fdorg_remaining(), "Лимит": FOOTBALL_DATA_ORG_DAILY_LIMIT},
            {"Сервис": "Автопроверка результатов", "Остаток": usage.settle_remaining(), "Лимит": 999},
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    except Exception as exc:
        st.warning(f"Счётчики локального хранилища недоступны: {type(exc).__name__}")
    budget_path = Path(DISK_CACHE_DIR) / "api_football" / "daily_budget.json"
    budget = {}
    try:
        budget = json.loads(budget_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        pass
    api_limit = int(os.environ.get("API_FOOTBALL_DAILY_LIMIT", "100") or 100)
    api_reserve = int(os.environ.get("API_FOOTBALL_DAILY_RESERVE", "10") or 10)
    used = int(budget.get("used", 0) or 0) if budget.get("date") == datetime.now(timezone.utc).date().isoformat() else 0
    st.markdown("**API-Football quota**")
    q1, q2, q3 = st.columns(3)
    q1.metric("Использовано сегодня", used)
    q2.metric("Лимит", api_limit)
    q3.metric("Доступно с учётом резерва", max(0, api_limit - api_reserve - used))
    st.caption("API-ключи не отображаются и не передаются в этот экран. Для реальной проверки доступности используй кнопки проверки в боковой панели.")
