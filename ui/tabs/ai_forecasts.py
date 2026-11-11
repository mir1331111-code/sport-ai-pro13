"""AI FORECASTS: transparent probabilities and historical evidence."""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from config import DB_FILE
from model.data_quality import assess_fixture_quality
from model.forecast_ledger import ForecastLedger

def render() -> None:
    data = st.session_state.get("data", {})
    cards = [card for card in (data.get("cards") or []) if isinstance(card, dict)]
    st.header("🧠 AI FORECASTS")
    st.caption("Вероятности основной модели, объём истории и факторы, использованные при прогнозе. Процент вероятности не является гарантией результата.")
    try:
        ledger_metrics = ForecastLedger(DB_FILE).metrics()
    except Exception:
        ledger_metrics = {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Завершённых прогнозов", int(ledger_metrics.get("n") or 0))
    m2.metric("Brier", f"{ledger_metrics['brier']:.3f}" if ledger_metrics.get("brier") is not None else "—")
    m3.metric("Log Loss", f"{ledger_metrics['log_loss']:.3f}" if ledger_metrics.get("log_loss") is not None else "—")
    m4.metric("Accuracy", f"{ledger_metrics['accuracy']:.1%}" if ledger_metrics.get("accuracy") is not None else "—")
    if int(ledger_metrics.get("n") or 0) < 50:
        st.warning("Исторической выборки пока недостаточно для уверенной оценки калибровки. Текущие вероятности показываются без искусственного диапазона уверенности.")
    if not cards:
        st.info("После сканирования здесь появятся вероятности и доказательная база матчей.")
        return
    ranked = []
    for card in cards:
        try:
            probabilities = {"П1": float(card.get("p1") or 0), "X": float(card.get("px") or 0), "П2": float(card.get("p2") or 0)}
        except (TypeError, ValueError):
            continue
        if not 0 < sum(probabilities.values()) <= 1.001:
            continue
        strongest = max(probabilities, key=probabilities.get)
        evidence = card.get("forecast_evidence") or {}
        quality = assess_fixture_quality(card, now=datetime.now(timezone.utc))
        ranked.append((card, probabilities, strongest, evidence, quality))
    ranked.sort(key=lambda item: item[1][item[2]], reverse=True)
    table = []
    for card, probabilities, strongest, evidence, quality in ranked:
        table.append({"Матч": str(card.get("match_ru") or card.get("match") or "—"), "Соревнование": str(card.get("league") or card.get("div") or "—"), "Категория": str(card.get("match_category") or "не уточнена"), "П1": probabilities["П1"], "X": probabilities["X"], "П2": probabilities["П2"], "Лидер модели": f"{strongest} · {probabilities[strongest]:.1%}", "История команд": int(card.get("games") or 0), "Полнота данных": quality["score"] if quality["score"] is not None else "—"})
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    if not ranked:
        return
    selected = st.selectbox("Детальная доказательная база", options=list(range(len(ranked))), format_func=lambda i: str(ranked[i][0].get("match_ru") or ranked[i][0].get("match") or f"Матч {i+1}"), key="ai_forecast_selected")
    card, probabilities, strongest, evidence, quality = ranked[selected]
    st.subheader(str(card.get("match_ru") or card.get("match") or "Матч"))
    columns = st.columns(3)
    for column, label in zip(columns, ("П1", "X", "П2")):
        column.metric(label, f"{probabilities[label]:.1%}")
    st.caption(f"Главный исход по модели: {strongest} ({probabilities[strongest]:.1%}). Качество данных: {quality['score'] if quality['score'] is not None else 'нет оценки'}/100.")
    ev = [
        ("Матчи хозяев в истории", evidence.get("home_history_n")),
        ("Матчи гостей в истории", evidence.get("away_history_n")),
        ("Хозяева дома — выборка", evidence.get("home_venue_history_n")),
        ("Гости в гостях — выборка", evidence.get("away_venue_history_n")),
        ("Форма хозяев (последние 5)", evidence.get("home_form")),
        ("Форма гостей (последние 5)", evidence.get("away_form")),
        ("Отдых хозяев, дней", evidence.get("home_rest_days")),
        ("Отдых гостей, дней", evidence.get("away_rest_days")),
        ("xG учтён", "Да" if card.get("xg_used") else "Нет / не хватило выборки"),
        ("Матчей с xG у хозяев", evidence.get("xg_home_samples")),
        ("Матчей с xG у гостей", evidence.get("xg_away_samples")),
        ("Средний Elo соперников хозяев", evidence.get("home_opponent_elo_avg")),
        ("Средний Elo соперников гостей", evidence.get("away_opponent_elo_avg")),
    ]
    st.dataframe(pd.DataFrame([{"Показатель": label, "Значение": value if value is not None else "Нет данных"} for label, value in ev]), width="stretch", hide_index=True)
    st.caption("Форма, отдых и сила соперников здесь — диагностические факторы. Их нельзя считать доказанным улучшением модели, пока не пройден walk-forward A/B-тест.")
