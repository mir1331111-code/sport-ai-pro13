"""DATA QUALITY: per-fixture source status, freshness and evidence coverage."""
from __future__ import annotations
from datetime import datetime, timezone
import pandas as pd
import streamlit as st
from model.data_quality import STATUSES, assess_fixture_quality

def render() -> None:
    data = st.session_state.get("data", {})
    cards = [card for card in (data.get("cards") or []) if isinstance(card, dict)]
    st.header("🧪 DATA QUALITY")
    st.caption("Проверка полноты и свежести источников по каждому матчу. Балл показывает полноту данных, а не вероятность исхода и не надёжность прогноза.")
    if not cards:
        st.info("Сначала выполните сканирование — здесь появится аудит найденных матчей.")
        return
    now = datetime.now(timezone.utc)
    assessed = [(card, assess_fixture_quality(card, now=now)) for card in cards]
    scores = [item[1]["score"] for item in assessed if item[1]["score"] is not None]
    conflicts = sum(bool(report["conflict"]) for _, report in assessed)
    api_connected = sum(any(row["key"] == "api_football" and row["status"] == "CONNECTED" for row in report["components"]) for _, report in assessed)
    complete_context = sum(bool((card.get("context_quality") or {}).get("strong_value_eligible")) for card, _ in assessed)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Матчей проверено", len(assessed))
    c2.metric("Средняя полнота", f"{sum(scores) / len(scores):.0f}/100" if scores else "—")
    c3.metric("API-Football сопоставлено", api_connected)
    c4.metric("Конфликты источников", conflicts)
    st.caption(f"Матчей с подтверждёнными XI обеих команд и судьёй: {complete_context}. Это отдельный строгий контекстный фильтр.")
    rows, labels = [], []
    for index, (card, report) in enumerate(assessed):
        match = str(card.get("match_ru") or card.get("match") or f"Матч {index + 1}")
        date = str(card.get("date_iso") or card.get("date") or "")
        rows.append({"Матч": match, "Дата": date[:16], "Полнота": report["score"] if report["score"] is not None else "—", "Категория": str(card.get("match_category") or card.get("league") or "Неизвестно"), "Статус": "Конфликт" if report["conflict"] else "Есть пробелы"})
        labels.append(f"{date[:10]} · {match}")
    st.subheader("Матчи и полнота данных")
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    selected = st.selectbox("Подробная проверка матча", options=list(range(len(assessed))), format_func=lambda item: labels[item], key="dq_selected_fixture")
    card, report = assessed[selected]
    score = report["score"]
    left, right = st.columns([1, 2])
    left.metric("Полнота данных", f"{score}/100" if score is not None else "Нет оценки", help="Взвешенная полнота доступных компонентов. Это не прогнозная вероятность.")
    if score is not None:
        left.progress(max(0.0, min(1.0, score / 100.0)))
    right.info(report["summary"])
    details = [{"Компонент": item["label"], "Статус": STATUSES.get(item["status"], item["status"]), "Код": item["status"], "Источник": item["source"], "Пояснение": item["note"]} for item in report["components"]]
    st.dataframe(pd.DataFrame(details), width="stretch", hide_index=True)
    if report["conflicts"]:
        st.error("Противоречия источников")
        for conflict in report["conflicts"]:
            st.write(f"• {conflict}")
    with st.expander("Как читаются статусы", expanded=False):
        for code in ("CONFIRMED", "CONNECTED", "NO_DATA", "UNAVAILABLE", "NOT_CONFIGURED", "NOT_DUE", "STALE", "CONFLICT", "UNKNOWN"):
            st.write(f"**{STATUSES[code]}** — {code}")
        st.caption("«Ответ получен, данных нет» не означает, что событие отсутствует. Например, пустой ответ injury-feed не доказывает, что у команд нет травм.")
