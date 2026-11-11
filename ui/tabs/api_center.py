"""API CENTER: credential presence, application quotas and safe diagnostics."""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import streamlit as st
from config import DISK_CACHE_DIR
from storage.secret_store import load_local_secrets
from storage import usage

def _secret_value(name: str) -> str:
    value = str(os.environ.get(name, "") or "").strip()
    if value:
        return value
    try:
        value = str(st.secrets.get(name, "") or "").strip()
        if value:
            return value
    except Exception:
        pass
    return ""

def _key_present(local: dict, storage_key: str, names: tuple[str, ...]) -> bool:
    return bool(local.get(storage_key)) or any(_secret_value(name) for name in names)

def render() -> None:
    st.header("⚙️ API CENTER")
    st.caption("Здесь отображается наличие ключа и локальные лимиты приложения. Наличие сохранённого ключа не означает, что провайдер подтвердил подключение.")
    local = load_local_secrets()
    providers = [
        ("API-Football", "api_football_key", ("API_FOOTBALL_KEY",), "Составы, травмы, судьи"),
        ("football-data.org", "fdorg_token", ("FDORG_TOKEN",), "Расписание и результаты"),
        ("SStats.net", "sstats_api_key", ("SSTATS_API_KEY", "sstats_api_key"), "Коэффициенты, угловые, карточки"),
        ("Highlightly", "highlightly_api_key", ("HIGHLIGHTLY_API_KEY",), "Коэффициенты"),
        ("Football API", "footballapi_key", ("FOOTBALLAPI_KEY",), "Исторические xG"),
        ("LLM", "llm_api_key", ("LLM_API_KEY",), "Текстовый анализ"),
    ]
    table = []
    for name, storage_key, aliases, purpose in providers:
        present = _key_present(local, storage_key, aliases)
        table.append({"Провайдер": name, "Ключ": "Сохранён / задан" if present else "Не настроен", "Назначение": purpose, "Проверка подключения": "Не выполнялась на этом экране"})
    st.dataframe(pd.DataFrame(table), width="stretch", hide_index=True)
    st.success("Секреты скрыты: экран показывает только факт наличия ключа.")
    st.subheader("Квоты приложения")
    try:
        from data.context_data import ApiFootballBudget
        limit = max(1, int(_secret_value("API_FOOTBALL_DAILY_LIMIT") or 100))
        reserve = max(0, min(limit - 1, int(_secret_value("API_FOOTBALL_DAILY_RESERVE") or 10)))
        budget = ApiFootballBudget(Path(DISK_CACHE_DIR) / "api_football" / "daily_budget.json", daily_limit=limit, reserve=reserve)
        used = budget.used_today()
        remaining = max(0, limit - reserve - used)
    except Exception:
        limit, reserve, used, remaining = 100, 10, None, None
    c1, c2, c3 = st.columns(3)
    c1.metric("API-Football лимит", limit)
    c2.metric("Использовано сегодня", used if used is not None else "—")
    c3.metric("Доступно сканеру", remaining if remaining is not None else "—")
    quota_rows = [
        {"Источник": "API-Football", "Счётчик": f"{used}/{max(0, limit-reserve)}" if used is not None else "—", "Резерв": reserve, "Примечание": "Собственный защитный лимит приложения"},
        {"Источник": "football-data.org", "Счётчик": "дневной лимит приложения", "Резерв": "—", "Примечание": f"Осталось: {usage.fdorg_remaining()}"},
        {"Источник": "Odds", "Счётчик": "дневной лимит приложения", "Резерв": "—", "Примечание": f"Осталось: {usage.odds_remaining()}"},
        {"Источник": "LLM", "Счётчик": "дневной лимит приложения", "Резерв": "—", "Примечание": f"Осталось: {usage.llm_remaining()}"},
    ]
    st.dataframe(pd.DataFrame(quota_rows), width="stretch", hide_index=True)
    st.caption("Счётчики отображают внутренние лимиты NEURO BET PRO, а не остаток тарифа у внешнего поставщика. Не отправляйте API-ключи в чат и не добавляйте их в Git.")
    report = [str(line) for line in (st.session_state.get("data", {}).get("report") or [])]
    if report:
        with st.expander("Последняя диагностика сканирования", expanded=False):
            for line in report[-40:]:
                st.write(line)
