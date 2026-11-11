"""MODEL LAB: validation status for independent market families."""
from __future__ import annotations
import pandas as pd
import streamlit as st
from config import DB_FILE
from model.forecast_ledger import MarketForecastLedger
from model.market_validation import validate_market_rows

def render() -> None:
    st.header("🔬 MODEL LAB")
    st.caption("Рынки оцениваются отдельно. Прогнозная вероятность сама по себе не даёт рынку статуса Value; для допуска нужны независимые подтверждённые результаты.")
    try:
        rows = MarketForecastLedger(DB_FILE).rows()
        audit = validate_market_rows(rows, min_samples=500)
    except Exception as exc:
        st.error(f"Журнал рынков недоступен: {type(exc).__name__}")
        return
    st.info("Строгий исследовательский порог: минимум 500 завершённых прогнозов на рынок, положительное улучшение Brier с нижней границей bootstrap-интервала выше нуля и улучшение Log Loss относительно сырой implied probability.")
    summary = [{"Рынок": item["market"], "N": item["n"], "Brier модели": item["model_brier"], "Brier цены": item["price_brier"], "Улучшение Brier": item["brier_improvement"], "Нижняя 95% граница": item["brier_improvement_ci_low_95"], "Улучшение Log Loss": item["log_loss_improvement"], "Статус": item["status"]} for item in audit["markets"]]
    if summary:
        st.dataframe(pd.DataFrame(summary), width="stretch", hide_index=True)
    else:
        st.info("Журнал пока не содержит завершённых прогнозов с валидной ценой.")
    st.warning("Даже пройденный сравнительный тест открывает только экспертную проверку, а не автоматический допуск в портфель. Цена сравнения здесь не очищена от маржи, а журнал содержит выбранные сигналы, не все матчи.")
    with st.expander("Методика и ограничения", expanded=False):
        st.write(audit["method"])
        st.write(audit["caveat"])
        st.write("Существующие строгие допуски 1X2 и тотала 2.5 не меняются этой страницей. BTTS, угловые, карточки и Asian Handicap продолжают оставаться наблюдательными рынками до отдельной проверки.")
