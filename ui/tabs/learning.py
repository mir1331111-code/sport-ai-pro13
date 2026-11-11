"""AI LEARNING CENTER: immutable results, calibration and error attribution."""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import math
import pandas as pd
import streamlit as st
from config import DB_FILE
from model.error_learner import OnlineErrorLearner
from model.forecast_ledger import ForecastLedger, MarketForecastLedger

def _score_rows(rows: list[dict]) -> dict:
    briers, losses, correct = [], [], 0
    for row in rows:
        probabilities = row.get("probabilities") or {}
        result = row.get("result")
        if result not in {"H", "D", "A"}:
            continue
        try:
            p = {key: float(probabilities[key]) for key in ("H", "D", "A")}
        except (TypeError, ValueError, KeyError):
            continue
        if any(value < 0 or value > 1 for value in p.values()):
            continue
        briers.append(sum((p[key] - float(key == result)) ** 2 for key in p))
        losses.append(-math.log(max(p[result], 1e-15)))
        correct += int(max(p, key=p.get) == result)
    n = len(briers)
    return {"n": n, "brier": sum(briers) / n if n else None, "log_loss": sum(losses) / n if n else None, "accuracy": correct / n if n else None}

def _segment_rows(rows: list[dict], key_fn) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    output = []
    for name, group in sorted(groups.items(), key=lambda item: str(item[0])):
        output.append({"Сегмент": str(name), **_score_rows(group)})
    return output

def render() -> None:
    st.header("🎓 AI LEARNING CENTER")
    st.caption("Разбор ошибок на неизменяемом журнале прогнозов. Система анализирует, но не меняет веса модели автоматически.")
    try:
        ledger = ForecastLedger(DB_FILE)
        rows = ledger.rows()
        metrics = ledger.metrics()
        market_metrics = MarketForecastLedger(DB_FILE).metrics()
    except Exception as exc:
        st.error(f"Журнал прогнозов недоступен: {type(exc).__name__}")
        return
    settled = [row for row in rows if row.get("result") in {"H", "D", "A"}]
    pending = len(rows) - len(settled)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Всего зафиксировано", len(rows))
    c2.metric("Результат подтверждён", len(settled))
    c3.metric("Ожидают результата", pending)
    c4.metric("xG использован", f"{metrics['xg_usage']['xg_used']}/{metrics['xg_usage']['tracked']}" if metrics.get("xg_usage", {}).get("tracked") else "Нет маркировки")
    m1, m2, m3 = st.columns(3)
    m1.metric("Brier", f"{metrics['brier']:.3f}" if metrics.get("brier") is not None else "—")
    m2.metric("Log Loss", f"{metrics['log_loss']:.3f}" if metrics.get("log_loss") is not None else "—")
    m3.metric("Accuracy", f"{metrics['accuracy']:.1%}" if metrics.get("accuracy") is not None else "—")
    if 20 <= len(settled) <= 300:
        try:
            rolling = OnlineErrorLearner(ledger).evaluate(as_of=datetime.now(timezone.utc))
            st.subheader("Хронологическая проверка калибровки")
            st.caption("Каждый прогноз калибруется только на исходах, уже известных к моменту прогноза. Это диагностическая проверка, а не разрешение включить коррекцию.")
            a, b, c, d = st.columns(4)
            a.metric("N", rolling["n"])
            b.metric("Raw Brier", f"{rolling['raw_brier']:.3f}" if rolling.get("raw_brier") is not None else "—")
            c.metric("Rolling Brier", f"{rolling['calibrated_brier']:.3f}" if rolling.get("calibrated_brier") is not None else "—")
            d.metric("Raw → rolling Log Loss", f"{rolling['raw_log_loss']:.3f} → {rolling['calibrated_log_loss']:.3f}" if rolling.get("raw_log_loss") is not None else "—")
            st.caption(rolling.get("note", ""))
            st.warning("Коррекция вероятностей остаётся выключенной по умолчанию. Сначала требуется отдельный нетронутый временной holdout.")
        except Exception as exc:
            st.warning(f"Хронологическая калибровка временно недоступна: {type(exc).__name__}")
    elif len(settled) < 20:
        st.info("Для хронологической оценки калибровки нужно больше завершённых прогнозов (минимум 20 для диагностики).")
    else:
        st.info("Хронологическая калибровка не запускается на экране при N>300, чтобы не задерживать интерфейс. Базовые метрики журнала доступны выше.")
    if settled:
        st.subheader("Где модель ошибается")
        segments = []
        segments.extend(_segment_rows(settled, lambda row: f"Категория: {row.get('match_category') or 'не записана'}"))
        segments.extend(_segment_rows(settled, lambda row: f"Контекст: {(row.get('context_quality') or {}).get('level', 'не записан')}"))
        segments.extend(_segment_rows(settled, lambda row: f"xG: {'использован' if row.get('xg_used') else 'не использован / неизвестно'}"))
        segments.extend(_segment_rows(settled, lambda row: f"Полнота данных: {int(row['data_quality_score']) // 20 * 20}–{min(100, int(row['data_quality_score']) // 20 * 20 + 19)}" if row.get("data_quality_score") is not None else "Полнота данных: не записана"))
        st.dataframe(pd.DataFrame(segments), width="stretch", hide_index=True)
        st.caption("Срезы по признакам показывают ассоциацию, а не причинное влияние. Нельзя делать вывод, что фактор улучшает модель, только по более высокому проценту побед.")
    st.subheader("Качество по рынкам")
    market_rows = market_metrics.get("by_market") or []
    if market_rows:
        st.dataframe(pd.DataFrame([{"Рынок": row.get("market"), "Записано": row.get("n"), "Завершено": row.get("settled"), "Won": row.get("wins"), "Lost": row.get("losses"), "Brier": row.get("brier"), "Log Loss": row.get("log_loss"), "ROI": row.get("roi"), "CLV": row.get("mean_clv")} for row in market_rows]), width="stretch", hide_index=True)
    else:
        st.info("Пока нет записей в журнале выбранных рынков.")
    if metrics.get("calibration_bins", {}).get("rows"):
        with st.expander("Таблица калибровки 1X2", expanded=False):
            st.dataframe(pd.DataFrame(metrics["calibration_bins"]["rows"]), width="stretch", hide_index=True)
