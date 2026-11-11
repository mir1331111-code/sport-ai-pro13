"""Data quality, audit learning and model/API centre screens."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import streamlit as st

from model.data_quality import (
    STATUS_LABELS, competition_category_known, evaluate_fixture_quality,
    summarize_quality,
)
from model.forecast_ledger import ForecastLedger, MarketForecastLedger
from model.error_learner import OnlineErrorLearner
import math


def _pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _status_label(status: str) -> str:
    return STATUS_LABELS.get(status, STATUS_LABELS["unknown"])


def _main_forecast_rows(ledger: ForecastLedger) -> list[dict]:
    return [
        row for row in ledger.rows()
        if str(row.get("model_version") or "").startswith("main-model-")
    ]


def _latest_main_version(rows: list[dict]) -> str | None:
    if not rows:
        return None
    latest = max(rows, key=lambda row: str(row.get("forecast_at") or ""))
    return str(latest.get("model_version") or "") or None


def render_data_quality(data: dict, *, api_keys: dict[str, bool] | None = None) -> None:
    st.header("🧭 DATA QUALITY")
    st.caption(
        "Проверка происхождения, полноты и свежести данных по каждому матчу. "
        "Оценка 0–100 — полнота входных данных, а не вероятность выигрыша."
    )
    cards = [
        card for card in (data.get("cards") or [])
        if isinstance(card, dict)
    ]
    skipped = [
        row for row in (data.get("quality_skipped_fixtures") or [])
        if isinstance(row, dict)
    ]
    if not cards and not skipped:
        st.info("Пока нет карточек последнего сканирования. Выполните скан, чтобы увидеть диагностику матчей.")
        return
    if skipped:
        st.warning(
            f"До построения прогнозов исключено матчей с неподтверждённой категорией: {len(skipped)}. "
            "Они не попали в мужскую/женскую модель и Value."
        )
        st.subheader("События, исключённые до прогнозирования")
        skipped_rows = [{
            "Матч": row.get("match_ru") or row.get("match") or "—",
            "Дата": " ".join(str(value) for value in (row.get("date_iso"), row.get("time")) if value),
            "Источник": ", ".join(row.get("fixture_sources") or []) or "Не записан",
            "Турнир от источника": row.get("league") or "Неизвестен",
            "Причина": row.get("reason") or "Категория не подтверждена",
        } for row in skipped]
        st.dataframe(pd.DataFrame(skipped_rows), use_container_width=True, hide_index=True)
    if not cards:
        return
    # The per-card API state reflects what was known at scan time. A key entered
    # after an old scan is not retroactive evidence that that fixture was queried.
    quality_rows = [
        (card, evaluate_fixture_quality(card, now=datetime.now(timezone.utc)))
        for card in cards
    ]
    summary = summarize_quality(cards, now=datetime.now(timezone.utc))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Матчей проверено", summary["fixtures"])
    c2.metric("Средняя полнота данных", f"{summary['mean_score']:.0f}/100")
    c3.metric("Матчей с котировками", summary["confirmed_odds"])
    c4.metric("Не сопоставлено с API", summary["unmatched_api"])
    if summary.get("women"):
        st.info(f"В последнем скане обнаружено женских матчей: {summary['women']}. Они выделены отдельно и используют отдельное пространство истории.")
    st.divider()

    table_rows = []
    for card, quality in quality_rows:
        table_rows.append({
            "Матч": str(card.get("match_ru") or card.get("match") or "—"),
            "Турнир": str(card.get("league") or "Неизвестен"),
            "Категория": (
                "Женский футбол" if quality["women"]
                else "Сборные" if quality["national"]
                else "Клубный футбол"
            ),
            "Полнота данных": quality["score"],
            "Составы": _status_label(quality["lineup_status"]),
            "Коэффициенты": _status_label(quality["odds_status"]),
            "История": _status_label(quality["history_status"]),
            "Источники": ", ".join(card.get("fixture_sources") or []) or "Неизвестно",
            "API": _status_label(quality["api_status"]),
        })
    st.subheader("Проверка матчей последнего сканирования")
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
    st.subheader("Подробности по источникам и статусам")
    for card, quality in quality_rows:
        title = str(card.get("match_ru") or card.get("match") or "Матч")
        with st.expander(f"{title} · {quality['score']}/100"):
            st.progress(quality["score"] / 100)
            source_rows = []
            for source_name, info in quality["sources"].items():
                source_rows.append({
                    "Проверка": source_name,
                    "Состояние": _status_label(info["status"]),
                    "Детали": info["detail"],
                })
            st.dataframe(pd.DataFrame(source_rows), use_container_width=True, hide_index=True)
            if quality["women"]:
                st.warning("Женское соревнование: прогноз использует отдельное пространство истории. Проверьте, что лига и команды подтверждены источником.")
            elif not competition_category_known(card):
                st.warning("Категория соревнования не подтверждена. Прогноз и Value по такому событию должны быть заблокированы.")
            if quality["history_status"] != "confirmed":
                st.warning("Недостаточно домашних/выездных наблюдений для сильной исторической поддержки.")
            if quality["api_status"] == "unmatched":
                st.warning("API ответил, но не сопоставил точный матч. Составы и судья не считаются подтверждёнными.")

    st.markdown(
        "**Статусы означают:** «Подтверждено» — источник вернул соответствующие данные; "
        "«Источник ответил, данных нет» — успешный пустой ответ, но не доказательство отсутствия травм; "
        "«Недоступен» — запрос завершился ошибкой; «Устарело» — временная метка вышла за TTL; "
        "«Не сопоставлен» — матч не удалось точно связать с записью провайдера."
    )


def render_learning_center(db_path: str) -> None:
    st.header("🧠 AI LEARNING CENTER")
    st.caption(
        "Центр оценки ошибок и калибровки. Он анализирует только сохранённые прогнозы "
        "и подтверждённые результаты; сам по себе не меняет параметры модели."
    )
    try:
        ledger = ForecastLedger(db_path)
        rows = _main_forecast_rows(ledger)
        llm_rows = [
            row for row in ledger.rows()
            if str(row.get("model_version") or "").startswith("independent-llm-")
        ]
        latest_version = _latest_main_version(rows)
        metrics = ledger.metrics(model_version=latest_version) if latest_version else {}
        settled = int(metrics.get("n") or 0)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Прогнозов 1X2", len(rows))
        c2.metric("Результат подтверждён", settled)
        c3.metric("Brier", _number(metrics.get("brier")))
        c4.metric("Log Loss", _number(metrics.get("log_loss")))
        c5.metric("CLV", _pct(metrics.get("mean_clv")))
        if settled < 50:
            st.warning(f"Недостаточно закрытых прогнозов основной модели для устойчивой оценки: {settled}/50. Калибровку пока нельзя считать подтверждённой.")
        elif settled < 200:
            st.info(f"Есть первичная выборка ({settled}), но для серьёзного пересмотра модели желательно накопить не менее 200 независимых завершённых прогнозов.")
        else:
            st.success(f"Накоплено {settled} завершённых прогнозов. Используйте временной holdout и сравнение с базовой моделью перед изменением параметров.")

        if latest_version:
            # Expanding-window diagnostic learns only from results settled
            # before each historical forecast and never mixes model versions.
            error_report = OnlineErrorLearner(ledger).evaluate(
                as_of=datetime.now(timezone.utc),
                model_version=latest_version,
            )
            st.subheader("Анализ прошлых ошибок · expanding window")
            e1, e2, e3, e4, e5 = st.columns(5)
            e1.metric("Прогнозов в тесте", error_report["n"])
            e2.metric("Brier · raw", _number(error_report["raw_brier"]))
            e3.metric("Brier · calibrated", _number(error_report["calibrated_brier"]))
            e4.metric("Log Loss · raw", _number(error_report["raw_log_loss"]))
            e5.metric("Log Loss · calibrated", _number(error_report["calibrated_log_loss"]))
            st.caption(
                "Каждый прогноз калибруется только по более ранним матчам, "
                "результат которых уже был известен на тот момент. Это исследовательский "
                "walk-forward отчёт, не разрешение включить коррекцию в live-модель."
            )
            settled_rows = ledger.rows(
                settled_only=True, model_version=latest_version,
            )[-20:]
            recent_errors = []
            for row in reversed(settled_rows):
                probabilities = row.get("probabilities") or {}
                result = row.get("result")
                if result not in {"H", "D", "A"}:
                    continue
                actual_probability = max(
                    float(probabilities.get(result) or 0.0), 1e-15,
                )
                brier = sum(
                    (float(probabilities.get(key) or 0.0) - float(key == result)) ** 2
                    for key in ("H", "D", "A")
                )
                recent_errors.append({
                    "Прогноз": str(row.get("forecast_at") or "")[:16],
                    "Матч": f"{row.get('home_team') or ''} — {row.get('away_team') or ''}",
                    "P1": _pct(probabilities.get("H")),
                    "X": _pct(probabilities.get("D")),
                    "P2": _pct(probabilities.get("A")),
                    "Факт": result,
                    "Brier": brier,
                    "Log Loss": -math.log(actual_probability),
                })
            if recent_errors:
                st.subheader("Последние 20 закрытых прогнозов")
                st.dataframe(pd.DataFrame(recent_errors), use_container_width=True, hide_index=True)
            calibration = ledger.reliability_bins(model_version=latest_version)
            st.subheader("Калибровка вероятностей 1X2")
            if calibration.get("rows"):
                frame = pd.DataFrame([
                    {
                        "Исход": row["outcome"],
                        "Диапазон P": f"{row['bin_from']:.0%}–{row['bin_to']:.0%}",
                        "Средняя P": row["mean_probability"],
                        "Фактическая частота": row["observed_frequency"],
                        "N": row["n"],
                        "Разница": row["observed_minus_predicted"],
                    }
                    for row in calibration["rows"]
                ])
                st.dataframe(frame, use_container_width=True, hide_index=True)
                ece = calibration.get("expected_calibration_error") or {}
                st.caption("ECE по исходам: " + " · ".join(f"{key}: {_number(value)}" for key, value in ece.items()))
            else:
                st.info("Калибровочные бины появятся после первых подтверждённых результатов.")
        else:
            st.info("Реестр основной модели пока пуст.")
        if llm_rows:
            st.subheader("Независимый LLM-прогноз")
            st.caption(f"Записано отдельных прогнозов ИИ: {len(llm_rows)}. Они не смешиваются с вероятностями основной модели.")
            llm_versions = sorted({str(row.get("model_version") or "") for row in llm_rows})
            comparison = []
            for version in llm_versions[-8:]:
                item = ledger.metrics(model_version=version)
                comparison.append({
                    "Версия ИИ": version,
                    "Закрыто": item.get("n", 0),
                    "Brier": item.get("brier"),
                    "Log Loss": item.get("log_loss"),
                    "Accuracy": item.get("accuracy"),
                })
            st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)
        else:
            st.caption("Отдельная оценка LLM появится после накопления сохранённых прогнозов и результатов.")
    except Exception as exc:
        st.error(f"Не удалось прочитать журнал прогнозов: {type(exc).__name__}")


def render_model_lab(db_path: str) -> None:
    st.header("🧪 MODEL LAB · MARKET VALIDATION")
    st.caption(
        "Показывает отдельную статистику по каждому рынку. Наличие прогноза или "
        "положительный ROI на малой выборке не означает независимую валидацию."
    )
    try:
        ledger = MarketForecastLedger(db_path)
        report = ledger.metrics()
        rows = report.get("by_market") or []
        if not rows:
            st.info("Рыночный журнал пока пуст. Сканер начнёт записывать отдельные сигналы по семействам рынков.")
        table = []
        for item in rows:
            # Brier/Log Loss and the displayed decisive sample exclude pushes/voids.
            n = int(item.get("wins") or 0) + int(item.get("losses") or 0)
            if n < 30:
                readiness = "Недостаточно данных"
            elif n < 100:
                readiness = "Наблюдение"
            else:
                readiness = "Выборка для независимой проверки"
            table.append({
                "Рынок": item.get("market"),
                "Прогнозов": item.get("n"),
                "Результат подтверждён": int(item.get("settled") or 0),
                "Решающих исходов": n,
                "Победы": item.get("wins"),
                "Поражения": item.get("losses"),
                "Brier": item.get("brier"),
                "Log Loss": item.get("log_loss"),
                "ROI (1 unit)": item.get("roi"),
                "Средний CLV": item.get("mean_clv"),
                "Статус": readiness,
            })
        if table:
            st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
        st.caption(
            f"Сырых записей: {report.get('raw_rows', 0)} · повторных сканирований исключено из summary: "
            f"{report.get('duplicate_rows_excluded', 0)}."
        )
        st.warning(
            "Ни один рынок не включается автоматически в Value или портфель только на основании этой таблицы. "
            "Для допуска нужны отдельный временной out-of-sample тест, сопоставимая букмекерская линия и "
            "улучшение proper scoring rules относительно базовой модели."
        )
        st.subheader("Текущие защитные правила")
        gates = [
            ("1X2", "Отдельная валидация вероятностей и полной линии одного букмекера."),
            ("Тоталы", "Для автоматического портфеля используется только независимо проверенная линия 2.5."),
            ("BTTS", "Прогнозный режим до собственной вневыборочной проверки."),
            ("Угловые и карточки", "Информационный режим; отдельная проверка распределений и settlement."),
            ("Asian Handicap", "Quarter-line расчёт отображается отдельно; без допуска в автоматический портфель."),
        ]
        st.dataframe(pd.DataFrame(gates, columns=["Рынок", "Текущий допуск"]), use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Не удалось прочитать рыночный журнал: {type(exc).__name__}")


def render_api_center(api_keys: dict[str, bool] | None = None) -> None:
    st.header("🔌 API CENTER")
    st.caption("Ключи по-прежнему вводятся в левой панели «API ключи». Значения ключей здесь никогда не отображаются.")
    details = [
        ("API-Football", "Составы, травмы, судьи и точная fixture-связка"),
        ("SStats.net", "Футбольное расписание, коэффициенты и статистика после матча"),
        ("Highlightly", "Prematch-коэффициенты"),
        ("Football API", "Исторический xG для поддерживаемого турнира"),
        ("football-data.org", "Результаты и соревнования"),
        ("LLM provider", "Независимое текстовое мнение, отдельно от основной модели"),
    ]
    statuses = api_keys or {}
    frame = []
    for name, purpose in details:
        configured = bool(statuses.get(name))
        frame.append({
            "Источник": name,
            "Назначение": purpose,
            "Ключ": "Введён" if configured else "Не настроен",
            "Статус": "Ключ присутствует; доступность проверяется отдельным тестом"
            if configured else "Требуется ключ для этого источника",
        })
    st.dataframe(pd.DataFrame(frame), use_container_width=True, hide_index=True)
    st.info("Состояние «ключ введён» не является проверкой API. Используйте кнопки проверки подключения в левой панели; успешный ответ и наличие данных отображаются отдельно в DATA QUALITY.")
    st.caption("Дневной бюджет API-Football и ограничения запросов задаются текущей конфигурацией приложения и планом провайдера.")
