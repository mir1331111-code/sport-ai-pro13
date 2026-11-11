"""Analytics terminal: dashboard, source quality, market lab, learning and API center."""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import streamlit as st

from config import APP_VERSION, DB_FILE
from model.forecast_ledger import ForecastLedger, MarketForecastLedger
from model.quality_control import assess_fixture_quality, portfolio_risk_report
from storage import sqlite_store as db


_STATUS_LABELS = {
    "returned": "API ответил",
    "not_configured": "Ключ не задан",
    "unavailable": "Источник недоступен",
    "not_matched": "Матч не сопоставлен",
    "confirmed": "Подтверждено",
    "reported": "Данные получены",
    "records_returned": "Записи получены",
    "responded_empty": "Источник ответил без записей",
    "responded_partial": "Частичные данные",
    "not_requested_yet": "Ожидается ближе к матчу",
    "missing": "Нет данных",
    "unknown": "Неизвестно",
    "stale": "Устарело",
    "future_timestamp": "Некорректное время",
    "received": "Котировки получены",
    "known": "Время известно",
    "conflict": "Есть противоречие",
    "none": "Не обнаружено",
}


def _api_key_present(session_key: str, *names: str) -> bool:
    if str(st.session_state.get(session_key, "") or "").strip():
        return True
    for name in names:
        if os.environ.get(name, "").strip():
            return True
    try:
        for name in names:
            if str(st.secrets.get(name, "") or "").strip():
                return True
            for section_name in ("api", "apis", "secrets", "highlightly", "sstats"):
                section = st.secrets.get(section_name, {})
                if hasattr(section, "get") and str(section.get(name, "") or "").strip():
                    return True
    except Exception:
        pass
    return False


def _safe_ledgers():
    main = ForecastLedger(DB_FILE)
    market = MarketForecastLedger(DB_FILE)
    return main, market


def _format_pct(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _format_number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _quality_rows(cards: list[dict], api_configured: bool, report: list[str]) -> list[dict]:
    rows = []
    now = datetime.now(timezone.utc)
    for card in cards:
        if not isinstance(card, dict):
            continue
        assessed = assess_fixture_quality(
            card, api_configured=api_configured, now=now, source_errors=report,
        )
        rows.append({"card": card, "quality": assessed})
    return rows


def render_dashboard(D: dict) -> None:
    """High-level terminal overview using actual app state and settled ledgers."""
    st.header("NEURO BET PRO · ANALYTICS TERMINAL")
    st.caption(
        "Единый обзор расписания, качества данных, результатов модели и риска портфеля. "
        "Показатели основаны на сохранённых данных приложения."
    )
    cards = [card for card in (D.get("cards") or []) if isinstance(card, dict)]
    bets = [bet for bet in (D.get("bets") or []) if isinstance(bet, dict)]
    pending = [bet for bet in bets if bet.get("status") == "pending"]
    api_ready = _api_key_present(
        "api_football_key_input", "API_FOOTBALL_KEY", "api_football_key",
    )
    rows = _quality_rows(cards, api_ready, D.get("report") or [])
    average_coverage = (
        sum(row["quality"]["score"] for row in rows) / len(rows) if rows else None
    )
    try:
        main_ledger, market_ledger = _safe_ledgers()
        model_metrics = main_ledger.metrics()
        market_metrics = market_ledger.metrics()["overall"]
    except Exception:
        model_metrics, market_metrics = {}, {}

    equity = float(D.get("bank") or 0.0) + sum(float(bet.get("stake") or 0) for bet in pending)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Матчи в последнем скане", len(cards))
    c2.metric(
        "Полнота данных",
        "—" if average_coverage is None else f"{average_coverage:.0f}/100",
        help="Средний индекс наличия и свежести данных. Это не оценка вероятности прогноза.",
    )
    c3.metric("Закрытые прогнозы 1X2", int(model_metrics.get("n") or 0))
    c4.metric("Позиции в портфеле", len(pending))
    c5.metric("Эквити виртуального банка", f"{equity:,.0f}")

    st.divider()
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("DATA QUALITY")
        if rows:
            distribution = defaultdict(int)
            for row in rows:
                score = row["quality"]["score"]
                distribution[
                    "Высокая полнота" if score >= 80 else
                    "Средняя полнота" if score >= 55 else "Низкая полнота"
                ] += 1
            for label in ("Высокая полнота", "Средняя полнота", "Низкая полнота"):
                st.write(f"**{label}:** {distribution[label]}")
            worst = sorted(rows, key=lambda item: item["quality"]["score"])[:5]
            for item in worst:
                match = item["card"].get("match_ru") or item["card"].get("match") or "Матч"
                st.caption(f"{match} · {item['quality']['score']}/100")
        else:
            st.info("После первого сканирования здесь появится проверка источников по каждому матчу.")
    with right:
        st.subheader("MODEL HEALTH")
        if model_metrics.get("n"):
            m1, m2 = st.columns(2)
            m1.metric("Brier · 1X2", _format_number(model_metrics.get("brier"), 4))
            m2.metric("Log loss · 1X2", _format_number(model_metrics.get("log_loss"), 4))
            st.caption(
                f"Accuracy: {_format_pct(model_metrics.get('accuracy'))} · "
                f"CLV наблюдений: {int(model_metrics.get('clv_n') or 0)}"
            )
        else:
            st.info("Пока нет завершённых предматчевых прогнозов для оценки качества.")
        if market_metrics.get("n"):
            st.caption(
                f"Рыночный журнал: {market_metrics.get('n')} записей, "
                f"{market_metrics.get('settled')} сверены. ROI — "
                f"{_format_pct(market_metrics.get('roi'))} на единичной ставке."
            )
        else:
            st.caption("Рыночный журнал ещё накапливает подтверждённые результаты.")
    st.divider()
    st.subheader("Ключевые сигналы последнего сканирования")
    funnel = D.get("funnel") or {}
    if funnel:
        metric_cols = st.columns(5)
        metric_cols[0].metric("Обучающая история", int(funnel.get("trained") or 0))
        metric_cols[1].metric("Источники расписания", int(funnel.get("src") or 0))
        metric_cols[2].metric("Найдено рынков", int(funnel.get("found") or 0))
        metric_cols[3].metric("Добавлено в paper", int(funnel.get("added") or 0))
        metric_cols[4].metric("Общий размер новых позиций", f"{float(funnel.get('frozen') or 0):.2f}")
    else:
        st.caption("Пока нет сохранённой статистики последнего сканирования.")
    report = [str(line) for line in (D.get("report") or []) if str(line).strip()]
    if report:
        with st.expander("Последняя диагностика сканера"):
            for line in report[-30:]:
                st.text(line)


def render_data_quality(D: dict) -> None:
    st.header("DATA QUALITY · Контроль источников")
    st.caption(
        "Состояния независимы: подтверждённые данные, ответ без записей, отсутствие данных, "
        "недоступность, несопоставленный матч, устаревшая информация и противоречия."
    )
    cards = [card for card in (D.get("cards") or []) if isinstance(card, dict)]
    if not cards:
        st.info("Нет карточек последнего сканирования. Выполни скан, чтобы построить отчёт.")
        return
    api_ready = _api_key_present(
        "api_football_key_input", "API_FOOTBALL_KEY", "api_football_key",
    )
    rows = _quality_rows(cards, api_ready, D.get("report") or [])
    average = sum(row["quality"]["score"] for row in rows) / len(rows)
    high = sum(row["quality"]["score"] >= 80 for row in rows)
    medium = sum(55 <= row["quality"]["score"] < 80 for row in rows)
    low = len(rows) - high - medium
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Матчи оценены", len(rows))
    k2.metric("Высокая полнота", high)
    k3.metric("Средняя полнота", medium)
    k4.metric("Низкая полнота", low)
    st.metric("Средний индекс полноты", f"{average:.1f}/100")
    st.caption(
        "Вес индекса: сопоставление API 10, оба состава 20+20, судья 10, "
        "ответ фида травм 10, коэффициенты 15, время 10, свежесть 5. "
        "Индекс измеряет комплектность, а не калибровку источника."
    )
    display = []
    for item in sorted(rows, key=lambda row: row["quality"]["score"]):
        card, quality = item["card"], item["quality"]
        statuses = quality["statuses"]
        display.append({
            "Матч": card.get("match_ru") or card.get("match") or "—",
            "Турнир": card.get("league") or card.get("div") or "—",
            "Категория": "Женский" if quality["gender"] == "women" else "Не определена",
            "Полнота": quality["score"],
            "API": _STATUS_LABELS.get(statuses["API-Football"], statuses["API-Football"]),
            "Составы": (
                _STATUS_LABELS.get(statuses["Состав хозяев"], statuses["Состав хозяев"])
                + " / "
                + _STATUS_LABELS.get(statuses["Состав гостей"], statuses["Состав гостей"])
            ),
            "Судья": _STATUS_LABELS.get(statuses["Судья"], statuses["Судья"]),
            "Травмы": _STATUS_LABELS.get(statuses["Травмы"], statuses["Травмы"]),
            "Коэффициенты": _STATUS_LABELS.get(statuses["Коэффициенты"], statuses["Коэффициенты"]),
            "Конфликт": _STATUS_LABELS.get(statuses["Противоречия"], statuses["Противоречия"]),
        })
    st.dataframe(display, width="stretch", hide_index=True)
    st.subheader("Подробности по каждому матчу")
    for index, item in enumerate(rows):
        card, quality = item["card"], item["quality"]
        title = f"{quality['score']}/100 · {card.get('match_ru') or card.get('match') or 'Матч'}"
        with st.expander(title, expanded=False):
            for name, status in quality["statuses"].items():
                st.write(f"**{name}:** {_STATUS_LABELS.get(status, status)}")
            for note in quality["notes"]:
                st.caption(note)
            if quality["conflicts"]:
                st.error("Обнаружены несоответствия: " + "; ".join(quality["conflicts"]))


def _market_validation_flags(cards: list[dict]) -> dict[str, bool]:
    flags = {"1X2": False, "OU": False, "BTTS": False, "STAT": False, "AH": False}
    for card in cards:
        if not isinstance(card, dict):
            continue
        flags["1X2"] = flags["1X2"] or bool(card.get("winner_market_validation"))
        flags["OU"] = flags["OU"] or bool(card.get("total_market_validation")) or any(
            bool(value) for value in (card.get("total_market_validation_by_line") or {}).values()
        )
        flags["BTTS"] = flags["BTTS"] or bool(card.get("btts_market_validation"))
        stat = card.get("stat_market_validation")
        flags["STAT"] = flags["STAT"] or (
            bool(stat) if not isinstance(stat, Mapping) else any(bool(value) for value in stat.values())
        )
    return flags


def render_model_lab(D: dict) -> None:
    st.header("MODEL LAB · Проверка модели и рынков")
    st.caption(
        "Исторические метрики считаются по предматчевым прогнозам с подтверждённым результатом. "
        "Показатели выбранных Value-кандидатов не заменяют полный независимый бэктест."
    )
    try:
        main, market = _safe_ledgers()
        metrics = main.metrics()
        market_report = market.metrics()
    except Exception as exc:
        st.error(f"Журналы модели недоступны: {type(exc).__name__}")
        return
    n = int(metrics.get("n") or 0)
    if n:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Сверенные прогнозы 1X2", n)
        c2.metric("Brier", _format_number(metrics.get("brier"), 4))
        c3.metric("Log loss", _format_number(metrics.get("log_loss"), 4))
        c4.metric("Accuracy", _format_pct(metrics.get("accuracy")))
        st.caption(
            f"CLV: {_format_pct(metrics.get('mean_clv'))} "
            f"по {int(metrics.get('clv_n') or 0)} сопоставимым ценовым наблюдениям."
        )
    else:
        st.info("Нет завершённых прогнозов 1X2. Метрики появятся после сверки результатов.")
    calibration = (metrics.get("calibration_bins") or {}).get("rows") or []
    if calibration:
        st.subheader("Калибровка вероятностей")
        rows = [{
            "Исход": {"H": "П1", "D": "X", "A": "П2"}.get(row["outcome"], row["outcome"]),
            "Диапазон": f"{row['bin_from']:.0%}–{row['bin_to']:.0%}",
            "N": row["n"],
            "Средняя вероятность": row["mean_probability"],
            "Фактическая частота": row["observed_frequency"],
            "Разница": row["observed_minus_predicted"],
        } for row in calibration]
        st.dataframe(rows, width="stretch", hide_index=True)
        ece = (metrics.get("calibration_bins") or {}).get("expected_calibration_error") or {}
        st.caption(
            "ECE по классам: "
            + " · ".join(f"{key} {_format_number(value, 4)}" for key, value in ece.items())
            + ". Бины без наблюдений не оцениваются."
        )
    st.divider()
    st.subheader("Независимая проверка рынков")
    flags = _market_validation_flags([
        card for card in (D.get("cards") or []) if isinstance(card, dict)
    ])
    by_market = {
        str(row.get("market") or "OTHER"): row
        for row in (market_report.get("by_market") or [])
    }
    labels = {
        "1X2": "Победитель матча", "OU": "Тоталы голов",
        "BTTS": "Обе забьют", "STAT": "Угловые / карточки",
        "AH": "Asian Handicap",
    }
    table = []
    for code in ("1X2", "OU", "BTTS", "STAT", "AH"):
        item = by_market.get(code, {})
        table.append({
            "Рынок": labels[code],
            "OOS gate текущего движка": "Пройден" if flags[code] else "Не пройден / не подтверждён",
            "Записей в market ledger": int(item.get("n") or 0),
            "Сверено": int(item.get("settled") or 0),
            "ROI paper": _format_pct(item.get("roi")),
            "Brier кандидатов": _format_number(item.get("brier"), 4),
            "CLV": _format_pct(item.get("mean_clv")),
        })
    st.dataframe(table, width="stretch", hide_index=True)
    st.warning(
        "Market ledger содержит сохранённые выбранные рекомендации, а не все возможные прогнозы. "
        "Его ROI/Brier подвержены отбору кандидатов и не являются независимой оценкой всей модели."
    )
    st.caption(
        "BTTS, угловые, карточки и Asian Handicap остаются справочными до собственного "
        "непересекающегося walk-forward теста. Наличие формулы расчёта не означает валидацию."
    )


def _binary_metrics(rows: list[dict]) -> dict[str, Any]:
    valid = []
    for row in rows:
        try:
            result = str(row.get("result") or "")
            if result not in {"H", "D", "A"}:
                continue
            probs = row.get("probabilities") or {}
            normalized = {key: float(probs[key]) for key in ("H", "D", "A")}
            if any(not 0 <= value <= 1 or not math_is_finite(value) for value in normalized.values()):
                continue
            if abs(sum(normalized.values()) - 1.0) > 1e-6:
                continue
            valid.append((normalized, result))
        except (TypeError, ValueError, KeyError):
            continue
    if not valid:
        return {"n": 0, "brier": None, "log_loss": None, "accuracy": None}
    brier, logloss, correct = [], [], 0
    for probs, result in valid:
        brier.append(sum((float(probs[k]) - float(k == result)) ** 2 for k in ("H", "D", "A")))
        logloss.append(-__import__("math").log(max(float(probs[result]), 1e-15)))
        correct += max(("H", "D", "A"), key=lambda key: float(probs[key])) == result
    n = len(valid)
    return {"n": n, "brier": sum(brier) / n, "log_loss": sum(logloss) / n, "accuracy": correct / n}


def math_is_finite(value: Any) -> bool:
    try:
        import math
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _feature_groups(rows: list[dict]) -> list[dict]:
    definitions = (
        ("XG отмечен", lambda row, feature: row.get("xg_used") in (1, True)),
        ("XG не использован", lambda row, feature: row.get("xg_used") in (0, False)),
        ("Оба состава подтверждены", lambda row, feature: bool(feature.get("lineup_home_confirmed") and feature.get("lineup_away_confirmed"))),
        ("Нет двух подтверждённых составов", lambda row, feature: not bool(feature.get("lineup_home_confirmed") and feature.get("lineup_away_confirmed"))),
        ("Судья указан", lambda row, feature: bool(feature.get("referee_present"))),
        ("Судья не указан", lambda row, feature: not bool(feature.get("referee_present"))),
        ("Женские соревнования", lambda row, feature: feature.get("competition_gender") == "women"),
        ("Мужская категория", lambda row, feature: feature.get("competition_gender") == "men"),
    )
    output = []
    for label, predicate in definitions:
        selected = []
        for row in rows:
            feature = row.get("feature_snapshot") or {}
            if not isinstance(feature, dict):
                continue
            try:
                if predicate(row, feature):
                    selected.append(row)
            except (TypeError, ValueError):
                continue
        report = _binary_metrics(selected)
        if report["n"]:
            output.append({"Группа": label, **report})
    return output


def render_ai_learning(D: dict) -> None:
    st.header("AI LEARNING CENTER · Анализ ошибок")
    st.caption(
        "Система анализирует только прогнозы, сохранённые до начала матча, и результаты "
        "после подтверждённого FT. Результаты групп по признакам — наблюдательные, не причинные."
    )
    try:
        ledger, market = _safe_ledgers()
        all_rows = ledger.rows()
        settled = [row for row in all_rows if row.get("result") in {"H", "D", "A"}]
        metrics = ledger.metrics()
        market_metrics = market.metrics()
    except Exception as exc:
        st.error(f"Журнал обучения недоступен: {type(exc).__name__}")
        return
    if not settled:
        st.info(
            "Пока нет сверенных предматчевых прогнозов. Нужны новые записи и подтверждённые "
            "результаты; система не будет заполнять пропуски догадками."
        )
        return
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Сверено", len(settled))
    c2.metric("Brier", _format_number(metrics.get("brier"), 4))
    c3.metric("Log loss", _format_number(metrics.get("log_loss"), 4))
    c4.metric("Accuracy", _format_pct(metrics.get("accuracy")))
    st.subheader("Сравнение по данным, которые были известны до матча")
    groups = _feature_groups(settled)
    if groups:
        st.dataframe([{
            "Группа": row["Группа"], "N": row["n"],
            "Brier": _format_number(row["brier"], 4),
            "Log loss": _format_number(row["log_loss"], 4),
            "Accuracy": _format_pct(row["accuracy"]),
        } for row in groups], width="stretch", hide_index=True)
        st.caption(
            "Группы могут различаться по лигам, сезону, соперникам и размеру выборки. "
            "Это диагностические сравнения, а не доказательство, что конкретный фактор улучшил прогноз."
        )
    else:
        st.info(
            "В старых записях нет снимков признаков. Новые прогнозы будут сохранять "
            "контекст и признаки на момент расчёта; старые записи задним числом не дополняются."
        )

    st.subheader("Наиболее неожиданные фактические результаты")
    surprises = []
    for row in settled:
        probabilities = row.get("probabilities") or {}
        try:
            actual_probability = float(probabilities[str(row["result"])])
            surprises.append({
                "Матч": f"{row.get('home_team')} — {row.get('away_team')}",
                "Лига": row.get("league") or "—",
                "Прогноз P фактического исхода": actual_probability,
                "Результат": row.get("result"),
                "Ошибка log loss": -__import__("math").log(max(actual_probability, 1e-15)),
                "Признаки записаны": "Да" if row.get("feature_snapshot") else "Нет",
            })
        except (TypeError, ValueError, KeyError):
            continue
    surprises.sort(key=lambda item: item["Ошибка log loss"], reverse=True)
    if surprises:
        st.dataframe([{
            **item,
            "Прогноз P фактического исхода": _format_pct(item["Прогноз P фактического исхода"]),
            "Ошибка log loss": _format_number(item["Ошибка log loss"], 3),
        } for item in surprises[:15]], width="stretch", hide_index=True)

    st.subheader("Память модели и контроль изменений")
    pending = sum(row.get("result") is None for row in all_rows)
    st.write(f"Непроверенных предматчевых прогнозов в журнале: **{pending}**.")
    st.write(
        f"Повторные рыночные записи, исключённые из агрегатов: "
        f"**{int(market_metrics.get('duplicate_rows_excluded') or 0)}**."
    )
    st.warning(
        "Автоматическое переобучение параметров по последним результатам выключено. "
        "Изменения модели должны пройти walk-forward сравнение на будущих датах и сохранить "
        "улучшение Brier и Log loss; иначе текущая версия остаётся без изменений."
    )


def render_portfolio_risk(D: dict) -> None:
    st.header("PORTFOLIO RISK · Экспозиция и корреляция")
    st.caption(
        "Контроль относится к виртуальному портфелю. Он не размещает ставки у букмекеров."
    )
    caps = st.columns(3)
    match_cap = caps[0].slider("Лимит на матч, %", 1, 10, 5, key="risk_limit_match")
    league_cap = caps[1].slider("Лимит на лигу, %", 2, 25, 12, key="risk_limit_league")
    day_cap = caps[2].slider("Лимит на день, %", 5, 40, 20, key="risk_limit_day")
    enabled = st.checkbox(
        "Блокировать новую виртуальную позицию при превышении лимитов",
        value=bool(st.session_state.get("risk_guard_enabled", True)),
        key="risk_guard_enabled",
        help="Ограничения применяются к новым позициям сканера. Существующие позиции не удаляются.",
    )
    try:
        history = db.bank_history(limit=10000)
    except Exception:
        history = []
    report = portfolio_risk_report(
        D.get("bets") or [], bank=float(D.get("bank") or 0.0),
        bank_history=history, max_match_pct=match_cap / 100,
        max_league_pct=league_cap / 100, max_day_pct=day_cap / 100,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Открытых позиций", report["pending_count"])
    c2.metric("Открытая экспозиция", f"{report['pending_stake']:.2f}")
    c3.metric("Экспозиция / equity", _format_pct(report["open_risk_share"]))
    c4.metric("Макс. просадка банка", _format_pct(report["max_drawdown"]))
    if not enabled:
        st.warning("Защитное блокирование выключено. Ниже остаётся только мониторинг концентрации.")
    if report["breaches"]:
        for breach in report["breaches"]:
            st.error(breach)
    elif report["pending_count"]:
        st.success("Текущая экспозиция не превышает заданные лимиты.")
    else:
        st.info("Открытых виртуальных позиций пока нет.")
    st.subheader("Концентрация открытых позиций")
    for title, key in (
        ("По матчу", "by_match"), ("По лиге", "by_league"), ("По дню", "by_day"),
    ):
        items = report[key]
        with st.expander(title, expanded=(title == "По матчу")):
            if not items:
                st.caption("Нет открытой экспозиции.")
            else:
                st.dataframe([{
                    "Группа": item["key"], "Позиции": item["positions"],
                    "Сумма": round(item["stake"], 2),
                    "Доля equity": _format_pct(item["share"]),
                } for item in items[:30]], width="stretch", hide_index=True)
    st.subheader("Зависимые позиции")
    if report["correlated_clusters"]:
        st.dataframe([{
            "Матч": item["match"], "Дата": item["date"],
            "Количество позиций": item["positions"],
            "Рынки": ", ".join(item["markets"]),
            "Сумма": round(item["stake"], 2),
        } for item in report["correlated_clusters"]], width="stretch", hide_index=True)
        st.caption(
            "Одинаковый матч объединён в кластер риска. Это индикатор возможной зависимости, "
            "а не статистически рассчитанный коэффициент корреляции исходов."
        )
    else:
        st.caption("Несколько позиций на один матч не обнаружены.")


def render_api_center(D: dict) -> None:
    st.header("API CENTER · Источники и квоты")
    st.caption(
        "Проверяется наличие настройки ключа. Наличие ключа само по себе не означает, "
        "что провайдер ответил успешно или что его данные полны для каждого турнира."
    )
    sources = [
        ("API-Football", "api_football_key_input", ("API_FOOTBALL_KEY", "api_football_key")),
        ("football-data.org", "fdorg_token_input", ("FDORG_TOKEN", "fdorg_token")),
        ("SStats.net", "sstats_api_key_input", ("SSTATS_API_KEY", "sstats_api_key")),
        ("Highlightly", "highlightly_api_key_input", ("HIGHLIGHTLY_API_KEY", "highlightly_api_key")),
        ("Football API · xG", "footballapi_key_input", ("FOOTBALLAPI_KEY", "footballapi_key")),
        ("LLM", "llm_api_key_input", ("LLM_API_KEY", "llm_api_key")),
    ]
    configured = []
    for label, session_key, env_names in sources:
        present = _api_key_present(session_key, *env_names)
        configured.append({"Источник": label, "Ключ": "Настроен" if present else "Не настроен", "Проверка ответа": "Не выполнялась на этом экране"})
    st.dataframe(configured, width="stretch", hide_index=True)
    st.subheader("Лимиты API-Football")
    def read_int(name: str, default: int) -> int:
        raw = os.environ.get(name, "")
        try:
            raw = raw or str(st.secrets.get(name, "") or "")
            return max(0, int(raw)) if raw else default
        except Exception:
            return default
    daily_limit = read_int("API_FOOTBALL_DAILY_LIMIT", 100)
    reserve = min(daily_limit, read_int("API_FOOTBALL_DAILY_RESERVE", 10))
    lineup_cap = read_int("API_FOOTBALL_MAX_LINEUP_CALLS", 12)
    c1, c2, c3 = st.columns(3)
    c1.metric("Дневной лимит", daily_limit)
    c2.metric("Резерв", reserve)
    c3.metric("Лимит запросов составов / скан", lineup_cap)
    st.caption("Составы запрашиваются только в шестичасовом окне до начала матча; кэшированные ответы не тратят новый слот.")
    st.subheader("Диагностика последнего сканирования")
    report = [str(line) for line in (D.get("report") or []) if str(line).strip()]
    relevant = [line for line in report if any(
        token in line.casefold() for token in (
            "api-football", "sstats", "highlightly", "football-data",
            "football api", "xg:", "источники расписания",
        )
    )]
    if relevant:
        for line in relevant[-40:]:
            st.text(line)
    else:
        st.info("После первого сканирования здесь появятся сообщения источников.")
    st.warning(
        "Панель не показывает и не экспортирует сами секреты. На хостингах с временной файловой "
        "системой локально сохранённые ключи могут потребовать повторного ввода после перезапуска."
    )
