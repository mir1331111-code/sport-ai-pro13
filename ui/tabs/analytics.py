"""Analytics terminal: data quality, learning audit, model lab and portfolio risk."""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import streamlit as st

from config import APP_VERSION, DB_FILE
from model.analytics_quality import card_quality, portfolio_risk
from model.forecast_ledger import ForecastLedger, MarketForecastLedger
from model.scan_ledger import ScanLedger
from model.error_learner import OnlineErrorLearner
from model.learning_service import ForecastLearningService


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
    render_scan_history()
    active_version = f"main-model-v{APP_VERSION}"
    try:
        ledger = ForecastLedger(DB_FILE)
        metrics = ledger.metrics(model_version=active_version)
        rows = ledger.rows(settled_only=True, model_version=active_version)
        market = MarketForecastLedger(DB_FILE).metrics(model_version=active_version)
        learning = ForecastLearningService(
            ledger,
            OnlineErrorLearner(ledger, model_version_prefix=active_version),
            enabled=False,
            min_history=100,
            auto_promote=True,
            promotion_min_samples=200,
            promotion_min_relative_improvement=0.01,
        )
        learning_status = learning.status(as_of=datetime.now(timezone.utc))
    except Exception as exc:
        st.error(f"Не удалось прочитать журнал: {type(exc).__name__}")
        return
    total_forecasts = len(ledger.rows(model_version=active_version))
    settled_forecasts = int(metrics.get("n", 0) or 0)
    pending_forecasts = max(0, total_forecasts - settled_forecasts)
    a, b, c, d = st.columns(4)
    a.metric("Прогнозов текущей версии", total_forecasts)
    b.metric("Ожидают подтверждённого FT", pending_forecasts)
    c.metric("Закрыто", settled_forecasts)
    d.metric("Средний CLV", f"{metrics['mean_clv']:+.2%}" if metrics.get("mean_clv") is not None else "—")
    e, f = st.columns(2)
    e.metric("Brier score · только FT", f"{metrics['brier']:.4f}" if metrics.get("brier") is not None else "—")
    f.metric("Log loss · только FT", f"{metrics['log_loss']:.4f}" if metrics.get("log_loss") is not None else "—")

    st.caption(
        f"Качество и калибровка ниже рассчитаны только для активной версии {active_version}; "
        "старые версии остаются в истории и не смешиваются с текущей."
    )
    st.subheader("Постоянное накопление и обучение")
    l1, l2, l3 = st.columns(3)
    l1.metric("Записано решений калибровки", learning_status["decisions"])
    l2.metric("Прогнозов с допущенной поправкой", learning_status["promoted_decisions"])
    shadow = learning_status["shadow_evaluation"]
    l3.metric("Хронологическая shadow-выборка", shadow.get("n", 0))
    gate = learning_status["promotion_gate"]
    if gate.get("passed"):
        st.success(
            "Калибровка допущена для новых прогнозов: Brier и Log Loss улучшились "
            "на требуемой хронологической выборке."
        )
    else:
        st.info(
            "Обучение работает в теневом режиме. Исходные вероятности сохраняются; "
            "поправки включатся только после "
            f"{gate.get('minimum_samples', 200)} завершённых shadow-прогнозов "
            "и улучшения Brier и Log Loss минимум на "
            f"{gate.get('minimum_relative_improvement', 0.01):.0%}."
        )
    if shadow.get("n"):
        st.dataframe([{
            "N": shadow["n"],
            "Brier исходный": shadow.get("raw_brier"),
            "Brier после обучения": shadow.get("shadow_brier"),
            "Улучшение Brier": shadow.get("brier_relative_improvement"),
            "Log Loss исходный": shadow.get("raw_log_loss"),
            "Log Loss после обучения": shadow.get("shadow_log_loss"),
            "Улучшение Log Loss": shadow.get("log_loss_relative_improvement"),
        }], width="stretch", hide_index=True)

    backup = st.session_state.get("_sqlite_backup_status") or {}
    restore = st.session_state.get("_sqlite_restore_status") or {}
    if backup.get("ok"):
        st.success(
            "Удалённая зашифрованная копия SQLite подтверждена: "
            + str(backup.get("backup_at") or backup.get("checked_at") or "сохранена")
        )
    elif backup.get("state") == "not_configured" or restore.get("state") == "not_configured":
        st.warning(
            "Удалённая копия не настроена. На Streamlit Cloud SQLite-файл "
            "может исчезнуть при перезапуске или redeploy. Добавь GIST_ID, "
            "GITHUB_TOKEN и NEURO_BACKUP_KEY в Streamlit → Manage app → Settings → Secrets. "
            "После этого каждый скан сохранит зашифрованную копию; пустая база "
            "сможет автоматически восстановиться."
        )
    elif backup.get("state") == "backup_error":
        st.error("Не удалось сохранить удалённую копию: " + str(backup.get("message") or "ошибка"))
    st.caption(
        "Журнал прогнозов, проверенные результаты, решения калибровки и рынок "
        "находятся в одном SQLite DB_FILE; резервная копия включает все таблицы."
    )

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
        st.dataframe(display, width="stretch", hide_index=True)
    else:
        st.info("Пока нет подтверждённых завершённых прогнозов в ledger. Данные не подменяются статистикой портфеля.")
    st.subheader("Результаты по рынкам")
    market_rows = market.get("by_market") or []
    if market_rows:
        st.dataframe([{"Рынок": x.get("market"), "Прогнозов": x.get("n"),
                       "Brier": x.get("brier"), "Log loss": x.get("log_loss"),
                       "ROI": x.get("roi")} for x in market_rows],
                     width="stretch", hide_index=True)
    else:
        st.info("Рыночный ledger пока не содержит закрытых прогнозов.")
    st.info(
        "Калибровка использует только прежние подтверждённые результаты основной модели; "
        "прогнозы ИИ не смешиваются с её обучающей выборкой. Базовые веса Poisson/Elo, "
        "пороги Value и риск-профиль автоматически не переписываются."
    )


def _render_fixed_holdout_runner():
    """Run and publish the frozen 1X2 Model Lab v1 on an operator-supplied CSV."""
    from model.holdout import (
        MARKET_SOURCES,
        evaluate_fixed_holdout,
        load_fixed_holdout_gate_status,
        read_historical_csv_bytes,
        validate_historical_rows,
        write_model_lab_artifacts,
    )
    from storage import user_datasets

    st.subheader("Fixed Holdout Model Lab v1 · независимая проверка 1X2")
    st.caption(
        "Загрузи архив завершённых матчей, задай один фиксированный holdout-период "
        "и выбери ровно одного букмекера. Модель обучается только на матчах до даты "
        "начала holdout; каждый игровой день прогнозируется до добавления результатов "
        "этого дня. Результаты holdout нельзя использовать для подгонки параметров."
    )

    try:
        gate_status = load_fixed_holdout_gate_status()
    except Exception as exc:
        gate_status = {"passed": False, "status": "NOT_READY", "gate_file": "неизвестен"}
        st.warning(f"Не удалось прочитать текущий gate: {type(exc).__name__}")
    if gate_status.get("passed"):
        st.success(
            "Текущий promotion gate: PASS · "
            f"{gate_status.get('holdout_start')} — {gate_status.get('holdout_end')}"
        )
    else:
        st.warning(
            "Текущий promotion gate: NOT_READY. O/U 2.5 и другие вторичные рынки "
            "не получают автоматический допуск только от этой панели."
        )

    data_source = st.radio(
        "Источник исторических матчей",
        ("Новый CSV", "Сохранённый архив из SQLite"),
        horizontal=True,
        key="model_lab_v1_data_source",
    )
    parsed_rows = None
    field_names = []
    raw = b""
    dataset_source = "uploaded_csv"
    if data_source == "Новый CSV":
        uploaded = st.file_uploader(
            "Исторический CSV (football-data.co.uk или совместимая таблица)",
            type=["csv"],
            key="model_lab_v1_csv_upload",
            help=(
                "Нужны дата, лига, хозяева, гости и счёт FT. Для подтверждённого "
                "pre-kickoff сравнения с рынком дополнительно нужны две timezone-aware "
                "колонки: время снятия коэффициента и время начала матча."
            ),
        )
        if uploaded is not None:
            raw = uploaded.getvalue()
            try:
                parsed_rows, field_names = read_historical_csv_bytes(raw)
                st.caption(
                    f"Файл: {uploaded.name} · строк матчей: {len(parsed_rows):,} · "
                    f"размер: {len(raw) / (1024 * 1024):.2f} МБ."
                )
            except ValueError as exc:
                st.error(str(exc))
    else:
        saved_rows = st.session_state.get("_model_lab_v1_saved_rows")
        refresh_saved = st.button(
            "Перечитать сохранённый архив из SQLite",
            key="model_lab_v1_reload_saved_archive",
        )
        if saved_rows is None or refresh_saved:
            try:
                saved_rows = user_datasets.load_user_dataset("custom_historical_match_rows")
                st.session_state["_model_lab_v1_saved_rows"] = saved_rows
            except Exception as exc:
                saved_rows = []
                st.error(f"Не удалось прочитать архив из SQLite: {type(exc).__name__}: {exc}")
        if saved_rows:
            try:
                parsed_rows, field_names = validate_historical_rows(saved_rows)
                raw = json.dumps(
                    parsed_rows, ensure_ascii=False, sort_keys=True,
                    separators=(",", ":"), default=str,
                ).encode("utf-8")
                dataset_source = "sqlite:user_datasets.custom_historical_match_rows"
                divisions = sorted({
                    str(row.get("Div") or row.get("div") or row.get("league") or "").strip()
                    for row in parsed_rows
                } - {""})
                st.caption(
                    f"SQLite архив: {len(parsed_rows):,} матчей · "
                    f"лиги/Div: {', '.join(divisions[:20]) or 'не указаны'} · "
                    f"SHA-256: {hashlib.sha256(raw).hexdigest()[:16]}…"
                )
                st.info(
                    "Model Lab v1 считает только E0, D1, I1, SP1 и F1. "
                    "Другие сохранённые дивизионы будут прозрачно исключены из отчёта."
                )
            except ValueError as exc:
                st.error(str(exc))
        else:
            st.warning(
                "В SQLite пока нет custom_historical_match_rows. Сначала импортируй "
                "архив в Сканере → Данные контекста и исторические архивы или переключись на CSV."
            )

    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        holdout_start = st.text_input(
            "Начало holdout (включительно)", key="model_lab_v1_start",
            placeholder="YYYY-MM-DD",
        ).strip()
    with c2:
        holdout_end = st.text_input(
            "Конец holdout (включительно)", key="model_lab_v1_end",
            placeholder="YYYY-MM-DD",
        ).strip()
    market_options = list(MARKET_SOURCES)
    with c3:
        market_source = st.selectbox(
            "Единый букмекерский benchmark",
            options=market_options,
            format_func=lambda key: MARKET_SOURCES[key]["label"],
            key="model_lab_v1_market_source",
        )

    ts1, ts2 = st.columns(2)
    timestamp_options = ["— не указано —"] + field_names
    default_odds_ts = (
        field_names.index("OddsCapturedAt") + 1
        if "OddsCapturedAt" in field_names else 0
    )
    default_kickoff_ts = (
        field_names.index("KickoffAt") + 1
        if "KickoffAt" in field_names else 0
    )
    with ts1:
        odds_timestamp = st.selectbox(
            "Колонка времени снятия коэффициента",
            options=timestamp_options,
            index=min(default_odds_ts, len(timestamp_options) - 1),
            key="model_lab_v1_odds_timestamp",
        )
    with ts2:
        kickoff_timestamp = st.selectbox(
            "Колонка времени начала матча",
            options=timestamp_options,
            index=min(default_kickoff_ts, len(timestamp_options) - 1),
            key="model_lab_v1_kickoff_timestamp",
        )

    if odds_timestamp == "— не указано —" and kickoff_timestamp == "— не указано —":
        st.info(
            "Без обеих временных колонок отчёт всё равно рассчитает качество 1X2 "
            "и диагностическое сравнение с выбранным букмекером, но market snapshot "
            "не будет считаться доказанно снятым до kickoff, поэтому строгий PASS невозможен."
        )
    elif odds_timestamp == "— не указано —" or kickoff_timestamp == "— не указано —":
        st.warning("Для проверки pre-kickoff snapshot нужно выбрать обе временные колонки.")

    can_run = bool(
        parsed_rows
        and holdout_start
        and holdout_end
        and (
            (odds_timestamp == "— не указано —" and kickoff_timestamp == "— не указано —")
            or (odds_timestamp != "— не указано —" and kickoff_timestamp != "— не указано —")
        )
    )
    run = st.button(
        "Запустить фиксированный holdout",
        type="primary",
        width="stretch",
        disabled=not can_run,
        key="model_lab_v1_run",
    )
    if run:
        try:
            with st.spinner(
                "Считаю четыре хронологических ветки на одних и тех же матчах, "
                "строю paired benchmark и доверительные интервалы…"
            ):
                report = evaluate_fixed_holdout(
                    parsed_rows,
                    holdout_start=holdout_start,
                    holdout_end=holdout_end,
                    market_source=market_source,
                    dataset_sha256=hashlib.sha256(raw).hexdigest(),
                    dataset_source=dataset_source,
                    odds_timestamp_field=(
                        odds_timestamp if odds_timestamp != "— не указано —" else None
                    ),
                    kickoff_timestamp_field=(
                        kickoff_timestamp if kickoff_timestamp != "— не указано —" else None
                    ),
                )
                configured_gate = os.environ.get("MODEL_LAB_GATE_PATH", "").strip()
                output_dir = (
                    Path(configured_gate).expanduser().parent
                    if configured_gate else Path("reports/model_lab")
                )
                artifacts = write_model_lab_artifacts(report, output_dir)
                if configured_gate:
                    target_gate = Path(configured_gate).expanduser()
                    generated_gate = Path(artifacts["promotion_gate"])
                    if target_gate.resolve() != generated_gate.resolve():
                        target_gate.parent.mkdir(parents=True, exist_ok=True)
                        target_gate.write_bytes(generated_gate.read_bytes())
                    artifacts["promotion_gate"] = str(target_gate)
                st.session_state["_model_lab_v1_report"] = report
                st.session_state["_model_lab_v1_artifacts"] = artifacts
        except Exception as exc:
            st.error(f"Fixed Holdout Model Lab не завершился: {type(exc).__name__}: {exc}")

    report = st.session_state.get("_model_lab_v1_report")
    artifacts = st.session_state.get("_model_lab_v1_artifacts") or {}
    if not isinstance(report, dict):
        return
    candidate = report.get("candidate") or {}
    market = report.get("market") or {}
    paired = market.get("candidate_minus_market") or {}
    difference = paired.get("differences") or {}
    intervals = paired.get("confidence_intervals_95") or {}
    log_ci = intervals.get("log_loss") or {}
    readiness = report.get("readiness") or {}
    data_quality = report.get("data_quality") or {}

    st.divider()
    status = readiness.get("status") or "NOT_READY"
    if status == "PASS":
        st.success("Fixed Holdout Model Lab v1: PASS · строгий gate пройден.")
    else:
        st.warning(
            "Fixed Holdout Model Lab v1: NOT_READY · "
            + str(readiness.get("explanation") or "Один или несколько обязательных критериев не выполнены.")
        )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Матчей holdout", f"{candidate.get('n', 0):,}")
    m2.metric("Brier · H/D/A", f"{candidate.get('brier'):.4f}" if candidate.get("brier") is not None else "—")
    m3.metric("Log Loss", f"{candidate.get('log_loss'):.4f}" if candidate.get("log_loss") is not None else "—")
    m4.metric("Accuracy", f"{candidate.get('accuracy'):.1%}" if candidate.get("accuracy") is not None else "—")

    market_candidate = market.get("candidate_on_same_fixtures") or {}
    st.subheader("Одна и та же выборка: модель против рынка")
    st.dataframe([{
        "Показатель": "Brier",
        "Рынок": (market.get("metrics") or {}).get("brier"),
        "Модель на тех же матчах": market_candidate.get("brier"),
        "Модель − рынок": difference.get("brier"),
    }, {
        "Показатель": "Log Loss",
        "Рынок": (market.get("metrics") or {}).get("log_loss"),
        "Модель на тех же матчах": market_candidate.get("log_loss"),
        "Модель − рынок": difference.get("log_loss"),
    }, {
        "Показатель": "Количество пар",
        "Рынок": market.get("n", 0),
        "Модель на тех же матчах": market.get("n", 0),
        "Модель − рынок": None,
    }], width="stretch", hide_index=True)
    st.caption(
        f"Букмекер: {market.get('source_label') or market.get('source') or market_source} · "
        f"95% CI для Δ Log Loss: [{log_ci.get('low', '—')}, {log_ci.get('high', '—')}] · "
        f"Pre-kickoff snapshot: "
        f"{'подтверждён' if readiness.get('market_snapshot_verified') else 'не подтверждён'}."
    )
    timestamp_audit = market.get("timestamp_audit") or {}
    st.info(str(timestamp_audit.get("note") or "Временные метки рынка отсутствуют."))

    league_rows = []
    for code, label in ((report.get("manifest") or {}).get("leagues") or {}).items():
        metric = (((report.get("stratified") or {}).get("by_league") or {}).get(code) or {}).get("arms", {}).get("full") or {}
        league_rows.append({
            "Лига": label,
            "N": metric.get("n", 0),
            "Brier": metric.get("brier"),
            "Log Loss": metric.get("log_loss"),
            "Accuracy": metric.get("accuracy"),
        })
    st.subheader("Пять фиксированных лиг")
    st.dataframe(league_rows, width="stretch", hide_index=True)

    st.subheader("Строгие критерии promotion gate")
    gate_rows = [
        {"Критерий": "Объём holdout / покрытие 5 лиг / число игровых дней",
         "Статус": "PASS" if readiness.get("sample_size_met") else "NOT MET"},
        {"Критерий": "Brier ≤ 0.57 и Log Loss ≤ 0.99",
         "Статус": "PASS" if readiness.get("absolute_targets_met") else "NOT MET"},
        {"Критерий": "≥ 3,000 пар с единым букмекером",
         "Статус": "PASS" if readiness.get("market_paired_sample_met") else "NOT MET"},
        {"Критерий": "Время снятия коэффициента строго до kickoff",
         "Статус": "PASS" if readiness.get("market_snapshot_verified") else "NOT VERIFIED"},
        {"Критерий": "Верхняя граница 95% CI Δ Log Loss ≤ +0.02",
         "Статус": "PASS" if readiness.get("not_worse_than_market") else "NOT MET"},
        {"Критерий": "Итоговый строгий gate",
         "Статус": "PASS" if readiness.get("strict_holdout_gate_passed") else "NOT READY"},
    ]
    st.dataframe(gate_rows, width="stretch", hide_index=True)
    st.caption(
        f"Training rows before holdout: {data_quality.get('training_rows_before_holdout', 0):,} · "
        f"invalid rows: {data_quality.get('invalid_rows', 0):,} · "
        f"duplicates skipped: {data_quality.get('duplicate_rows_skipped', 0):,} · "
        f"experiment ID: {report.get('experiment_id')}."
    )

    download_map = (
        ("JSON", "json", "application/json"),
        ("Markdown", "markdown", "text/markdown"),
        ("Reliability SVG", "reliability_svg", "image/svg+xml"),
        ("Promotion gate JSON", "promotion_gate", "application/json"),
    )
    download_columns = st.columns(4)
    for column, (label, key, mime) in zip(download_columns, download_map):
        path = artifacts.get(key)
        if not path:
            continue
        try:
            data = Path(path).read_bytes()
        except OSError:
            continue
        file_path = Path(path)
        column.download_button(
            f"Скачать {label}",
            data=data,
            file_name=file_path.name,
            mime=mime,
            key=f"model_lab_v1_download_{key}",
            width="stretch",
        )


def render_model_lab():
    _render_fixed_holdout_runner()
    try:
        ledger = ForecastLedger(DB_FILE)
        active_version = f"main-model-v{APP_VERSION}"
        metrics = ledger.metrics(model_version=active_version)
        reliability = ledger.reliability_bins(bins=10, model_version=active_version)
        market_metrics = MarketForecastLedger(DB_FILE).metrics(model_version=active_version)
    except Exception as exc:
        st.error(f"Ошибка чтения модельного аудита: {type(exc).__name__}")
        return
    all_forecasts = ledger.rows(model_version=active_version)
    n = int(metrics.get("n", 0) or 0)
    total_forecasts = len(all_forecasts)
    pending_forecasts = max(0, total_forecasts - n)
    xg = metrics.get("xg_usage") or {}
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Всего pre-match прогнозов", total_forecasts)
    c2.metric("Ожидают результата", pending_forecasts)
    c3.metric("Закрыто · FT", n)
    c4.metric("xG coverage", f"{xg.get('share'):.1%}" if xg.get("share") is not None else "—")
    st.caption(
        f"Все показатели этого блока относятся к активной версии {active_version}. "
        "Исторические версии показаны отдельно в журнале сканирований/прогнозов."
    )
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
                      for x in reliability["rows"]], width="stretch", hide_index=True)
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
        } for item in market_rows], width="stretch", hide_index=True)
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
                 width="stretch", hide_index=True) if report["fixture_top"] else st.info("Открытых позиций пока нет.")
    st.subheader("Риск по лигам")
    st.dataframe([{"Лига": k, "Открытый риск": round(v,2)} for k,v in report["league_top"]],
                 width="stretch", hide_index=True) if report["league_top"] else st.info("Нет открытых позиций.")
    st.subheader("Риск по дням")
    st.dataframe([{"Дата": k, "Открытый риск": round(v,2)} for k,v in report["day_top"]],
                 width="stretch", hide_index=True) if report["day_top"] else st.info("Нет открытых позиций.")
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
        st.dataframe(rows, width="stretch", hide_index=True)
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
