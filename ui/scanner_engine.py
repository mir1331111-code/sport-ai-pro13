"""Model training and fitted-engine cache for the match scanner.

This module owns the slow historical training path and model reuse lifecycle;
the scanner UI imports the three established private entry points below.
"""
from __future__ import annotations

import copy
import os
import threading
import time
from datetime import datetime

import streamlit as st

from config import (
    APP_VERSION, DATA_VERSION, DIV_NAMES, DIV_TO_TSDB, DISK_CACHE_DIR,
)
from model.engine_cache import (
    engine_cache_fingerprint, incremental_engine_checkpoint_key,
)
from data.sources import load_seasonal, season_str, parse_date
from model.engine import Engine, _norm
from model.shot_xg import SHOT_XG_VERSION, enrich_history_with_shot_xg
from model.validation_cache import (
    select_validation_rows, validation_cache_key, validation_schedule_key,
)
from model.training import (
    TRAINING_STATE_SCHEMA, plan_incremental_training,
    prepare_completed_training_rows, training_day_digests,
    train_engine_chronologically,
)

# Keep validation leagues module-scoped: training and validation are separate
# functions, so a local variable inside the history loader is not visible here.
_PRIORITY_DIVS = (
    "E0", "SP1", "I1", "D1", "F1", "E1", "SP2", "I2",
    "D2", "F2", "N1", "B1", "P1", "T1", "R1",
)

# Source-level seasonal CSVs already have a three-day disk TTL. Keep one
# process-local, read-only assembled snapshot for 15 minutes so repeated Streamlit
# scans do not decompress, parse and sort every league-season archive again.
_HISTORY_SNAPSHOT_TTL_SEC = 15 * 60
_HISTORY_SNAPSHOT_CACHE = {}
_HISTORY_SNAPSHOT_LOCK = threading.RLock()


def _validation_divisions(train_divs):
    """Return priority validation leagues in stable order, limited to trained divisions."""
    available = set(train_divs or ())
    return tuple(division for division in _PRIORITY_DIVS if division in available)


def _load_seasonal_history(logs, update_loader):
    """Load a deterministic seasonal snapshot with a short-lived process cache."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cur_year = today.year if today.month >= 7 else today.year - 1
    history_years = list(range(cur_year - 4, cur_year + 1))
    priority_divs = list(_PRIORITY_DIVS)
    supported_divs = list(DIV_TO_TSDB.keys()) + ["E2", "E3", "D3"]
    train_divs = list(dict.fromkeys(priority_divs + supported_divs))
    cache_key = (DATA_VERSION, tuple(history_years), tuple(train_divs))
    now_mono = time.monotonic()

    with _HISTORY_SNAPSHOT_LOCK:
        cached = _HISTORY_SNAPSHOT_CACHE.get(cache_key)
        if (
            isinstance(cached, tuple)
            and len(cached) == 2
            and cached[0] > now_mono
            and isinstance(cached[1], dict)
        ):
            history = cached[1]
            logs.append(
                "⚡ Исторический архив: собранный снимок из памяти процесса "
                "(повторное чтение сезонных CSV пропущено)."
            )
            for i, division in enumerate(train_divs):
                pct = 0.04 + (i + 1) / max(1, len(train_divs)) * 0.14
                update_loader(
                    f"История из памяти [{i+1}/{len(train_divs)}] — "
                    f"{DIV_NAMES.get(division, division)}",
                    pct, logs,
                )
                seasons = history.get(division, [])
                for year, rows in zip(history_years, seasons):
                    if rows:
                        logs.append(
                            f"  {season_str(year)}: {len(rows)} матчей (memory)"
                        )
                league_rows = sum(len(rows) for rows in seasons)
                logs.append(
                    f"OK {DIV_NAMES.get(division, division)}: {league_rows} матчей "
                    f"за {len(history_years)} сезонов (memory)"
                )
            return {
                "history_years": history_years,
                "train_divs": train_divs,
                "history": history,
            }

        history = {}
        for i, division in enumerate(train_divs):
            pct = 0.04 + (i + 1) / max(1, len(train_divs)) * 0.14
            update_loader(
                f"Проверка истории [{i+1}/{len(train_divs)}] — "
                f"{DIV_NAMES.get(division, division)}",
                pct, logs,
            )
            seasons = []
            for year in history_years:
                rows = load_seasonal(division, season_str(year))
                rows = sorted(
                    rows,
                    key=lambda row: parse_date(row.get("Date", "")) or datetime.min,
                )
                seasons.append(rows)
                if rows:
                    logs.append(f"  {season_str(year)}: {len(rows)} матчей")
            history[division] = seasons
            league_rows = sum(len(rows) for rows in seasons)
            logs.append(
                f"OK {DIV_NAMES.get(division, division)}: {league_rows} матчей "
                f"за {len(history_years)} сезонов"
            )

        _HISTORY_SNAPSHOT_CACHE.clear()
        _HISTORY_SNAPSHOT_CACHE[cache_key] = (
            time.monotonic() + _HISTORY_SNAPSHOT_TTL_SEC,
            history,
        )
        return {
            "history_years": history_years,
            "train_divs": train_divs,
            "history": history,
        }


def _train_engine(
    matrix_n, logs, update_loader, footballapi_key="",
    footballapi_xg_rows=None, custom_historical_rows=None,
    seasonal_snapshot=None, incremental_checkpoint=None,
    incremental_key=None,
):
    if seasonal_snapshot is None:
        seasonal_snapshot = _load_seasonal_history(logs, update_loader)
    history_years = list(seasonal_snapshot["history_years"])
    train_divs = list(seasonal_snapshot["train_divs"])
    history = copy.deepcopy(seasonal_snapshot["history"])
    engine = Engine(matrix_n=matrix_n)
    from model.historical_match_stats import (
        HistoricalMatchStats, walk_forward_market_validation,
    )
    engine.match_stats = HistoricalMatchStats()

    # Optional user-supplied histories extend base training to competitions
    # not covered by football-data.co.uk (women's leagues, national teams, etc.).
    # They are kept in their own Div namespace and excluded from the existing
    # major-league bookmaker validation sample until separately benchmarked.
    custom_by_division = {}
    for row in custom_historical_rows or []:
        if not isinstance(row, dict):
            continue
        division = str(row.get("Div") or "").strip()
        if division:
            custom_by_division.setdefault(division, []).append(dict(row))
    custom_added = 0
    for division, custom_rows in custom_by_division.items():
        existing_keys = set()
        for existing_season in history.get(division, []) or []:
            for existing in existing_season or []:
                stamp = parse_date(existing.get("Date") or existing.get("date"))
                key = (
                    stamp.date().isoformat() if stamp else "",
                    _norm(existing.get("HomeTeam") or existing.get("home_team") or ""),
                    _norm(existing.get("AwayTeam") or existing.get("away_team") or ""),
                )
                if all(key):
                    existing_keys.add(key)
        unique_rows = []
        for custom in custom_rows:
            stamp = parse_date(custom.get("Date") or custom.get("date"))
            key = (
                stamp.date().isoformat() if stamp else "",
                _norm(custom.get("HomeTeam") or ""),
                _norm(custom.get("AwayTeam") or ""),
            )
            if not all(key) or key in existing_keys:
                continue
            existing_keys.add(key)
            custom["Div"] = division
            unique_rows.append(custom)
        if unique_rows:
            unique_rows.sort(
                key=lambda row: parse_date(row.get("Date") or "") or datetime.min
            )
            history.setdefault(division, []).append(unique_rows)
            if division not in train_divs:
                train_divs.append(division)
            custom_added += len(unique_rows)
    logs.append(
        f"📥 Пользовательский исторический архив: добавлено {custom_added} матчей "
        f"по {len(custom_by_division)} кодам соревнований; новые лиги не допущены "
        "к Value без отдельной рыночной валидации."
    )

    # Enrich exact matching EPL history rows with genuine provider xG.
    # Unmatched provider events must not become duplicate historical fixtures.
    xg_rows = list(footballapi_xg_rows or [])
    if xg_rows and "E0" in history and history["E0"]:
        current_rows = history["E0"][-1]
        index = {}
        for row in current_rows:
            date_value = parse_date(row.get("Date") or "")
            if not date_value:
                continue
            key = (
                date_value.date().isoformat(),
                _norm(row.get("HomeTeam") or ""),
                _norm(row.get("AwayTeam") or ""),
            )
            if all(key):
                index[key] = row
        enriched = already_present = replaced_proxy = unmatched = 0
        for xg_row in xg_rows:
            date_value = parse_date(xg_row.get("Date") or "")
            if not date_value:
                unmatched += 1
                continue
            key = (
                date_value.date().isoformat(),
                _norm(xg_row.get("HomeTeam") or ""),
                _norm(xg_row.get("AwayTeam") or ""),
            )
            if not all(key):
                unmatched += 1
                continue
            existing = index.get(key)
            if existing is None:
                unmatched += 1
                continue
            if existing.get("HxG") not in (None, "") and existing.get("AxG") not in (None, ""):
                if str(existing.get("xg_kind") or "").casefold() == "modelled_proxy":
                    for field in ("HxG", "AxG", "xg_source", "xg_kind",
                                  "xg_model_version", "xg_training_sides"):
                        existing.pop(field, None)
                    replaced_proxy += 1
                else:
                    already_present += 1
                    continue
            existing["HxG"] = xg_row["HxG"]
            existing["AxG"] = xg_row["AxG"]
            existing["xg_source"] = xg_row.get("xg_source", "footballapi.org")
            existing.pop("xg_kind", None)
            existing.pop("xg_model_version", None)
            existing.pop("xg_training_sides", None)
            enriched += 1
        logs.append(
            f"✅ Football API xG: обогащено={enriched} (включая замену proxy={replaced_proxy}), "
            f"уже были={already_present}, не сопоставлено={unmatched}; "
            f"дубликаты исторических матчей не создавались."
        )
    elif not footballapi_key:
        logs.append(
            "ℹ️ Football API xG-ключ не задан; доступны архивные и модельные xG."
        )

    # Estimate xG coverage from aggregate shot counts with a separate, dated model.
    # The function estimates a whole calendar day before learning that day's results.
    enrich_history_with_shot_xg(history, train_divs, logs)

    # Prepare the exact post-enrichment training sequence before deciding
    # whether the saved Engine can be continued safely.
    training_rows, skipped_rows = prepare_completed_training_rows(
        history, train_divs,
    )
    if not training_rows:
        raise RuntimeError(
            "Не найдено ни одного исторического матча с валидной датой, "
            "командами и финальным счётом; обучение остановлено без подстановки 0:0."
        )
    current_day_digests = training_day_digests(training_rows)
    checkpoint_for_plan = (
        incremental_checkpoint
        if isinstance(incremental_checkpoint, dict)
        and incremental_checkpoint.get("config_key") == incremental_key
        else None
    )
    incremental_plan = plan_incremental_training(
        checkpoint_for_plan, current_day_digests,
    )
    force_market_validation = bool(
        checkpoint_for_plan is not None
        and incremental_plan["mode"] == "full"
        and incremental_plan["reason"] not in {"no_checkpoint"}
    )

    # The validation sample is intentionally bounded: full replays over every
    # competition/season on each cold start made Streamlit appear frozen.
    # 15 main leagues × latest 500 complete fixtures retains >500 observations
    # across competitions for each supported market while limiting CPU time.
    validation_divs = _validation_divisions(train_divs)
    # Keep a broad multi-league validation set, but avoid replaying 7k+
    # fixtures through eight chronological model arms on the first cold start.
    # The global >=500 paired-observation promotion gates remain unchanged.
    validation_rows = select_validation_rows(
        history, validation_divs, max_rows_per_league=250,
    )
    from security import cache_get, cache_put
    from model.walk_forward import walk_forward_compare, totals_market_is_validated
    validation_key = validation_cache_key(
        validation_rows, matrix_n=matrix_n, model_version=APP_VERSION,
    )
    schedule_key = validation_schedule_key(
        matrix_n=matrix_n, model_version=APP_VERSION,
        divisions=validation_divs, max_rows_per_league=250,
        min_history=30, min_market_n=500,
    )
    try:
        validation_refresh_days = max(
            1, int(os.environ.get("MODEL_MARKET_VALIDATION_REFRESH_DAYS", "7"))
        )
    except (TypeError, ValueError, OverflowError):
        validation_refresh_days = 7

    cached_validation = None
    if (
        not force_market_validation
        and incremental_plan["mode"] in {"append", "unchanged"}
    ):
        # New results are admitted incrementally every scan, but expensive
        # market walk-forward is refreshed on a separate seven-day cadence.
        cached_validation = cache_get(
            schedule_key, validation_refresh_days * 86400,
        )
        if isinstance(cached_validation, dict):
            logs.append(
                f"⏱️ Рыночная walk-forward проверка не старше {validation_refresh_days} дн.; "
                "повторный replay в обычном сканировании пропущен."
            )
    if (
        cached_validation is None
        and not force_market_validation
        and incremental_plan["mode"] == "full"
    ):
        # Exact-content cache is appropriate for a full build. Incremental
        # runs obey the schedule TTL above and refresh when that TTL expires.
        cached_validation = cache_get(validation_key, 30 * 86400)

    if isinstance(cached_validation, dict):
        totals_backtest = cached_validation.get("totals_backtest") or {}
        stat_report = cached_validation.get("stat_report") or {}
        logs.append(
            f"⚡ Историческая проверка из кэша: {len(validation_rows)} матчей, "
            f"{len(validation_divs)} лиг."
        )
    else:
        update_loader(
            f"Историческая проверка рынков: подготовлено {len(validation_rows)} матчей",
            0.52, logs,
        )

        def _walk_forward_progress(processed, total):
            ratio = processed / max(1, total)
            update_loader(
                f"Проверка 1X2 / тоталов / xG: {processed}/{total} матчей",
                0.52 + 0.10 * ratio,
                logs,
            )

        def _stat_market_progress(processed, total):
            ratio = processed / max(1, total)
            update_loader(
                f"Проверка угловых / карточек: {processed}/{total} матчей",
                0.63 + 0.10 * ratio,
                logs,
            )

        try:
            totals_backtest = walk_forward_compare(
                validation_rows, matrix_n=matrix_n, bootstrap_samples=0,
                totals_bootstrap_samples=1000, market_bootstrap_samples=1000,
                progress_callback=_walk_forward_progress,
            )
            update_loader(
                "Хронологическая проверка угловых и карточек",
                0.63, logs,
            )
            stat_report = walk_forward_market_validation(
                validation_rows, min_history=30, min_n=500,
                bootstrap_samples=1000,
                progress_callback=_stat_market_progress,
            )
            validation_payload = {
                "totals_backtest": totals_backtest,
                "stat_report": stat_report,
            }
            cache_put(validation_key, validation_payload)
            cache_put(schedule_key, validation_payload)
            logs.append(
                f"Историческая проверка рассчитана: {len(validation_rows)} матчей, "
                f"{len(validation_divs)} лиг; результат сохранён в кэш."
            )
        except Exception as exc:
            totals_backtest = {}
            stat_report = {}
            engine.winner_market_validation = False
            engine.winner_market_validation_report = {
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"
            }
            engine.total_market_validation = False
            engine.total_market_validation_by_line = {}
            engine.total_market_validation_report = {
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"
            }
            engine.btts_market_validation = False
            engine.btts_market_validation_report = {
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"
            }
            engine.stat_market_validation = {}
            engine.stat_market_validation_report = {
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"
            }
            logs.append(
                "⚠️ Историческая проверка рынков не завершилась: "
                f"{type(exc).__name__}: {str(exc)[:160]}. "
                "Непроверенные рынки закрыты для рекомендаций."
            )

    if isinstance(totals_backtest, dict):
        reliability = (
            (totals_backtest.get("candidate") or {}).get("reliability") or {}
        )
        for code, label in (("H", "П1"), ("D", "X"), ("A", "П2")):
            class_report = (reliability.get("classes") or {}).get(code) or {}
            populated = [
                item for item in (class_report.get("bins") or [])
                if int(item.get("count") or 0) > 0
            ]
            summary = "; ".join(
                f"{float(item['lower']):.1f}–{float(item['upper']):.1f}: "
                f"n={int(item['count'])}, "
                f"прогноз={float(item['mean_predicted']):.1%}, "
                f"факт={float(item['observed_frequency']):.1%}"
                for item in populated
            ) or "нет заполненных диапазонов"
            ece = class_report.get("ece")
            mce = class_report.get("mce")
            logs.append(
                f"🎯 Калибровка 1X2 {label}: n={class_report.get('n', 0)}, "
                f"ECE={float(ece):.4f}" if ece is not None else
                f"🎯 Калибровка 1X2 {label}: n={class_report.get('n', 0)}, ECE=нет данных"
            )
            logs.append(
                f"   Надёжность {label}: MCE={float(mce):.4f}; {summary}"
                if mce is not None
                else f"   Надёжность {label}: MCE=нет данных; {summary}"
            )

    xg_ablation_report = (
        totals_backtest.get("xg_ablation") if isinstance(totals_backtest, dict) else None
    ) or {}
    engine.xg_ablation_report = xg_ablation_report

    if xg_ablation_report:
        xg_arms = xg_ablation_report.get("arms") or {}
        xg_usage = xg_ablation_report.get("usage") or {}
        xg_source_rows = xg_ablation_report.get("source_rows") or {}
        logs.append(
            "📊 xG абляция на одинаковых матчах: "
            f"provider-строк={xg_source_rows.get('provider', 0)}, "
            f"proxy-строк={xg_source_rows.get('modelled_proxy', 0)}; "
            "95% CI рассчитаны bootstrap по календарным игровым дням."
        )
        for arm in ("goals_only", "provider_xg_only", "proxy_xg_only", "mixed_provider_preferred"):
            report = xg_arms.get(arm) or {}
            usage = xg_usage.get(arm) or {}
            logs.append(
                f"xG arm {arm}: n={report.get('n', 0)}, "
                f"Brier={report.get('brier')}, LogLoss={report.get('log_loss')}, "
                f"xG использован в {usage.get('forecasts_using_xg', 0)} прогнозах "
                f"(provider={usage.get('provider_xg_used', 0)}, proxy={usage.get('proxy_xg_used', 0)})."
            )
        paired = xg_ablation_report.get("paired_differences") or {}
        for comparison in ("provider_vs_goals_only", "proxy_vs_goals_only", "mixed_vs_goals_only"):
            report = paired.get(comparison) or {}
            diff = report.get("differences") or {}
            ci = report.get("confidence_intervals_95") or {}
            logs.append(
                f"xG A/B {comparison}: n={report.get('n', 0)}, "
                f"ΔBrier={diff.get('brier')}, "
                f"CI={((ci.get('brier') or {}).get('low'))}…{((ci.get('brier') or {}).get('high'))}, "
                f"ΔLogLoss={diff.get('log_loss')}, "
                f"CI={((ci.get('log_loss') or {}).get('low'))}…{((ci.get('log_loss') or {}).get('high'))}."
            )

        stratified = xg_ablation_report.get("stratified") or {}
        league_seasons = stratified.get("by_league_season") or {}
        for league in sorted(league_seasons):
            for season, report in sorted(league_seasons[league].items()):
                if int(report.get("n", 0) or 0) < 30:
                    continue
                comparisons = report.get("paired_differences") or {}
                parts = []
                for comparison in (
                    "provider_vs_goals_only", "proxy_vs_goals_only",
                    "mixed_vs_goals_only",
                ):
                    paired_report = comparisons.get(comparison) or {}
                    differences = paired_report.get("differences") or {}
                    parts.append(
                        f"{comparison}: ΔBrier={differences.get('brier')}, "
                        f"ΔLogLoss={differences.get('log_loss')}"
                    )
                confidence_tag = (
                    "инференциальная выборка"
                    if report.get("inference_eligible") else "описательная выборка"
                )
                logs.append(
                    f"xG лига/сезон {league} {season}: n={report.get('n')}, "
                    f"игровых дней={report.get('cluster_count')}, {confidence_tag}; "
                    + "; ".join(parts)
                )
        if stratified:
            logs.append(
                "ℹ️ Сезоны: явное поле источника имеет приоритет; иначе используется "
                "оценка июль–июнь. Для лиг с календарным сезоном проверьте источник поля Season."
            )

    if totals_backtest:
        # Value requires an out-of-sample improvement over a de-vigged
        # bookmaker benchmark, not merely a high model probability.
        engine.winner_market_validation_report = (
            totals_backtest.get("candidate_minus_market") or {}
        )
        engine.winner_market_validation = totals_market_is_validated(
            engine.winner_market_validation_report, min_n=500
        )
        engine.total_market_validation_report = totals_backtest.get(
            "goals_over25_candidate_minus_market", {})
        line_reports = totals_backtest.get("goals_totals_by_line") or {}
        engine.total_market_validation_by_line = {
            str(line): bool(report.get("validation_passed"))
            for line, report in line_reports.items()
        }
        # Line-vs-league-baseline validation is useful for forecast screening,
        # but it is NOT proof of an edge over bookmakers. Automatic total-goals
        # staking requires the separate paired candidate-vs-market benchmark.
        engine.total_market_validation = totals_market_is_validated(
            engine.total_market_validation_report, min_n=500
        )
        engine.btts_market_validation_report = totals_backtest.get("btts") or {}
        engine.btts_market_validation = bool(
            engine.btts_market_validation_report.get("validation_passed")
        )
        total_diff = engine.total_market_validation_report.get("differences", {})
        total_ci = engine.total_market_validation_report.get("confidence_intervals_95", {})
        logs.append(
            f"Исторический walk-forward тоталов: прогнозов="
            f"{totals_backtest.get('goals_over25', {}).get('n', 0)}, "
            f"парных с рынком={engine.total_market_validation_report.get('n', 0)}, "
            f"источники={totals_backtest.get('goals_over25_market_sources', {})}"
        )
        winner_diff = engine.winner_market_validation_report.get("differences", {})
        winner_ci = engine.winner_market_validation_report.get("confidence_intervals_95", {})
        logs.append(
            f"1X2 vs рынок: пар={engine.winner_market_validation_report.get('n', 0)}, "
            f"ΔBrier={winner_diff.get('brier')}, ΔLogLoss={winner_diff.get('log_loss')}, "
            f"CI Brier={winner_ci.get('brier', {}).get('low')}…{winner_ci.get('brier', {}).get('high')}, "
            f"CI LogLoss={winner_ci.get('log_loss', {}).get('low')}…{winner_ci.get('log_loss', {}).get('high')}; "
            f"bootstrap={engine.winner_market_validation_report.get('bootstrap_unit', 'unknown')}, "
            f"clusters={engine.winner_market_validation_report.get('cluster_count', 0)}; "
            f"Value-допуск={'да' if engine.winner_market_validation else 'нет'}"
        )
        if total_diff:
            logs.append(
                f"ТБ/ТМ vs рынок: ΔBrier={total_diff.get('brier'):.4f}, "
                f"ΔLogLoss={total_diff.get('log_loss'):.4f}; "
                f"95% CI Brier={total_ci.get('brier', {}).get('low')}…{total_ci.get('brier', {}).get('high')}, "
                f"LogLoss={total_ci.get('log_loss', {}).get('low')}…{total_ci.get('log_loss', {}).get('high')}"
            )
        logs.append(
            "1X2 допущен к Value только при полной линии одного букмекера."
            if engine.winner_market_validation else
            "1X2 пока forecast-only: нет ≥500 парных матчей с доказанным улучшением над de-vig рынком."
        )
        logs.append(
            "ТБ/ТМ 2.5 допущены к Value только при полной линии одного букмекера."
            if engine.total_market_validation else
            "Тоталы пока forecast-only: нет ≥500 парных матчей с доказанным улучшением над de-vig рынком."
        )
        for line, line_report in line_reports.items():
            metric = line_report.get("metrics") or {}
            paired = line_report.get("candidate_minus_baseline") or {}
            logs.append(
                f"ТБ/ТМ {line}: WF n={metric.get('n', 0)}, "
                f"baseline-пар={paired.get('n', 0)}, "
                f"ΔBrier={(paired.get('differences') or {}).get('brier')}, "
                f"ΔLogLoss={(paired.get('differences') or {}).get('log_loss')}, "
                f"лучше средней лиги={'да' if line_report.get('validation_passed') else 'нет'}; "
                f"букмекерский Value=не проверен"
            )
        logs.append(
            f"BTTS: WF n={engine.btts_market_validation_report.get('candidate', {}).get('n', 0)}, "
            f"пар={engine.btts_market_validation_report.get('candidate_minus_baseline', {}).get('n', 0)}, "
            f"лучше средней лиги={'да' if engine.btts_market_validation else 'нет'}; "
            "букмекерский Value=не проверен"
        )

    if stat_report:
        engine.stat_market_validation_report = stat_report
        engine.stat_market_validation = {
            market: {
                str(line): bool(line_report.get("validation_passed"))
                for line, line_report in line_reports.items()
            }
            for market, line_reports in (stat_report.get("markets") or {}).items()
        }
        for market, line_reports in engine.stat_market_validation.items():
            for line, passed in line_reports.items():
                report = (stat_report.get("markets") or {}).get(market, {}).get(line, {})
                diff = (report.get("candidate_minus_baseline") or {}).get("differences") or {}
                logs.append(
                    f"WF {market} {line}: n={report.get('n', 0)}, "
                    f"ΔBrier={diff.get('brier')}, ΔLogLoss={diff.get('log_loss')}, "
                    f"лучше средней лиги={'да' if passed else 'нет'}; "
                    "букмекерский Value=не проверен"
                )
    if not totals_backtest and not stat_report:
        engine.winner_market_validation = False
        engine.winner_market_validation_report = {}
        engine.total_market_validation = False
        engine.total_market_validation_by_line = {}
        engine.btts_market_validation = False
        engine.stat_market_validation = {}
        update_loader(
            "Проверка рынков не завершена; непроверенные рынки отключены",
            0.74, logs,
        )
    else:
        update_loader(
            "Исторические рынки проверены; завершаю обучение модели",
            0.76, logs,
        )

    def _training_progress(processed, total):
        if processed % 50 == 0 or processed == total:
            update_loader(
                f"Хронологическое обучение [{processed}/{total}]",
                0.76 + processed / max(1, total) * 0.22, logs,
            )

    if incremental_plan["mode"] in {"append", "unchanged"}:
        # Never mutate the previously published checkpoint: a failed update
        # must leave the last-good model available for the next scan.
        previous_engine = copy.deepcopy(checkpoint_for_plan["engine"])
        validation_attributes = (
            "winner_market_validation", "winner_market_validation_report",
            "total_market_validation", "total_market_validation_by_line",
            "total_market_validation_report", "btts_market_validation",
            "btts_market_validation_report", "stat_market_validation",
            "stat_market_validation_report", "xg_ablation_report",
        )
        for attribute in validation_attributes:
            if hasattr(engine, attribute):
                setattr(
                    previous_engine, attribute,
                    copy.deepcopy(getattr(engine, attribute)),
                )
        engine = previous_engine
        match_stats = engine.match_stats
        start_index = int(incremental_plan["start_index"])
        new_rows = training_rows[start_index:]
        if new_rows:
            update_loader(
                f"Догружаю только новые результаты: {len(new_rows)} матчей",
                0.76, logs,
            )
            training_summary = train_engine_chronologically(
                engine, match_stats, new_rows,
                on_progress=_training_progress,
                start_index=start_index,
                total_rows=len(training_rows),
            )
            logs.append(
                f"✅ Инкрементальное обучение: добавлено {len(new_rows)} новых матчей "
                f"за {training_summary['days']} игровых дней; всего в модели "
                f"{training_summary['trained']} матчей."
            )
        else:
            training_summary = {
                "trained": start_index, "days": 0, "skipped": 0,
            }
            logs.append(
                f"⚡ История не изменилась: хронологическое обучение пропущено "
                f"({start_index} матчей уже в Engine)."
            )
    else:
        # No checkpoint, a corrected historical row, xG backfill, deleted
        # result, or a late fixture in an already-trained day: replay all rows.
        reason = incremental_plan.get("reason", "checkpoint_invalid")
        if incremental_checkpoint is not None:
            logs.append(
                f"♻️ Полное хронологическое переобучение: обнаружено изменение "
                f"ранее обученной истории ({reason})."
            )
        else:
            logs.append(
                "🧠 Первичное полное хронологическое обучение: checkpoint ещё нет."
            )
        training_summary = train_engine_chronologically(
            engine, engine.match_stats, training_rows,
            on_progress=_training_progress,
            start_index=0,
            total_rows=len(training_rows),
        )

    engine.trained_n = training_summary["trained"]
    engine._incremental_training_manifest = {
        "schema": TRAINING_STATE_SCHEMA,
        "config_key": incremental_key,
        "day_digests": current_day_digests,
        "trained_rows": len(training_rows),
        "last_training_date": (
            current_day_digests[-1]["date"] if current_day_digests else None
        ),
    }
    logs.append(
        f"📚 История обучения: {training_summary['trained']} матчей; "
        f"пропущено без даты={skipped_rows['missing_date']}, "
        f"без команд={skipped_rows['missing_teams']}, "
        f"без валидного финального счёта={skipped_rows['missing_or_invalid_score']}, "
        f"невалидных строк={skipped_rows['invalid_row']}."
    )
    return engine


@st.cache_resource(show_spinner=False)
def _shared_engine_cache():
    # Shared across Streamlit reruns and browser sessions in the same process.
    return {
        "engines": {}, "incremental": {}, "locks": {},
        "lock": threading.RLock(),
    }


def _load_engine(
    matrix_n, logs, update_loader, footballapi_key="",
    footballapi_xg_rows=None, custom_historical_rows=None,
):
    """Reuse a fitted model and append only new, complete chronological matchdays."""
    import pickle, gzip, os, json
    from config import DISK_CACHE_DIR
    from model.cache_locks import get_keyed_lock

    # Load the historical inputs once, before cache lookup, so cache reuse is
    # conditional on the actual data currently returned by the source cache.
    seasonal_snapshot = _load_seasonal_history(logs, update_loader)
    xg_enabled = bool(str(footballapi_key or "").strip())
    cache_key = engine_cache_fingerprint(
        app_version=APP_VERSION,
        data_version=DATA_VERSION,
        matrix_n=matrix_n,
        xg_enabled=xg_enabled,
        seasonal_history=seasonal_snapshot["history"],
        history_years=seasonal_snapshot["history_years"],
        divisions=seasonal_snapshot["train_divs"],
        footballapi_xg_rows=footballapi_xg_rows,
        custom_historical_rows=custom_historical_rows,
    )
    incremental_key = incremental_engine_checkpoint_key(
        app_version=APP_VERSION,
        data_version=DATA_VERSION,
        matrix_n=matrix_n,
        xg_enabled=xg_enabled,
        shot_xg_version=SHOT_XG_VERSION,
        training_state_schema=TRAINING_STATE_SCHEMA,
    )
    session_key = f"_trained_engine_{cache_key}"

    update_loader("🔎 Проверяю кэш модели по отпечатку истории и xG...", 0.20, logs)
    try:
        session_engine = st.session_state.get(session_key)
        if (session_engine is not None
                and hasattr(session_engine, "match_stats")
                and getattr(session_engine, "trained_n", 0) > 0):
            logs.append(
                f"🧠 Обученная модель из памяти сессии "
                f"({session_engine.trained_n} матчей; повторного обучения нет)"
            )
            return session_engine
    except Exception:
        pass

    shared = _shared_engine_cache()
    guard = shared.setdefault("lock", threading.RLock())
    shared.setdefault("incremental", {})

    def _cached_engine():
        with guard:
            candidate = shared.setdefault("engines", {}).get(cache_key)
        if (
            candidate is not None
            and hasattr(candidate, "match_stats")
            and getattr(candidate, "trained_n", 0) > 0
        ):
            return candidate
        return None

    cached = _cached_engine()
    if cached is not None:
        try:
            st.session_state[session_key] = cached
        except Exception:
            pass
        logs.append(
            f"🧠 Модель из общего кэша ({cached.trained_n} матчей; повторного обучения нет)"
        )
        return cached

    # Serialize updates for one model configuration, not one history hash:
    # two scans that see different newly completed fixtures must not race to
    # publish the same append-only checkpoint.
    model_lock = get_keyed_lock(shared, incremental_key)
    acquired = model_lock.acquire(blocking=False)
    waited_seconds = 0
    while not acquired:
        cached = _cached_engine()
        if cached is not None:
            try:
                st.session_state[session_key] = cached
            except Exception:
                pass
            logs.append(
                f"🧠 Модель подготовлена параллельным запуском "
                f"({cached.trained_n} матчей); использую готовый кэш."
            )
            return cached

        waited_seconds += 5
        update_loader(
            f"⏳ Модель этого набора уже обучается в другой вкладке — "
            f"ожидание готового кэша ({waited_seconds} сек.)",
            min(0.24, 0.20 + waited_seconds / 600.0),
            logs,
        )
        acquired = model_lock.acquire(timeout=5)
        if waited_seconds >= 180 and not acquired:
            raise RuntimeError(
                "Та же модель более 3 минут обучается в другом запуске. "
                "Не запускайте сканирование в нескольких вкладках одновременно; "
                "проверьте вкладку, где началось первое обучение."
            )

    try:
        cached = _cached_engine()
        if cached is not None:
            try:
                st.session_state[session_key] = cached
            except Exception:
                pass
            logs.append(
                f"🧠 Модель из общего кэша ({cached.trained_n} матчей; повторного обучения нет)"
            )
            return cached

        p = os.path.join(DISK_CACHE_DIR, f"{cache_key}.pkl.gz")
        update_loader("💾 Проверяю сохранённую модель на диске...", 0.22, logs)
        if os.path.exists(p):
            try:
                if os.path.getsize(p) > 350 * 1024 * 1024:
                    raise ValueError("cache file exceeds 350 MB safety limit")
                with open(p, "rb") as file_obj:
                    data = pickle.loads(gzip.decompress(file_obj.read()))
                cached_engine = data.get("engine") if isinstance(data, dict) else None
                if (
                    isinstance(data, dict)
                    and data.get("fp") == APP_VERSION
                    and data.get("cache_key") == cache_key
                    and hasattr(cached_engine, "match_stats")
                    and getattr(cached_engine, "trained_n", 0) > 0
                ):
                    with guard:
                        shared.setdefault("engines", {})[cache_key] = cached_engine
                    try:
                        st.session_state[session_key] = cached_engine
                    except Exception:
                        pass
                    logs.append(
                        f"🧠 Модель из дискового кэша "
                        f"({cached_engine.trained_n} матчей; повторного обучения нет)"
                    )
                    return cached_engine
            except Exception as exc:
                logs.append(
                    f"⚠️ Дисковый кэш не прочитан ({type(exc).__name__}); "
                    "создаю модель заново."
                )

        # Prefer the last-good append-only checkpoint. It is copied before
        # training so an exception cannot corrupt the published model.
        incremental_path = os.path.join(
            DISK_CACHE_DIR, f"{incremental_key}.incremental.pkl.gz",
        )
        incremental_checkpoint = None
        with guard:
            memory_checkpoint = shared.setdefault("incremental", {}).get(incremental_key)
        if isinstance(memory_checkpoint, dict):
            incremental_checkpoint = memory_checkpoint
            logs.append("⚡ Найдено состояние Engine для инкрементального продолжения.")
        elif os.path.exists(incremental_path):
            try:
                if os.path.getsize(incremental_path) > 350 * 1024 * 1024:
                    raise ValueError("incremental checkpoint exceeds 350 MB safety limit")
                with open(incremental_path, "rb") as file_obj:
                    candidate = pickle.loads(gzip.decompress(file_obj.read()))
                candidate_engine = candidate.get("engine") if isinstance(candidate, dict) else None
                candidate_manifest = (
                    getattr(candidate_engine, "_incremental_training_manifest", None)
                    if candidate_engine is not None else None
                )
                if (
                    isinstance(candidate, dict)
                    and candidate.get("schema") == TRAINING_STATE_SCHEMA
                    and candidate.get("config_key") == incremental_key
                    and candidate_manifest
                    and candidate_manifest.get("config_key") == incremental_key
                    and hasattr(candidate_engine, "match_stats")
                    and int(getattr(candidate_engine, "trained_n", -1))
                    == int(candidate.get("trained_rows", -2))
                ):
                    incremental_checkpoint = candidate
                    with guard:
                        shared.setdefault("incremental", {})[incremental_key] = candidate
                    logs.append("💾 Загружен сохранённый checkpoint хронологического обучения.")
                else:
                    logs.append("⚠️ Checkpoint обучения несовместим; будет безопасный полный replay.")
            except Exception as exc:
                logs.append(
                    f"⚠️ Checkpoint обучения не прочитан ({type(exc).__name__}); "
                    "будет безопасный полный replay."
                )

        update_loader("🧠 Проверяю новые результаты для обучения...", 0.24, logs)
        engine = _train_engine(
            matrix_n, logs, update_loader,
            footballapi_key=footballapi_key,
            footballapi_xg_rows=footballapi_xg_rows,
            custom_historical_rows=custom_historical_rows,
            seasonal_snapshot=seasonal_snapshot,
            incremental_checkpoint=incremental_checkpoint,
            incremental_key=incremental_key,
        )
        update_loader("💾 Сохраняю обученную модель...", 0.99, logs)
        try:
            os.makedirs(DISK_CACHE_DIR, exist_ok=True)
            packed = gzip.compress(
                pickle.dumps({"fp": APP_VERSION, "cache_key": cache_key, "engine": engine}),
                compresslevel=1,
            )
            temporary_path = p + ".tmp"
            with open(temporary_path, "wb") as file_obj:
                file_obj.write(packed)
            os.replace(temporary_path, p)
            logs.append("💾 Обученная модель сохранена в дисковый кэш.")
        except Exception as exc:
            logs.append(
                f"ℹ️ Дисковый кэш недоступен ({type(exc).__name__}); "
                "модель останется в памяти текущего процесса."
            )
        with guard:
            shared.setdefault("engines", {})[cache_key] = engine
        manifest = getattr(engine, "_incremental_training_manifest", None)
        if isinstance(manifest, dict):
            checkpoint = {
                "schema": TRAINING_STATE_SCHEMA,
                "config_key": incremental_key,
                "day_digests": manifest.get("day_digests") or [],
                "trained_rows": int(manifest.get("trained_rows") or 0),
                "last_training_date": manifest.get("last_training_date"),
                "engine": engine,
            }
            with guard:
                shared.setdefault("incremental", {})[incremental_key] = checkpoint
            try:
                os.makedirs(DISK_CACHE_DIR, exist_ok=True)
                packed_checkpoint = gzip.compress(
                    pickle.dumps(checkpoint), compresslevel=1,
                )
                temporary_checkpoint = incremental_path + ".tmp"
                with open(temporary_checkpoint, "wb") as file_obj:
                    file_obj.write(packed_checkpoint)
                    file_obj.flush()
                    os.fsync(file_obj.fileno())
                os.replace(temporary_checkpoint, incremental_path)
                logs.append(
                    f"💾 Инкрементальный checkpoint сохранён "
                    f"(дата {checkpoint.get('last_training_date')}, "
                    f"{checkpoint['trained_rows']} матчей)."
                )
            except Exception as exc:
                logs.append(
                    f"ℹ️ Checkpoint не сохранён на диск ({type(exc).__name__}); "
                    "состояние останется в памяти процесса."
                )
        try:
            st.session_state[session_key] = engine
        except Exception:
            pass
        return engine
    finally:
        model_lock.release()

