"""Model training and fitted-engine cache for the match scanner.

This module owns the slow historical training path and model reuse lifecycle;
the scanner UI imports the three established private entry points below.
"""
from __future__ import annotations

import copy
import os
import threading
import time
from datetime import datetime, timedelta

import streamlit as st

from config import (
    APP_VERSION, DATA_VERSION, DIV_NAMES, DIV_TO_TSDB, DISK_CACHE_DIR,
)
from model.engine_cache import engine_cache_fingerprint
from data.sources import load_seasonal, season_str, parse_date
from model.engine import Engine, _norm
from model.shot_xg import enrich_history_with_shot_xg
from model.validation_cache import select_validation_rows, validation_cache_key
from model.training import (
    prepare_completed_training_rows, train_engine_chronologically,
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


def _history_catalog():
    """Return the fixed training catalogue without reading historical rows."""
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cur_year = today.year if today.month >= 7 else today.year - 1
    history_years = list(range(cur_year - 4, cur_year + 1))
    priority_divs = list(_PRIORITY_DIVS)
    supported_divs = list(DIV_TO_TSDB.keys()) + ["E2", "E3", "D3"]
    train_divs = list(dict.fromkeys(priority_divs + supported_divs))
    return history_years, train_divs


def _stable_training_engine_key(
    matrix_n, *, footballapi_key="", custom_historical_rows=None,
):
    """Key only the actual training configuration, never today's fixture slate."""
    from model.incremental import stable_engine_fingerprint

    history_years, catalogue_divisions = _history_catalog()
    # Custom archive contents are checked by the immutable training manifest
    # during an explicit refresh; they must not fragment the fitted-model cache.
    training_divisions = tuple(catalogue_divisions)
    key = stable_engine_fingerprint(
        matrix_n=matrix_n,
        xg_enabled=bool(str(footballapi_key or "").strip()),
        divisions=training_divisions,
        history_years=history_years,
    )
    return key, history_years, training_divisions


def _legacy_live_slate_engine_key(
    matrix_n, *, footballapi_key="", custom_historical_rows=None,
    extra_divisions=None,
):
    """Read-only compatibility key written by PR #140; never use it for new writes."""
    from model.incremental import stable_engine_fingerprint

    base_key, history_years, training_divisions = _stable_training_engine_key(
        matrix_n,
        footballapi_key=footballapi_key,
        custom_historical_rows=custom_historical_rows,
    )
    custom_divisions = [
        str(row.get("Div") or "").strip()
        for row in (custom_historical_rows or [])
        if isinstance(row, dict) and str(row.get("Div") or "").strip()
    ]
    legacy_divisions = tuple(dict.fromkeys(
        list(training_divisions)
        + custom_divisions
        + [
            str(value).strip() for value in (extra_divisions or [])
            if str(value).strip()
        ]
    ))
    if legacy_divisions == training_divisions:
        return base_key
    return stable_engine_fingerprint(
        matrix_n=matrix_n,
        xg_enabled=bool(str(footballapi_key or "").strip()),
        divisions=legacy_divisions,
        history_years=history_years,
    )


def _valid_fitted_engine(candidate):
    return bool(
        candidate is not None
        and hasattr(candidate, "match_stats")
        and getattr(candidate, "trained_n", 0) > 0
        and getattr(candidate, "_incremental_manifest", None)
    )


def _promote_stable_engine_cache(candidate, stable_key, logs=None):
    """Publish a legacy-key engine under the canonical training-only key."""
    import gzip
    import os
    import pickle

    shared = _shared_engine_cache()
    guard = shared.setdefault("lock", threading.RLock())
    with guard:
        shared.setdefault("engines", {})[stable_key] = candidate
    try:
        stable_session_key = f"_trained_engine_{stable_key}"
        st.session_state[stable_session_key] = candidate
    except Exception:
        pass
    try:
        path = os.path.join(DISK_CACHE_DIR, f"{stable_key}.pkl.gz")
        os.makedirs(DISK_CACHE_DIR, exist_ok=True)
        packed = gzip.compress(
            pickle.dumps({"fp": APP_VERSION, "cache_key": stable_key, "engine": candidate}),
            compresslevel=1,
        )
        temp_path = path + ".tmp"
        with open(temp_path, "wb") as handle:
            handle.write(packed)
        os.replace(temp_path, path)
        if logs is not None:
            logs.append("♻️ Существующая модель перенесена на стабильный ключ обучения.")
    except Exception as exc:
        if logs is not None:
            logs.append(
                f"ℹ️ Совместимый кэш работает в памяти ({type(exc).__name__} при переносе)."
            )


def _validation_divisions(train_divs):
    """Return priority validation leagues in stable order, limited to trained divisions."""
    available = set(train_divs or ())
    return tuple(division for division in _PRIORITY_DIVS if division in available)


def _load_seasonal_history(logs, update_loader, *, force_refresh=False):
    """Load the assembled history snapshot; ordinary scans must never call this."""
    history_years, train_divs = _history_catalog()
    cache_key = (DATA_VERSION, tuple(history_years), tuple(train_divs))
    now_mono = time.monotonic()

    with _HISTORY_SNAPSHOT_LOCK:
        cached = None if force_refresh else _HISTORY_SNAPSHOT_CACHE.get(cache_key)
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

        from concurrent.futures import ThreadPoolExecutor

        def _load_one_division(division):
            seasons = []
            for year in history_years:
                rows = load_seasonal(division, season_str(year), force_refresh=force_refresh)
                rows = sorted(
                    rows,
                    key=lambda row: parse_date(row.get("Date", "")) or datetime.min,
                )
                seasons.append(rows)
            return division, seasons

        # Independent CSV downloads run in parallel; map preserves deterministic order.
        with ThreadPoolExecutor(max_workers=8) as pool:
            loaded = list(pool.map(_load_one_division, train_divs))

        history = {}
        for i, (division, seasons) in enumerate(loaded):
            history[division] = seasons
            pct = 0.04 + (i + 1) / max(1, len(train_divs)) * 0.14
            update_loader(
                f"Проверка истории [{i+1}/{len(train_divs)}] — "
                f"{DIV_NAMES.get(division, division)}",
                pct, logs,
            )
            for year, rows in zip(history_years, seasons):
                if rows:
                    logs.append(f"  {season_str(year)}: {len(rows)} матчей")
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


def _prepare_training_history(
    seasonal_snapshot, *, footballapi_key="", footballapi_xg_rows=None,
    custom_historical_rows=None, logs=None,
):
    """Apply custom rows and dated xG enrichment identically on full/delta paths."""
    history_years = list(seasonal_snapshot["history_years"])
    train_divs = list(seasonal_snapshot["train_divs"])
    history = copy.deepcopy(seasonal_snapshot["history"])
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

    return history, train_divs


def _refresh_engine_market_validation(
    engine, history, train_divs, matrix_n, logs, update_loader,
):
    """Reuse a frozen validation snapshot, recomputing only when its epoch changes."""
    from model.historical_match_stats import walk_forward_market_validation
    validation_divs = _validation_divisions(train_divs)
    # Keep a broad multi-league validation set, but avoid replaying 7k+
    # fixtures through eight chronological model arms on the first cold start.
    # The global >=500 paired-observation promotion gates remain unchanged.
    # Freeze recent validation data in fixed 48-hour epochs.
    today = datetime.now().date()
    epoch_start = today - timedelta(days=today.toordinal() % 2)
    freeze_day = epoch_start - timedelta(days=2)

    def _before_freeze(row):
        stamp = parse_date(row.get("Date") or row.get("date") or "")
        return stamp is not None and stamp.date() < freeze_day

    frozen_history = {
        division: [
            [row for row in season if _before_freeze(row)]
            for season in seasons
        ]
        for division, seasons in history.items()
    }
    # Keep a bounded recent holdout per league for scoring, but replay the
    # complete frozen history before it so teams and league priors are warmed
    # exactly as they would be in production. The benchmark excludes warm-up
    # rows from metrics; they are only ingested chronologically.
    validation_rows = select_validation_rows(
        frozen_history, validation_divs, max_rows_per_league=250,
    )
    walk_forward_rows = select_validation_rows(
        frozen_history, validation_divs, max_rows_per_league=1_000_000,
    )
    evaluation_start_by_league = {}
    for row in validation_rows:
        division = str(row.get("Div") or "").strip()
        stamp = parse_date(row.get("Date") or row.get("date") or "")
        if division and stamp is not None:
            previous = evaluation_start_by_league.get(division)
            if previous is None or stamp.date() < previous:
                evaluation_start_by_league[division] = stamp.date()
    from security import cache_get, cache_put
    from model.walk_forward import walk_forward_compare, totals_market_is_validated
    validation_key = validation_cache_key(
        walk_forward_rows, matrix_n=matrix_n,
        model_version=f"{APP_VERSION}-warmup-holdout-v1",
    )
    cached_validation = cache_get(validation_key, 30 * 86400)

    if isinstance(cached_validation, dict):
        totals_backtest = cached_validation.get("totals_backtest") or {}
        stat_report = cached_validation.get("stat_report") or {}
        logs.append(
            f"⚡ Историческая проверка из кэша: warm-up {len(walk_forward_rows)} матчей, "
            f"holdout {len(validation_rows)} матчей, {len(validation_divs)} лиг."
        )
    else:
        update_loader(
            f"Историческая проверка рынков: warm-up {len(walk_forward_rows)} + "
            f"holdout {len(validation_rows)} матчей",
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
                walk_forward_rows, matrix_n=matrix_n, bootstrap_samples=0,
                totals_bootstrap_samples=1000, market_bootstrap_samples=1000,
                progress_callback=_walk_forward_progress,
                evaluation_start_by_league=evaluation_start_by_league,
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
            cache_put(validation_key, {
                "totals_backtest": totals_backtest,
                "stat_report": stat_report,
            })
            logs.append(
                f"Историческая проверка рассчитана: warm-up {len(walk_forward_rows)} матчей, "
                f"holdout {len(validation_rows)} матчей, {len(validation_divs)} лиг; "
                "результат сохранён в кэш."
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
        form_report = totals_backtest.get("recent_form_ablation") or {}
        form_candidate = form_report.get("candidate") or {}
        form_delta = (form_report.get("paired_vs_base_model") or {}).get("differences") or {}
        if form_candidate.get("n"):
            logs.append(
                f"Экспериментальная форма (вес={form_report.get('configured_research_weight')}): "
                f"n={form_candidate.get('n')}, Brier={form_candidate.get('brier')}, "
                f"LogLoss={form_candidate.get('log_loss')}; "
                f"Δ к базовой модели Brier/LogLoss={form_delta.get('brier')}/"
                f"{form_delta.get('log_loss')}. В production вес остаётся 0."
            )
        blend_report = totals_backtest.get("market_blends") or {}
        blend_arms = (blend_report.get("weights_model_market") or {})
        for weight in ("0.25", "0.50", "0.75"):
            arm = blend_arms.get(weight) or {}
            blend_metrics = arm.get("metrics") or {}
            model_delta = (arm.get("vs_model") or {}).get("differences") or {}
            market_delta = (arm.get("vs_market") or {}).get("differences") or {}
            if blend_metrics.get("n"):
                logs.append(
                    f"Экспериментальный blend модель/рынок {weight}/{1-float(weight):.2f}: "
                    f"n={blend_metrics.get('n')}, Brier={blend_metrics.get('brier')}, "
                    f"LogLoss={blend_metrics.get('log_loss')}; "
                    f"Δ к модели Brier/LogLoss={model_delta.get('brier')}/"
                    f"{model_delta.get('log_loss')}; "
                    f"Δ к рынку={market_delta.get('brier')}/{market_delta.get('log_loss')}. "
                    "Только диагностика — боевые вероятности не меняются."
                )
        for league, weights in sorted((blend_report.get("by_league") or {}).items()):
            for weight, arm in sorted(weights.items()):
                metric = arm.get("metrics") or {}
                model_delta = (arm.get("vs_model") or {}).get("differences") or {}
                market_delta = (arm.get("vs_market") or {}).get("differences") or {}
                if int(metric.get("n") or 0) >= 30:
                    logs.append(
                        f"Blend по лиге {league}, вес модели {weight}: n={metric.get('n')}, "
                        f"Δ Brier к модели/рынку={model_delta.get('brier')}/"
                        f"{market_delta.get('brier')}; "
                        f"Δ LogLoss={model_delta.get('log_loss')}/{market_delta.get('log_loss')}."
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

    engine._market_validation_cache_key = validation_key
    engine._market_validation_freeze_day = freeze_day.isoformat()
    return validation_key


def _defer_market_validation(engine, logs):
    """Keep every market fail-closed when the expensive benchmark is deferred."""
    message = (
        "Рыночная проверка отложена до ручного обновления архивов; "
        "непроверенные рынки остаются forecast-only, Value и автодопуск закрыты."
    )
    report = {"state": "deferred", "reason": message}
    engine.winner_market_validation = False
    engine.winner_market_validation_report = dict(report)
    engine.total_market_validation = False
    engine.total_market_validation_by_line = {}
    engine.total_market_validation_report = dict(report)
    engine.btts_market_validation = False
    engine.btts_market_validation_report = dict(report)
    engine.stat_market_validation = {}
    engine.stat_market_validation_report = dict(report)
    engine.xg_ablation_report = {}
    engine._market_validation_cache_key = None
    engine._market_validation_freeze_day = None
    engine._market_validation_deferred = True
    logs.append("⏭ " + message)


def _train_engine(
    matrix_n, logs, update_loader, footballapi_key="",
    footballapi_xg_rows=None, custom_historical_rows=None,
    seasonal_snapshot=None, run_market_validation=False,
):
    if seasonal_snapshot is None:
        seasonal_snapshot = _load_seasonal_history(logs, update_loader)
    history, train_divs = _prepare_training_history(
        seasonal_snapshot,
        footballapi_key=footballapi_key,
        footballapi_xg_rows=footballapi_xg_rows,
        custom_historical_rows=custom_historical_rows,
        logs=logs,
    )
    engine = Engine(matrix_n=matrix_n)
    from model.historical_match_stats import HistoricalMatchStats
    engine.match_stats = HistoricalMatchStats()
    if run_market_validation:
        _refresh_engine_market_validation(
            engine, history, train_divs, matrix_n, logs, update_loader,
        )
    else:
        _defer_market_validation(engine, logs)
        update_loader("⏭ Рыночный benchmark отложен; обучаю базовую модель", 0.24, logs)
    training_rows, skipped_rows = prepare_completed_training_rows(
        history, train_divs,
    )
    if not training_rows:
        raise RuntimeError(
            "Не найдено ни одного исторического матча с валидной датой, "
            "командами и финальным счётом; обучение остановлено без подстановки 0:0."
        )

    training_start = 0.76 if run_market_validation else 0.24

    def _training_progress(processed, total):
        if processed % 50 == 0 or processed == total:
            update_loader(
                f"Хронологическое обучение [{processed}/{total}]",
                training_start + processed / max(1, total) * (0.98 - training_start),
                logs,
            )

    training_summary = train_engine_chronologically(
        engine, engine.match_stats, training_rows,
        on_progress=_training_progress,
    )
    engine.trained_n = training_summary["trained"]
    from model.incremental import record_training_manifest
    record_training_manifest(engine, training_rows)
    logs.append(
        f"Обучено хронологически: {training_summary['trained']} матчей, "
        f"{training_summary['days']} календарных дней; "
        f"пропущено без даты={skipped_rows['missing_date']}, "
        f"без команд={skipped_rows['missing_teams']}, "
        f"без валидного финального счёта={skipped_rows['missing_or_invalid_score']}, "
        f"невалидных строк={skipped_rows['invalid_row']}."
    )
    return engine


@st.cache_resource(show_spinner=False)
def _shared_engine_cache():
    # Shared across Streamlit reruns and browser sessions in the same process.
    return {"engines": {}, "locks": {}, "lock": threading.RLock()}


def has_cached_engine(
    matrix_n, *, footballapi_key="", custom_historical_rows=None,
    extra_divisions=None,
):
    """Check the fitted model without reading history; live leagues are legacy lookup only."""
    import gzip
    import os
    import pickle

    stable_key, _, _ = _stable_training_engine_key(
        matrix_n,
        footballapi_key=footballapi_key,
        custom_historical_rows=custom_historical_rows,
    )
    legacy_key = _legacy_live_slate_engine_key(
        matrix_n,
        footballapi_key=footballapi_key,
        custom_historical_rows=custom_historical_rows,
        extra_divisions=extra_divisions,
    )
    keys = [stable_key]
    if legacy_key != stable_key:
        keys.append(legacy_key)

    shared = _shared_engine_cache()
    guard = shared.setdefault("lock", threading.RLock())
    for key in keys:
        with guard:
            candidate = shared.setdefault("engines", {}).get(key)
        if _valid_fitted_engine(candidate):
            if key != stable_key:
                _promote_stable_engine_cache(candidate, stable_key)
            return True

        try:
            candidate = st.session_state.get(f"_trained_engine_{key}")
        except Exception:
            candidate = None
        if _valid_fitted_engine(candidate):
            with guard:
                shared.setdefault("engines", {})[key] = candidate
            if key != stable_key:
                _promote_stable_engine_cache(candidate, stable_key)
            return True

        path = os.path.join(DISK_CACHE_DIR, f"{key}.pkl.gz")
        if not os.path.exists(path):
            continue
        try:
            if os.path.getsize(path) > 350 * 1024 * 1024:
                continue
            with open(path, "rb") as handle:
                data = pickle.loads(gzip.decompress(handle.read()))
            candidate = data.get("engine") if isinstance(data, dict) else None
            if not (
                isinstance(data, dict)
                and data.get("fp") == APP_VERSION
                and data.get("cache_key") == key
                and _valid_fitted_engine(candidate)
            ):
                continue
            with guard:
                shared.setdefault("engines", {})[key] = candidate
            try:
                st.session_state[f"_trained_engine_{key}"] = candidate
            except Exception:
                pass
            if key != stable_key:
                _promote_stable_engine_cache(candidate, stable_key)
            return True
        except Exception:
            continue
    return False


def _load_engine(
    matrix_n, logs, update_loader, footballapi_key="",
    footballapi_xg_rows=None, custom_historical_rows=None,
    *, refresh_history=False, force_source_refresh=False,
    extra_divisions=None, run_market_validation=False,
):
    """Use the fitted engine for normal scans; read/replay history only on explicit refresh or cold start."""
    import pickle, gzip, os, json
    from config import DISK_CACHE_DIR
    from model.cache_locks import get_keyed_lock

    from model.incremental import (
        IncrementalHistoryChanged, bring_engine_up_to_date,
    )
    xg_enabled = bool(str(footballapi_key or "").strip())
    stable_key, history_years, stable_divisions = _stable_training_engine_key(
        matrix_n,
        footballapi_key=footballapi_key,
        custom_historical_rows=custom_historical_rows,
    )
    legacy_key = _legacy_live_slate_engine_key(
        matrix_n,
        footballapi_key=footballapi_key,
        custom_historical_rows=custom_historical_rows,
        extra_divisions=extra_divisions,
    )
    stable_session_key = f"_trained_engine_{stable_key}"
    shared_stable = _shared_engine_cache()
    stable_guard = shared_stable.setdefault("lock", threading.RLock())
    stable_candidate = None
    with stable_guard:
        stable_candidate = shared_stable.setdefault("engines", {}).get(stable_key)
    if stable_candidate is None:
        try:
            stable_candidate = st.session_state.get(stable_session_key)
        except Exception:
            stable_candidate = None
    stable_path = os.path.join(DISK_CACHE_DIR, f"{stable_key}.pkl.gz")
    lookup_keys = [stable_key]
    if legacy_key != stable_key:
        lookup_keys.append(legacy_key)
    if stable_candidate is None:
        for lookup_key in lookup_keys:
            lookup_path = os.path.join(DISK_CACHE_DIR, f"{lookup_key}.pkl.gz")
            if not os.path.exists(lookup_path):
                continue
            try:
                if os.path.getsize(lookup_path) > 350 * 1024 * 1024:
                    raise ValueError("cache file exceeds 350 MB safety limit")
                with open(lookup_path, "rb") as file_obj:
                    cached_data = pickle.loads(gzip.decompress(file_obj.read()))
                disk_engine = cached_data.get("engine") if isinstance(cached_data, dict) else None
                if (
                    isinstance(cached_data, dict)
                    and cached_data.get("fp") == APP_VERSION
                    and cached_data.get("cache_key") == lookup_key
                    and _valid_fitted_engine(disk_engine)
                ):
                    stable_candidate = disk_engine
                    logs.append(
                        "💾 Найден стабильный дисковый кэш обученной модели."
                        if lookup_key == stable_key else
                        "♻️ Найден прежний кэш модели; переношу на ключ обучения без текущих лиг."
                    )
                    break
            except Exception as exc:
                logs.append(f"ℹ️ Дисковый кэш {lookup_key[:18]} не использован ({type(exc).__name__}).")

    # Fast path: the normal live scanner only reads the fitted model. It does
    # not download/reparse the 5-season archive, rebuild validation samples,
    # recalculate the historical walk-forward benchmark, or replay training.
    if (
        stable_candidate is not None
        and hasattr(stable_candidate, "match_stats")
        and getattr(stable_candidate, "trained_n", 0) > 0
        and getattr(stable_candidate, "_incremental_manifest", None)
        and not refresh_history
    ):
        if legacy_key != stable_key:
            _promote_stable_engine_cache(stable_candidate, stable_key, logs)
        with stable_guard:
            shared_stable.setdefault("engines", {})[stable_key] = stable_candidate
        try:
            st.session_state[stable_session_key] = stable_candidate
        except Exception:
            pass
        logs.append(
            f"⚡ Исторический архив не перечитывался: готовая модель из кэша "
            f"({stable_candidate.trained_n} обучающих матчей). "
            "Для обновления истории используй отдельную кнопку в блоке архивов."
        )
        return stable_candidate

    # Only a cold start or an explicit history refresh enters this block.
    seasonal_snapshot = _load_seasonal_history(
        logs, update_loader, force_refresh=bool(force_source_refresh)
    )
    if (
        stable_candidate is not None
        and hasattr(stable_candidate, "match_stats")
        and getattr(stable_candidate, "trained_n", 0) > 0
        and getattr(stable_candidate, "_incremental_manifest", None)
    ):
        stable_lock = get_keyed_lock(shared_stable, stable_key)
        acquired_stable = stable_lock.acquire(blocking=False)
        waited_stable = 0
        while not acquired_stable:
            waited_stable += 5
            update_loader(
                f"⏳ Инкрементальная модель обновляется ({waited_stable} сек.)",
                min(0.24, 0.20 + waited_stable / 600.0), logs,
            )
            if waited_stable >= 900:
                raise RuntimeError("Стабильная модель не освободила блокировку за 15 минут.")
            acquired_stable = stable_lock.acquire(timeout=5)
        try:
            with stable_guard:
                latest = shared_stable.setdefault("engines", {}).get(stable_key)
            if latest is not None:
                stable_candidate = latest
            engine = copy.deepcopy(stable_candidate)
            history, train_divs = _prepare_training_history(
                seasonal_snapshot,
                footballapi_key=footballapi_key,
                footballapi_xg_rows=footballapi_xg_rows,
                custom_historical_rows=custom_historical_rows,
                logs=logs,
            )
            added = bring_engine_up_to_date(engine, history, train_divs, logs)
            if run_market_validation:
                _refresh_engine_market_validation(
                    engine, history, train_divs, matrix_n, logs, update_loader,
                )
            elif added:
                _defer_market_validation(engine, logs)
                update_loader(
                    "⏭ История обновлена; рыночный benchmark отложен, Value-гейты закрыты",
                    0.24, logs,
                )
            with stable_guard:
                shared_stable.setdefault("engines", {})[stable_key] = engine
            try:
                st.session_state[stable_session_key] = engine
            except Exception:
                pass
            logs.append(
                f"🧠 Модель из стабильного кэша +{added} новых матчей; "
                "полного переобучения нет."
            )
            try:
                os.makedirs(DISK_CACHE_DIR, exist_ok=True)
                packed = gzip.compress(
                    pickle.dumps({"fp": APP_VERSION, "cache_key": stable_key, "engine": engine}),
                    compresslevel=1,
                )
                temp_path = stable_path + ".tmp"
                with open(temp_path, "wb") as file_obj:
                    file_obj.write(packed)
                os.replace(temp_path, stable_path)
            except Exception as exc:
                logs.append(
                    f"ℹ️ Стабильный кэш остался в памяти ({type(exc).__name__} при записи)."
                )
            return engine
        except IncrementalHistoryChanged as exc:
            logs.append(
                f"ℹ️ Инкремент небезопасен: {str(exc)[:180]}; запускаю полное обучение."
            )
        except Exception as exc:
            logs.append(
                f"⚠️ Инкремент не удался ({type(exc).__name__}: {str(exc)[:160]}); "
                "запускаю полное обучение."
            )
        finally:
            stable_lock.release()

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

    # A single global lock used to cover the whole 30k–40k match training pass.
    # That made unrelated sessions and different model keys wait silently.
    # Use one lock per exact xG/data/matrix fingerprint instead.
    model_lock = get_keyed_lock(shared, cache_key)
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
        if waited_seconds >= 900 and not acquired:
            raise RuntimeError(
                "Та же модель более 15 минут обучается в другом запуске. "
                "Проверьте первую вкладку и её журнал сканирования."
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

        update_loader("🧠 Кэш не найден — начинаю обучение модели...", 0.24, logs)
        engine = _train_engine(
            matrix_n, logs, update_loader,
            footballapi_key=footballapi_key,
            footballapi_xg_rows=footballapi_xg_rows,
            custom_historical_rows=custom_historical_rows,
            seasonal_snapshot=seasonal_snapshot,
            run_market_validation=run_market_validation,
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
            stable_packed = gzip.compress(
                pickle.dumps({"fp": APP_VERSION, "cache_key": stable_key, "engine": engine}),
                compresslevel=1,
            )
            stable_temp_path = stable_path + ".tmp"
            with open(stable_temp_path, "wb") as stable_file:
                stable_file.write(stable_packed)
            os.replace(stable_temp_path, stable_path)
            logs.append("💾 Обученная модель сохранена в обычный и стабильный дисковый кэш.")
        except Exception as exc:
            logs.append(
                f"ℹ️ Дисковый кэш недоступен ({type(exc).__name__}); "
                "модель останется в памяти текущего процесса."
            )
        with guard:
            shared.setdefault("engines", {})[cache_key] = engine
            shared.setdefault("engines", {})[stable_key] = engine
        try:
            st.session_state[session_key] = engine
            st.session_state[stable_session_key] = engine
        except Exception:
            pass
        return engine
    finally:
        model_lock.release()

