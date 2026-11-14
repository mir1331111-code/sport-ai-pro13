"""Model training and fitted-engine cache for the match scanner.

This module owns the slow historical training path and model reuse lifecycle;
the scanner UI imports the three established private entry points below.
"""
from __future__ import annotations

import hashlib
import os
import threading
from datetime import datetime

import streamlit as st

from config import (
    APP_VERSION, DIV_NAMES, DIV_TO_TSDB, DISK_CACHE_DIR,
)
from data.sources import load_seasonal, season_str, parse_date
from model.engine import Engine, _norm
from model.shot_xg import enrich_history_with_shot_xg
from model.validation_cache import select_validation_rows, validation_cache_key
from model.training import (
    prepare_completed_training_rows, train_engine_chronologically,
)

def _train_engine(
    matrix_n, logs, update_loader, footballapi_key="",
    footballapi_xg_rows=None, custom_historical_rows=None,
):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cur_year = today.year if today.month >= 7 else today.year - 1
    history_years = list(range(cur_year - 4, cur_year + 1))
    priority_divs = ["E0", "SP1", "I1", "D1", "F1", "E1", "SP2", "I2",
                     "D2", "F2", "N1", "B1", "P1", "T1", "R1"]
    supported_divs = list(DIV_TO_TSDB.keys()) + ["E2", "E3", "D3"]
    train_divs = list(dict.fromkeys(priority_divs + supported_divs))
    engine = Engine(matrix_n=matrix_n)
    from model.historical_match_stats import (
        HistoricalMatchStats, walk_forward_market_validation,
    )
    engine.match_stats = HistoricalMatchStats()
    history = {}
    for i, dv in enumerate(train_divs):
        pct = 0.1 + (i + 1) / len(train_divs) * 0.4
        update_loader(
            f"История [{i+1}/{len(train_divs)}] — {DIV_NAMES.get(dv, dv)}",
            pct, logs)
        seasons = []
        for year in history_years:
            rows = load_seasonal(dv, season_str(year))
            rows = sorted(rows, key=lambda row: parse_date(row.get("Date", "")) or datetime.min)
            seasons.append(rows)
            if rows:
                logs.append(f"  {season_str(year)}: {len(rows)} матчей")
        history[dv] = seasons
        league_rows = [row for season_rows in seasons for row in season_rows]
        # Per-division walk-forward was redundant: the unified market validation
        # below evaluates the selected multi-league sample once and is cached.
        logs.append(f"OK {DIV_NAMES.get(dv, dv)}: {len(league_rows)} матчей за {len(history_years)} сезонов")

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

    # The validation sample is intentionally bounded: full replays over every
    # competition/season on each cold start made Streamlit appear frozen.
    # 15 main leagues × latest 500 complete fixtures retains >500 observations
    # across competitions for each supported market while limiting CPU time.
    validation_divs = tuple(priority_divs)
    validation_rows = select_validation_rows(
        history, validation_divs, max_rows_per_league=500,
    )
    from security import cache_get, cache_put
    from model.walk_forward import walk_forward_compare, totals_market_is_validated
    validation_key = validation_cache_key(
        validation_rows, matrix_n=matrix_n, model_version=APP_VERSION,
    )
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
        try:
            totals_backtest = walk_forward_compare(
                validation_rows, matrix_n=matrix_n, bootstrap_samples=0,
                totals_bootstrap_samples=1000, market_bootstrap_samples=1000,
            )
            update_loader(
                "Проверка угловых и карточек (хронологический тест)",
                0.68, logs,
            )
            stat_report = walk_forward_market_validation(
                validation_rows, min_history=30, min_n=500, bootstrap_samples=1000,
            )
            cache_put(validation_key, {
                "totals_backtest": totals_backtest,
                "stat_report": stat_report,
            })
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

    training_rows, skipped_rows = prepare_completed_training_rows(
        history, train_divs,
    )
    if not training_rows:
        raise RuntimeError(
            "Не найдено ни одного исторического матча с валидной датой, "
            "командами и финальным счётом; обучение остановлено без подстановки 0:0."
        )

    def _training_progress(processed, total):
        if processed % 50 == 0 or processed == total:
            update_loader(
                f"Хронологическое обучение [{processed}/{total}]",
                0.76 + processed / max(1, total) * 0.22, logs,
            )

    training_summary = train_engine_chronologically(
        engine, engine.match_stats, training_rows,
        on_progress=_training_progress,
    )
    engine.trained_n = training_summary["trained"]
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


def _load_engine(
    matrix_n, logs, update_loader, footballapi_key="",
    footballapi_xg_rows=None, custom_historical_rows=None,
):
    """Reuse a fitted model; serialize training only for an identical cache key."""
    import pickle, gzip, os, json
    from config import DISK_CACHE_DIR
    from model.cache_locks import get_keyed_lock

    xg_cache_tag = "xg_on" if str(footballapi_key or "").strip() else "xg_off"
    xg_ids = sorted({
        str(row.get("xg_match_id") or (
            f"{row.get('Date','')}|{row.get('HomeTeam','')}|{row.get('AwayTeam','')}"
        ))
        for row in (footballapi_xg_rows or [])
        if isinstance(row, dict)
    })
    xg_data_tag = hashlib.sha256("|".join(xg_ids).encode("utf-8")).hexdigest()[:12] if xg_ids else "empty"
    custom_payload = json.dumps(
        list(custom_historical_rows or []),
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    )
    custom_data_tag = hashlib.sha256(custom_payload.encode("utf-8")).hexdigest()[:12] if custom_historical_rows else "none"
    cache_key = f"{APP_VERSION}:{int(matrix_n)}:{xg_cache_tag}:{xg_data_tag}:{custom_data_tag}"
    session_key = f"_trained_engine_{APP_VERSION}_{int(matrix_n)}_{xg_cache_tag}_{xg_data_tag}_{custom_data_tag}"

    update_loader("🔎 Проверяю кэш обученной модели...", 0.20, logs)
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

        p = os.path.join(
            DISK_CACHE_DIR,
            f"engine_{APP_VERSION}_{matrix_n}_{xg_cache_tag}_{xg_data_tag}_{custom_data_tag}.pkl.gz",
        )
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
        )
        update_loader("💾 Сохраняю обученную модель...", 0.99, logs)
        try:
            os.makedirs(DISK_CACHE_DIR, exist_ok=True)
            packed = gzip.compress(
                pickle.dumps({"fp": APP_VERSION, "engine": engine}),
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
        try:
            st.session_state[session_key] = engine
        except Exception:
            pass
        return engine
    finally:
        model_lock.release()

