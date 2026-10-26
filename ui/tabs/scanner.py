"""ui/tabs/scanner.py — Сканер с ручным вводом кэфов."""
from __future__ import annotations
import os
import json
import csv
import io
import html
import time
import hashlib
import threading
from datetime import datetime, timedelta

import streamlit as st

from config import APP_VERSION, DIV_NAMES, DIV_TO_ODDS, DIV_TO_TSDB, DB_FILE
from storage import usage
from storage import sqlite_store as db
from data.sources import (load_seasonal, season_str, fdorg_matches, fdorg_match_result,
                          parse_date, tsdb_matches, espn_matches, highlightly_odds_fixture)
from data.fixture_merge import merge_fixture_sources
from data.marathon import marathon_odds
from model.engine import Engine
from betting.verdict import build_verdict
from betting.best_picks import best_value_market
from betting.kelly import kelly, market_type
from llm.analyst import analyze_match
from model.independent_forecasts import historical_features, record_scan_forecasts
from ui.cards import (render_verdict_card, translate_team,
                      parse_card_date, day_label, render_fixture_header)

LLM_TOP_N = 8
MAX_STAKE_PCT = 0.05
AUTO_PORTFOLIO_MIN_PROB = 0.65  # Hard confidence gate for virtual-portfolio additions.


def _runtime_secret(session_key: str, *names: str) -> str:
    """Read credentials from the current password widget, environment, or Streamlit secrets."""
    value = str(st.session_state.get(session_key, "") or "").strip()
    if value:
        return value
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        for name in names:
            value = str(st.secrets.get(name, "") or "").strip()
            if value:
                return value
    except Exception:
        pass
    return ""


def _safe_filter(rows):
    if not isinstance(rows, list):
        return []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if not r.get("HomeTeam") or not r.get("AwayTeam"):
            continue
        out.append(r)
    return out


def _print_log(msg: str):
    try:
        import sys
        print(f"[SCAN] {msg}", file=sys.stdout, flush=True)
    except Exception:
        pass


def _train_engine(matrix_n, logs, update_loader):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cur_year = today.year if today.month >= 7 else today.year - 1
    history_years = list(range(cur_year - 4, cur_year + 1))
    priority_divs = ["E0", "SP1", "I1", "D1", "F1", "E1", "SP2", "I2",
                     "D2", "F2", "N1", "B1", "P1", "T1", "R1"]
    supported_divs = list(DIV_TO_TSDB.keys()) + ["E2", "E3", "D3"]
    train_divs = list(dict.fromkeys(priority_divs + supported_divs))
    engine = Engine(matrix_n=matrix_n)
    from model.historical_match_stats import HistoricalMatchStats, walk_forward_evaluate
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
        evaluation = walk_forward_evaluate(league_rows, dv, min_history=30)
        for market_name, metric in evaluation.items():
            if metric["n"]:
                logs.append(
                    f"WF {DIV_NAMES.get(dv, dv)} {market_name}: "
                    f"n={metric['n']}, MAE={metric['mae']:.2f}"
                )
        logs.append(f"OK {DIV_NAMES.get(dv, dv)}: {len(league_rows)} матчей за {len(history_years)} сезонов")
    # Separate leakage-safe historical walk-forward check for totals.
    # Other markets skip bootstrap here to keep the one-time cache rebuild practical.
    historical_rows = []
    for dv in train_divs:
        for season_rows in history.get(dv, []):
            for original in season_rows:
                row = dict(original)
                row["Div"] = dv
                historical_rows.append(row)
    try:
        from model.walk_forward import walk_forward_compare, totals_market_is_validated
        totals_backtest = walk_forward_compare(
            historical_rows, matrix_n=matrix_n, bootstrap_samples=0,
            totals_bootstrap_samples=250,
        )
        engine.total_market_validation_report = totals_backtest.get(
            "goals_over25_candidate_minus_market", {})
        engine.total_market_validation = totals_market_is_validated(
            engine.total_market_validation_report, min_n=500)
        total_diff = engine.total_market_validation_report.get("differences", {})
        total_ci = engine.total_market_validation_report.get("confidence_intervals_95", {})
        logs.append(
            f"Исторический walk-forward тоталов: прогнозов={totals_backtest.get('goals_over25', {}).get('n', 0)}, "
            f"парных с рынком={engine.total_market_validation_report.get('n', 0)}, "
            f"источники={totals_backtest.get('goals_over25_market_sources', {})}"
        )
        if total_diff:
            logs.append(
                f"ТБ/ТМ vs рынок: ΔBrier={total_diff.get('brier'):.4f}, "
                f"ΔLogLoss={total_diff.get('log_loss'):.4f}; "
                f"95% CI Brier={total_ci.get('brier', {}).get('low')}…{total_ci.get('brier', {}).get('high')}, "
                f"LogLoss={total_ci.get('log_loss', {}).get('low')}…{total_ci.get('log_loss', {}).get('high')}"
            )
        logs.append(
            "Тоталы допущены к ручному автодобавлению."
            if engine.total_market_validation else
            "Тоталы не допущены: нужно ≥500 парных матчей и статистически подтверждённое улучшение Brier и LogLoss."
        )
        for line, line_report in (totals_backtest.get("goals_totals_by_line") or {}).items():
            metric = line_report.get("metrics") or {}
            logs.append(
                f"ТБ/ТМ {line}: walk-forward n={metric.get('n', 0)}, "
                f"Brier={metric.get('brier')}, LogLoss={metric.get('log_loss')}"
            )
    except Exception as exc:
        engine.total_market_validation = False
        engine.total_market_validation_report = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        logs.append(f"Историческая проверка тоталов не выполнена: {type(exc).__name__}: {str(exc)[:160]}")
    total = sum(len(src) for dv in train_divs for src in history.get(dv, []))
    processed = 0
    trained = 0
    for dv in train_divs:
        for src in history.get(dv, []):
            for r in src:
                try:
                    hg = float(r.get("FTHG", 0))
                    ag = float(r.get("FTAG", 0))
                    engine.learn_step(
                        r["HomeTeam"], r["AwayTeam"], hg, ag, r,
                        lg=dv, match_num=processed, total=total,
                        match_date=parse_date(r.get("Date", "")))
                    engine.match_stats.learn_step(r, dv)
                    trained += 1
                except Exception:
                    pass
                processed += 1
                if processed % 50 == 0:
                    update_loader(
                        f"Обучение [{processed}/{total}]",
                        0.5 + processed / max(1, total) * 0.3, logs)
    engine.trained_n = trained
    logs.append(f"Обучено: {trained}")
    return engine


@st.cache_resource(show_spinner=False)
def _shared_engine_cache():
    # Shared across Streamlit reruns and browser sessions in the same process.
    return {"engines": {}, "lock": threading.RLock()}


def _load_engine(matrix_n, logs, update_loader):
    """Reuse a fitted model across reruns/sessions; rebuild only on cache miss."""
    import pickle, gzip, os
    from config import DISK_CACHE_DIR

    cache_key = f"{APP_VERSION}:{int(matrix_n)}"
    session_key = f"_trained_engine_{APP_VERSION}_{int(matrix_n)}"
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
    with shared["lock"]:
        cached = shared["engines"].get(cache_key)
        if (cached is not None and hasattr(cached, "match_stats")
                and getattr(cached, "trained_n", 0) > 0):
            try:
                st.session_state[session_key] = cached
            except Exception:
                pass
            logs.append(
                f"🧠 Модель из общего кэша ({cached.trained_n} матчей; "
                "повторного обучения нет)"
            )
            return cached

        p = os.path.join(DISK_CACHE_DIR, f"engine_{APP_VERSION}_{matrix_n}.pkl.gz")
        if os.path.exists(p):
            try:
                with open(p, "rb") as f:
                    data = pickle.loads(gzip.decompress(f.read()))
                cached_engine = data.get("engine")
                if (data.get("fp") == APP_VERSION
                        and hasattr(cached_engine, "match_stats")
                        and getattr(cached_engine, "trained_n", 0) > 0):
                    shared["engines"][cache_key] = cached_engine
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
                logs.append(f"⚠️ Кэш модели не прочитан ({type(exc).__name__}); создаю заново.")

        engine = _train_engine(matrix_n, logs, update_loader)
        try:
            with open(p, "wb") as f:
                f.write(gzip.compress(pickle.dumps(
                    {"fp": APP_VERSION, "engine": engine})))
        except Exception as exc:
            logs.append(f"ℹ️ Дисковый кэш недоступен ({type(exc).__name__}); модель сохранена в памяти.")
        shared["engines"][cache_key] = engine
        try:
            st.session_state[session_key] = engine
        except Exception:
            pass
        return engine

def _render_manual_odds_block(c, D, kelly_frac, min_prob=0.40, min_edge=0.03, min_ev=0.03, market_filter=None):
    """Понятный ввод букмекерского коэффициента для выбранного рынка."""
    v = c.get("verdict", {})
    stat_preds = c.get("historical_stats") or {}
    from model.historical_match_stats import poisson_over_probability
    market_options = {
        "Победа хозяев (П1)": ("П1", float(c.get("p1") or 0)),
        "Ничья (X)": ("X", float(c.get("px") or 0)),
        "Победа гостей (П2)": ("П2", float(c.get("p2") or 0)),
        "Обе команды забьют — да": ("BTTS Да", float(c.get("btts") or 0)),
        "Обе команды забьют — нет": ("BTTS Нет", 1.0 - float(c.get("btts") or 0)),
    }
    # Evaluate every common full-match goals line, not only 2.5.
    try:
        total_mean = max(0.05, float(c.get("lam_h") or 0) + float(c.get("lam_a") or 0))
    except (TypeError, ValueError):
        total_mean = 0.0
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        over_p = (float(c.get("over") or 0) if line == 2.5 else
                  poisson_over_probability(total_mean, line) if total_mean else None)
        if over_p is None or not 0.0 < over_p < 1.0:
            continue
        line_label = f"{line:.1f}"
        market_options[f"Тотал голов больше {line_label}"] = (f"ТБ {line_label}", over_p)
        market_options[f"Тотал голов меньше {line_label}"] = (f"ТМ {line_label}", 1.0 - over_p)
    for stat_market, lines in (("corners", (8.5, 9.5, 10.5)), ("cards", (3.5, 4.5, 5.5))):
        pred = stat_preds.get(stat_market) or {}
        if pred.get("total") is not None and pred.get("history_matches", 0) >= 30:
            for line in lines:
                over_p = poisson_over_probability(pred["total"], line)
                if over_p is not None:
                    prefix = "Угловые" if stat_market == "corners" else "Карточки"
                    market_options[f"{prefix}: больше {line}"] = (f"{'Угл' if stat_market == 'corners' else 'Карты'} ТБ {line}", over_p)
                    market_options[f"{prefix}: меньше {line}"] = (f"{'Угл' if stat_market == 'corners' else 'Карты'} ТМ {line}", 1.0 - over_p)

    # Automatically select the strongest forecast; no manual market cycling.
    source_odds = c.get("market_odds") or {}
    total_odds = c.get("total_market_odds") or {}
    btts_odds = c.get("btts_market_odds") or {}
    stat_odds = c.get("stat_market_odds") or {}

    def quote_for(candidate_pick):
        kind = market_type(candidate_pick)
        # Extra markets are usable only when the detail page explicitly labels
        # a quote belongs to the exact market. A page-wide bookmaker mention is not proof
        # that every parsed price belongs to that bookmaker.
        extra_source = str(c.get("extra_market_source") or "").casefold()
        if kind in {"OU", "BTTS", "STAT"} and extra_source != "highlightly":
            return None
        if kind == "1X2":
            raw = source_odds.get(candidate_pick)
        elif kind == "OU":
            raw = total_odds.get(candidate_pick)
        elif kind == "BTTS":
            raw = btts_odds.get(candidate_pick)
        elif kind == "STAT":
            raw = stat_odds.get(candidate_pick)
        else:
            raw = None
        try:
            value = float(raw or 0)
            return value if 1.01 < value <= 100 else None
        except (TypeError, ValueError):
            return None

    filter_types = {
        "winner": {"1X2"},
        "goals": {"OU"},
        "btts": {"BTTS"},
        "corners": {"STAT"},
        "cards": {"STAT"},
    }
    # In "all matches" mode, compare every modeled market. Dedicated tabs
    # restrict the list to their own market type.
    allowed_types = filter_types.get(market_filter, {"1X2", "OU", "BTTS", "STAT"})
    candidate_options = {
        label: item for label, item in market_options.items()
        if market_type(item[0]) in allowed_types
    }
    if market_filter == "corners":
        candidate_options = {label: item for label, item in candidate_options.items()
                             if item[0].startswith("Угл ")}
    elif market_filter == "cards":
        candidate_options = {label: item for label, item in candidate_options.items()
                             if item[0].startswith("Карты ")}

    # Show all available exact-market prices in the current market view.
    displayed_quotes = [
        (item[0], quote_for(item[0]))
        for item in candidate_options.values()
        if quote_for(item[0]) is not None
    ]
    if displayed_quotes:
        quote_text = " · ".join(f"{pick}: **{odd:.2f}**" for pick, odd in displayed_quotes)
        if market_filter is None:
            st.caption("Найденные реальные коэффициенты по рынкам (каждая цена относится только к указанному исходу/линии): " + quote_text)
        else:
            st.caption("Коэффициенты букмекера именно для выбранного рынка: " + quote_text)
    elif market_filter in {"goals", "btts", "corners", "cards"}:
        st.caption("Highlightly не вернул цену именно для этого рынка; цены 1X2 не подставляются.")

    quoted_candidates = [
        (label, item[0], item[1], quote_for(item[0]))
        for label, item in candidate_options.items()
        if quote_for(item[0]) is not None and 0 < item[1] < 1
    ]
    # Prefer the strongest model forecast among markets with a real,
    # exact-market quote. If none is priced, keep the best model forecast
    # visible but mark it as unpriced and do not allow value/portfolio actions.
    if not candidate_options:
        st.info("Для этого рынка модель пока не сформировала прогноз с достаточными данными.")
        return
    # Rank priced picks by expected value, not raw probability. A 38% outcome
    # at 2.36 is not a recommendation when its EV is negative. Keep unpriced
    # model forecasts visible in market-specific views, but never present them
    # as a bet recommendation.
    priced_metrics = [
        (label, pick_name, probability, odd, probability * odd - 1,
         probability - (1.0 / odd))
        for label, pick_name, probability, odd in quoted_candidates
    ]
    positive_value = [row for row in priced_metrics if row[4] > 0 and row[5] > 0]
    if positive_value:
        selected_label, pick, model_prob, selected_odd, selected_ev, selected_edge = max(
            positive_value, key=lambda row: (row[4], row[5], row[2])
        )
        auto_quote_exists = True
        has_positive_value = True
    elif priced_metrics:
        selected_label, pick, model_prob, selected_odd, selected_ev, selected_edge = max(
            priced_metrics, key=lambda row: (row[4], row[5], row[2])
        )
        auto_quote_exists = True
        has_positive_value = False
    else:
        selected_label, (pick, model_prob) = max(
            candidate_options.items(), key=lambda row: row[1][1]
        )
        selected_odd = None
        selected_ev = None
        selected_edge = None
        auto_quote_exists = False
        has_positive_value = False

    if has_positive_value:
        st.markdown("**🎯 Лучший вариант с положительным value**")
        st.markdown(f"### {selected_label}")
        st.caption(
            f"Выбран по EV среди рынков с реальным коэффициентом: "
            f"EV {selected_ev * 100:+.1f}% · Edge {selected_edge * 100:+.1f}%."
        )
    elif auto_quote_exists:
        st.markdown("**📊 Прогноз модели — ставки с положительным value не найдено**")
        st.markdown(f"### {selected_label}")
        st.warning(
            f"Лучший из проверенных коэффициентов всё ещё без преимущества: "
            f"EV {selected_ev * 100:+.1f}% · Edge {selected_edge * 100:+.1f}%. "
            "Это прогноз для анализа, не рекомендация к ставке."
        )
    else:
        st.markdown("**📊 Прогноз модели — без подтверждённого коэффициента**")
        st.markdown(f"### {selected_label}")
        st.caption("Реального коэффициента на выбранный исход не получено; ставку и value подтвердить нельзя.")
    market_is_validated = (market_type(pick) == "1X2" or
                           (market_type(pick) == "OU" and bool(c.get("total_market_validation"))))
    fair_odd = (1.0 / model_prob) if model_prob > 0 else None

    st.caption(
        f"Вероятность модели: {model_prob * 100:.1f}% · "
        + (f"справочный кэф модели: {fair_odd:.2f}" if fair_odd else "справочный кэф модели: нет расчёта")
    )
    # Only markets with an actual bookmaker quote can be auto-priced.
    # Model fair odds are not a substitute for a real quote.
    st.info(
        "Все реальные коэффициенты берутся только из Highlightly и только для точного матча "
        "и совпадающего рынка/линии. Если Highlightly не вернул цену, система не подставляет "
        "коэффициенты с других сайтов или справочный кэф модели."
    )

    examples = {
        "1X2": "Пример: выбрали «Победа хозяев (П1)» → берите коэффициент П1.",
        "OU": "Пример: выбрали «ТБ 2.5» → берите кэф именно на тотал больше 2.5, не на ТМ 2.5.",
        "BTTS": "Пример: выбрали «Обе команды забьют — да» → берите кэф на «Да», а не на «Нет».",
        "STAT": "Пример: берите кэф именно на выбранную линию угловых или карточек. Эти рынки пока только для просмотра.",
    }
    st.caption(examples.get(market_type(pick), "Коэффициент должен соответствовать выбранному исходу и линии."))

    key_base = f"manual_{c.get('match', '')}_{pick}".replace(" ", "_").replace("/", "_")
    input_key = f"odd_{key_base}"
    btn_key = f"btn_{key_base}"

    col_input, col_metrics, col_btn = st.columns([2, 2, 1])

    # Reuse only the exact-market quote fetched from Highlightly.
    # Never make the user retype a price the scanner already has, and never treat a
    # model fair price as a bookmaker quote.
    # Automatic selection and pricing use the same exact-market quote lookup.
    # There is deliberately no manual-odds fallback.
    source_odd = quote_for(pick)

    if source_odd is not None:
        manual_odd = source_odd
        source_label = str((
            c.get("extra_market_source", c.get("total_odds_source")) if market_type(pick) in {"OU", "BTTS", "STAT"}
            else c.get("odds_source")
        ) or "Highlightly")
        col_input.success(f"Кэф загружен автоматически ({source_label}): **{manual_odd:.2f}**")
    else:
        manual_odd = None
        col_input.warning(
            "Highlightly не вернул реальный коэффициент именно на этот рынок и линию. "
            "Расчёт value и добавление в портфель отключены — коэффициент 1X2 нельзя "
            "использовать для «Обе забьют», тоталов, угловых или карточек."
        )

    if manual_odd is not None and manual_odd > 1.01 and 0.0 < model_prob < 1.0:
        market_implied = 1.0 / manual_odd
        # Do not blend bookmaker implied probability into the model probability:
        # that can make a bad quote look like a stronger prediction than the model.
        prob = model_prob
        ev = prob * manual_odd - 1
        edge = prob - market_implied
        stake = kelly(prob, manual_odd, D["bank"], kelly_frac, cap=0.02)

        col_metrics.markdown(
            f"**Расчёт для выбранного исхода**  \n"
            f"- Вероятность модели (без подмешивания кэфа): **{prob * 100:.1f}%**  \n"
            f"- Вероятность из кэфа (без снятия маржи): **{market_implied * 100:.1f}%**  \n"
            f"- Edge модели: **{edge * 100:+.1f}%**  \n"
            f"- EV по вероятности модели: **{ev * 100:+.1f}%**"
        )
        if abs(model_prob - market_implied) >= 0.20:
            col_metrics.warning(
                "Сильное расхождение модели и котировки (20 п.п. или больше). "
                "Проверьте матч, выбранный исход и линию. Не используйте этот кэф "
                "для ставки, пока причина расхождения не выяснена."
            )
        if market_is_validated:
            col_metrics.caption(f"Предварительный размер по дробному Kelly: {stake:.2f} у.е. (лимит 2% банка).")

        passed = (
            market_is_validated
            and prob >= max(float(min_prob), AUTO_PORTFOLIO_MIN_PROB)
            and edge >= min_edge
            and ev >= min_ev
            and stake > 0
        )
        if not market_is_validated:
            col_btn.info("Автодобавление отключено: для этого рынка не пройдена историческая проверка или он не поддерживается.")
        elif passed:
            existing = any(
                f"{b.get('match', '')}|{b.get('pick', '')}" == f"{c['match']}|{pick}"
                and b.get("status") == "pending"
                for b in (D.get("bets") or [])
            )
            if existing:
                col_btn.caption("Этот исход уже находится в портфеле.")
            else:
                D2 = dict(D)
                new_bet = {
                    "match": c["match"], "match_ru": c["match_ru"],
                    "div": c["div"], "league": c["league"],
                    "market": market_type(pick), "pick": pick,
                    "odds": manual_odd, "stake": stake,
                    "prob": prob, "model_prob": model_prob,
                    "market_implied_raw": market_implied,
                    "probability_method": "market_blend_0.4_model_delta",
                    "status": "pending", "strat": "scanner_auto_recommendation",
                    "odds_source": str((c.get("extra_market_source", c.get("total_odds_source")) if market_type(pick) in {"OU", "BTTS", "STAT"} else c.get("odds_source")) or "Highlightly"),
                    "mode": "real",
                    "date": (parse_date(c.get("date_iso", "")) or datetime.now()).strftime("%d.%m.%Y"),
                    "date_iso": c.get("date_iso"),
                    "match_time": c.get("time", ""),
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "fixture_id": c.get("fixture_id"), "score": None, "ev": ev,
                }
                D2["bets"] = list(D.get("bets") or []) + [new_bet]
                D2["bank"] = max(0.0, float(D.get("bank") or 0.0) - stake)
                st.session_state.data = D2
                if db.SQLITE_BOOT_OK:
                    db.insert_bets_batch([new_bet])
                    db.log_bank(D2["bank"], event="scanner_auto_recommendation")
                    db.invalidate_caches()
                usage.set_local_data(D2)
                st.toast("✅ Прогноз прошёл фильтры и добавлен в виртуальный портфель.", icon="✅")
                st.rerun()
        else:
            col_btn.caption("Автодобавление не сработало: не пройдены пороги вероятности, Edge, EV или Kelly.")
    elif source_odd is None:
        col_metrics.caption("Ожидается реальный коэффициент выбранного рынка из Highlightly.")

def render(min_prob, kelly_frac, matrix_n, min_edge=0.03):
    D = st.session_state.data
    # Modern scanner header and clear workflow: fixtures -> odds -> value -> portfolio.
    st.markdown(
        "<div class='scanner-hero'>"
        "<div class='scanner-kicker'>NEURO BET PRO · MATCH INTELLIGENCE</div>"
        "<div class='scanner-title'>Футбольный сканер</div>"
        "<div class='scanner-subtitle'>Матчи, рыночные коэффициенты и value-сигналы — в одном рабочем пространстве.</div>"
        "<div class='scanner-steps'><span>01 Матчи</span><i>→</i><span>02 Коэффициенты</span><i>→</i><span>03 Проверка value</span><i>→</i><span>04 Виртуальный портфель</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    # Compact dashboard summary keeps the first screen useful before and after a scan.
    previous_cards = [c for c in (D.get("cards") or []) if isinstance(c, dict)]
    priced_count = sum(bool(c.get("market_odds")) for c in previous_cards)
    qualified_count = sum(
        bool(c.get("best") is not None
             and (c.get("verdict") or {}).get("real_odds")
             and (c.get("verdict") or {}).get("value_passed"))
        for c in previous_cards
    )
    pending_count = sum(1 for b in (D.get("bets") or []) if b.get("status") == "pending")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Матчи в последнем скане", len(previous_cards))
    k2.metric("С реальными кэфами", priced_count)
    k3.metric("Value-проходы", qualified_count)
    k4.metric("Виртуальный портфель", pending_count)

    st.markdown("<div class='scanner-section-label'>БЫСТРЫЙ ЗАПУСК</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.25, 1], vertical_alignment="bottom")
    days = c1.slider("Горизонт матчей", 1, 14, 3,
                     help="На сколько дней вперёд искать расписание.")
    scan = c2.button("⚡ Запустить сканирование", type="primary",
                     use_container_width=True,
                     disabled=st.session_state.get("_scan_in_progress", False))

    with st.expander("📥 Импорт коэффициентов", expanded=False):
        st.markdown(
            "Вставь котировки, полученные вручную или из разрешённого API. "
            "Они сопоставляются с найденными матчами по командам и дате. "
            "Импорт не размещает ставки у букмекера."
        )
        odds_file = st.file_uploader(
            "Загрузить файл коэффициентов (.csv или .json)",
            type=["csv", "json"],
            key="scanner_odds_file",
            help="CSV: date, home, away, П1, X, П2. Также распознаются home_team, away_team, commence_time, home_odds, draw_odds, away_odds.",
        )
        st.download_button(
            "⬇️ Скачать шаблон CSV",
            data="date,home,away,П1,X,П2,source\\n,,,,,,\\n",
            file_name="neuro_bet_odds_template.csv",
            mime="text/csv",
            use_container_width=False,
        )
        odds_json = st.text_area(
            "Или вставь JSON",
            key="scanner_manual_odds_json",
            height=100,
            placeholder='[{"date":"2026-10-26","home":"Santa Clara","away":"Benfica","П1":9.0,"X":5.0,"П2":1.3,"source":"manual"}]',
        )
        try:
            parsed_quotes = []
            import_source_label = "JSON"
            if odds_file is not None:
                raw_file = odds_file.getvalue().decode("utf-8-sig", errors="replace")
                if odds_file.name.lower().endswith(".json"):
                    parsed_quotes = json.loads(raw_file)
                    import_source_label = odds_file.name
                else:
                    reader = csv.DictReader(io.StringIO(raw_file))
                    for row in reader:
                        lower = {str(k or "").strip().casefold(): v for k, v in row.items()}
                        def _csv_value(*names):
                            for name in names:
                                value = lower.get(name.casefold())
                                if value is not None and str(value).strip():
                                    return str(value).strip()
                            return ""
                        parsed_quotes.append({
                            "date": _csv_value("date", "match_date", "commence_time", "datetime", "kickoff")[:10],
                            "home": _csv_value("home", "home_team", "hometeam", "host"),
                            "away": _csv_value("away", "away_team", "awayteam", "visitor"),
                            "П1": _csv_value("П1", "1", "home_odds", "home_win", "odds_home"),
                            "X": _csv_value("X", "draw", "draw_odds", "odds_draw"),
                            "П2": _csv_value("П2", "2", "away_odds", "away_win", "odds_away"),
                            "source": _csv_value("source", "bookmaker") or odds_file.name,
                        })
                    import_source_label = odds_file.name
            elif odds_json.strip():
                parsed_quotes = json.loads(odds_json)
            else:
                parsed_quotes = []

            if isinstance(parsed_quotes, dict):
                parsed_quotes = [parsed_quotes]
            if not isinstance(parsed_quotes, list):
                raise ValueError("Нужен JSON-объект, список объектов или CSV-файл.")
            normalized_quotes = []
            for quote in parsed_quotes:
                if not isinstance(quote, dict):
                    continue
                quote_date = str(quote.get("date") or quote.get("commence_time") or "").strip()[:10]
                home_quote = str(quote.get("home") or quote.get("home_team") or "").strip()
                away_quote = str(quote.get("away") or quote.get("away_team") or "").strip()
                raw_prices = quote.get("odds") if isinstance(quote.get("odds"), dict) else quote
                aliases = {
                    "П1": ("П1", "1", "home_odds", "home_win", "odds_home"),
                    "X": ("X", "draw", "draw_odds", "odds_draw"),
                    "П2": ("П2", "2", "away_odds", "away_win", "odds_away"),
                }
                prices = {}
                for market, keys in aliases.items():
                    value = next((raw_prices[k] for k in keys if raw_prices.get(k) not in (None, "")), None)
                    prices[market] = float(str(value).replace(",", ".")) if value is not None else 0.0
                try:
                    datetime.strptime(quote_date, "%Y-%m-%d")
                except ValueError:
                    continue
                if (not home_quote or not away_quote
                        or not all(1.01 < price <= 100 for price in prices.values())):
                    continue
                normalized_quotes.append({
                    "date": quote_date, "home": home_quote, "away": away_quote,
                    "odds": prices, "source": str(quote.get("source") or quote.get("bookmaker") or "manual_import"),
                })
            st.session_state["_scanner_manual_odds_quotes"] = normalized_quotes
            if normalized_quotes:
                st.success(f"Готово: {len(normalized_quotes)} набор(а) коэффициентов из {import_source_label}.")
                st.caption("Нажми «Запустить скан», чтобы сопоставить котировки с матчами.")
            elif odds_file is not None or odds_json.strip():
                st.warning("Не удалось прочитать валидные записи. Проверь дату, названия команд и коэффициенты П1 / X / П2.")
            elif "_scanner_manual_odds_quotes" not in st.session_state:
                st.session_state["_scanner_manual_odds_quotes"] = []
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            st.error(f"Импорт не прочитан: {exc}")

    with st.expander("⚙️ Настройки сканера", expanded=False):
        st.caption("Дополнительные параметры скрыты, чтобы не перегружать основной экран.")
        source_col, portfolio_col = st.columns(2)
        with source_col:
            st.checkbox(
                "Разрешить источник Prognozist / Marathon",
                value=False,
                key="prognozist_permission_input",
                help="Источник используется только в рамках доступных условий и разрешений.",
            )
        with portfolio_col:
            auto_add = st.checkbox(
                "Автодобавление в виртуальный портфель",
                value=True,
                key="scanner_auto_add_qualified_v3",
                help="Добавляются только прогнозы с реальными коэффициентами, прошедшие value-фильтры. Реальные ставки не размещаются.",
            )
    auto_add = st.session_state.get("scanner_auto_add_qualified_v3", True)

    if not scan:
        fn = D.get("funnel")
        if fn:
            st.success(
                f"🧠 Обучено {fn.get('trained', 0)} · "
                f"📡 Источников {fn.get('src', 0)} · "
                f"🎯 Найдено {fn.get('found', 0)} · "
                f"🤖 Автодобавлено {fn.get('added', 0)} · "
                f"💰 Заморожено {fn.get('frozen', 0):.0f} · "
                f"🤖 LLM: {fn.get('llm', 0)}"
            )
        with st.expander("🔌 Диагностика"):
            for line in D.get("report", []):
                st.text(line)

        all_cards = [c for c in D.get("cards", []) if isinstance(c, dict)]
        if not all_cards:
            st.info("Нажми ⚡ СКАН.")
            return

        qualified = [c for c in all_cards
                     if c.get("best") is not None
                     and (c.get("verdict") or {}).get("real_odds")
                     and (c.get("verdict") or {}).get("value_passed")]
        try:
            from betting.ranking import rank_score
            qualified.sort(key=lambda c: (
                -rank_score(c),
                -float((c.get("verdict") or {}).get("ev") or -1),
                -float((c.get("verdict") or {}).get("prob") or 0),
            ))
        except Exception:
            qualified.sort(key=lambda c: -float(
                (c.get("verdict") or {}).get("ev") or -1))

        st.markdown(
            "<div style='display:flex;align-items:end;justify-content:space-between;gap:12px;margin:8px 0 4px'>"
            "<div><div class='scanner-kicker'>SHORTLIST · AI MATCH DESK</div>"
            "<h2 style='margin:0'>Что играем</h2>"
            "<div style='color:#91a4ad;font-size:.88rem;margin-top:5px'>Только отобранные исходы. Расчёты — внутри, здесь главное решение и причина.</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if not qualified:
            st.markdown(
                "<div class='pick-story' style='animation:none'><div class='pick-choice'>Пока нет ставки, которую можно подтвердить</div>"
                "<div class='pick-ai'><b>Почему так?</b>Для рекомендации нужны совпавшие реальные коэффициенты и пройденные фильтры. "
                "Сканер не будет придумывать котировки или добавлять слабый вариант ради заполнения списка.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='pick-feed'>", unsafe_allow_html=True)
            for rank, card in enumerate(qualified[:6], start=1):
                verdict = card.get("verdict") or {}
                match_name = str(card.get("match_ru") or card.get("match") or "Матч")
                league_name = str(card.get("league") or "Футбол")
                pick_name = str(card.get("best") or verdict.get("pick") or verdict.get("best_pick") or "Исход выбран моделью")
                odd = verdict.get("real_odds") or verdict.get("odds") or card.get("best_odds")
                try:
                    odd_text = f"Коэффициент {float(odd):.2f}" if odd is not None else ""
                except (TypeError, ValueError):
                    odd_text = ""
                date_text = str(card.get("date") or "")
                time_text = str(card.get("time") or "")
                kickoff = " · ".join(v for v in (date_text, time_text) if v)
                comment = next((str(card.get(k)).strip() for k in (
                    "ai_comment", "ai_analysis", "llm_comment", "commentary",
                    "explanation", "reason", "analysis_text", "why_pick"
                ) if card.get(k) and str(card.get(k)).strip()), "")
                status_text = "ОТОБРАНО" if verdict.get("value_passed") else "НА ПРОВЕРКЕ"
                status_class = "" if verdict.get("value_passed") else " watch"
                probability = verdict.get("prob")
                prob_chip = ""
                try:
                    if probability is not None:
                        prob_chip = f"<span class='pick-chip'>Оценка модели {float(probability)*100:.0f}%</span>"
                except (TypeError, ValueError):
                    pass
                comment_html = (
                    "<div class='pick-ai'><b>Комментарий ИИ</b>" + html.escape(comment) + "</div>"
                    if comment else
                    "<div class='pick-ai pick-ai-empty'><b>Комментарий</b>Отдельный текстовый комментарий ИИ для этого матча пока не сформирован.</div>"
                )
                odd_html = f"<span class='pick-chip'>{html.escape(odd_text)}</span>" if odd_text else ""
                st.markdown(
                    f"<article class='pick-story' style='animation-delay:{min(rank-1,5)*55}ms'>"
                    f"<div class='pick-story-top'><span class='pick-league'>{html.escape(league_name)}</span>"
                    f"<span class='pick-status{status_class}'>{status_text}</span></div>"
                    f"{render_fixture_header(card)}" + f"<div class='pick-match'>{html.escape(match_name)}</div>"
                    f"<div class='pick-time'>{html.escape(kickoff or 'Время начала уточняется')}</div>"
                    f"<div class='pick-divider'></div><div class='pick-choice-label'>Наш выбор</div>"
                    f"<div class='pick-choice'>{html.escape(pick_name)}</div>"
                    f"<div class='pick-meta'>{odd_html}{prob_chip}</div>"
                    f"{comment_html}</article>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 2, 2])
        sort_mode = c1.selectbox(
            "Сортировка",
            ["По общему рейтингу",
             "По EV (высокие сначала)",
             "По вероятности (высокие сначала)",
             "По дате (ближайшие сначала)"],
            key="scanner_sort")
        show_hidden = c2.checkbox("Только сильные модели", False,
                                   key="scanner_show_hidden")
        group_by_day = c3.checkbox("Группировать по дням", True,
                                    key="scanner_group_by_day")

        if show_hidden:
            cards_view = [c for c in all_cards
                          if (c.get("verdict") or {}).get("is_action")]
        else:
            # Manual bookmaker odds should be available for every scanned fixture.
            cards_view = all_cards

        if sort_mode.startswith("По дате"):
            cards_view.sort(key=lambda c: parse_card_date(c))
        elif sort_mode.startswith("По вероятности"):
            cards_view.sort(
                key=lambda c: -(c.get("verdict", {}).get("prob") or 0))
        elif sort_mode.startswith("По EV"):
            cards_view.sort(
                key=lambda c: -(c.get("verdict", {}).get("ev") or -1))
        else:
            try:
                from betting.ranking import rank_score
                cards_view.sort(key=lambda c: -rank_score(c))
            except Exception:
                cards_view.sort(key=lambda c: -float(
                    (c.get("verdict", {}).get("prob") or 0)))

        if not cards_view:
            st.warning("Ничего не найдено.")
            return

        # Until verified Marathon odds are available for extra markets, keep
        # the scanner focused on 1X2 only. This prevents model-only totals/BTTS
        # forecasts from being presented beside winner odds or entering the portfolio.
        market_tabs = st.tabs([
            f"Все матчи ({len(cards_view)})",
            f"Победа / 1X2 ({sum(bool(c.get('market_odds')) for c in cards_view)})",
        ])

        with market_tabs[0]:
            if group_by_day:
                grouped = {}
                for c in cards_view:
                    grouped.setdefault(day_label(parse_card_date(c)), []).append(c)
                sorted_groups = sorted(grouped.items(), key=lambda kv: parse_card_date(kv[1][0]))
                for label, items in sorted_groups:
                    st.markdown(f"### {label} · {len(items)} матч.")
                    for c in items:
                        st.markdown(render_fixture_header(c), unsafe_allow_html=True)
                        _render_manual_odds_block(
                            c, D, kelly_frac, min_prob, min_edge, market_filter="winner"
                        )
            else:
                for c in cards_view:
                    st.markdown(render_fixture_header(c), unsafe_allow_html=True)
                    _render_manual_odds_block(
                        c, D, kelly_frac, min_prob, min_edge, market_filter="winner"
                    )

        with market_tabs[1]:
            winner_cards = [c for c in cards_view if c.get("p1") is not None and c.get("p2") is not None]
            if not winner_cards:
                st.info("Модель не сформировала прогноз 1X2.")
            for c in winner_cards:
                st.markdown(render_fixture_header(c), unsafe_allow_html=True)
                _render_manual_odds_block(
                    c, D, kelly_frac, min_prob, min_edge, market_filter="winner"
                )

        st.caption("Пока доступны только подтверждённые коэффициенты 1X2. Тоталы, «Обе забьют», угловые и карточки временно скрыты до проверки реальных котировок Marathon.")
        st.caption("✅ Показано " + str(len(cards_view)) + " из "
                   + str(len(all_cards)))
        return

    # ==================== СКАН ====================
    st.session_state["_scan_in_progress"] = True
    try:
        loader_ph = st.empty()
        log_ph = st.empty()

        def update_loader(text, pct, logs=None):
            loader_ph.markdown(
                f"<div class='nbr-loader'><div class='nbr-ring'></div>"
                f"<div class='nbr-text'><b>{text}</b>"
                f"<div class='nbr-bar'><div class='nbr-bar-fill' "
                f"style='width:{pct*100:.0f}%'></div></div></div></div>",
                unsafe_allow_html=True)
            if logs:
                log_ph.markdown(
                    "<div style='color:#8b93a7;font-size:.8rem;"
                    "background:rgba(10,14,24,.6);padding:10px;"
                    "border-radius:10px;max-height:180px;overflow-y:auto'>"
                    + "<br>".join(logs[-12:]) + "</div>",
                    unsafe_allow_html=True)

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        logs = []

        token = _runtime_secret("fdorg_token_input", "FDORG_TOKEN", "fdorg_token")
        fixture_sources = []
        try:
            fixture_sources.extend(tsdb_matches(days, logs))
        except Exception as exc:
            logs.append(f"⚠️ TheSportsDB fixtures: {type(exc).__name__}")
        try:
            fixture_sources.extend(espn_matches(days, logs))
        except Exception as exc:
            logs.append(f"⚠️ ESPN fixtures: {type(exc).__name__}")
        if token:
            try:
                fixture_sources.extend(fdorg_matches(days, token, logs))
            except Exception as exc:
                logs.append(f"⚠️ football-data.org fixtures: {type(exc).__name__}")
        try:
            rows = _safe_filter(merge_fixture_sources(fixture_sources))
        except Exception:
            # Keep scanner usable even if cross-provider deduplication fails.
            rows = _safe_filter(fixture_sources)
        logs.append(f"📡 Матчи для анализа из футбольных источников: {len(rows)}")
        if not rows:
            logs.append("⚠️ Матчи не найдены. Проверьте доступность источников расписания.")

        update_loader("🧠 Загрузка модели...", 0.20, logs)
        engine = _load_engine(matrix_n, logs, update_loader)

        update_loader("🧠 Анализ матчей...", 0.80, logs)
        cards = []
        matches_with_best = 0
        xg_used_count = 0
        highlightly_key = _runtime_secret("highlightly_api_key_input", "HIGHLIGHTLY_API_KEY", "highlightly_api_key")

        funnel_counts = {"model": 0, "market_req": 0, "market_got": 0,
                          "bet": 0, "watch": 0, "skip": 0,
                          "model_only": 0, "estimated": 0, "real": 0}
        if not highlightly_key:
            logs.append(
                "❌ Highlightly API-ключ не найден: odds-запросы не выполнялись. "
                "Добавьте ключ в поле Highlightly API key или секрет HIGHLIGHTLY_API_KEY."
            )

        for match_idx, r in enumerate(rows, 1):
            if match_idx == 1 or match_idx % 5 == 0:
                update_loader(
                    f"🧠 Анализ матчей... {match_idx}/{len(rows)}",
                    min(0.88, 0.80 + 0.08 * match_idx / max(1, len(rows))),
                    logs,
                )
            d = parse_date(r.get("Date", ""))
            if not d or not (today <= d < today + timedelta(days=days)):
                continue
            h_en = (r.get("HomeTeam") or "").strip()
            a_en = (r.get("AwayTeam") or "").strip()
            if not h_en or not a_en:
                continue
            h_ru = translate_team(h_en)
            a_ru = translate_team(a_en)
            lg = r.get("Div") or "G"

            funnel_counts["model"] += 1

            try:
                P = engine.predict(h_en, a_en, lg, match_date=d,
                                   cup=lg in ("C1", "EL", "EC"))
            except Exception:
                funnel_counts["skip"] += 1
                continue
            if P.get("xg_used"):
                xg_used_count += 1

            fh = engine.form_str(h_en)
            fa = engine.form_str(a_en)
            verdict, rows_, _ = build_verdict(
                P, min_prob, D["bank"], kelly_frac,
                h_ru, a_ru, fh, fa, P.get("h2h_n", 0))

            best = None
            odds_source = "model_only"
            market_odds = {}
            total_market_odds = dict(r.get("total_market_odds") or {})
            btts_market_odds = dict(r.get("btts_market_odds") or {})
            stat_market_odds = dict(r.get("stat_market_odds") or {})
            total_odds_source = "highlightly" if total_market_odds else "no_market"
            extra_market_source = str(r.get("extra_market_source") or "")
            # Fail closed: secondary-market odds are accepted only from Highlightly.
            if extra_market_source.casefold() != "highlightly":
                total_market_odds = {}
                btts_market_odds = {}
                stat_market_odds = {}
            api_market_data = None
            odds_source = "no_market"

            # Fetch real 1X2 odds for every displayed fixture, not only when
            # the model already crosses the probability threshold. The strict
            # value gates below decide whether a bet is eligible; odds display
            # and source diagnostics must not depend on model confidence.
            if rows_ is not None:
                try:
                    candidates = []

                    # Highlightly is the sole permitted bookmaker-odds source.
                    if highlightly_key:
                        funnel_counts["market_req"] += 1
                        try:
                            highlighted = highlightly_odds_fixture(
                                d.strftime("%Y-%m-%d"), h_en, a_en, highlightly_key, logs
                            )
                            if isinstance(highlighted, dict):
                                if all(key in highlighted for key in ("П1", "X", "П2")):
                                    candidates.append((highlighted.get("source", "highlightly"), highlighted))
                                    market_odds = {key: highlighted[key] for key in ("П1", "X", "П2")}
                                    odds_source = highlighted.get("source", "highlightly")
                                total_market_odds = dict(highlighted.get("total_market_odds") or {})
                                btts_market_odds = dict(highlighted.get("btts_market_odds") or {})
                                stat_market_odds = dict(highlighted.get("stat_market_odds") or {})
                                total_odds_source = "highlightly" if total_market_odds else "no_market"
                                extra_market_source = "highlightly"
                        except Exception as exc:
                            logs.append(f"⚠️ Highlightly lookup failed: {type(exc).__name__}: {str(exc)[:120]}")

                    # Fallback: Prognozist/Marathon real 1X2 line, using strict team and date matching.
                    # This runs when Highlightly is unavailable or has no line for this fixture.
                    if not market_odds:
                        try:
                            marathon_line = marathon_odds(
                                h_en, a_en, d.strftime("%Y-%m-%d"), lg
                            )
                            if isinstance(marathon_line, dict) and all(
                                key in marathon_line for key in ("П1", "X", "П2")
                            ):
                                candidate = {}
                                for key in ("П1", "X", "П2"):
                                    value = float(marathon_line[key])
                                    if not 1.01 < value <= 100:
                                        raise ValueError(f"invalid {key} odds")
                                    candidate[key] = value
                                market_odds = candidate
                                odds_source = "prognozist_marathon"
                                funnel_counts["market_req"] += 1
                                logs.append(
                                    f"🏁 Prognozist/Marathon: {h_ru} — {a_ru} | "
                                    f"П1 {candidate['П1']:.2f} / X {candidate['X']:.2f} / П2 {candidate['П2']:.2f}"
                                )
                        except Exception as exc:
                            logs.append(f"⚠️ Prognozist/Marathon: {type(exc).__name__}: {str(exc)[:100]}")

                    # User-provided/manual quote import: exact date + exact home/away
                    # names only. Highlightly remains preferred; manual prices are a fallback.
                    if not market_odds:
                        def _team_key(value):
                            # Ignore punctuation, accents and common club prefixes.
                            import unicodedata
                            raw = unicodedata.normalize("NFKD", str(value or ""))
                            raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
                            key = "".join(ch.lower() for ch in raw if ch.isalnum())
                            for prefix in ("clubde", "cd", "fc", "sc", "cf", "sporting"):
                                if key.startswith(prefix) and len(key) > len(prefix) + 3:
                                    key = key[len(prefix):]
                            return key

                        fixture_home_key, fixture_away_key = _team_key(h_en), _team_key(a_en)
                        fixture_date = d.strftime("%Y-%m-%d")
                        for quote in st.session_state.get("_scanner_manual_odds_quotes", []):
                            quote_home_key = _team_key(quote.get("home"))
                            quote_away_key = _team_key(quote.get("away"))
                            # Exact normalized match first; allow a clear containment
                            # match for harmless prefixes such as "CD Santa Clara".
                            home_match = (
                                fixture_home_key == quote_home_key
                                or (min(len(fixture_home_key), len(quote_home_key)) >= 5
                                    and (fixture_home_key in quote_home_key or quote_home_key in fixture_home_key))
                            )
                            away_match = (
                                fixture_away_key == quote_away_key
                                or (min(len(fixture_away_key), len(quote_away_key)) >= 5
                                    and (fixture_away_key in quote_away_key or quote_away_key in fixture_away_key))
                            )
                            quote_date = str(quote.get("date") or "")
                            try:
                                date_delta = abs((datetime.strptime(quote_date, "%Y-%m-%d").date()
                                                  - d.date()).days)
                            except ValueError:
                                date_delta = 999
                            if date_delta <= 1 and home_match and away_match:
                                imported = dict(quote.get("odds") or {})
                                if all(key in imported and 1.01 < float(imported[key]) <= 100
                                       for key in ("П1", "X", "П2")):
                                    market_odds = {key: float(imported[key]) for key in ("П1", "X", "П2")}
                                    odds_source = str(quote.get("source") or "manual_import")
                                    logs.append(
                                        f"📥 Ручной импорт коэффициентов ({odds_source}, дата ±{date_delta} дн.): "
                                        f"{h_en} — {a_en} | П1 {market_odds['П1']:.2f} / "
                                        f"X {market_odds['X']:.2f} / П2 {market_odds['П2']:.2f}"
                                    )
                                    break

                    # If no provider has a verified line, leave odds empty.

                    for source_name, candidate in candidates:
                        if not isinstance(candidate, dict):
                            continue
                        valid = {}
                        for key in ("П1", "X", "П2"):
                            try:
                                value = float(candidate.get(key))
                                if 1.01 < value <= 100:
                                    valid[key] = value
                            except (TypeError, ValueError):
                                continue
                        if all(key in valid for key in ("П1", "X", "П2")):
                            market_odds = valid
                            odds_source = source_name
                            break

                    if market_odds:
                        funnel_counts["market_got"] += 1
                        logs.append(
                            f"🏁 Реальные кэфы ({odds_source}): {h_ru} — {a_ru} | "
                            f"П1 {market_odds['П1']:.2f} / X {market_odds['X']:.2f} / "
                            f"П2 {market_odds['П2']:.2f}"
                        )
                except Exception as exc:
                    logs.append(f"⚠️ Ошибка получения линии: {type(exc).__name__}")

            # Если есть реальные кэфы — используем best_value_market
            if market_odds:
                chosen_verdict, best = best_value_market(
                    verdict, rows_, market_odds, D["bank"], kelly_frac,
                    min_prob, min_edge=max(min_edge, 0.05), min_ev=0.05,
                )
                if best is not None:
                    verdict = chosen_verdict
                    funnel_counts["real"] += 1
                else:
                    funnel_counts["model_only"] += 1
            else:
                funnel_counts["model_only"] += 1

            # No bookmaker price means no bet candidate. Fair odds are display-only.
            if best is None:
                verdict["real_odds"] = False
                verdict["odd"] = None
                verdict["ev"] = None
                verdict["edge"] = None
                verdict["stake"] = 0.0
                verdict["is_bet"] = False
                # Keep the fetched line/source for display even when it fails
                # value gates. "No bet" is not the same as "no odds found".
                if not market_odds:
                    odds_source = "no_market"

            if best:
                matches_with_best += 1
                funnel_counts["bet"] += 1
                logs.append(
                    f"✅ BET: {h_ru} — {a_ru} | "
                    f"{best[1]} @ {best[2]:.2f} ({odds_source})")
            else:
                logs.append(
                    f"❌ NO BEST: {h_ru} — {a_ru} | "
                    f"is_action={verdict.get('is_action')}")

            cards.append({
                "div": lg,
                "league": r.get("League") or DIV_NAMES.get(lg, "Лига"),
                "match": f"{h_en} vs {a_en}",
                "match_ru": f"{h_ru} — {a_ru}",
                "date": d.strftime("%d.%m") +
                        (f" {r.get('Time','')}" if r.get("Time") else ""),
                "verdict": verdict, "best": best,
                "games": P["games"], "fh": fh, "fa": fa,
                "fixture_id": r.get("fixture_id"),
                "historical_stats": (engine.match_stats.predict(h_en, a_en, lg) if hasattr(engine, "match_stats") else {}),
                "date_iso": d.strftime("%Y-%m-%d"),
                "time": r.get("Time") or "",
                "lam_h": P["lams"][0], "lam_a": P["lams"][1],
                # LLM receives observed venue-specific goal averages, not the primary model's lambdas.
                "independent_features": historical_features(engine, h_en, a_en),
                "xg_used": bool(P.get("xg_used")),
                "total_market_validation": bool(getattr(engine, "total_market_validation", False)),
                "total_market_validation_report": getattr(engine, "total_market_validation_report", {}),
                "p1": P["p1"], "px": P["x"], "p2": P["p2"],
                "over": P["over"], "btts": P["btts"],
                "odds_source": odds_source,
                "market_odds": dict(market_odds),
                "total_market_odds": dict(total_market_odds),
                "total_odds_source": total_odds_source if total_market_odds else (str(r.get("extra_market_source") or "") if r.get("total_market_odds") else total_odds_source),
                "extra_market_source": extra_market_source,
                "btts_market_odds": dict(btts_market_odds),
                "stat_market_odds": dict(stat_market_odds),
                "home_badge": r.get("home_badge") or "",
                "away_badge": r.get("away_badge") or "",
                "league_badge": r.get("league_badge") or "",
                "venue": r.get("venue") or r.get("stadium") or "",
                "stadium_image": r.get("stadium_image") or r.get("venue_image") or "",
                "context": {},
            })

        logs.append(f"📈 xG: использован в {xg_used_count} из {funnel_counts['model']} прогнозов")
        logs.append(f"🎯 BET: {matches_with_best}")

        # ============ LLM ============
        llm_key = _runtime_secret("llm_api_key_input", "LLM_API_KEY", "llm_api_key")
        llm_prov = D.get("meta", {}).get("llm_provider",
                                          "Groq (бесплатно, быстро)")
        llm_model = D.get("meta", {}).get("llm_model", "")

        # Review fixtures with the most historical observations first, not only
        # fixtures already selected by the value model. The LLM still receives no
        # model probabilities, pick, confidence, EV, or bookmaker prices.
        action_cards = [
            c for c in cards
            if isinstance(c.get("verdict"), dict)
            and c.get("lam_h") is not None and c.get("lam_a") is not None
        ]
        # Deterministic hash sampling spreads the limited LLM budget across fixtures
        # without prioritizing the main model's favourite pick, confidence, or odds value.
        action_cards.sort(key=lambda c: hashlib.sha256(
            f"{c.get('date_iso','')}|{c.get('match','')}".encode("utf-8")
        ).hexdigest())
        llm_done = 0

        logs.append(
            f"🤖 LLM подготовка: key={'есть' if llm_key else 'НЕТ'} | "
            f"provider={llm_prov} | "
            f"analysis_cards={len(action_cards)} | "
            f"remaining={usage.llm_remaining()}")

        if llm_key and usage.llm_remaining() > 0 and action_cards:
            for i_, card in enumerate(action_cards[:LLM_TOP_N]):
                if usage.llm_remaining() <= 0:
                    logs.append("🤖 LLM: лимит")
                    break
                if i_ > 0:
                    time.sleep(2)
                update_loader(
                    f"🤖 Независимый ИИ-анализ [{i_+1}/{min(LLM_TOP_N, len(action_cards))}]",
                    0.90 + 0.08 * (i_+1) / LLM_TOP_N, logs)
                v_ = card.get("verdict", {})
                # Blind AI context: the statistical model's probabilities, selected outcome,
                # confidence, EV and market prices are deliberately not passed to the LLM.
                # Model outcome is used only after the LLM responds, for a separate comparison.
                ctx_llm = {
                    "api_key": llm_key, "provider": llm_prov, "model": llm_model,
                    "home": card.get("match_ru", "").split(" — ")[0],
                    "away": card.get("match_ru", "").split(" — ")[-1],
                    "league": card.get("league", ""),
                    "date": f"{card.get('date_iso') or card.get('date', '')} {card.get('time') or ''}".strip(),
                    "independent_features": card.get("independent_features") or {},
                    "fh": card.get("fh", "—"), "fa": card.get("fa", "—"),
                    # Main-model label is derived locally after the blind LLM response.
                    "model_pick_code": max(
                        (("П1", float(card.get("p1") or 0)),
                         ("X", float(card.get("px") or 0)),
                         ("П2", float(card.get("p2") or 0))),
                        key=lambda item: item[1],
                    )[0],
                }
                try:
                    opinion = analyze_match(ctx_llm)
                except Exception as exc:
                    opinion = None
                    logs.append(
                        f"🤖 LLM ошибка [{i_+1}]: {type(exc).__name__}: {str(exc)[:160]}"
                    )
                if opinion:
                    card["llm_opinion"] = opinion
                    llm_done += 1
                    logs.append(f"🤖 OK [{i_+1}]: {opinion[:60]}")
                else:
                    logs.append(f"🤖 FAIL [{i_+1}]: ответ не получен")
            logs.append(f"🤖 LLM итог: {llm_done} мнений")
        elif not llm_key:
            logs.append("🤖 LLM: ключ не задан")
        elif not action_cards:
            logs.append("🤖 LLM: нет матчей с данными для независимого анализа")

        # Persist both forecasters before the fixture starts. The ledger keeps the
        # first prediction immutable, making repeated scans idempotent and auditable.
        try:
            tracked = record_scan_forecasts(
                cards, provider=llm_prov, model=llm_model, db_path=DB_FILE,
            )
            logs.append(
                "📚 Журнал прогнозов: "
                f"модель +{tracked['main_recorded']}, ИИ +{tracked['ai_recorded']}, "
                f"без времени старта пропущено {tracked['skipped_no_time']}, "
                f"после старта пропущено {tracked['skipped_started']}, "
                f"ошибок {tracked['invalid']}"
            )
        except Exception as exc:
            logs.append(f"⚠️ Журнал прогнозов ИИ недоступен: {type(exc).__name__}: {str(exc)[:120]}")

        # Автодобавление — только в виртуальный портфель, не в аккаунт букмекера.
        # Учитываем только реальные котировки 1X2, прошедшие все value-фильтры.
        new_bets = []
        total_stake = 0.0
        available_bank = float(D.get("bank") or 0.0)
        existing_keys = {
            (str(b.get("match") or ""), str(b.get("pick") or ""))
            for b in (D.get("bets") or [])
            if b.get("status") == "pending"
        }
        if auto_add:
            for card in cards:
                verdict_card = card.get("verdict") or {}
                best_card = card.get("best")
                if not best_card or not verdict_card.get("real_odds") or not verdict_card.get("value_passed"):
                    continue
                pick = str(best_card[1])
                if market_type(pick) != "1X2":
                    continue
                match_key = (str(card.get("match") or ""), pick)
                if match_key in existing_keys:
                    continue
                try:
                    odds_value = float(best_card[2])
                    probability = float(verdict_card.get("prob") or 0.0)
                    ev_value = float(verdict_card.get("ev") or 0.0)
                    stake_value = kelly(
                        probability, odds_value, available_bank,
                        kelly_frac, cap=0.02,
                    )
                except (TypeError, ValueError, IndexError):
                    continue
                # High confidence is a hard portfolio gate, in addition to value/EV.
                if probability < max(float(min_prob), AUTO_PORTFOLIO_MIN_PROB):
                    continue
                if stake_value <= 0 or stake_value > available_bank:
                    continue
                new_bet = {
                    "match": card.get("match", ""),
                    "match_ru": card.get("match_ru", card.get("match", "")),
                    "div": card.get("div", ""),
                    "league": card.get("league", ""),
                    "market": "1X2",
                    "pick": pick,
                    "odds": odds_value,
                    "stake": stake_value,
                    "prob": probability,
                    "model_prob": verdict_card.get("model_prob"),
                    "market_implied_raw": verdict_card.get("market_implied_raw"),
                    "probability_method": verdict_card.get("probability_method", "market_blend_0.4_model_delta"),
                    "status": "pending",
                    "strat": "scanner_auto",
                    "odds_source": card.get("odds_source", "unknown"),
                    "mode": "real",
                    "date": (parse_date(card.get("date_iso", "")) or datetime.now()).strftime("%d.%m.%Y"),
                    "date_iso": card.get("date_iso"),
                    "match_time": card.get("time", ""),
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "fixture_id": card.get("fixture_id"),
                    "score": None,
                    "ev": ev_value,
                }
                new_bets.append(new_bet)
                existing_keys.add(match_key)
                available_bank -= stake_value
                total_stake += stake_value
                logs.append(f"🤖 AUTO-PORTFOLIO: {card.get('match_ru', card.get('match', ''))} | {pick} @ {odds_value:.2f} | stake {stake_value:.2f}")

        # Explain why the portfolio stayed empty instead of silently showing zero additions.
        if auto_add and not new_bets:
            pending_count = len(existing_keys)
            priced_count = sum(1 for card in cards if card.get("market_odds"))
            qualified_count = sum(
                1 for card in cards
                if card.get("best")
                and (card.get("verdict") or {}).get("real_odds")
                and (card.get("verdict") or {}).get("value_passed")
            )
            logs.append(
                "ℹ️ AUTO-PORTFOLIO: новых ставок 0; "
                f"матчей с реальными кэфами 1X2={priced_count}/{len(cards)}, "
                f"прошедших value-фильтры={qualified_count}, "
                f"текущих ожидающих ставок в ключах={pending_count}. "
                "Если кэфы есть, но value=0, ни один исход не прошёл пороги "
                "вероятности/edge/EV/Kelly; ставки без value не добавляются."
            )
        elif not auto_add:
            logs.append("ℹ️ AUTO-PORTFOLIO выключен переключателем; ставки автоматически не добавлялись.")

        # Totals are auto-added only after the historical market benchmark passes.
        if auto_add:
            for card in cards:
                if not card.get("total_market_validation"):
                    continue
                try:
                    over_p = float(card.get("over"))
                except (TypeError, ValueError):
                    continue
                if not 0.0 < over_p < 1.0:
                    continue
                odds_by_pick = card.get("total_market_odds") or {}
                for pick, model_p in (("ТБ 2.5", over_p), ("ТМ 2.5", 1.0 - over_p)):
                    match_key = (str(card.get("match") or ""), pick)
                    if pick not in odds_by_pick or match_key in existing_keys:
                        continue
                    try:
                        odd = float(odds_by_pick[pick])
                        if not 1.01 < odd <= 100:
                            continue
                        implied = 1.0 / odd
                        prob = max(1e-6, min(1.0 - 1e-6, implied + 0.40 * (model_p - implied)))
                        edge_value = prob - implied
                        ev_value = prob * odd - 1.0
                        stake = kelly(prob, odd, available_bank, kelly_frac, cap=0.02)
                    except (TypeError, ValueError, ZeroDivisionError):
                        continue
                    if (prob < max(float(min_prob), AUTO_PORTFOLIO_MIN_PROB)
                            or edge_value < float(min_edge) or ev_value < 0.03
                            or stake <= 0 or stake > available_bank):
                        continue
                    new_bets.append({
                        "match": card.get("match", ""), "match_ru": card.get("match_ru", card.get("match", "")),
                        "div": card.get("div", ""), "league": card.get("league", ""),
                        "market": "OU", "pick": pick, "odds": odd, "stake": stake,
                        "prob": prob, "model_prob": model_p, "market_implied_raw": implied,
                        "probability_method": "market_blend_0.4_model_delta",
                        "status": "pending", "strat": "scanner_auto_totals",
                        "odds_source": card.get("total_odds_source", "highlightly"), "mode": "real",
                        "date": (parse_date(card.get("date_iso", "")) or datetime.now()).strftime("%d.%m.%Y"),
                        "date_iso": card.get("date_iso"), "match_time": card.get("time", ""),
                        "date_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "fixture_id": card.get("fixture_id"), "score": None, "ev": ev_value,
                    })
                    existing_keys.add(match_key)
                    available_bank -= stake
                    total_stake += stake
                    logs.append("AUTO-TOTALS: " + str(card.get("match_ru") or card.get("match") or "") + " | " + pick + " @ " + format(odd, ".2f"))

        D2 = dict(D)
        D2["cards"] = cards
        D2["report"] = logs
        D2["bets"] = list(D.get("bets") or []) + new_bets
        D2["bank"] = max(0.0, float(D.get("bank") or 0.0) - total_stake)
        D2["funnel"] = {
            "trained": getattr(engine, "trained_n", 0),
            "src": len(rows), "found": matches_with_best,
            "added": len(new_bets), "frozen": total_stake, "llm": llm_done,
        }
        st.session_state.data = D2
        usage.set_local_data(D2)
        if db.SQLITE_BOOT_OK:
            db.insert_bets_batch(new_bets)
            db.log_bank(D2["bank"], event="scan")
            db.invalidate_caches()

        # Surface source diagnostics in deployment logs, not only the in-app report.
        for source_line in logs:
            if "Highlightly" in source_line or "матчей после фильтрации" in source_line:
                _print_log(f"SOURCE: {source_line}")

        _print_log(f"FUNNEL: {funnel_counts}")

        update_loader(
            f"✅ Готово! автодобавлено {len(new_bets)} ставок · 🤖 {llm_done} LLM",
            1.0, logs)
        time.sleep(1.2)
        loader_ph.empty()
        log_ph.empty()
    except Exception as exc:
        message = f"Сканирование прервано: {type(exc).__name__}: {str(exc)[:240]}"
        _print_log(message)
        try:
            D2 = dict(st.session_state.get("data") or D)
            report = list(D2.get("report") or [])
            report.append(message)
            D2["report"] = report[-100:]
            st.session_state.data = D2
            usage.set_local_data(D2)
        except Exception:
            pass
        st.error(message)
        st.info("Проверь раздел «Диагностика» и последние строки журнала.")
        return
    finally:
        st.session_state["_scan_in_progress"] = False
    st.rerun()
