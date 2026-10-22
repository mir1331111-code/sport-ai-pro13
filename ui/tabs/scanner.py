"""ui/tabs/scanner.py — Сканер с ручным вводом кэфов."""
from __future__ import annotations
import os
import time
from datetime import datetime, timedelta

import streamlit as st

from config import APP_VERSION, DIV_NAMES, DIV_TO_ODDS, DIV_TO_TSDB, DB_FILE
from storage import usage
from storage import sqlite_store as db
from data.sources import (load_seasonal, season_str,
                          fdorg_matches, fdorg_match_result, odds_api_fixture, parse_date,
                          tsdb_matches, espn_matches, therundown_odds_fixture,
                          prognozist_marathon_odds)
from data.marathon import marathon_odds
from data.fixture_merge import merge_fixture_sources
from model.engine import Engine
from betting.verdict import build_verdict
from betting.best_picks import best_value_market
from betting.kelly import kelly, market_type
from llm.analyst import analyze_match
from ui.cards import (render_verdict_card, translate_team,
                      parse_card_date, day_label)

LLM_TOP_N = 8
MAX_STAKE_PCT = 0.05


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


def _load_engine(matrix_n, logs, update_loader):
    import pickle, gzip, os
    from config import DISK_CACHE_DIR
    p = os.path.join(DISK_CACHE_DIR, f"engine_{matrix_n}.pkl.gz")
    if os.path.exists(p):
        try:
            with open(p, "rb") as f:
                data = pickle.loads(gzip.decompress(f.read()))
            cached_engine = data.get("engine")
            if (data.get("fp") == APP_VERSION
                    and hasattr(cached_engine, "match_stats")):
                logs.append("Engine из кэша (включая историю угловых/карточек)")
                return cached_engine
        except Exception:
            pass
    engine = _train_engine(matrix_n, logs, update_loader)
    try:
        with open(p, "wb") as f:
            f.write(gzip.compress(pickle.dumps(
                {"fp": APP_VERSION, "engine": engine})))
    except Exception:
        pass
    return engine


def _render_manual_odds_block(c, D, kelly_frac, min_prob=0.40, min_edge=0.03, min_ev=0.03):
    """Понятный ввод букмекерского коэффициента для выбранного рынка."""
    v = c.get("verdict", {})
    stat_preds = c.get("historical_stats") or {}
    from model.historical_match_stats import poisson_over_probability
    market_options = {
        "Победа хозяев (П1)": ("П1", float(c.get("p1") or 0)),
        "Ничья (X)": ("X", float(c.get("px") or 0)),
        "Победа гостей (П2)": ("П2", float(c.get("p2") or 0)),
        "Тотал голов больше 2.5 (ТБ 2.5)": ("ТБ 2.5", float(c.get("over") or 0)),
        "Тотал голов меньше 2.5 (ТМ 2.5)": ("ТМ 2.5", 1.0 - float(c.get("over") or 0)),
        "Обе команды забьют — да": ("BTTS Да", float(c.get("btts") or 0)),
        "Обе команды забьют — нет": ("BTTS Нет", 1.0 - float(c.get("btts") or 0)),
    }
    for stat_market, lines in (("corners", (8.5, 9.5, 10.5)), ("cards", (3.5, 4.5, 5.5))):
        pred = stat_preds.get(stat_market) or {}
        if pred.get("total") is not None and pred.get("history_matches", 0) >= 30:
            for line in lines:
                over_p = poisson_over_probability(pred["total"], line)
                if over_p is not None:
                    prefix = "Угловые" if stat_market == "corners" else "Карточки"
                    market_options[f"{prefix}: больше {line}"] = (f"{'Угл' if stat_market == 'corners' else 'Карты'} ТБ {line}", over_p)
                    market_options[f"{prefix}: меньше {line}"] = (f"{'Угл' if stat_market == 'corners' else 'Карты'} ТМ {line}", 1.0 - over_p)

    default_label = next(
        (name for name, item in market_options.items()
         if item[0].casefold() == str(v.get("pick") or "").casefold()),
        "Победа хозяев (П1)",
    )
    labels = list(market_options)
    selected_label = st.selectbox(
        "1. Выберите исход, на который смотрите коэффициент",
        labels,
        index=labels.index(default_label),
        key=f"market_{c.get('match','').replace(' ','_')}",
    )
    pick, model_prob = market_options[selected_label]
    market_is_validated = (market_type(pick) == "1X2" or
                           (market_type(pick) == "OU" and bool(c.get("total_market_validation"))))
    fair_odd = (1.0 / model_prob) if model_prob > 0 else None

    st.caption(
        f"Вероятность модели: {model_prob * 100:.1f}% · "
        + (f"справочный кэф модели: {fair_odd:.2f}" if fair_odd else "справочный кэф модели: нет расчёта")
    )
    st.info(
        "2. Введите десятичный коэффициент именно из линии букмекера для выбранного исхода. "
        "Например, если у букмекера рядом с «ТБ 2.5» написано 1.85 — вводите 1.85. "
        "Не вводите справочный кэф модели и не вводите сумму ставки."
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
    manual_odd = col_input.number_input(
        "3. Коэффициент букмекера",
        min_value=1.01,
        max_value=100.0,
        value=1.01,
        step=0.01,
        key=input_key,
        help="Введите десятичный кэф, например 1.85 или 2.10. Значение 1.01 означает, что кэф ещё не введён.",
    )

    if manual_odd > 1.01 and 0.0 < model_prob < 1.0:
        market_implied = 1.0 / manual_odd
        prob = max(1e-6, min(1.0 - 1e-6,
                             market_implied + 0.40 * (model_prob - market_implied)))
        ev = prob * manual_odd - 1
        edge = prob - market_implied
        stake = kelly(prob, manual_odd, D["bank"], kelly_frac, cap=0.02)

        col_metrics.markdown(
            f"**Расчёт для выбранного исхода**  \n"
            f"- Вероятность после смешивания: **{prob * 100:.1f}%**  \n"
            f"- Вероятность из кэфа (без снятия маржи): **{market_implied * 100:.1f}%**  \n"
            f"- Edge: **{edge * 100:+.1f}%**  \n"
            f"- EV: **{ev * 100:+.1f}%**"
        )
        if market_is_validated:
            col_metrics.caption(f"Предварительный размер по дробному Kelly: {stake:.2f} у.е. (лимит 2% банка).")

        passed = (
            market_is_validated
            and prob >= min_prob
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
                    "status": "pending", "strat": "manual",
                    "odds_source": "manual",
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
                usage.set_local_data(D2)
                st.toast("✅ Условия выполнены — автоматически добавлено в виртуальный портфель.", icon="✅")
                st.rerun()
        else:
            col_btn.caption("Автодобавление не сработало: не пройдены пороги вероятности, Edge, EV или Kelly.")
    else:
        col_metrics.caption("Введите коэффициент выше 1.01, чтобы увидеть расчёт.")

def render(min_prob, kelly_frac, matrix_n, min_edge=0.03):
    D = st.session_state.data
    c1, c2 = st.columns([4, 1])
    days = c1.slider("Период матчей: 1 = только сегодня", 1, 14, 1)
    scan = c2.button("⚡ СКАН", type="primary",
                     disabled=st.session_state.get("_scan_in_progress", False))
    auto_add = st.checkbox(
        "Автоматически добавлять в виртуальный портфель подходящие 1X2 и подтверждённые тоталы",
        value=True,
        key="scanner_auto_add_qualified",
        help="1X2 проходит обычные value-фильтры; ТБ/ТМ 2.5 добавляются только при ≥500 исторических котировок и статистически подтверждённом преимуществе модели над рынком. Только виртуальный учёт, не размещение ставки у букмекера.",
    )

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

        with st.expander("🏆 Лучшие варианты дня · все лиги", expanded=True):
            st.caption(
                "Единый рейтинг матчей по вероятности, реальному коэффициенту, "
                "EV и преимуществу над рынком."
            )
            if not qualified:
                st.info("Пока нет подтверждённых value-вариантов.")
            else:
                for rank, card in enumerate(qualified[:5], start=1):
                    st.markdown(f"### #{rank} · {card.get('league', 'Лига')}")
                    st.markdown(render_verdict_card(card, min_prob),
                                unsafe_allow_html=True)

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

        if group_by_day:
            grouped = {}
            for c in cards_view:
                dt = parse_card_date(c)
                label = day_label(dt)
                grouped.setdefault(label, []).append(c)

            sorted_groups = sorted(
                grouped.items(),
                key=lambda kv: parse_card_date(kv[1][0]))

            for label, items in sorted_groups:
                st.markdown(
                    "<div style='margin-top:20px;margin-bottom:12px;"
                    "padding:8px 16px;background:rgba(34,211,238,.08);"
                    "border-left:3px solid #22d3ee;border-radius:8px;"
                    "color:#a5f3fc;font-weight:800;font-size:1.05rem;"
                    "letter-spacing:1px;'>" + label + " · "
                    "<span style='color:#8b93a7;font-weight:500;'>"
                    + str(len(items)) + " матч.</span></div>",
                    unsafe_allow_html=True)
                for c in items:
                    st.markdown(render_verdict_card(c, min_prob),
                                unsafe_allow_html=True)
                    _render_manual_odds_block(c, D, kelly_frac, min_prob, min_edge)
        else:
            for c in cards_view:
                st.markdown(render_verdict_card(c, min_prob),
                            unsafe_allow_html=True)
                _render_manual_odds_block(c, D, kelly_frac, min_prob, min_edge)

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
        if token:
            logs.append(f"🔑 football-data.org: {usage.fdorg_remaining()} запросов")
        else:
            logs.append("ℹ️ football-data.org token не задан: использую доступные публичные источники TSDB/ESPN.")

        update_loader("📡 Сбор матчей из всех доступных источников...", 0.05, logs)
        source_rows = {}
        for source_name, source_fn in (
            ("football-data.org", lambda: fdorg_matches(days, token, logs) if token else []),
            ("TheSportsDB", lambda: tsdb_matches(days, logs)),
            ("ESPN", lambda: espn_matches(days, logs)),
        ):
            try:
                source_rows[source_name] = source_fn() or []
                logs.append(f"📡 {source_name}: {len(source_rows[source_name])} матчей")
            except Exception as exc:
                source_rows[source_name] = []
                logs.append(f"⚠️ {source_name}: источник недоступен ({type(exc).__name__})")
        rows = _safe_filter(merge_fixture_sources(
            source_rows.get("football-data.org", []),
            source_rows.get("TheSportsDB", []),
            source_rows.get("ESPN", []),
        ))
        logs.append(f"📡 Всего после объединения: {len(rows)} уникальных матчей")

        update_loader("🧠 Загрузка модели...", 0.20, logs)
        engine = _load_engine(matrix_n, logs, update_loader)

        update_loader("🧠 Анализ матчей...", 0.80, logs)
        cards = []
        matches_with_best = 0
        xg_used_count = 0
        odds_key = _runtime_secret("odds_api_key_input", "ODDS_API_KEY", "odds_api_key")

        funnel_counts = {"model": 0, "market_req": 0, "market_got": 0,
                          "bet": 0, "watch": 0, "skip": 0,
                          "model_only": 0, "estimated": 0, "real": 0}

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
            total_market_odds = {}
            total_odds_source = "no_market"
            api_market_data = None
            odds_source = "no_market"

            has_model_candidate = any(
                float(item.get("prob") or 0) >= float(min_prob)
                for item in rows_
            )

            if has_model_candidate or (odds_key and getattr(engine, "total_market_validation", False)):
                try:
                    candidates = []

                    # Fetch current bookmaker prices for totals separately from the 1X2 source.
                    if odds_key and DIV_TO_ODDS.get(lg):
                        try:
                            api_market_data = odds_api_fixture(
                                DIV_TO_ODDS.get(lg), h_en, a_en, odds_key, logs
                            )
                            if isinstance(api_market_data, dict):
                                for total_pick in ("ТБ 2.5", "ТМ 2.5"):
                                    try:
                                        total_odd = float(api_market_data.get(total_pick))
                                        if 1.01 < total_odd <= 100:
                                            total_market_odds[total_pick] = total_odd
                                    except (TypeError, ValueError):
                                        continue
                                if total_market_odds:
                                    total_odds_source = "the_odds_api"
                        except Exception as exc:
                            logs.append(f"⚠️ Odds API totals: {type(exc).__name__}")

                    # 1. Сначала пробуем Prognozist через уже подключённый
                    # адаптер data.sources. Не импортируем отсутствующий модуль.
                    # Если имена команд не совпали на английском, пробуем русские.
                    try:
                        prog_odds = prognozist_marathon_odds(h_en, a_en)
                        if not prog_odds:
                            prog_odds = prognozist_marathon_odds(h_ru, a_ru)
                        if isinstance(prog_odds, dict):
                            valid_prog = {}
                            for source_key, target_key in (("П1", "П1"), ("X", "X"), ("П2", "П2"),
                                                           ("1", "П1"), ("2", "П2")):
                                if target_key in valid_prog or source_key not in prog_odds:
                                    continue
                                try:
                                    value = float(prog_odds[source_key])
                                except (TypeError, ValueError):
                                    continue
                                if 1.01 < value < 100:
                                    valid_prog[target_key] = value
                            if all(key in valid_prog for key in ("П1", "X", "П2")):
                                candidates.append(("prognozist", valid_prog))
                                logs.append(f"✅ Prognozist нашёл линию: {h_ru} — {a_ru}")
                            else:
                                logs.append("ℹ️ Prognozist: линия неполная или коэффициенты некорректны")
                        else:
                            logs.append("ℹ️ Prognozist: матч не найден, проверяю другие источники")
                    except Exception as exc:
                        logs.append(f"⚠️ Prognozist error: {type(exc).__name__}")

                    # 2. Если Prognozist не дал — пробуем Marathon
                    if not candidates:
                        try:
                            candidates.append(("marathon_category",
                                               marathon_odds(h_en, a_en, r.get("Date", ""), lg)))
                        except Exception:
                            pass
                    
                    # 3. Если всё ещё нет — TheRundown
                    if not candidates:
                        rundown_key = _runtime_secret(
                            "therundown_api_key_input", "THERUNDOWN_API_KEY",
                            "therundown_api_key", "rundown_key")
                        if rundown_key:
                            rundown = therundown_odds_fixture(
                                h_en, a_en, d.strftime("%Y-%m-%d"), lg, rundown_key, logs
                            )
                            if rundown:
                                candidates.append(("therundown", rundown))

                    if not candidates and isinstance(api_market_data, dict):
                        candidates.append(("the_odds_api", api_market_data))

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
                    min_prob, min_edge=min_edge, min_ev=0.03,
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
                "xg_used": bool(P.get("xg_used")),
                "total_market_validation": bool(getattr(engine, "total_market_validation", False)),
                "total_market_validation_report": getattr(engine, "total_market_validation_report", {}),
                "p1": P["p1"], "px": P["x"], "p2": P["p2"],
                "over": P["over"], "btts": P["btts"],
                "odds_source": odds_source,
                "total_market_odds": total_market_odds,
                "total_odds_source": total_odds_source,
                "home_badge": r.get("home_badge") or "",
                "away_badge": r.get("away_badge") or "",
                "league_badge": r.get("league_badge") or "",
                "context": {},
            })

        logs.append(f"📈 xG: использован в {xg_used_count} из {funnel_counts['model']} прогнозов")
        logs.append(f"🎯 BET: {matches_with_best}")

        # ============ LLM ============
        llm_key = _runtime_secret("llm_api_key_input", "LLM_API_KEY", "llm_api_key")
        llm_prov = D.get("meta", {}).get("llm_provider",
                                          "Groq (бесплатно, быстро)")
        llm_model = D.get("meta", {}).get("llm_model", "")

        action_cards = [c for c in cards if c.get("best") is not None]
        action_cards.sort(key=lambda c: -(c.get("verdict", {}).get("prob") or 0))
        llm_done = 0

        logs.append(
            f"🤖 LLM подготовка: key={'есть' if llm_key else 'НЕТ'} | "
            f"provider={llm_prov} | "
            f"action_cards={len(action_cards)} | "
            f"remaining={usage.llm_remaining()}")

        if llm_key and usage.llm_remaining() > 0 and action_cards:
            for i_, card in enumerate(action_cards[:LLM_TOP_N]):
                if usage.llm_remaining() <= 0:
                    logs.append("🤖 LLM: лимит")
                    break
                if i_ > 0:
                    time.sleep(2)
                update_loader(
                    f"🤖 ИИ-анализ [{i_+1}/{min(LLM_TOP_N, len(action_cards))}]",
                    0.90 + 0.08 * (i_+1) / LLM_TOP_N, logs)
                v_ = card.get("verdict", {})
                ctx_llm = {
                    "api_key": llm_key, "provider": llm_prov, "model": llm_model,
                    "home": card.get("match_ru", "").split(" — ")[0],
                    "away": card.get("match_ru", "").split(" — ")[-1],
                    "league": card.get("league", ""),
                    "date": card.get("date", ""),
                    "lam_h": card.get("lam_h", 0), "lam_a": card.get("lam_a", 0),
                    "p1": card.get("p1", 0), "px": card.get("px", 0),
                    "p2": card.get("p2", 0), "over": card.get("over", 0),
                    "btts": card.get("btts", 0),
                    "fh": card.get("fh", "—"), "fa": card.get("fa", "—"),
                    "pick": v_.get("label", ""), "prob": v_.get("prob", 0),
                    "confidence": v_.get("confidence", ""), "ev": v_.get("ev", 0),
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
            logs.append("🤖 LLM: нет action_cards (best=None у всех)")

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
                    if prob < float(min_prob) or edge_value < float(min_edge) or ev_value < 0.03 or stake <= 0 or stake > available_bank:
                        continue
                    new_bets.append({
                        "match": card.get("match", ""), "match_ru": card.get("match_ru", card.get("match", "")),
                        "div": card.get("div", ""), "league": card.get("league", ""),
                        "market": "OU", "pick": pick, "odds": odd, "stake": stake,
                        "prob": prob, "model_prob": model_p, "market_implied_raw": implied,
                        "probability_method": "market_blend_0.4_model_delta",
                        "status": "pending", "strat": "scanner_auto_totals",
                        "odds_source": card.get("total_odds_source", "the_odds_api"), "mode": "real",
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
