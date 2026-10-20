"""ui/tabs/scanner.py — Сканер с правильной логикой best + LLM."""
from __future__ import annotations
import time
from datetime import datetime, timedelta, timezone

import streamlit as st

from config import APP_VERSION, DIV_NAMES, DIV_TO_ODDS, DIV_TO_TSDB, DB_FILE
from storage import usage
from storage import sqlite_store as db
from data.sources import (load_seasonal, season_str,
                          fdorg_matches, fdorg_match_result, odds_api_fixture, parse_date,
                          tsdb_matches, espn_matches)
from data.fixture_merge import merge_fixture_sources
from model.engine import Engine
from betting.verdict import build_verdict
from betting.best_picks import best_value_market
from betting.kelly import kelly, market_type
from llm.analyst import analyze_match
from ui.cards import (render_verdict_card, translate_team,
                      parse_card_date, day_label)

LLM_TOP_N = 8
MIN_STAKE_PCT = 0.005   # 0.5% от банка — минимальный стейк
MAX_STAKE_PCT = 0.05    # 5% от банка — максимальный

def _forecast_learning_service():
    """Open persistent audit + shadow calibration; never block the live scanner."""
    try:
        from model.forecast_ledger import ForecastLedger
        from model.learning_service import ForecastLearningService
        ledger = ForecastLedger(str(DB_FILE) + ".forecast-ledger.sqlite")
        # Guarded online calibration: requires 100 settled results and eligible bins.
        return ForecastLearningService(ledger, enabled=True, min_history=100)
    except Exception as exc:
        _print_log(f"forecast learning service unavailable: {type(exc).__name__}: {exc}")
        return None

def _reconcile_forecast_results(ledger, token, logs):
    """Settle old forecasts only when a score provider confirms regular-time FT."""
    if ledger is None:
        return 0
    settled = 0
    try:
        from data.scores import find_match_score
        now = datetime.now(timezone.utc)
        pending = [row for row in ledger.rows() if row.get("result") is None]
        for row in pending:
            fixture_id = row.get("fixture_id")
            kickoff = row.get("kickoff_at")
            if not kickoff:
                continue
            try:
                kickoff_dt = datetime.fromisoformat(str(kickoff).replace("Z", "+00:00"))
                if kickoff_dt.tzinfo is None:
                    kickoff_dt = kickoff_dt.replace(tzinfo=timezone.utc)
                if kickoff_dt > now - timedelta(hours=2, minutes=15):
                    continue
                result, source = None, None
                if token and fixture_id and str(fixture_id).isdigit():
                    result = fdorg_match_result(str(fixture_id), token)
                    if result and result.get("status") == "FT":
                        source = "football-data.org:verified-FT"
                    else:
                        result = None
                if result is None:
                    result = find_match_score(
                        str(row.get("home_team") or ""),
                        str(row.get("away_team") or ""),
                        kickoff_dt.date().isoformat(),
                    )
                    if not result or result.get("status") != "FT":
                        continue
                    source = "score-provider:" + str(result.get("source") or "verified-FT")
                hg, ag = int(result["home"]), int(result["away"])
                if hg < 0 or ag < 0:
                    continue
                outcome = "H" if hg > ag else "A" if hg < ag else "D"
                ledger.settle(
                    row["forecast_key"], outcome, home_score=hg, away_score=ag,
                    settled_at=datetime.now(timezone.utc),
                    closing_odds_source=source or "verified-FT",
                )
                settled += 1
            except Exception as exc:
                logs.append(f"Журнал: не удалось закрыть {fixture_id or row.get('forecast_key')}: {type(exc).__name__}")
        if settled:
            logs.append(f"📚 Журнал: подтверждено итогов FT — {settled}")
    except Exception as exc:
        logs.append(f"📚 Журнал результатов недоступен: {type(exc).__name__}")
    return settled

def _record_raw_forecast(service, known_keys, row, home, away, league,
                         probabilities, logs, *, variant="h2h"):
    """Append one immutable A/B forecast variant; never changes live picks."""
    if service is None:
        return
    fixture_id = row.get("fixture_id")
    kickoff = row.get("kickoff_utc")
    if fixture_id in (None, "") or not kickoff:
        return
    if variant not in ("h2h", "no-h2h"):
        raise ValueError("unknown forecast audit variant")
    model_version = f"{APP_VERSION}-{variant}-v1"
    key = f"fdorg:{fixture_id}:{model_version}"
    try:
        if key in known_keys:
            return
        prices_raw = row.get("odds") or {}
        prices = {}
        for outcome, name in (("H", "П1"), ("D", "X"), ("A", "П2")):
            try:
                odd = float(prices_raw.get(name))
                if 1 < odd < float("inf"):
                    prices[outcome] = odd
            except (TypeError, ValueError):
                pass
        decision = service.forecast(
            forecast_key=key, fixture_id=str(fixture_id),
            home_team=home, away_team=away, model_version=model_version,
            probabilities=probabilities,
            forecast_at=datetime.now(timezone.utc),
            kickoff_at=kickoff, league=str(league or ""),
            entry_odds=prices or None,
            odds_source=row.get("odds_source") or None,
        )
        known_keys.add(key)
        if decision.get("history_n", 0) >= 100:
            logs.append(
                f"🧠 Калибровка в теневом режиме: история={decision['history_n']}, "
                f"коррекция={'готова к оценке' if decision.get('learning_report') else 'нет данных'}"
            )
    except Exception as exc:
        logs.append(f"📚 Журнал: прогноз не записан ({type(exc).__name__})")

def _paired_ab_metrics(rows_a, rows_b):
    """Score A/B variants only on fixture IDs settled in both arms."""
    a = {row["fixture_id"]: row for row in rows_a
         if row.get("result") is not None and row.get("fixture_id")}
    b = {row["fixture_id"]: row for row in rows_b
         if row.get("result") is not None and row.get("fixture_id")}
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"n": 0, "a_brier": None, "b_brier": None,
                "a_log_loss": None, "b_log_loss": None}
    def scores(row):
        probs, actual = row["probabilities"], row["result"]
        brier = sum((probs[k] - float(k == actual)) ** 2 for k in ("H", "D", "A"))
        loss = -math.log(max(probs[actual], 1e-15))
        return brier, loss
    import math
    sa = [scores(a[key]) for key in ids]
    sb = [scores(b[key]) for key in ids]
    n = len(ids)
    return {
        "n": n,
        "a_brier": sum(x[0] for x in sa) / n,
        "a_log_loss": sum(x[1] for x in sa) / n,
        "b_brier": sum(x[0] for x in sb) / n,
        "b_log_loss": sum(x[1] for x in sb) / n,
    }

def _render_learning_status():
    """Show audit coverage and scoring metrics without changing picks."""
    try:
        service = _forecast_learning_service()
        if service is None:
            st.info("Журнал обучения пока недоступен.")
            return
        with st.expander("🧠 Обучение на ошибках и аудит прогнозов", expanded=False):
            all_rows = service.ledger.rows()
            metrics = service.ledger.metrics()
            total = len(all_rows)
            settled = metrics.get("n", 0)
            a, b, c, d = st.columns(4)
            a.metric("Прогнозов в журнале", total)
            b.metric("Матчей с итогом", settled)
            brier, loss, accuracy = metrics.get("brier"), metrics.get("log_loss"), metrics.get("accuracy")
            c.metric("Brier (ниже лучше)", f"{brier:.4f}" if brier is not None else "—")
            d.metric("Log loss (ниже лучше)", f"{loss:.4f}" if loss is not None else "—")
            st.caption(
                f"Точность top-1: {accuracy:.1%}" if accuracy is not None
                else "Точность top-1 появится после подтверждённых результатов."
            )
            h2h_version = f"{APP_VERSION}-h2h-v1"
            no_h2h_version = f"{APP_VERSION}-no-h2h-v1"
            h2h_metrics = service.ledger.metrics(model_version=h2h_version)
            no_h2h_metrics = service.ledger.metrics(model_version=no_h2h_version)
            h2h_rows = service.ledger.rows(model_version=h2h_version)
            no_h2h_rows = service.ledger.rows(model_version=no_h2h_version)
            paired = _paired_ab_metrics(h2h_rows, no_h2h_rows)
            left, right = st.columns(2)
            with left:
                st.markdown("**A · С H2H**")
                st.metric("Прогнозов / результатов",
                          f"{len(h2h_rows)} / {h2h_metrics.get('n', 0)}")
                bm, lm = paired["a_brier"], paired["a_log_loss"]
                st.caption(f"Парный Brier: {bm:.4f}" if bm is not None else "Парный Brier: ждём общие результаты")
                st.caption(f"Парный Log loss: {lm:.4f}" if lm is not None else "Парный Log loss: ждём общие результаты")
            with right:
                st.markdown("**B · Без H2H**")
                st.metric("Прогнозов / результатов",
                          f"{len(no_h2h_rows)} / {no_h2h_metrics.get('n', 0)}")
                bm, lm = paired["b_brier"], paired["b_log_loss"]
                st.caption(f"Парный Brier: {bm:.4f}" if bm is not None else "Парный Brier: ждём общие результаты")
                st.caption(f"Парный Log loss: {lm:.4f}" if lm is not None else "Парный Log loss: ждём общие результаты")
            st.caption(
                f"Парных завершённых прогнозов: {paired['n']}. Метрики рассчитаны "
                "только на одинаковых fixture_id; меньшие Brier и log loss лучше."
            )
            st.warning(
                "Калибровка работает в теневом режиме: текущие вероятности, ставки "
                "и размеры ставок не изменяются. Это защищает от включения обучения "
                "до проверки на отдельной хронологической выборке."
            )
            if settled < 100:
                st.info(f"Для первичного порога нужно больше истории: сейчас {settled}/100 завершённых прогнозов.")
            else:
                st.caption(
                    "Порог истории достигнут. Это не доказательство улучшения: "
                    "сначала сравни Brier и log loss на отдельном holdout."
                )
    except Exception as exc:
        st.info(f"Диагностика обучения временно недоступна: {type(exc).__name__}")

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
    prev_year = cur_year - 1
    priority_divs = ["E0", "SP1", "I1", "D1", "F1", "E1", "SP2", "I2",
                     "D2", "F2", "N1", "B1", "P1", "T1", "R1"]
    supported_divs = list(DIV_TO_TSDB.keys()) + ["E2", "E3", "D3"]
    train_divs = list(dict.fromkeys(priority_divs + supported_divs))
    engine = Engine(matrix_n=matrix_n)
    dp, dc = {}, {}
    for i, dv in enumerate(train_divs):
        pct = 0.1 + (i + 1) / len(train_divs) * 0.4
        update_loader(
            f"История [{i+1}/{len(train_divs)}] — {DIV_NAMES.get(dv, dv)}",
            pct, logs)
        dp[dv] = load_seasonal(dv, season_str(prev_year))
        dc[dv] = load_seasonal(dv, season_str(cur_year))
        logs.append(f"OK {DIV_NAMES.get(dv, dv)}: {len(dp[dv])+len(dc[dv])}")
    total = sum(len(dp.get(dv, [])) + len(dc.get(dv, [])) for dv in train_divs)
    processed = 0
    trained = 0
    for dv in train_divs:
        for src in (dp.get(dv, []), dc.get(dv, [])):
            for r in src:
                try:
                    hg = float(r.get("FTHG", 0))
                    ag = float(r.get("FTAG", 0))
                    engine.learn_step(
                        r["HomeTeam"], r["AwayTeam"], hg, ag, r,
                        lg=dv, match_num=processed, total=total,
                        match_date=parse_date(r.get("Date", "")))
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
            if data.get("fp") == APP_VERSION:
                logs.append("Engine из кэша")
                return data.get("engine")
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

def render(min_prob, kelly_frac, matrix_n):
    D = st.session_state.data
    c1, c2 = st.columns([4, 1])
    days = c1.slider("Горизонт, дней", 1, 14, 7)
    scan = c2.button("⚡ СКАН", type="primary",
                     disabled=st.session_state.get("_scan_in_progress", False))

    if not scan:
        _render_learning_status()
        fn = D.get("funnel")
        if fn:
            st.success(
                f"🧠 Обучено {fn.get('trained', 0)} · "
                f"📡 Источников {fn.get('src', 0)} · "
                f"🎯 Найдено {fn.get('found', 0)} · "
                f"➕ В портфель {fn.get('added', 0)} · "
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
        # Daily view and picks use only the latest scan; never fabricate a value pick.
        today_iso = datetime.now().date().isoformat()
        todays_cards = [c for c in all_cards if c.get("date_iso") == today_iso]
        with st.expander(f"📅 Матчи дня · {len(todays_cards)}", expanded=True):
            if not todays_cards:
                st.info("Матчи на сегодня не найдены в последнем скане. Нажми ⚡ СКАН, чтобы обновить источники.")
            else:
                todays_qualified = [c for c in todays_cards if c in qualified]
                st.caption(f"Всего матчей: {len(todays_cards)} · value с реальными коэффициентами: {len(todays_qualified)}")
                for card in sorted(todays_cards, key=lambda c: str(c.get("time") or "99:99")):
                    v = card.get("verdict") or {}
                    odds_note = f" · кэф {float(v.get('odd')):.2f}" if v.get("odd") else ""
                    st.markdown(f"**{card.get('time') or 'Время н/д'} · {card.get('match_ru') or card.get('match')}** — {card.get('league') or 'Лига'}{odds_note}")
                    if card in todays_qualified:
                        st.markdown(render_verdict_card(card, min_prob), unsafe_allow_html=True)
        with st.expander("🎯 Ставка дня / экспресс дня", expanded=True):
            if not qualified:
                st.info("Ставка дня не назначается: нет вариантов, прошедших фильтры реального коэффициента, EV и value.")
            else:
                daily_ranked = sorted(qualified, key=lambda c: (
                    -float((c.get("verdict") or {}).get("ev") or -1),
                    -float((c.get("verdict") or {}).get("edge") or -1),
                    -float((c.get("verdict") or {}).get("prob") or 0),
                ))
                st.markdown("**Ставка дня — лучший одиночный value-вариант**")
                st.markdown(render_verdict_card(daily_ranked[0], min_prob), unsafe_allow_html=True)
                express = daily_ranked[:3]
                if len(express) >= 2:
                    st.markdown("**Экспресс дня — до 3 вариантов, прошедших фильтры**")
                    st.warning("Это подборка кандидатов, не гарантия выигрыша; экспресс несёт повышенный риск.")
                    for card in express:
                        v = card.get("verdict") or {}
                        st.markdown(f"- {card.get('match_ru') or card.get('match')} — {v.get('label') or v.get('pick') or 'рынок'}; P {float(v.get('prob') or 0):.0%}; EV {float(v.get('ev') or 0):+.1%}")
                else:
                    st.caption("Для экспресса нужно минимум два варианта, прошедших фильтры.")
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
        show_hidden = c2.checkbox("Показывать слабые", False,
                                   key="scanner_show_hidden")
        group_by_day = c3.checkbox("Группировать по дням", True,
                                    key="scanner_group_by_day")

        if show_hidden:
            cards_view = all_cards
        else:
            cards_view = [c for c in all_cards
                          if (c.get("verdict") or {}).get("is_action")]

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
        else:
            for c in cards_view:
                st.markdown(render_verdict_card(c, min_prob),
                            unsafe_allow_html=True)

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

        token = D.get("meta", {}).get("fdorg_token", "").strip()
        if token:
            logs.append(f"🔑 football-data.org: {usage.fdorg_remaining()} запросов")
        else:
            logs.append("ℹ️ football-data.org token не задан: использую доступные публичные источники TSDB/ESPN.")

        learning_service = _forecast_learning_service()
        audit_ledger = learning_service.ledger if learning_service is not None else None
        audit_keys = set()
        if audit_ledger is not None:
            _reconcile_forecast_results(audit_ledger, token, logs)
            try:
                audit_keys = {item.get("forecast_key") for item in audit_ledger.rows()}
            except Exception:
                audit_keys = set()

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
        logs.append(f"📡 Всего после объединения источников: {len(rows)} уникальных матчей")

        update_loader("🧠 Загрузка модели...", 0.20, logs)
        engine = _load_engine(matrix_n, logs, update_loader)

        update_loader("🧠 Анализ матчей...", 0.80, logs)
        cards = []
        matches_with_best = 0
        xg_used_count = 0
        odds_key = D.get("meta", {}).get("odds_api_key", "")

        funnel_counts = {"model": 0, "market_req": 0, "market_got": 0,
                          "bet": 0, "watch": 0, "skip": 0,
                          "model_only": 0, "estimated": 0, "real": 0}

        for r in rows:
            d = parse_date(r.get("Date", ""))
            if not d or not (today <= d <= today + timedelta(days=days)):
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

            # Live A/B audit (shadow only, не влияет на picks)
            try:
                _record_raw_forecast(
                    learning_service, audit_keys, r, h_en, a_en, lg,
                    {"H": float(P["p1"]), "D": float(P["x"]), "A": float(P["p2"])},
                    logs, variant="h2h",
                )
                P_no_h2h = engine.predict(
                    h_en, a_en, lg, match_date=d,
                    cup=lg in ("C1", "EL", "EC"), apply_h2h=False,
                )
                _record_raw_forecast(
                    learning_service, audit_keys, r, h_en, a_en, lg,
                    {"H": float(P_no_h2h["p1"]), "D": float(P_no_h2h["x"]),
                     "A": float(P_no_h2h["p2"])},
                    logs, variant="no-h2h",
                )
            except Exception as exc:
                logs.append(f"📚 Журнал A/B: ошибка записи ({type(exc).__name__})")

            fh = engine.form_str(h_en)
            fa = engine.form_str(a_en)
            verdict, rows_, _ = build_verdict(
                P, min_prob, D["bank"], kelly_frac,
                h_ru, a_ru, fh, fa, P.get("h2h_n", 0))

            best = None
            odds_source = "model_only"

            # Пробуем взять кэфы: сначала из источника, потом из API
            market_odds = dict(r.get("odds") or {})
            if market_odds:
                odds_source = r.get("odds_source") or "fixture_source"

            sport_key = DIV_TO_ODDS.get(lg)
            has_model_candidate = any(
                float(item.get("prob") or 0) >= float(min_prob)
                for item in rows_
            )
            if (has_model_candidate and sport_key and odds_key
                    and usage.odds_remaining() > 0
                    and funnel_counts["market_req"] < 15):
                funnel_counts["market_req"] += 1
                api_odds = odds_api_fixture(sport_key, h_en, a_en, odds_key)
                if api_odds:
                    market_odds.update(api_odds)
                    odds_source = "market"
                    funnel_counts["market_got"] += 1

            # Если есть реальные кэфы — используем best_value_market
            if market_odds:
                chosen_verdict, best = best_value_market(
                    verdict, rows_, market_odds, D["bank"], kelly_frac,
                    min_prob, min_edge=0.03, min_ev=0.03,
                )
                if best is not None:
                    verdict = chosen_verdict
                    funnel_counts["real"] += 1
                else:
                    funnel_counts["model_only"] += 1
            else:
                funnel_counts["model_only"] += 1

            # Не подменяем рыночные котировки математической оценкой.
            # Без подтверждённого коэффициента букмекера матч остаётся в аналитике,
            # но не превращается в ставку с фиктивной ценой.
            if best is None:
                verdict["real_odds"] = False
                verdict["odds_source"] = "unavailable"
                verdict["price_warning"] = "Нет подтверждённого коэффициента букмекера"

            # Отладка
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

            # Контекст
            context = {}
            try:
                from context_football import analyze_match_context
                context = analyze_match_context(
                    engine,
                    home_en=h_en, away_en=a_en,
                    home_ru=h_ru, away_ru=a_ru,
                    match_date=d,
                    referee=r.get("referee", "") or ""
                )
            except Exception:
                context = {}

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
                "date_iso": d.strftime("%Y-%m-%d"),
                "time": r.get("Time") or "",
                "lam_h": P["lams"][0], "lam_a": P["lams"][1],
                "xg_used": bool(P.get("xg_used")),
                "xg_weight": float(P.get("xg_weight") or 0.0),
                "xg_home_samples": int(P.get("xg_home_samples") or 0),
                "xg_away_samples": int(P.get("xg_away_samples") or 0),
                "p1": P["p1"], "px": P["x"], "p2": P["p2"],
                "over": P["over"], "btts": P["btts"],
                "odds_source": odds_source,
                "home_badge": r.get("home_badge") or "",
                "away_badge": r.get("away_badge") or "",
                "league_badge": r.get("league_badge") or "",
                "context": context,
            })

        logs.append(f"📈 xG: использован в {xg_used_count} из {funnel_counts['model']} прогнозов")
        logs.append(f"🎯 BET: {matches_with_best}")

        # ============ LLM ============
        llm_key = D.get("meta", {}).get("llm_api_key", "")
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

        _print_log(f"LLM: {len(action_cards)} cards, "
                   f"key={'ok' if llm_key else 'none'}")

        if llm_key and usage.llm_remaining() > 0:
            for i_, card in enumerate(action_cards[:LLM_TOP_N]):
                if usage.llm_remaining() <= 0:
                    logs.append("🤖 LLM: лимит")
                    break
                if i_ > 0:
                    time.sleep(2)   # обход rate limit Groq
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

        # ============ СТАВКИ ============
        new_bets = []
        existing = {
            f"{b['match']}|{b['pick']}"
            for b in D["bets"]
            if isinstance(b, dict) and b.get("status") == "pending"
        }
        for c in cards:
            b = c.get("best")
            if not b:
                continue
            mkt, pick, odd, ev, prob, stake = b
            stake = round(min(max(stake, D["bank"] * MIN_STAKE_PCT),
                              D["bank"] * MAX_STAKE_PCT), 2)
            if stake <= 0:
                continue
            bk = f"{c['match']}|{pick}"
            if bk in existing:
                continue
            new_bets.append({
                "match": c["match"], "match_ru": c["match_ru"],
                "div": c["div"], "league": c["league"],
                "market": mkt, "pick": pick, "odds": odd, "stake": stake,
                "prob": prob, "status": "pending", "strat": "value",
                "odds_source": c.get("odds_source", "estimated"),
                "mode": "real" if c.get("odds_source") in ("real", "market", "prognostiz") else "paper",
                "date": datetime.now().strftime("%d.%m.%Y"),
                "date_iso": c.get("date_iso"),
                "match_time": c.get("time", ""),
                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "fixture_id": c.get("fixture_id"), "score": None, "ev": ev,
            })
            existing.add(bk)

        D2 = dict(D)
        D2["cards"] = cards
        D2["report"] = logs
        D2["bets"] = D["bets"] + new_bets
        total_stake = sum(b["stake"] for b in new_bets)
        D2["bank"] = max(0.0, D["bank"] - total_stake)
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
            f"✅ Готово! +{len(new_bets)} ставок · 🤖 {llm_done} LLM",
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
