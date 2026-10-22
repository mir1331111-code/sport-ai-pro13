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
MIN_STAKE_PCT = 0.005
MAX_STAKE_PCT = 0.05


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


def _render_manual_odds_block(c, D, kelly_frac, min_prob=0.40, min_edge=0.03, min_ev=0.03):
    """Блок для ручного ввода кэфа под карточкой."""
    v = c.get("verdict", {})
    pick = v.get("pick", "—")
    label = v.get("label", "—")
    fair_odd = v.get("fair_odd", 1.0)
    prob = v.get("prob", 0.0)
    
    key_base = f"manual_{c.get('match', '')}_{pick}".replace(" ", "_").replace("/", "_")
    input_key = f"odd_{key_base}"
    btn_key = f"btn_{key_base}"
    
    st.markdown(
        "<div style='background:rgba(52,211,153,.1);"
        "border:1px solid rgba(52,211,153,.3);"
        "padding:10px 14px;border-radius:10px;margin-bottom:14px;"
        "display:flex;align-items:center;gap:10px;'>"
        "<span style='color:#34d399;font-weight:bold;font-size:1.1rem;'>✍️</span>"
        "<span style='color:#a5f3fc;font-size:.85rem;'>Ручной ввод кэфа (если нет реального):</span>"
        "</div>", unsafe_allow_html=True
    )
    
    col_input, col_metrics, col_btn = st.columns([2, 2, 1])
    
    manual_odd = col_input.number_input(
        "Кэф", 
        min_value=1.01, max_value=100.0, 
        value=1.01,
        step=0.01,
        key=input_key,
        label_visibility="collapsed",
        help=f"Введи реальный кэф для {label}"
    )
    
    if manual_odd > 1.01:
        ev = prob * manual_odd - 1
        edge = prob - (1.0 / manual_odd)
        stake = kelly(prob, manual_odd, D["bank"], kelly_frac)
        
        col_metrics.markdown(
            f"<div style='font-size:.9rem;line-height:1.5;'>"
            f"EV: <b style='color:{'#34d399' if ev>0 else '#f87171'}'>{ev*100:+.1f}%</b><br>"
            f"Edge: <b>{edge*100:+.1f}%</b><br>"
            f"Kelly: <b style='color:#fbbf24'>{stake:.2f} у.е.</b>"
            f"</div>", unsafe_allow_html=True
        )
        
        passed = (prob >= min_prob and edge >= min_edge and ev >= min_ev and stake > 0)
        if col_btn.button("➕ В портфель", key=btn_key, use_container_width=True, disabled=not passed):
            D2 = dict(D)
            new_bet = {
                "match": c["match"], "match_ru": c["match_ru"],
                "div": c["div"], "league": c["league"],
                "market": market_type(pick), "pick": pick, 
                "odds": manual_odd, "stake": stake,
                "prob": prob, "status": "pending", "strat": "manual",
                "odds_source": "manual",
                "mode": "real",
                "date": datetime.now().strftime("%d.%m.%Y"),
                "date_iso": c.get("date_iso"),
                "match_time": c.get("time", ""),
                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "fixture_id": c.get("fixture_id"), "score": None, "ev": ev,
            }
            existing = f"{c['match']}|{pick}"
            if not any(f"{b['match']}|{b['pick']}" == existing for b in D2["bets"] if b.get("status") == "pending"):
                D2["bets"].append(new_bet)
                D2["bank"] = max(0.0, D2["bank"] - stake)
                st.session_state.data = D2
                usage.set_local_data(D2)
                st.toast("✅ Добавлено в портфель!", icon="✅")
                st.rerun()
            else:
                st.warning("Эта ставка уже есть в портфеле.")
    else:
        col_metrics.caption("Введи кэф > 1.01")


def render(min_prob, kelly_frac, matrix_n):
    D = st.session_state.data
    c1, c2 = st.columns([4, 1])
    days = c1.slider("Горизонт, дней", 1, 14, 7)
    scan = c2.button("⚡ СКАН", type="primary",
                     disabled=st.session_state.get("_scan_in_progress", False))

    if not scan:
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
                    _render_manual_odds_block(c, D, kelly_frac, min_prob)
        else:
            for c in cards_view:
                st.markdown(render_verdict_card(c, min_prob),
                            unsafe_allow_html=True)
                _render_manual_odds_block(c, D, kelly_frac)

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
        odds_key = D.get("meta", {}).get("odds_api_key", "")

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

            fh = engine.form_str(h_en)
            fa = engine.form_str(a_en)
            verdict, rows_, _ = build_verdict(
                P, min_prob, D["bank"], kelly_frac,
                h_ru, a_ru, fh, fa, P.get("h2h_n", 0))

            best = None
            odds_source = "model_only"
            market_odds = {}
            odds_source = "no_market"

            has_model_candidate = any(
                float(item.get("prob") or 0) >= float(min_prob)
                for item in rows_
            )

            if has_model_candidate:
                try:
                    candidates = []
                    
                    # 1. СНАЧАЛА ПРОБУЕМ PROGNOZIST (рабочий парсер)
                    try:
                        from data.prognozist_odds import find_match_odds
                        prog_data = find_match_odds(h_en, a_en, kickoff=d)
                        if prog_data and prog_data.get("odds"):
                            odds_dict = prog_data["odds"]
                            p1 = float(odds_dict["1"]["best"])
                            px = float(odds_dict["X"]["best"])
                            p2 = float(odds_dict["2"]["best"])
                            if all(1.01 < v < 100 for v in (p1, px, p2)):
                                candidates.append(("prognozist", {"П1": p1, "X": px, "П2": p2}))
                                logs.append(f"✅ Prognozist нашёл: {h_ru} — {a_ru}")
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
                        rundown_key = (
                            D.get("meta", {}).get("rundown_key")
                            or D.get("meta", {}).get("therundown_api_key")
                            or os.environ.get("THERUNDOWN_API_KEY", "")
                        )
                        if rundown_key:
                            rundown = therundown_odds_fixture(
                                h_en, a_en, d.strftime("%Y-%m-%d"), lg, rundown_key, logs
                            )
                            if rundown:
                                candidates.append(("therundown", rundown))

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
                    min_prob, min_edge=0.03, min_ev=0.03,
                )
                if best is not None:
                    verdict = chosen_verdict
                    funnel_counts["real"] += 1
                else:
                    funnel_counts["model_only"] += 1
            else:
                funnel_counts["model_only"] += 1

            # ============ FALLBACK: ESTIMATED ============
            if best is None and verdict.get("is_action"):
                est_odd = verdict.get("fair_odd")
                if est_odd and est_odd > 1.01:
                    prob_ = float(verdict["prob"])
                    ev_ = prob_ * est_odd - 1
                    min_stake = round(D["bank"] * MIN_STAKE_PCT, 2)
                    stake_ = max(kelly(prob_, est_odd, D["bank"], kelly_frac), min_stake)
                    pick_ = verdict["pick"]
                    
                    best = (market_type(pick_), pick_, est_odd, ev_, prob_, stake_)
                    verdict["real_odds"] = False
                    verdict["odd"] = est_odd
                    verdict["ev"] = ev_
                    odds_source = "estimated"
                    funnel_counts["estimated"] += 1
                    logs.append(
                        f"📐 ESTIMATED BET: {h_ru} — {a_ru} | {pick_} @ {est_odd:.2f}"
                    )

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
                "date_iso": d.strftime("%Y-%m-%d"),
                "time": r.get("Time") or "",
                "lam_h": P["lams"][0], "lam_a": P["lams"][1],
                "xg_used": bool(P.get("xg_used")),
                "p1": P["p1"], "px": P["x"], "p2": P["p2"],
                "over": P["over"], "btts": P["btts"],
                "odds_source": odds_source,
                "home_badge": r.get("home_badge") or "",
                "away_badge": r.get("away_badge") or "",
                "league_badge": r.get("league_badge") or "",
                "context": {},
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
                "mode": "real" if c.get("odds_source") in {
                    "prognozist_marathon", "marathon_category",
                    "prognozist_match_centre", "therundown", "prognozist"
                } else "paper",
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
