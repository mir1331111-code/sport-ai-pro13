"""ui/tabs/scanner.py — Сканер v13.5: TSDB + fdorg + ESPN + Grok + счётчики воронки."""
from __future__ import annotations

import time
from datetime import datetime, timedelta

import streamlit as st

from config import APP_VERSION, DIV_NAMES, DIV_TO_ODDS
from storage import usage
from storage import sqlite_store as db
from data.sources import (
    load_seasonal,
    season_str,
    fdorg_matches,
    espn_matches,
    tsdb_matches,
    odds_api_fixture,
    grok_web_odds,
    parse_date,
)
from model.engine import Engine
from betting.verdict import (
    build_verdict, refine_with_real_odds, refine_with_manual_odd,
)
from betting.kelly import kelly, market_type
from llm.analyst import analyze_match
from ui.cards import render_verdict_card, translate_team
from betting.ranking import sort_cards, value_score, rank_score
from betting.adaptive import build_profiles as build_adaptive_profiles, priority as adaptive_priority

try:
    from context_football import analyze_match_context
    HAS_CONTEXT = True
except Exception:
    HAS_CONTEXT = False


LLM_TOP_N = 8
GROK_ODDS_TOP_N = 8
ODDS_SCAN_TOP_N = 30


def _safe_filter(rows):
    if not isinstance(rows, list):
        return []
    return [
        r for r in rows
        if isinstance(r, dict)
        and r.get("HomeTeam")
        and r.get("AwayTeam")
    ]


def _decision_reason(decision, edge, ev, kelly_pct, real_odds=True):
    if not real_odds:
        return "Нет реального кэфа — ждём цену букмекера."
    try:
        edge_f = float(edge or 0.0)
    except (TypeError, ValueError):
        edge_f = 0.0
    try:
        ev_f = float(ev or 0.0)
    except (TypeError, ValueError):
        ev_f = 0.0
    try:
        kelly_f = float(kelly_pct or 0.0)
    except (TypeError, ValueError):
        kelly_f = 0.0

    if decision == "BET":
        return "Edge ≥ 3%, EV ≥ 3% и Kelly > 0 — цена подтверждена."
    if decision == "WATCH":
        if kelly_f <= 0:
            return "Kelly ≤ 0 — ждём более высокий кэф."
        if edge_f < 0.03 and ev_f < 0.03:
            return "Edge и EV ниже 3% — нужен более выгодный кэф."
        if edge_f < 0.03:
            return "Edge ниже 3% — нужен более выгодный кэф."
        if ev_f < 0.03:
            return "EV ниже 3% — нужен более выгодный кэф."
        return "Value положительный, но цена пока не проходит полный фильтр."
    if kelly_f <= 0:
        return "Kelly ≤ 0 — ставка не оправдывает риск."
    if edge_f <= 0 and ev_f <= 0:
        return "Edge и EV ≤ 0 — положительного преимущества нет."
    if edge_f <= 0:
        return "Edge ≤ 0 — рыночная цена хуже fair."
    if ev_f <= 0:
        return "EV ≤ 0 — математического преимущества нет."
    return "Цена не проходит строгий фильтр BET."


def _match_start_dt(row):
    d = parse_date(row.get("Date", ""))
    if not d:
        return None
    raw_time = str(row.get("Time", "") or "").strip()
    if not raw_time:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.combine(
                d.date(),
                datetime.strptime(raw_time[:8], fmt).time(),
            )
        except Exception:
            continue
    return None


def _train_engine(matrix_n, logs, update_loader):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cur_year = today.year if today.month >= 7 else today.year - 1
    prev_year = cur_year - 1
    train_divs = [
        "E0", "SP1", "I1", "D1", "F1",
        "E1", "SP2", "I2", "D2", "F2",
        "N1", "B1", "P1", "T1", "R1",
    ]
    engine = Engine(matrix_n=matrix_n)
    dp = {}
    dc = {}
    for i, dv in enumerate(train_divs):
        pct = 0.1 + (i + 1) / len(train_divs) * 0.4
        update_loader(
            f"История [{i + 1}/{len(train_divs)}] — {DIV_NAMES.get(dv, dv)}",
            pct, logs,
        )
        dp[dv] = load_seasonal(dv, season_str(prev_year))
        dc[dv] = load_seasonal(dv, season_str(cur_year))
        logs.append(
            f"OK {DIV_NAMES.get(dv, dv)}: {len(dp[dv]) + len(dc[dv])}"
        )
    total = sum(
        len(dp.get(dv, [])) + len(dc.get(dv, [])) for dv in train_divs
    )
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
                        match_date=parse_date(r.get("Date", "")),
                    )
                    trained += 1
                except Exception:
                    pass
                processed += 1
                if processed % 50 == 0:
                    update_loader(
                        f"Обучение [{processed}/{total}]",
                        0.5 + processed / max(1, total) * 0.3,
                        logs,
                    )
    engine.trained_n = trained
    logs.append(f"Обучено: {trained}")
    return engine


def _load_engine(matrix_n, logs, update_loader):
    import gzip, os, pickle
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
                {"fp": APP_VERSION, "engine": engine}
            )))
    except Exception:
        pass
    return engine


def render(min_prob, kelly_frac, matrix_n):
    D = st.session_state.data

    # Adaptive Rules Engine
    adaptive_profile = {}
    if db.SQLITE_BOOT_OK:
        try:
            adaptive_profile = build_adaptive_profiles(
                db.fetch_decision_snapshots(limit=100000)
            )
        except Exception:
            adaptive_profile = {}

    # ============ СЧЁТЧИКИ ВОРОНКИ ============
    model_candidates = 0        # is_action = True
    market_checked = 0          # сколько раз пробовали запросить кэф
    market_received = 0         # сколько раз получили реальный кэф
    value_rejected = 0          # реальный кэф есть, но value не прошло
    matches_with_best = 0       # best создан (BET)
    quality_real_market = 0     # реальный кэф
    quality_missing_market = 0  # только estimated
    decision_bet = 0
    decision_watch = 0
    decision_skip = 0
    decision_model_only = 0
    risk_rejected = 0
    risk_reasons = []
    women_bet_count = 0
    national_bet_count = 0
    ctx_count = 0
    llm_done = 0

    c1, c2 = st.columns([4, 1])
    days = c1.slider("Горизонт, дней", 1, 14, 7)
    scan = c2.button(
        "⚡ СКАН",
        type="primary",
        disabled=st.session_state.get("_scan_in_progress", False),
    )

    if not scan:
        fn = D.get("funnel")
        if fn:
            st.markdown("#### Последний скан · воронка сигналов")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Модель", fn.get("model_candidates", 0))
            f2.metric("Рынок проверен", fn.get("market_checked", 0))
            f3.metric("Value прошло", fn.get("found", 0))
            f4.metric("В портфель", fn.get("added", 0))
            st.caption(
                f"📡 {fn.get('src', 0)} матчей · "
                f"💰 заморожено {fn.get('frozen', 0):.0f} · "
                f"🛡️ Risk Guard: {fn.get('risk_rejected', 0)} · "
                f"♻️ Дубли: {fn.get('duplicate_rejected', 0)} · "
                f"🤖 LLM {fn.get('llm', 0)} · "
                f"🔍 контекст {fn.get('ctx', 0)}"
            )
        if HAS_CONTEXT:
            st.caption("🟢 Context Engine: подключён")

        # ============ PORTFOLIO RISK ============
        pending_stake = sum(
            float(b.get("stake", 0) or 0)
            for b in D.get("bets", [])
            if isinstance(b, dict) and b.get("status") == "pending"
        )
        current_bank = float(D.get("bank") or 0.0)
        initial_bank = float(
            (D.get("meta") or {}).get("initial_bank")
            or (current_bank + pending_stake)
            or 1.0
        )
        risk_ratio = pending_stake / max(initial_bank, 1.0)
        risk_pct = risk_ratio * 100
        risk_color = "#34d399" if risk_pct <= 10 else ("#fbbf24" if risk_pct <= 20 else "#f87171")
        risk_label = "LOW" if risk_pct <= 10 else ("WATCH" if risk_pct <= 20 else "HIGH")
        rr1, rr2, rr3 = st.columns([2.2, 1, 1])
        with rr1:
            st.markdown(
                f"**BANK RISK · <span style='color:{risk_color}'>{risk_label}</span>**",
                unsafe_allow_html=True,
            )
            st.progress(min(risk_ratio, 1.0), text=f"Заморожено {risk_pct:.1f}% банка")
        with rr2:
            st.metric("Pending exposure", f"{pending_stake:.0f}")
        with rr3:
            st.metric("Свободный банк", f"{current_bank:.0f}")
        st.caption("Риск = сумма незакрытых ставок / начальный банк. Ориентир: до 20%.")

        with st.expander("🔌 Диагностика"):
            for line in D.get("report", []):
                st.text(line)

        all_cards = D.get("cards", [])
        cards_view = sort_cards(
            [c for c in all_cards if isinstance(c, dict)],
            mode="rank",
        )
        shown = 0
        hidden = 0

        # ============ TOP VALUE ============
        real_value_cards = [
            c for c in cards_view
            if c.get("best")
            and (c.get("odds_source") in ("market", "grok_web", "manual", "fdorg"))
            and (c.get("verdict") or {}).get("real_odds", False)
        ]
        real_value_cards.sort(
            key=lambda c: (
                -float(value_score(c) or 0.0),
                -float((c.get("verdict") or {}).get("ev") or 0.0),
                -float((c.get("verdict") or {}).get("edge") or 0.0),
            )
        )
        if real_value_cards:
            st.markdown("### 🏆 TOP VALUE")
            st.caption("Лучшие реальные цены по Value Score · максимум 5.")
            tv_cols = st.columns(min(5, len(real_value_cards)))
            for ti, tc in enumerate(real_value_cards[:5]):
                tv = tc.get("verdict") or {}
                tb = tc.get("best") or ()
                odd = float(tb[2] or 0.0) if len(tb) > 2 else float(tv.get("odd") or 0.0)
                ev = tv.get("ev")
                edge = tv.get("edge")
                stake = float(tb[5] or 0.0) if len(tb) > 5 else float(tv.get("stake") or 0.0)
                fair = float(tv.get("fair_odd") or 0.0)
                vs = float(value_score(tc) or 0.0)
                status = (
                    "🟢" if ev is not None and edge is not None
                    and float(ev) >= 0.03 and float(edge) >= 0.03
                    else "🟡" if ev is not None and edge is not None
                    and float(ev) >= 0 and float(edge) >= 0
                    else "🔴"
                )
                with tv_cols[ti % len(tv_cols)]:
                    st.markdown(
                        f"**{status} {tc.get('match_ru', tc.get('match', '—'))}**"
                    )
                    st.caption(
                        f"{tv.get('pick', '—')} · **@ {odd:.2f}** · Fair {fair:.2f}"
                    )
                    st.markdown(
                        f"Edge **{float(edge):.1%}** · EV **{float(ev):.1%}**"
                        if ev is not None and edge is not None
                        else "Edge / EV —"
                    )
                    st.caption(f"Value Score {vs:.2f} · ставка {stake:.0f}")
        else:
            st.info("🏆 TOP VALUE появится после получения реальных кэфов.")

        # ============ DECISION BOARD ============
        bet_cards = [
            c for c in cards_view
            if str(c.get("decision") or (c.get("verdict") or {}).get("decision") or "").upper() == "BET"
        ]
        watch_cards = [
            c for c in cards_view
            if str(c.get("decision") or (c.get("verdict") or {}).get("decision") or "").upper() == "WATCH"
        ]
        skip_cards = [
            c for c in cards_view
            if str(c.get("decision") or (c.get("verdict") or {}).get("decision") or "").upper() == "SKIP"
        ]

        st.markdown("### 🎯 DECISION BOARD")
        db1, db2, db3 = st.columns(3)
        with db1:
            st.metric("BET · в портфель", len(bet_cards))
        with db2:
            st.metric("WATCH · ждать цену", len(watch_cards))
        with db3:
            st.metric("SKIP · не входить", len(skip_cards))

        if bet_cards:
            st.markdown("#### 🟢 BEST BETS TODAY")
            st.caption("Реальные цены · Edge ≥ 3% · EV ≥ 3% · Kelly > 0.")
            best_cols = st.columns(min(3, len(bet_cards)))
            for bi, bc in enumerate(bet_cards):
                bv = bc.get("verdict") or {}
                odd_b = float(bv.get("odd") or 0.0)
                ev_b = float(bv.get("ev") or 0.0)
                edge_b = float(bv.get("edge") or 0.0)
                with best_cols[bi % len(best_cols)]:
                    st.markdown(f"**🟢 {bc.get('match_ru', bc.get('match', '—'))}**")
                    st.caption(
                        f"{bv.get('label', '—')} · @ {odd_b:.2f} · "
                        f"Fair {float(bv.get('fair_odd') or 0.0):.2f}"
                    )
                    st.markdown(
                        f"Edge **{edge_b:.1%}** · EV **{ev_b:.1%}** · "
                        f"Kelly **{float(bv.get('kelly_pct') or 0.0):.1%}**"
                    )

        if watch_cards:
            st.markdown("#### 🟡 WATCHLIST · ЖДЁМ ЛУЧШУЮ ЦЕНУ")
            watch_cols = st.columns(min(3, len(watch_cards)))
            for wi, wc in enumerate(watch_cards):
                wv = wc.get("verdict") or {}
                with watch_cols[wi % len(watch_cols)]:
                    st.markdown(f"**🟡 {wc.get('match_ru', wc.get('match', '—'))}**")
                    st.caption(
                        f"{wv.get('label', '—')} · @ {float(wv.get('odd') or 0.0):.2f} · "
                        f"Fair {float(wv.get('fair_odd') or 0.0):.2f}"
                    )
                    st.markdown(
                        f"Edge **{float(wv.get('edge') or 0.0):.1%}** · "
                        f"EV **{float(wv.get('ev') or 0.0):.1%}** · "
                        f"Kelly **{float(wv.get('kelly_pct') or 0.0):.1%}**"
                    )
                    st.caption("Причина: " + str(
                        wc.get('decision_reason') or wv.get('decision_reason') or "Цена ждёт улучшения."
                    ))

        if skip_cards:
            with st.expander(f"🔴 SKIP · {len(skip_cards)} цен не проходят"):
                for sc_ in skip_cards:
                    sv = sc_.get("verdict") or {}
                    st.markdown(
                        f"**{sc_.get('match_ru', sc_.get('match', '—'))}** · "
                        f"{sv.get('label', '—')} · "
                        f"@ {float(sv.get('odd') or 0.0):.2f} · "
                        f"Edge {float(sv.get('edge') or 0.0):.1%} · "
                        f"EV {float(sv.get('ev') or 0.0):.1%} · "
                        f"Kelly {float(sv.get('kelly_pct') or 0.0):.1%}"
                    )
                    st.caption("Причина: " + str(
                        sc_.get('decision_reason') or sv.get('decision_reason') or "Цена не проходит фильтр."
                    ))

        waiting = [
            c for c in cards_view
            if (c.get("verdict") or {}).get("is_action", False)
            and not (c.get("verdict") or {}).get("real_odds", False)
            and not c.get("best")
        ]
        if waiting:
            st.markdown("### ⏳ WAITING FOR ODDS")
            st.caption(
                f"{len(waiting)} модельных сигналов ждут подходящей цены."
            )
            wait_cols = st.columns(min(3, len(waiting)))
            for wi, wc in enumerate(waiting):
                wv = wc.get("verdict") or {}
                fair_w = float(wv.get("fair_odd") or 0.0)
                entry_w = float(wc.get("min_entry_odd") or 0.0)
                with wait_cols[wi % len(wait_cols)]:
                    st.markdown(
                        f"**{wc.get('match_ru', wc.get('match', '—'))}**  \n"
                        f"{wv.get('pick', '—')} · "
                        f"**{wv.get('prob', 0) * 100:.1f}%**  \n"
                        f"Fair **{fair_w:.2f}** → вход **{entry_w:.2f}+**"
                    )

        for idx, c in enumerate(cards_view):
            v = c.get("verdict") or {}
            if not v.get("is_action", False):
                hidden += 1
                continue

            st.markdown(
                render_verdict_card(c, min_prob),
                unsafe_allow_html=True,
            )
            shown += 1

            # Ручная котировка букмекера
            if not v.get("real_odds", False) and not c.get("best"):
                fair = float(v.get("fair_odd") or 0.0)
                prob = float(v.get("prob") or 0.0)
                min_entry = float(c.get("min_entry_odd") or 0.0)
                key_base = str(c.get("fixture_id") or c.get("match")) + "_" + str(v.get("pick") or "pick")
                st.markdown("**🎯 Ввести кэф букмекера вручную**")
                q1, q2, q3 = st.columns([1.2, 1.2, 1])
                manual_odd = q1.number_input(
                    "Кэф",
                    min_value=1.01,
                    max_value=100.0,
                    value=1.50,
                    step=0.01,
                    format="%.2f",
                    key=f"manual_odd_{key_base}",
                )
                q2.caption(f"Fair: **{fair:.2f}** · мин. вход: **{min_entry:.2f}**")
                q3.caption("EV / Edge / Kelly")
                if st.button(
                    "Проверить кэф → добавить",
                    key=f"manual_add_{key_base}",
                    type="secondary",
                ):
                    try:
                        manual_v, manual_best = refine_with_manual_odd(
                            dict(v), [], manual_odd,
                            float(D.get("bank") or 0.0), kelly_frac,
                        )
                    except Exception as exc:
                        manual_v, manual_best = dict(v), None
                        st.error(f"Ошибка расчёта кэфа: {exc}")

                    edge_v = manual_v.get("edge")
                    ev_v = manual_v.get("ev")
                    try:
                        edge_f = float(edge_v) if edge_v is not None else 0.0
                    except (TypeError, ValueError):
                        edge_f = 0.0
                    try:
                        ev_f = float(ev_v) if ev_v is not None else 0.0
                    except (TypeError, ValueError):
                        ev_f = 0.0
                    try:
                        kelly_f = float(manual_v.get("kelly_pct") or 0.0)
                    except (TypeError, ValueError):
                        kelly_f = 0.0

                    if (manual_best is None or kelly_f <= 0
                            or edge_f < 0.03 or ev_f < 0.03):
                        st.warning(
                            f"Кэф не стал BET: Edge {edge_f:.1%} · "
                            f"EV {ev_f:.1%} · Kelly {kelly_f:.1%}. "
                            "Нужно Edge ≥ 3%, EV ≥ 3%, Kelly > 0."
                        )
                    else:
                        mkt, pick, odd, ev, prob, stake = manual_best
                        stake = round(min(max(float(stake), 0.0),
                                          float(D.get("bank") or 0.0) * 0.05), 2)
                        existing_keys = {
                            f"{b.get('match')}|{b.get('pick')}"
                            for b in D.get("bets", [])
                            if isinstance(b, dict) and b.get("status") == "pending"
                        }
                        bk = f"{c['match']}|{pick}"
                        if bk in existing_keys:
                            st.info("Такая ставка уже есть в pending.")
                        elif stake <= 0:
                            st.warning("Kelly дал нулевую ставку.")
                        elif stake > float(D.get("bank") or 0.0):
                            st.warning("Недостаточно свободного банка.")
                        else:
                            manual_v["decision"] = "BET"
                            manual_v["is_bet"] = True
                            c["verdict"] = manual_v
                            c["best"] = (mkt, pick, odd, ev, prob, stake)
                            c["decision"] = "BET"
                            c["odds_source"] = "manual"
                            bet = {
                                "match": c["match"],
                                "match_ru": c["match_ru"],
                                "div": c["div"], "league": c["league"],
                                "market": mkt, "pick": pick,
                                "odds": odd, "stake": stake, "prob": prob,
                                "status": "pending", "strat": "value",
                                "odds_source": "manual", "mode": "real",
                                "date": datetime.now().strftime("%d.%m.%Y"),
                                "date_iso": c.get("date_iso"),
                                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "fixture_id": c.get("fixture_id"),
                                "score": None, "ev": ev,
                                "decision": "BET",
                                "decision_at": datetime.now().isoformat(),
                                "women": c.get("women", False),
                                "national": c.get("national", False),
                            }
                            D2 = dict(D)
                            D2["cards"] = cards_view
                            D2["bets"] = list(D.get("bets", [])) + [bet]
                            D2["bank"] = max(0.0, float(D.get("bank") or 0.0) - stake)
                            st.session_state.data = D2
                            usage.set_local_data(D2)
                            if db.SQLITE_BOOT_OK:
                                db.insert_bets_batch([bet])
                                db.log_bank(D2["bank"], event="manual_odds")
                                db.invalidate_caches()
                            st.success(
                                f"Добавлено: {pick} @ {odd:.2f} · "
                                f"EV {ev:.1%} · ставка {stake:.2f}"
                            )
                            st.rerun()

        if shown == 0 and hidden == 0:
            st.info("Нажми ⚡ СКАН.")
        elif shown == 0 and hidden > 0:
            st.warning(
                f"⚠️ Ни один матч не прошёл порог "
                f"**{min_prob * 100:.0f}%**. Скрыто **{hidden}**."
            )
        elif hidden > 0:
            st.caption(f"✅ Показано **{shown}** · скрыто **{hidden}**")
        return

    st.session_state["_scan_in_progress"] = True

    try:
        loader_ph = st.empty()
        log_ph = st.empty()

        def update_loader(text, pct, logs=None):
            loader_ph.markdown(
                f"""
                <div class='nbr-loader'>
                    <div class='nbr-ring'></div>
                    <div class='nbr-text'>
                        <b>{text}</b>
                        <div class='nbr-bar'>
                            <div class='nbr-bar-fill'
                                 style='width:{pct * 100:.0f}%'></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if logs:
                log_ph.markdown(
                    "<div style='color:#8b93a7;font-size:.8rem;"
                    "background:rgba(10,14,24,.6);padding:10px;"
                    "border-radius:10px;max-height:200px;"
                    "overflow-y:auto'>"
                    + "<br>".join(logs[-15:]) + "</div>",
                    unsafe_allow_html=True,
                )

        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        logs = []

        # ============ 1. THESPORTSDB ============
        update_loader("📡 TheSportsDB: сбор всех лиг...", 0.05, logs)
        try:
            tsdb_rows = tsdb_matches(days, logs)
        except Exception as e:
            logs.append(f"⚠️ TSDB ошибка: {e}")
            tsdb_rows = []

        # ============ 2. FOOTBALL-DATA.ORG ============
        token = D.get("meta", {}).get("fdorg_token", "").strip()
        fdorg_rows = []
        if token:
            update_loader("📡 football-data.org: дополнение...", 0.10, logs)
            try:
                fdorg_rows = fdorg_matches(days, token, logs)
            except Exception as e:
                logs.append(f"⚠️ fdorg ошибка: {e}")
        else:
            logs.append("📡 football-data.org: токен не задан")

        # ============ 3. ESPN ============
        espn_rows = []
        update_loader("📡 ESPN: дополнение покрытия...", 0.15, logs)
        try:
            espn_rows = espn_matches(days, logs)
        except Exception as e:
            logs.append(f"⚠️ ESPN ошибка: {e}")

        # ============ ОБЪЕДИНЕНИЕ ============
        rows_by_key = {}
        key_order = []

        for r in tsdb_rows + fdorg_rows + espn_rows:
            fid = str(r.get("fixture_id", "")).strip()
            home_key = "".join(
                ch for ch in str(r.get("HomeTeam", "")).lower
