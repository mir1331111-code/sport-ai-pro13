"""ui/tabs/scanner.py — Сканер v13.7: TSDB + fdorg + ESPN + Marathon + Grok."""
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
from betting.adaptive import (
    build_profiles as build_adaptive_profiles,
    priority as adaptive_priority,
)

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
        return "Edge >= 3%, EV >= 3% и Kelly > 0 — цена подтверждена."
    if decision == "WATCH":
        if kelly_f <= 0:
            return "Kelly <= 0 — ждём более высокий кэф."
        if edge_f < 0.03 and ev_f < 0.03:
            return "Edge и EV ниже 3% — нужен более выгодный кэф."
        if edge_f < 0.03:
            return "Edge ниже 3% — нужен более выгодный кэф."
        if ev_f < 0.03:
            return "EV ниже 3% — нужен более выгодный кэф."
        return "Value положительный, но цена пока не проходит полный фильтр."
    if kelly_f <= 0:
        return "Kelly <= 0 — ставка не оправдывает риск."
    if edge_f <= 0 and ev_f <= 0:
        return "Edge и EV <= 0 — положительного преимущества нет."
    if edge_f <= 0:
        return "Edge <= 0 — рыночная цена хуже fair."
    if ev_f <= 0:
        return "EV <= 0 — математического преимущества нет."
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

    adaptive_profile = {}
    if db.SQLITE_BOOT_OK:
        try:
            adaptive_profile = build_adaptive_profiles(
                db.fetch_decision_snapshots(limit=100000)
            )
        except Exception:
            adaptive_profile = {}

    model_candidates = 0
    market_checked = 0
    market_received = 0
    value_rejected = 0
    matches_with_best = 0
    quality_real_market = 0
    quality_missing_market = 0
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
    marathon_found = 0
    marathon_calls = 0
    marathon_errors = 0

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
                f"🔍 контекст {fn.get('ctx', 0)} · "
                f"🏁 Марафон: {fn.get('marathon', 0)} "
                f"(вызовов {fn.get('marathon_calls', 0)}, "
                f"ошибок {fn.get('marathon_errors', 0)})"
            )
        if HAS_CONTEXT:
            st.caption("🟢 Context Engine: подключён")

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
            st.progress(min(risk_ratio, 1.0),
                        text=f"Заморожено {risk_pct:.1f}% банка")
        with rr2:
            st.metric("Pending exposure", f"{pending_stake:.0f}")
        with rr3:
            st.metric("Свободный банк", f"{current_bank:.0f}")
        st.caption("Риск = сумма незакрытых ставок / начальный банк.")

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

        real_value_cards = [
            c for c in cards_view
            if c.get("best")
            and (c.get("odds_source") in ("market", "grok_web", "manual",
                                           "fdorg", "marathon"))
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

        bet_cards = [
            c for c in cards_view
            if str(c.get("decision")
                   or (c.get("verdict") or {}).get("decision") or "").upper() == "BET"
        ]
        watch_cards = [
            c for c in cards_view
            if str(c.get("decision")
                   or (c.get("verdict") or {}).get("decision") or "").upper() == "WATCH"
        ]
        skip_cards = [
            c for c in cards_view
            if str(c.get("decision")
                   or (c.get("verdict") or {}).get("decision") or "").upper() == "SKIP"
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
            best_cols = st.columns(min(3, len(bet_cards)))
            for bi, bc in enumerate(bet_cards):
                bv = bc.get("verdict") or {}
                with best_cols[bi % len(best_cols)]:
                    st.markdown(f"**🟢 {bc.get('match_ru', bc.get('match', '—'))}**")
                    st.caption(
                        f"{bv.get('label', '—')} · @ {float(bv.get('odd') or 0.0):.2f} · "
                        f"Fair {float(bv.get('fair_odd') or 0.0):.2f}"
                    )
                    st.markdown(
                        f"Edge **{float(bv.get('edge') or 0.0):.1%}** · "
                        f"EV **{float(bv.get('ev') or 0.0):.1%}** · "
                        f"Kelly **{float(bv.get('kelly_pct') or 0.0):.1%}**"
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

            if not v.get("real_odds", False) and not c.get("best"):
                fair = float(v.get("fair_odd") or 0.0)
                prob = float(v.get("prob") or 0.0)
                min_entry = float(c.get("min_entry_odd") or 0.0)
                key_base = (str(c.get("fixture_id") or c.get("match"))
                            + "_" + str(v.get("pick") or "pick"))
                st.markdown("**🎯 Ввести кэф букмекера вручную**")
                q1, q2, q3 = st.columns([1.2, 1.2, 1])
                manual_odd = q1.number_input(
                    "Кэф", min_value=1.01, max_value=100.0,
                    value=1.50, step=0.01, format="%.2f",
                    key=f"manual_odd_{key_base}",
                )
                q2.caption(f"Fair: **{fair:.2f}** · мин. вход: **{min_entry:.2f}**")
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
                        st.error(f"Ошибка расчёта: {exc}")

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
                            "Нужно Edge >= 3%, EV >= 3%, Kelly > 0."
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
                                "match": c["match"], "match_ru": c["match_ru"],
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

        update_loader("📡 Сбор матчей...", 0.05, logs)

        # TSDB
        try:
            tsdb_rows = tsdb_matches(days, logs)
        except Exception as e:
            logs.append(f"⚠️ TSDB ошибка: {e}")
            tsdb_rows = []

        # fdorg
        token = D.get("meta", {}).get("fdorg_token", "").strip()
        fdorg_rows = []
        if token:
            update_loader("📡 fdorg...", 0.10, logs)
            try:
                fdorg_rows = fdorg_matches(days, token, logs)
            except Exception as e:
                logs.append(f"⚠️ fdorg ошибка: {e}")
        else:
            logs.append("📡 fdorg: токен не задан")

        # ESPN
        espn_rows = []
        update_loader("📡 ESPN...", 0.15, logs)
        try:
            espn_rows = espn_matches(days, logs)
        except Exception as e:
            logs.append(f"⚠️ ESPN ошибка: {e}")

        # Объединение
        rows_by_key = {}
        key_order = []
        for r in tsdb_rows + fdorg_rows + espn_rows:
            fid = str(r.get("fixture_id", "")).strip()
            home_key = "".join(
                ch for ch in str(r.get("HomeTeam", "")).lower()
                if ch.isalnum())
            away_key = "".join(
                ch for ch in str(r.get("AwayTeam", "")).lower()
                if ch.isalnum())
            date_key = str(r.get("Date", ""))[:10]
            match_key = (date_key, home_key, away_key)
            key = ("id", fid) if fid else ("match",) + match_key
            if key not in rows_by_key:
                rows_by_key[key] = dict(r)
                key_order.append(key)
                continue
            existing = rows_by_key[key]
            for field, value in r.items():
                if value in (None, "", [], {}):
                    continue
                if field == "odds":
                    merged = dict(existing.get("odds") or {})
                    merged.update({k: v for k, v in (value or {}).items()
                                   if v not in (None, "", 0)})
                    if merged:
                        existing["odds"] = merged
                    continue
                if not existing.get(field):
                    existing[field] = value

        rows_raw = [rows_by_key[k] for k in key_order]
        rows = _safe_filter(rows_raw)
        w_count = sum(1 for r in rows if r.get("women"))
        n_count = sum(1 for r in rows if r.get("national"))
        c_count = len(rows) - w_count - n_count
        logs.append(
            f"📡 ИСТОЧНИКИ: TSDB {len(tsdb_rows)} · "
            f"fdorg {len(fdorg_rows)} · ESPN {len(espn_rows)}"
        )
        logs.append(
            f"📡 ИТОГО: {len(rows)} · 🏟 {c_count} · 👩 {w_count} · 🌍 {n_count}"
        )

        update_loader("🧠 Модель...", 0.20, logs)
        engine = _load_engine(matrix_n, logs, update_loader)

        update_loader("🧠 Анализ...", 0.50, logs)
        cards = []
        odds_key = D.get("meta", {}).get("odds_api_key", "")
        llm_provider_now = D.get("meta", {}).get("llm_provider", "")
        grok_key = str(D.get("meta", {}).get("llm_api_key", "") or "").strip()
        grok_remaining = usage.llm_remaining()
        grok_enabled = (
            llm_provider_now == "Grok (x.ai)"
            and bool(grok_key) and grok_remaining > 0
        )
        if grok_enabled:
            logs.append(f"🌐 GROK WEB: ON · лимит LLM {grok_remaining}")
        elif llm_provider_now != "Grok (x.ai)":
            logs.append(f"🌐 GROK WEB: OFF · провайдер {llm_provider_now or 'не задан'}")
        elif not grok_key:
            logs.append("🌐 GROK WEB: OFF · не задан LLM key")
        else:
            logs.append("🌐 GROK WEB: OFF · дневной лимит LLM")

        skipped_started = 0
        skipped_bad_time = 0
        skipped_bad_date = 0
        skipped_missing_teams = 0
        skipped_model_error = 0

        for r in rows:
            d = parse_date(r.get("Date", ""))
            if not d:
                skipped_bad_date += 1
                continue
            if not (today <= d <= today + timedelta(days=days)):
                continue
            start_dt = _match_start_dt(r)
            if start_dt is None:
                if d.date() <= now.date():
                    skipped_bad_time += 1
                    continue
            elif start_dt <= now:
                skipped_started += 1
                continue
            h_en = (r.get("HomeTeam") or "").strip()
            a_en = (r.get("AwayTeam") or "").strip()
            if not h_en or not a_en:
                skipped_missing_teams += 1
                continue
            h_ru = translate_team(h_en)
            a_ru = translate_team(a_en)
            lg = r.get("Div") or "G"
            try:
                P = engine.predict(h_en, a_en, lg, match_date=d,
                                   cup=lg in ("C1", "EL", "EC", "W_C1",
                                              "NT_WC", "NT_EURO", "NT_CA",
                                              "NT_ACN", "NT_ASC"))
            except Exception:
                skipped_model_error += 1
                continue
            fh = engine.form_str(h_en)
            fa = engine.form_str(a_en)
            verdict, rows_, _ = build_verdict(
                P, min_prob, D["bank"], kelly_frac,
                h_ru, a_ru, fh, fa, P.get("h2h_n", 0),
            )

            best = None
            odds_source = "estimated"

            if verdict.get("is_action"):
                model_candidates += 1
                verdict["real_odds"] = False
                verdict["odd"] = verdict.get("fair_odd")
                verdict["ev"] = None
                verdict["edge"] = None
                verdict["stake"] = 0.0
                verdict["kelly_pct"] = 0.0
                verdict["is_bet"] = False
                odds_source = "estimated" if verdict.get("fair_odd") else "unavailable"
            else:
                verdict["real_odds"] = False
                verdict["odd"] = None
                verdict["ev"] = None
                verdict["edge"] = None
                verdict["stake"] = 0.0
                verdict["is_bet"] = False

            p_top = float(verdict.get("prob") or 0.0)
            min_edge_price = 1.0 / (p_top - 0.03) if p_top > 0.03 else 0.0
            min_ev_price = 1.03 / p_top if p_top > 0 else 0.0
            min_entry_odd = max(min_edge_price, min_ev_price, 0.0)
            adaptive_factor, adaptive_reason = adaptive_priority(
                adaptive_profile, {
                    "date_iso": d.strftime("%Y-%m-%d"),
                    "league": r.get("League") or DIV_NAMES.get(lg, "Лига"),
                    "div": lg, "verdict": verdict,
                },
            )

            cards.append({
                "div": lg,
                "league": r.get("League") or DIV_NAMES.get(lg, "Лига"),
                "match": f"{h_en} vs {a_en}",
                "match_ru": f"{h_ru} — {a_ru}",
                "date": d.strftime("%d.%m")
                        + (f" {r.get('Time', '')}" if r.get("Time") else ""),
                "verdict": verdict, "best": best,
                "games": P["games"], "fh": fh, "fa": fa,
                "fixture_id": r.get("fixture_id"),
                "date_iso": d.strftime("%Y-%m-%d"),
                "lam_h": P["lams"][0], "lam_a": P["lams"][1],
                "p1": P["p1"], "px": P["x"], "p2": P["p2"],
                "over": P["over"], "btts": P["btts"],
                "odds_source": odds_source,
                "fdorg_odds": {
                    k: float(v) for k, v in (r.get("odds") or {}).items()
                    if v is not None and float(v) > 1.01
                },
                "grok_comment": "", "grok_bookmaker": "",
                "min_entry_odd": round(min_entry_odd, 2) if min_entry_odd > 0 else None,
                "women": r.get("women", False),
                "national": r.get("national", False),
                "kind": r.get("kind", "club"),
                "home_badge": r.get("home_badge") or "",
                "away_badge": r.get("away_badge") or "",
                "league_badge": r.get("league_badge") or "",
                "adaptive_factor": adaptive_factor,
                "adaptive_reason": adaptive_reason,
            })

        # === МАРКЕТ ===
        market_candidates = [
            c for c in cards
            if (c.get("verdict") or {}).get("is_action", False)
        ]
        market_candidates.sort(
            key=lambda c: (-float(rank_score(c) or 0.0)
                           * float(c.get("adaptive_factor") or 1.0))
        )
        top_ex
