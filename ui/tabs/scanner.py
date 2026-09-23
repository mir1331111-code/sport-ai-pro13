"""ui/tabs/scanner.py — Сканер v13.3: TheSportsDB + Context + LLM + сборные."""
from __future__ import annotations
import time
from datetime import datetime, timedelta

import streamlit as st

from config import APP_VERSION, DIV_NAMES, DIV_TO_ODDS
from storage import usage
from storage import sqlite_store as db
from data.sources import (load_seasonal, season_str,
                          fdorg_matches, tsdb_matches,
                          odds_api_fixture, parse_date)
from model.engine import Engine
from betting.verdict import build_verdict, refine_with_real_odds
from betting.kelly import kelly, market_type
from llm.analyst import analyze_match
from ui.cards import render_verdict_card, translate_team

try:
    from context_football import analyze_match_context, render_context_flags
    HAS_CONTEXT = True
except Exception:
    HAS_CONTEXT = False

LLM_TOP_N = 8


def _safe_filter(rows):
    if not isinstance(rows, list): return []
    return [r for r in rows if isinstance(r, dict) and r.get("HomeTeam") and r.get("AwayTeam")]


def _train_engine(matrix_n, logs, update_loader):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    cur_year = today.year if today.month >= 7 else today.year - 1
    prev_year = cur_year - 1
    train_divs = ["E0", "SP1", "I1", "D1", "F1", "E1", "SP2", "I2",
                  "D2", "F2", "N1", "B1", "P1", "T1", "R1"]
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
    processed = 0; trained = 0
    for dv in train_divs:
        for src in (dp.get(dv, []), dc.get(dv, [])):
            for r in src:
                try:
                    hg = float(r.get("FTHG", 0)); ag = float(r.get("FTAG", 0))
                    engine.learn_step(
                        r["HomeTeam"], r["AwayTeam"], hg, ag, r,
                        lg=dv, match_num=processed, total=total,
                        match_date=parse_date(r.get("Date", "")))
                    trained += 1
                except Exception: pass
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
                logs.append("Engine из кэша"); return data.get("engine")
        except Exception: pass
    engine = _train_engine(matrix_n, logs, update_loader)
    try:
        with open(p, "wb") as f:
            f.write(gzip.compress(pickle.dumps({"fp": APP_VERSION, "engine": engine})))
    except Exception: pass
    return engine


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
                f"📡 Матчей {fn.get('src', 0)} · "
                f"🎯 Найдено {fn.get('found', 0)} · "
                f"➕ В портфель {fn.get('added', 0)} · "
                f"💰 Заморожено {fn.get('frozen', 0):.0f} · "
                f"🤖 LLM: {fn.get('llm', 0)} · "
                f"🔍 Контекст: {fn.get('ctx', 0)} · "
                f"👩 Женских: {fn.get('women', 0)} · "
                f"🌍 Сборных: {fn.get('national', 0)}"
            )
        if HAS_CONTEXT:
            st.caption("🟢 Context Engine: подключён")
        with st.expander("🔌 Диагностика"):
            for line in D.get("report", []):
                st.text(line)

        all_cards = D.get("cards", [])
        cards_view = sorted(
            [c for c in all_cards if isinstance(c, dict)],
            key=lambda c: (c.get("verdict", {}).get("prob") or 0), reverse=True)
        shown = 0; hidden = 0
        for c in cards_view:
            v = c.get("verdict") or {}
            if not v.get("is_action", False): hidden += 1; continue
            st.markdown(render_verdict_card(c, min_prob), unsafe_allow_html=True)
            shown += 1
        if shown == 0 and hidden == 0:
            st.info("Нажми ⚡ СКАН.")
        elif shown == 0 and hidden > 0:
            st.warning(f"⚠️ Ни один матч не прошёл порог **{min_prob*100:.0f}%**. Скрыто **{hidden}**.")
        elif hidden > 0:
            st.caption(f"✅ Показано **{shown}** · скрыто **{hidden}**")
        return

    st.session_state["_scan_in_progress"] = True
    try:
        loader_ph = st.empty(); log_ph = st.empty()

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
                    "border-radius:10px;max-height:200px;overflow-y:auto'>"
                    + "<br>".join(logs[-15:]) + "</div>", unsafe_allow_html=True)

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        logs = []

        # ===== 1. THESPORTSDB — ОСНОВНОЙ (все лиги + женские + сборные) =====
        update_loader("📡 TheSportsDB: сбор всех лиг...", 0.05, logs)
        tsdb_rows = tsdb_matches(days, logs)

        # ===== 2. FOOTBALL-DATA.ORG — ДОПОЛНИТЕЛЬНЫЙ (если есть токен) =====
        token = D.get("meta", {}).get("fdorg_token", "").strip()
        fdorg_rows = []
        if token:
            update_loader("📡 football-data.org: дополнение...", 0.10, logs)
            fdorg_rows = fdorg_matches(days, token, logs)
        else:
            logs.append("📡 football-data.org: токен не задан (работаем на TheSportsDB)")

        # Объединяем, убираем дубликаты
        seen_ids = set()
        rows_raw = []
        for r in tsdb_rows + fdorg_rows:
            fid = str(r.get("fixture_id", ""))
            if fid and fid in seen_ids: continue
            if fid: seen_ids.add(fid)
            rows_raw.append(r)

        rows = _safe_filter(rows_raw)
        w_count = sum(1 for r in rows if r.get("women"))
        n_count = sum(1 for r in rows if r.get("national"))
        c_count = len(rows) - w_count - n_count
        logs.append(f"📡 ИТОГО: {len(rows)} · 🏟 клубные {c_count} · 👩 женские {w_count} · 🌍 сборные {n_count}")

        # ===== 3. ОБУЧЕНИЕ МОДЕЛИ =====
        update_loader("🧠 Загрузка модели...", 0.20, logs)
        engine = _load_engine(matrix_n, logs, update_loader)

        # ===== 4. АНАЛИЗ МАТЧЕЙ =====
        update_loader("🧠 Анализ матчей...", 0.50, logs)
        cards = []; matches_with_best = 0
        odds_key = D.get("meta", {}).get("odds_api_key", "")

        for r in rows:
            d = parse_date(r.get("Date", ""))
            if not d or not (today <= d <= today + timedelta(days=days)): continue
            h_en = (r.get("HomeTeam") or "").strip()
            a_en = (r.get("AwayTeam") or "").strip()
            if not h_en or not a_en: continue
            h_ru = translate_team(h_en); a_ru = translate_team(a_en)
            lg = r.get("Div") or "G"

            try:
                P = engine.predict(h_en, a_en, lg, match_date=d,
                                   cup=lg in ("C1", "EL", "EC", "W_C1", "NT_WC", "NT_EURO", "NT_CA", "NT_ACN", "NT_ASC"))
            except Exception: continue
            fh = engine.form_str(h_en); fa = engine.form_str(a_en)
            verdict, rows_, _ = build_verdict(
                P, min_prob, D["bank"], kelly_frac,
                h_ru, a_ru, fh, fa, P.get("h2h_n", 0))

            best = None; odds_source = "estimated"

            if verdict.get("is_action"):
                sport_key = DIV_TO_ODDS.get(lg)
                real_odds = None
                if sport_key and odds_key and usage.odds_remaining() > 0 and matches_with_best < 15:
                    real_odds = odds_api_fixture(sport_key, h_en, a_en, odds_key)
                if real_odds:
                    verdict, best = refine_with_real_odds(
                        verdict, rows_, real_odds, D["bank"], kelly_frac)
                    odds_source = "market"
                if best is None:
                    est_odd = verdict.get("fair_odd")
                    if est_odd and est_odd > 1.01:
                        prob_ = verdict["prob"]; ev_ = prob_ * est_odd - 1
                        min_stake = round(D["bank"] * 0.005, 2)
                        stake_ = max(kelly(prob_, est_odd, D["bank"], kelly_frac), min_stake)
                        verdict["real_odds"] = False; verdict["odd"] = est_odd; verdict["ev"] = ev_
                        best = (market_type(verdict["pick"]), verdict["pick"], est_odd, ev_, prob_, stake_)
                        odds_source = "estimated"

            if best: matches_with_best += 1

            cards.append({
                "div": lg, "league": r.get("League") or DIV_NAMES.get(lg, "Лига"),
                "match": f"{h_en} vs {a_en}", "match_ru": f"{h_ru} — {a_ru}",
                "date": d.strftime("%d.%m") + (f" {r.get('Time','')}" if r.get("Time") else ""),
                "verdict": verdict, "best": best,
                "games": P["games"], "fh": fh, "fa": fa,
                "fixture_id": r.get("fixture_id"),
                "date_iso": d.strftime("%Y-%m-%d"),
                "lam_h": P["lams"][0], "lam_a": P["lams"][1],
                "p1": P["p1"], "px": P["x"], "p2": P["p2"],
                "over": P["over"], "btts": P["btts"],
                "odds_source": odds_source,
                "women": r.get("women", False),
                "national": r.get("national", False),
                "kind": r.get("kind", "club"),
                "home_badge": r.get("home_badge") or "",
                "away_badge": r.get("away_badge") or "",
                "league_badge": r.get("league_badge") or "",
            })

        logs.append(f"🎯 Найдено с P≥{min_prob*100:.0f}%: {matches_with_best}")

        # ===== 5. LLM =====
        llm_key = D.get("meta", {}).get("llm_api_key", "")
        llm_prov = D.get("meta", {}).get("llm_provider", "Groq (бесплатно, быстро)")
        llm_model = D.get("meta", {}).get("llm_model", "")
        action_cards = [c for c in cards if c.get("best") is not None]
        action_cards.sort(key=lambda c: -(c.get("verdict", {}).get("prob") or 0))
        llm_done = 0

        if llm_key and usage.llm_remaining() > 0:
            for i_, card in enumerate(action_cards[:LLM_TOP_N]):
                if usage.llm_remaining() <= 0: break
                update_loader(
                    f"🤖 ИИ-анализ [{i_+1}/{min(LLM_TOP_N, len(action_cards))}]",
                    0.70 + 0.10 * (i_+1) / max(1, LLM_TOP_N), logs)
                v_ = card.get("verdict", {})
                ctx = {
                    "api_key": llm_key, "provider": llm_prov, "model": llm_model,
                    "home": card.get("match_ru", "").split(" — ")[0],
                    "away": card.get("match_ru", "").split(" — ")[-1],
                    "league": card.get("league", ""), "date": card.get("date", ""),
                    "lam_h": card.get("lam_h", 0), "lam_a": card.get("lam_a", 0),
                    "p1": card.get("p1", 0), "px": card.get("px", 0),
                    "p2": card.get("p2", 0), "over": card.get("over", 0),
                    "btts": card.get("btts", 0),
                    "fh": card.get("fh", "—"), "fa": card.get("fa", "—"),
                    "pick": v_.get("label", ""), "prob": v_.get("prob", 0),
                    "confidence": v_.get("confidence", ""), "ev": v_.get("ev", 0),
                }
                opinion = analyze_match(ctx)
                if opinion: card["llm_opinion"] = opinion; llm_done += 1
            logs.append(f"🤖 LLM: {llm_done} мнений")
        elif not llm_key:
            logs.append("🤖 LLM: ключ не задан")

        # ===== 6. CONTEXT ENGINE =====
        ctx_count = 0
        if HAS_CONTEXT:
            update_loader("🔍 Контекстный анализ...", 0.85, logs)
            cur_y = today.year if today.month >= 7 else today.year - 1
            s_ctx = season_str(cur_y)
            for c in cards:
                if not c.get("verdict", {}).get("is_action"): continue
                if c.get("national"): continue  # контекст только для клубных
                try:
                    parts_m = c["match"].split(" vs ")
                    if len(parts_m) == 2:
                        ctx = analyze_match_context(
                            parts_m[0], parts_m[1], c.get("div", ""), s_ctx)
                        c["context"] = ctx; ctx_count += 1
                except Exception: pass
            logs.append(f"🔍 Контекст: {ctx_count} клубных матчей")

        # ===== 7. СТАВКИ =====
        update_loader("💼 Формирование портфеля...", 0.95, logs)
        new_bets = []
        existing = {f"{b['match']}|{b['pick']}" for b in D["bets"]
                    if isinstance(b, dict) and b.get("status") == "pending"}
        women_bet_count = 0; national_bet_count = 0
        for c in cards:
            b = c.get("best")
            if not b: continue
            mkt, pick, odd, ev, prob, stake = b
            stake = round(min(max(stake, D["bank"] * 0.005), D["bank"] * 0.05), 2)
            if stake <= 0: continue
            bk = f"{c['match']}|{pick}"
            if bk in existing: continue
            if c.get("women"): women_bet_count += 1
            if c.get("national"): national_bet_count += 1
            new_bets.append({
                "match": c["match"], "match_ru": c["match_ru"],
                "div": c["div"], "league": c["league"],
                "market": mkt, "pick": pick, "odds": odd, "stake": stake,
                "prob": prob, "status": "pending", "strat": "value",
                "odds_source": c.get("odds_source", "estimated"),
                "mode": "paper" if c.get("odds_source") == "estimated" else "real",
                "date": datetime.now().strftime("%d.%m.%Y"),
                "date_iso": c.get("date_iso"),
                "date_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "fixture_id": c.get("fixture_id"), "score": None, "ev": ev,
                "women": c.get("women", False),
                "national": c.get("national", False),
            })
            existing.add(bk)

        # ===== 8. СОХРАНЕНИЕ =====
        D2 = dict(D); D2["cards"] = cards; D2["report"] = logs
        D2["bets"] = D["bets"] + new_bets
        total_stake = sum(b["stake"] for b in new_bets)
        D2["bank"] = max(0.0, D["bank"] - total_stake)
        D2["funnel"] = {
            "trained": getattr(engine, "trained_n", 0),
            "src": len(rows), "found": matches_with_best,
            "added": len(new_bets), "frozen": total_stake,
            "llm": llm_done, "ctx": ctx_count,
            "women": women_bet_count, "national": national_bet_count,
        }
        st.session_state.data = D2; usage.set_local_data(D2)
        if db.SQLITE_BOOT_OK:
            db.insert_bets_batch(new_bets)
            db.log_bank(D2["bank"], event="scan"); db.invalidate_caches()

        update_loader(
            f"✅ Готово! +{len(new_bets)} ставок · 📡 {len(rows)} матчей · 🤖 {llm_done} LLM · 👩 {women_bet_count} женских · 🌍 {national_bet_count} сборных",
            1.0, logs)
        time.sleep(1.5); loader_ph.empty(); log_ph.empty()
    finally:
        st.session_state["_scan_in_progress"] = False
    st.rerun()
