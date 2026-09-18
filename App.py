"""App.py — NEURO BET PRO v13.0 (compact)."""
from __future__ import annotations
import os
from datetime import datetime, timedelta

import streamlit as st

for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy"]:
    os.environ.pop(_k, None)

st.set_page_config(page_title="NEURO BET PRO v13",
                   page_icon="🏟", layout="wide")

from config import (APP_VERSION, LLM_PROVIDERS, ODDS_LIMIT_DAILY,
                    AUTO_SETTLE_LIMIT, LLM_DAILY_LIMIT, DATA_VERSION)
from storage import sqlite_store as db
from storage import usage
from data.sources import parse_date, tsdb_match_result
from betting.settlement import determine_outcome
from ui import theme
from ui.tabs import scanner, portfolio, stats, calculator, backtest

ERR = []

db.db_init()
theme.inject()

if "data" not in st.session_state:
    st.session_state.data = usage.get_local_data() or {
        "version": DATA_VERSION, "bank": 10000.0, "bets": [], "cards": [],
        "funnel": None, "report": [], "meta": {},
        "stats": {"won": 0, "lost": 0, "profit": 0, "push": 0, "void": 0},
        "mode": "paper",
    }

D = st.session_state.data
D.setdefault("meta", {})
if "initial_bank" not in D["meta"]:
    D["meta"]["initial_bank"] = float(D.get("bank", 10000.0))
    usage.set_local_data(D)


def _auto_settle(D):
    if usage.settle_remaining() <= 0:
        return D, 0
    D2 = dict(D)
    bets = list(D2["bets"])
    changed = 0
    now = datetime.now()
    for idx, b in enumerate(bets):
        if b.get("status") != "pending" or usage.settle_remaining() <= 0:
            continue
        fid = b.get("fixture_id")
        if not fid:
            continue
        bd = parse_date(b.get("date_iso") or b.get("date") or "")
        if bd and bd > now:
            continue
        res = tsdb_match_result(fid)
        usage.settle_increment(1)
        if not res or res.get("status") not in ("Match Finished", "FT", "AET", "PEN"):
            continue
        outcome, _ = determine_outcome(
            b.get("market"), b.get("pick"), res["home"], res["away"])
        if not outcome:
            continue
        b2 = dict(b)
        b2["score"] = f"{res['home']}:{res['away']}"
        s = dict(D2.get("stats", {}))
        if outcome == "push":
            b2["status"] = "push"
            D2["bank"] += b2["stake"]
            s["push"] = s.get("push", 0) + 1
        elif outcome == "void":
            b2["status"] = "void"
            D2["bank"] += b2["stake"]
            s["void"] = s.get("void", 0) + 1
        elif outcome == "won":
            b2["status"] = "won"
            D2["bank"] += b2["stake"] * b2["odds"]
            s["won"] = s.get("won", 0) + 1
            s["profit"] = s.get("profit", 0) + b2["stake"] * (b2["odds"] - 1)
        elif outcome == "lost":
            b2["status"] = "lost"
            s["lost"] = s.get("lost", 0) + 1
            s["profit"] = s.get("profit", 0) - b2["stake"]
        bets[idx] = b2
        D2["stats"] = s
        changed += 1
    D2["bets"] = bets
    return D2, changed


_now = datetime.now().timestamp()
_last = st.session_state.get("_last_auto_settle_ts", 0)
if _now - _last > 21600:
    D2, n = _auto_settle(D)
    if n > 0:
        st.session_state.data = D2
        usage.set_local_data(D2)
        db.log_bank(D2.get("bank", 10000.0), event="auto_settle")
        db.invalidate_caches()
        D = D2
        st.toast(f"Авто-закрыто {n} ставок")
    st.session_state["_last_auto_settle_ts"] = _now

pending_count = sum(1 for b in D["bets"]
                    if isinstance(b, dict) and b.get("status") == "pending")

st.markdown(f"""
<div class="hero"><h1>NEURO BET PRO</h1>
<p>v{APP_VERSION} · 100% FREE · ИИ-аналитик · SQLite · CLV</p>
<div class="kpis">
 <div class="kpi"><div class="t">Банкролл</div>
  <div class="v y">{D['bank']:.0f} у.е.</div></div>
 <div class="kpi"><div class="t">В работе</div><div class="v">{pending_count}</div></div>
 <div class="kpi"><div class="t">Всего</div><div class="v">{len(D['bets'])}</div></div>
 <div class="kpi"><div class="t">Ошибок</div>
  <div class="v {'r' if ERR else 'g'}">{len(ERR)}</div></div>
</div></div>""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Настройки")
    st.markdown(f"**SQLite:** {'OK' if db.SQLITE_BOOT_OK else 'FAIL'}")
    st.markdown(f"**Odds API:** {usage.odds_remaining()} / {ODDS_LIMIT_DAILY}")
    st.markdown(f"**Auto-settle:** {usage.settle_remaining()} / {AUTO_SETTLE_LIMIT}")
    st.markdown(f"**LLM:** {usage.llm_remaining()} / {LLM_DAILY_LIMIT}")

    odds_key = st.text_input("The Odds API key",
        value=D.get("meta", {}).get("odds_api_key", ""), type="password")
    if odds_key != D["meta"].get("odds_api_key", ""):
        D["meta"]["odds_api_key"] = odds_key
        usage.set_local_data(D)

    prov_keys = list(LLM_PROVIDERS.keys())
    cur = D["meta"].get("llm_provider", prov_keys[0])
    llm_prov = st.selectbox("LLM провайдер", prov_keys,
        index=prov_keys.index(cur) if cur in prov_keys else 0)
    llm_key = st.text_input("LLM key",
        value=D.get("meta", {}).get("llm_api_key", ""), type="password")
    if llm_key != D["meta"].get("llm_api_key", ""):
        D["meta"]["llm_api_key"] = llm_key
        D["meta"]["llm_provider"] = llm_prov
        usage.set_local_data(D)

    min_prob = st.slider("Мин. вероятность %", 50, 85, 55, 1) / 100
    kelly_frac = st.slider("Kelly доля", 0.10, 0.40, 0.25, 0.05)
    matrix_n = st.slider("Матрица голов", 6, 15, 12, 1)

    if st.button("Сбросить счётчики"):
        usage.settle_reset()
        usage.llm_reset()
        usage.odds_reset()
        st.rerun()

    if st.button("Очистить портфель"):
        D["bets"] = []
        D["cards"] = []
        D["bank"] = 10000.0
        D["stats"] = {"won": 0, "lost": 0, "profit": 0, "push": 0, "void": 0}
        usage.set_local_data(D)
        db.invalidate_caches()
        st.rerun()


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Сканер", "Портфель", "Статистика", "Калькулятор", "Бэктест"])

with tab1:
    scanner.render(min_prob, kelly_frac, matrix_n)
with tab2:
    portfolio.render()
with tab3:
    stats.render()
with tab4:
    calculator.render(matrix_n)
with tab5:
    backtest.render(matrix_n)
