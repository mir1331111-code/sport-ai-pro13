"""App.py — NEURO BET PRO v13.0. Точка входа Streamlit."""
from __future__ import annotations
import os
from datetime import datetime, timedelta

import streamlit as st

for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy", "FTP_PROXY", "ftp_proxy"]:
    os.environ.pop(_k, None)

st.set_page_config(page_title="NEURO BET PRO v13",
                   page_icon="🏟", layout="wide",
                   initial_sidebar_state="expanded")

from config import (APP_VERSION, LLM_PROVIDERS, ODDS_LIMIT_DAILY,
                    AUTO_SETTLE_LIMIT, LLM_DAILY_LIMIT, DATA_VERSION)
from storage import sqlite_store as db
from storage import usage
from data.sources import parse_date, tsdb_match_result
from betting.settlement import determine_outcome
from ui import theme
from ui.tabs import scanner, portfolio, stats, calculator, backtest


ERR: list = []


def log_err(tag: str, e: Exception) -> None:
    line = (f"[{datetime.now():%Y-%m-%d %H:%M:%S}][{tag}] "
            f"{type(e).__name__}: {str(e)[:200]}")
    ERR.append(line)
    if len(ERR) > 100:
        ERR.pop(0)


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
        if b.get("status") != "pending":
            continue
        if usage.settle_remaining() <= 0:
            break
        fid = b.get("fixture_id")
        if not fid:
            continue
        bd = parse_date(b.get("date_iso") or b.get("date") or "")
        if bd and bd > now:
            continue
        res = tsdb_match_result(fid)
        usage.settle_increment(1)
        if not res:
            continue
        status = res.get("status", "")
        if status not in ("Match Finished", "FT", "AET", "PEN"):
            continue
        outcome, _reason = determine_outcome(
            b.get("market"), b.get("pick"), res["home"], res["away"])
        if outcome is None:
            continue
        b2 = dict(b)
        b2["score"] = f"{res['home']}:{res['away']}"
        stats_d = dict(D2.get("stats", {}))
        if outcome == "push":
            b2["status"] = "push"
            D2["bank"] = D2.get("bank", 10000.0) + b2["stake"]
            stats_d["push"] = stats_d.get("push", 0) + 1
        elif outcome == "void":
            b2["status"] = "void"
            D2["bank"] = D2.get("bank", 10000.0) + b2["stake"]
            stats_d["void"] = stats_d.get("void", 0) + 1
        elif outcome == "won":
            b2["status"] = "won"
            D2["bank"] = D2.get("bank", 10000.0) + b2["stake"] * b2["odds"]
            stats_d["won"] = stats_d.get("won", 0) + 1
            stats_d["profit"] = stats_d.get("profit", 0) + b2["stake"] * (b2["odds"] - 1)
        elif outcome == "lost":
            b2["status"] = "lost"
            stats_d["lost"] = stats_d.get("lost", 0) + 1
            stats_d["profit"] = stats_d.get("profit", 0) - b2["stake"]
        bets[idx] = b2
        D2["stats"] = stats_d
        changed += 1

    for idx, b in enumerate(bets):
        if b.get("status") != "pending":
            continue
        bd = parse_date(b.get("date_iso") or b.get("date") or "")
        if bd and now - bd > timedelta(hours=48):
            b2 = dict(b)
            b2["status"] = "void"
            b2["score"] = "void (no result)"
            D2["bank"] = D2.get("bank", 10000.0) + b2["stake"]
            stats_d = dict(D2.get("stats", {}))
            stats_d["void"] = stats_d.get("void", 0) + 1
            D2["stats"] = stats_d
            bets[idx] = b2
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
<p>v{APP_VERSION} · 100% FREE · ИИ-аналитик · прогнозы→портфель ·
SQLite · CLV · real/paper разделение</p>
<div class="kpis">
 <div class="kpi"><div class="t">Банкролл</div>
  <div class="v y">{D['bank']:.0f} у.е.</div></div>
 <div class="kpi"><div class="t">В работе</div><div class="v">{pending_count}</div></div>
 <div class="kpi"><div class="t">Всего ставок</div>
  <div class="v">{len(D['bets'])}</div></div>
 <div class="kpi"><div class="t">Ошибок</div>
  <div class="v {'r' if ERR else 'g'}">{len(ERR)}</div></div>
</div></div>""", unsafe_allow_html=True)


with st.sidebar:
    st.header("Настройки")

    st.markdown(f"""
<div style="background:rgba(34,197,94,.10);border:1px solid rgba(34,197,94,.4);
 border-radius:12px;padding:10px 14px;margin-bottom:10px;">
<div style="color:#86efac;font-size:.7rem;text-transform:uppercase;
 font-weight:700;">Источники (все бесплатные)</div>
<div style="font-size:.78rem;color:#e6eaf2;line-height:1.5;">
football-data.co.uk — история + кэфы<br>
TheSportsDB — матчи + авто-сеттл<br>
The Odds API — 500 запросов/мес<br>
LLM-аналитик — Gemini/Groq/Grok</div></div>""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.10);
 border-radius:12px;padding:10px 14px;margin-bottom:10px;">
<div style="color:#7dd3fc;font-size:.7rem;text-transform:uppercase;
 font-weight:700;">SQLite</div>
<div style="font-size:.75rem;color:#e6eaf2;">
Status: <b style="color:{'#34d399' if db.SQLITE_BOOT_OK else '#f87171'};">
{'OK' if db.SQLITE_BOOT_OK else 'FAIL'}</b></div></div>""",
        unsafe_allow_html=True)

    o_rem = usage.odds_remaining()
    s_rem = usage.settle_remaining()
    llm_rem = usage.llm_remaining()
    st.markdown(f"""
<div style="background:rgba(34,197,94,.10);border:1px solid rgba(34,197,94,.35);
 border-radius:12px;padding:10px 14px;margin-bottom:10px;">
<div style="color:#86efac;font-size:.7rem;text-transform:uppercase;
 font-weight:700;">The Odds API</div>
<div style="font-size:1.3rem;font-weight:800;color:#86efac;">{o_rem} / {ODDS_LIMIT_DAILY}</div>
<div style="color:#8b93a7;font-size:.75rem;">реальных кэфов сегодня</div></div>

<div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.10);
 border-radius:12px;padding:10px 14px;margin-bottom:10px;">
<div style="color:#7dd3fc;font-size:.7rem;text-transform:uppercase;
 font-weight:700;">Auto-settle</div>
<div style="font-size:1.3rem;font-weight:800;color:#34d399;">{s_rem} / {AUTO_SETTLE_LIMIT}</div></div>

<div style="background:rgba(139,92,246,.10);border:1px solid rgba(139,92,246,.35);
 border-radius:12px;padding:10px 14px;margin-bottom:10px;">
<div style="color:#c4b5fd;font-size:.7rem;text-transform:uppercase;
 font-weight:700;">LLM-аналитик</div>
<div style="font-size:1.3rem;font-weight:800;color:#c4b5fd;">{llm_rem} / {LLM_DAILY_LIMIT}</div></div>""",
        unsafe_allow_html=True)

    if db.SQLITE_BOOT_OK:
        clv = db.clv_summary()
        st.markdown(f"""
<div style="background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.10);
 border-radius:12px;padding:10px 14px;margin-bottom:10px;">
<div style="color:#7dd3fc;font-size:.7rem;text-transform:uppercase;
 font-weight:700;">CLV</div>
<div style="font-size:.9rem;color:#e6eaf2;">
Avg: <b style="color:{'#34d399' if clv['avg_clv']>0 else '#f87171'}">
{clv['avg_clv']*100:+.2f}%</b> · N={clv['n']}</div></div>""",
            unsafe_allow_html=True)

    if st.button("Сбросить счётчики"):
        usage.settle_reset()
        usage.llm_reset()
        usage.odds_reset()
        st.toast("Счётчики сброшены")
        st.rerun()

    st.success("Local mode")

    st.markdown("**Ключи**")
    odds_key = st.text_input(
        "The Odds API (500/мес бесплатно)",
        value=D.get("meta", {}).get("odds_api_key", ""),
        type="password",
        help="the-odds-api.com -> Dashboard -> API Key")
    if odds_key != D["meta"].get("odds_api_key", ""):
        D["meta"]["odds_api_key"] = odds_key
        usage.set_local_data(D)

    st.markdown("**ИИ-аналитик**")
    prov_keys = list(LLM_PROVIDERS.keys())
    cur = D["meta"].get("llm_provider", prov_keys[0])
    llm_prov = st.selectbox("Провайдер", prov_keys,
                            index=prov_keys.index(cur) if cur in prov_keys else 0)
    cfg = LLM_PROVIDERS[llm_prov]
    st.caption(f"Ключ: [{cfg['key_url']}]({cfg['key_url']})")
    llm_key = st.text_input("LLM API Key",
                            value=D["meta"].get("llm_api_key", ""),
                            type="password")
    llm_model = st.text_input("Модель (override)",
                              value=D["meta"].get("llm_model", ""),
                              placeholder=cfg["model"])
    if (llm_prov != D["meta"].get("llm_provider")
            or llm_key != D["meta"].get("llm_api_key", "")
            or llm_model != D["meta"].get("llm_model", "")):
        D["meta"]["llm_provider"] = llm_prov
        D["meta"]["llm_api_key"] = llm_key
        D["meta"]["llm_model"] = llm_model
        usage.set_local_data(D)

    st.markdown("**Минимальная вероятность**")
    min_prob = st.slider("", 50, 85, 55, 1, label_visibility="collapsed") / 100
    st.caption(f"Порог: {min_prob*100:.0f}%")
    kelly_frac = st.slider("Келли (доля)", 0.10, 0.40, 0.25, 0.05)
    matrix_n = st.slider("Матрица голов", 6, 15, 12, 1)

    st.info("Используйте проверенные кэфы. Paper-ставки не влияют на реальную статистику.")

    with st.expander(f"Ошибки ({len(ERR)})"):
        for line in ERR[-15:]:
            st.text(line)
    if st.button("Очистить лог"):
        ERR.clear()
        st.rerun()

    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False
    if not st.session_state.confirm_clear:
        if st.button("Очистить портфель"):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("Удалить ВСЕ ставки и сбросить банк?")
        cc1, cc2 = st.columns(2)
        if cc1.button("Да", key="confirm_yes"):
            D["bets"] = []
            D["cards"] = []
            D["bank"] = 10000.0
            D["stats"] = {"won": 0, "lost": 0, "profit": 0,
                          "push": 0, "void": 0}
            D["meta"]["initial_bank"] = 10000.0
            usage.set_local_data(D)
            db.invalidate_caches()
            st.session_state.confirm_clear = False
            st.rerun()
        if cc2.button("Нет", key="confirm_no"):
            st.session_state.confirm_clear = False
            st.rerun()


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Сканер", "Портфель", "Статистика", "Калькулятор PRO", "Бэктест"])

with tab1:
    scanner.render(min_prob, kelly_frac, matrix_n)
with tab2:
    portfolio.render()
with tab3:
    stats.render()
with tab4:
    calculator.render(matrix_n)
with tab5:
