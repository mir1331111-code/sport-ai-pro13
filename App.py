"""App.py — NEURO BET PRO v13.1. Точка входа Streamlit."""
from __future__ import annotations

try:
    from context_football import analyze_match_context, render_context_flags
    HAS_CONTEXT = True
except Exception:
    HAS_CONTEXT = False

import os
import json
import re
from datetime import datetime, timedelta

import streamlit as st
import requests as _requests

for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
           "ALL_PROXY", "all_proxy", "FTP_PROXY", "ftp_proxy"]:
    os.environ.pop(_k, None)

st.set_page_config(page_title="NEURO BET PRO",
                   page_icon="🏟", layout="wide",
                   initial_sidebar_state="expanded")

from config import (APP_VERSION, LLM_PROVIDERS, DATA_VERSION)
from storage import sqlite_store as db
from storage import usage
from data.sources import parse_date, fdorg_match_result
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

# ==================== BOOT ====================
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


# ==================== AUTO-SETTLE ====================
def _tsdb_find_result(home: str, away: str, date_iso: str) -> dict | None:
    """TheSportsDB fallback: поиск результата по командам + дате."""
    if not home or not away or not date_iso:
        return None
    try:
        r = _requests.get(
            "https://www.thesportsdb.com/api/v1/json/3/eventsday.php",
            params={"d": date_iso, "s": "Soccer"}, timeout=10)
        if r.status_code != 200:
            return None
        events = (r.json() or {}).get("events") or []
        hn = re.sub(r"[^a-z0-9]", "", home.lower())
        an = re.sub(r"[^a-z0-9]", "", away.lower())
        for ev in events:
            eh = re.sub(r"[^a-z0-9]", "", (ev.get("strHomeTeam") or "").lower())
            ea = re.sub(r"[^a-z0-9]", "", (ev.get("strAwayTeam") or "").lower())
            if (eh == hn and ea == an) or \
               ((eh in hn or hn in eh) and (ea in an or an in ea)):
                hg = ev.get("intHomeScore")
                ag = ev.get("intAwayScore")
                st_status = ev.get("strStatus", "")
                if hg is not None and ag is not None and \
                   st_status in ("Match Finished", "FT", "AET", "PEN", "Result Pending"):
                    return {"home": int(hg), "away": int(ag)}
    except Exception:
        pass
    return None


def _auto_settle(D):
    token = D.get("meta", {}).get("fdorg_token", "").strip()
    if not token:
        return D, 0
    D2 = json.loads(json.dumps(D))
    bets = D2["bets"]
    changed = 0
    now = datetime.now()

    for idx, b in enumerate(bets):
        if not isinstance(b, dict) or b.get("status") != "pending":
            continue
        fid = b.get("fixture_id")
        if not fid:
            continue
        bd = parse_date(b.get("date_iso") or b.get("date") or "")
        if bd and bd > now:
            continue

        # Источник 1: football-data.org
        res = fdorg_match_result(fid, token)

        # Источник 2: TheSportsDB (бесплатно, без ключа)
        if not res and bd:
            parts = (b.get("match") or "").split(" vs ")
            if len(parts) == 2:
                res = _tsdb_find_result(parts[0], parts[1], bd.strftime("%Y-%m-%d"))

        if not res:
            continue

        outcome, _reason = determine_outcome(
            b.get("market"), b.get("pick"), res["home"], res["away"])
        if outcome is None:
            continue

        b["score"] = f"{res['home']}:{res['away']}"
        if outcome == "push":
            b["status"] = "push"
            D2["bank"] = D2.get("bank", 10000.0) + b["stake"]
            D2["stats"]["push"] = D2["stats"].get("push", 0) + 1
        elif outcome == "void":
            b["status"] = "void"
            D2["bank"] = D2.get("bank", 10000.0) + b["stake"]
            D2["stats"]["void"] = D2["stats"].get("void", 0) + 1
        elif outcome == "won":
            b["status"] = "won"
            D2["bank"] = D2.get("bank", 10000.0) + b["stake"] * b["odds"]
            D2["stats"]["won"] = D2["stats"].get("won", 0) + 1
            D2["stats"]["profit"] = (D2["stats"].get("profit", 0)
                                     + b["stake"] * (b["odds"] - 1))
        elif outcome == "lost":
            b["status"] = "lost"
            D2["stats"]["lost"] = D2["stats"].get("lost", 0) + 1
            D2["stats"]["profit"] = D2["stats"].get("profit", 0) - b["stake"]
        changed += 1

    # Void через 48ч
    for idx, b in enumerate(bets):
        if not isinstance(b, dict) or b.get("status") != "pending":
            continue
        bd = parse_date(b.get("date_iso") or b.get("date") or "")
        if bd and now - bd > timedelta(hours=48):
            b["status"] = "void"
            b["score"] = "void (no result)"
            D2["bank"] = D2.get("bank", 10000.0) + b["stake"]
            D2["stats"]["void"] = D2["stats"].get("void", 0) + 1
            changed += 1

    return D2, changed


_now = datetime.now().timestamp()
_last = st.session_state.get("_last_auto_settle_ts", 0)
if _now - _last > 3600:
    D2, n = _auto_settle(D)
    if n > 0:
        st.session_state.data = D2
        usage.set_local_data(D2)
        db.log_bank(D2.get("bank", 10000.0), event="auto_settle")
        db.invalidate_caches()
        D = D2
        st.toast(f"✅ Авто-закрыто {n} ставок")
    st.session_state["_last_auto_settle_ts"] = _now

pending_count = sum(1 for b in D["bets"]
                    if isinstance(b, dict) and b.get("status") == "pending")

# ==================== HERO ====================
st.markdown(f"""
<div class="hero"><h1>NEURO BET PRO</h1>
<p>v{APP_VERSION} · 100% FREE · ИИ-аналитик · SQLite · CLV · 🔍 Контекст</p>
<div class="kpis">
 <div class="kpi"><div class="t">Банкролл</div>
  <div class="v y">{D['bank']:.0f} у.е.</div></div>
 <div class="kpi"><div class="t">В работе</div><div class="v">{pending_count}</div></div>
 <div class="kpi"><div class="t">Всего</div><div class="v">{len(D['bets'])}</div></div>
 <div class="kpi"><div class="t">Ошибок</div>
  <div class="v {'r' if ERR else 'g'}">{len(ERR)}</div></div>
</div></div>""", unsafe_allow_html=True)

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:800;color:#e6eaf2;"
        "margin-bottom:12px;'>🔑 API ключи</div>",
        unsafe_allow_html=True)

    fdorg_token = st.text_input(
        "football-data.org token",
        value=D.get("meta", {}).get("fdorg_token", ""),
        type="password",
        help="Регистрация: football-data.org/client/register")
    if fdorg_token != D["meta"].get("fdorg_token", ""):
        D["meta"]["fdorg_token"] = fdorg_token
        usage.set_local_data(D)

    odds_key = st.text_input(
        "The Odds API key (опционально)",
        value=D.get("meta", {}).get("odds_api_key", ""),
        type="password",
        help="the-odds-api.com → Dashboard → API Key")
    if odds_key != D["meta"].get("odds_api_key", ""):
        D["meta"]["odds_api_key"] = odds_key
        usage.set_local_data(D)

    st.markdown(
        "<div style='font-size:.72rem;color:#8b93a7;margin-top:14px;"
        "margin-bottom:6px;'>🤖 LLM-аналитик</div>",
        unsafe_allow_html=True)
    prov_keys = list(LLM_PROVIDERS.keys())
    cur = D["meta"].get("llm_provider", prov_keys[0])
    llm_prov = st.selectbox("Провайдер", prov_keys,
                            index=prov_keys.index(cur) if cur in prov_keys else 0)
    llm_key = st.text_input(
        "LLM key",
        value=D["meta"].get("llm_api_key", ""),
        type="password",
        help=LLM_PROVIDERS[llm_prov]["key_url"])
    if (llm_prov != D["meta"].get("llm_provider")
            or llm_key != D["meta"].get("llm_api_key", "")):
        D["meta"]["llm_provider"] = llm_prov
        D["meta"]["llm_api_key"] = llm_key
        usage.set_local_data(D)

    st.markdown("<hr style='border-color:rgba(255,255,255,.08);"
                "margin:18px 0;'>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:.72rem;color:#8b93a7;"
        "margin-bottom:8px;'>🎯 Параметры скана</div>",
        unsafe_allow_html=True)
    min_prob = st.slider("Мин. вероятность %", 50, 85, 55, 1) / 100
    kelly_frac = st.slider("Kelly доля", 0.05, 0.40, 0.25, 0.05)
    matrix_n = st.slider("Матрица голов", 6, 15, 12, 1)

    st.markdown("<hr style='border-color:rgba(255,255,255,.08);"
                "margin:18px 0;'>", unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:.75rem;color:#8b93a7;"
        "margin-bottom:8px;'>💾 Резервная копия</div>",
        unsafe_allow_html=True)

    _backup_json = json.dumps(D, ensure_ascii=False, indent=2, default=str)
    st.download_button(
        "📥 Скачать данные",
        _backup_json,
        file_name=f"neuro_data_{datetime.now():%Y%m%d_%H%M}.json",
        mime="application/json",
        use_container_width=True,
        help="Сохрани перед выключением ПК")

    _uploaded = st.file_uploader(
        "📤 Загрузить данные",
        type=["json"],
        key="restore_upload",
        label_visibility="collapsed")
    if _uploaded is not None:
        try:
            _restored = json.loads(_uploaded.read().decode("utf-8"))
            if isinstance(_restored, dict) and "data" in _restored:
                st.session_state.data = _restored["data"]
                usage.set_local_data(_restored["data"])
                st.success("✅ Данные восстановлены!")
                st.rerun()
            elif isinstance(_restored, dict):
                st.session_state.data = _restored
                usage.set_local_data(_restored)
                st.success("✅ Данные восстановлены!")
                st.rerun()
            else:
                st.error("❌ Неверный формат файла")
        except Exception as e:
            st.error(f"❌ Ошибка: {e}")

    st.markdown("<hr style='border-color:rgba(255,255,255,.08);"
                "margin:18px 0;'>", unsafe_allow_html=True)

    if st.button("♻️ Сбросить счётчики", use_container_width=True):
        usage.settle_reset()
        usage.llm_reset()
        usage.odds_reset()
        usage.fdorg_reset()
        st.toast("Счётчики сброшены")
        st.rerun()

    if "confirm_clear" not in st.session_state:
        st.session_state.confirm_clear = False
    if not st.session_state.confirm_clear:
        if st.button("🗑️ Очистить портфель", use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()
    else:
        st.warning("Удалить ВСЕ ставки?")
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

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏟 Сканер", "💼 Портфель", "📈 Статистика",
     "🧮 Калькулятор", "🧪 Бэктест"])

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
