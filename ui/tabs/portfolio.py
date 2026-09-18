"""ui/tabs/portfolio.py — вкладка Портфель."""
from __future__ import annotations
import csv, io, re
from datetime import datetime

import streamlit as st

from storage import sqlite_store as db
from storage import usage
from ui.cards import translate_match


def _apply_settle(D, idx, outcome, score=None):
    D2 = dict(D)
    bets = list(D2["bets"])
    if idx < 0 or idx >= len(bets):
        return D
    b = dict(bets[idx])
    if b.get("status") != "pending":
        return D
    if score:
        b["score"] = score
    stats = dict(D2.get("stats", {}))
    if outcome == "push":
        b["status"] = "push"
        D2["bank"] = D2["bank"] + b["stake"]
        stats["push"] = stats.get("push", 0) + 1
    elif outcome == "void":
        b["status"] = "void"
        D2["bank"] = D2["bank"] + b["stake"]
        stats["void"] = stats.get("void", 0) + 1
    elif outcome == "won":
        b["status"] = "won"
        D2["bank"] = D2["bank"] + b["stake"] * b["odds"]
        stats["won"] = stats.get("won", 0) + 1
        stats["profit"] = stats.get("profit", 0) + b["stake"] * (b["odds"] - 1)
    elif outcome == "lost":
        b["status"] = "lost"
        stats["lost"] = stats.get("lost", 0) + 1
        stats["profit"] = stats.get("profit", 0) - b["stake"]
    bets[idx] = b
    D2["bets"] = bets
    D2["stats"] = stats
    return D2


def _cancel_bet(D, idx):
    D2 = dict(D)
    bets = list(D2["bets"])
    if idx < 0 or idx >= len(bets):
        return D
    b = bets[idx]
    if b.get("status") != "pending":
        return D
    D2["bank"] = D2["bank"] + float(b.get("stake") or 0)
    bets.pop(idx)
    D2["bets"] = bets
    return D2


def _persist(D):
    usage.set_local_data(D)
    if db.SQLITE_BOOT_OK:
        db.log_bank(D.get("bank", 10000.0), event="manual_settle")
        db.invalidate_caches()


def render():
    D = st.session_state.data
    st.header("Портфель")
    if not D["bets"]:
        st.warning("Пусто. После СКАНа ставки появятся здесь.")
        return

    pending = sum(1 for b in D["bets"]
                  if isinstance(b, dict) and b.get("status") == "pending")
    st.caption(f"Всего: {len(D['bets'])} · В работе: {pending}")

    if st.button("Экспорт CSV"):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Match", "League", "Pick", "Market", "Odds", "OddsSource",
                    "Mode", "Stake", "Prob", "EV", "Status", "Score", "Date"])
        for b in D["bets"]:
            if not isinstance(b, dict):
                continue
            md = b.get("match_ru") or translate_match(b.get("match", ""))
            w.writerow([md, b.get("league", ""), b.get("pick", ""),
                        b.get("market", ""), b.get("odds", ""),
                        b.get("odds_source", ""), b.get("mode", ""),
                        b.get("stake", ""), b.get("prob", ""), b.get("ev", ""),
                        b.get("status", ""), b.get("score", ""), b.get("date", "")])
        st.download_button(
            "Скачать", buf.getvalue(),
            file_name=f"neuro_bets_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv")

    for i, b in enumerate(D["bets"]):
        if not isinstance(b, dict):
            continue
        st_ = b.get("status", "pending")
        icon = {"pending": "⏳", "won": "🟢", "lost": "🔴",
                "push": "⚪", "void": "⬜"}.get(st_, "⏳")
        score = f" — счёт {b.get('score','')}" if b.get("score") else ""
        prob = float(b.get("prob") or 0)
        odds = float(b.get("odds") or 1.0)
        stake = float(b.get("stake") or 0.0)
        ev = float(b.get("ev") or 0)
        md = b.get("match_ru") or translate_match(b.get("match", "—"))
        mode = b.get("mode", "")
        mode_badge = {"paper": "paper", "real": "real"}.get(mode, "")

        st.markdown(f"""
<div class="betcard {st_}" style="padding:14px 18px;">
 <div style="display:flex;justify-content:space-between;align-items:flex-start;">
  <div><div style="font-size:1rem;font-weight:800;color:#fff;">{icon} {md}{score}</div>
   <div style="color:#8b93a7;font-size:.78rem;margin-top:4px;">
     {b.get('league','—')} · {b.get('date','—')} · {mode_badge}</div></div>
  <div style="text-align:right;">
   <div style="color:#fbbf24;font-size:1.1rem;font-weight:800;">{b.get('pick','—')}</div>
   <div style="color:#34d399;font-weight:700;">P {prob*100:.0f}% · EV {ev*100:+.1f}%</div>
  </div></div>
 <div style="color:#c9d2e3;font-size:.82rem;margin-top:8px;">
   Кэф: <b>{odds:.2f}</b> · Ставка: <b>{stake:.2f} у.е.</b></div></div>""",
            unsafe_allow_html=True)

        if b.get("status") == "pending":
            cc = st.columns([1, 1, 1, 1])
            sin = cc[0].text_input("Счёт", key=f"sc{i}",
                                   label_visibility="collapsed", placeholder="2:1")
            sc = sin.strip() if re.match(r"^\d+\s*:\s*\d+$", sin.strip()) else None
            if cc[1].button("OK", key=f"w{i}"):
                st.session_state.data = _apply_settle(D, i, "won", score=sc)
                _persist(st.session_state.data)
                st.rerun()
            if cc[2].button("NO", key=f"l{i}"):
                st.session_state.data = _apply_settle(D, i, "lost", score=sc)
                _persist(st.session_state.data)
                st.rerun()
            if cc[3].button("X", key=f"c{i}"):
                st.session_state.data = _cancel_bet(D, i)
                _persist(st.session_state.data)
                st.toast("Ставка отменена")
                st.rerun()
