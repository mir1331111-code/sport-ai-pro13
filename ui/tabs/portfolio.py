"""ui/tabs/portfolio.py — Портфель с сортировкой по дням."""
from __future__ import annotations
import csv, io, re
from datetime import datetime, timedelta

import streamlit as st

from storage import sqlite_store as db
from storage import usage
from ui.cards import translate_match, day_label


def _bet_date(b: dict):
    """Дата ставки (по date_iso)."""
    try:
        iso = b.get("date_iso") or ""
        if iso:
            return datetime.strptime(iso, "%Y-%m-%d")
    except Exception:
        pass
    return datetime(2099, 1, 1)


def _bet_pick_label(b: dict) -> str:
    """Понятная подпись ставки: '🏠 Победа Arsenal' или '⚽ Больше 2.5'."""
    pick = b.get("pick") or "—"
    match_ru = b.get("match_ru") or b.get("match") or "—"
    parts = match_ru.split(" — ") if " — " in match_ru else match_ru.split(" vs ")
    home = parts[0].strip() if parts else "?"
    away = parts[-1].strip() if len(parts) > 1 else "?"

    mapping = {
        "П1": f"🏠 Победа {home}",
        "X": "🤝 Ничья",
        "П2": f"✈️ Победа {away}",
        "ТБ 2.5": "⚽ Больше 2.5 голов",
        "ТМ 2.5": "🛡️ Меньше 2.5 голов",
        "BTTS да": "⚽🤝 Обе забьют — Да",
        "BTTS нет": "🛡️ Обе забьют — Нет",
        "1X": f"🏠🤝 {home} или ничья",
        "X2": f"🤝✈️ Ничья или {away}",
        "12": f"🏠✈️ {home} или {away}",
    }
    return mapping.get(pick, pick)


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


def _render_bet(b, D, index):
    """Отрисовка одной карточки ставки с кнопками управления."""
    st_ = b.get("status", "pending")
    icon = {"pending": "⏳", "won": "🟢", "lost": "🔴",
            "push": "⚪", "void": "⬜"}.get(st_, "⏳")

    prob = float(b.get("prob") or 0)
    odds = float(b.get("odds") or 1.0)
    stake = float(b.get("stake") or 0.0)
    ev = float(b.get("ev") or 0)
    match_ru = b.get("match_ru") or translate_match(b.get("match", "—"))
    pick_label = _bet_pick_label(b)
    league = b.get("league", "—")
    date_str = b.get("date", "—")
    time_str = b.get("match_time", "")
    mode = b.get("mode", "")
    mode_badge = {"paper": "📐 paper", "real": "🎯 real"}.get(mode, "")

    score_html = ""
    if b.get("score"):
        score_html = (f"<span style='background:rgba(52,211,153,.25);"
                      f"color:#6ee7b7;padding:2px 10px;border-radius:9px;"
                      f"font-weight:900;margin-left:8px;'>"
                      f"{b.get('score')}</span>")

    pnl_html = ""
    if st_ == "won":
        pr = stake * (odds - 1)
        pnl_html = (f"<span style='color:#34d399;font-weight:700;'>"
                    f"+{pr:.2f} у.е.</span>")
    elif st_ == "lost":
        pnl_html = (f"<span style='color:#f87171;font-weight:700;'>"
                    f"−{stake:.2f} у.е.</span>")
    elif st_ in ("push", "void"):
        pnl_html = f"<span style='color:#94a3b8;'>возврат {stake:.2f}</span>"

    st.markdown(f"""
<div class="betcard {st_}" style="padding:16px 20px;">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:16px;">
    <div style="flex:1;">
      <div style="font-size:1.05rem;font-weight:800;color:#fff;margin-bottom:6px;">
        {icon} {match_ru}{score_html}
      </div>
      <div style="color:#a5f3fc;font-size:.9rem;font-weight:700;margin-bottom:4px;">
        {pick_label}
      </div>
      <div style="color:#8b93a7;font-size:.75rem;">
        {league} · 📅 {date_str}{(" " + time_str) if time_str else ""} · {mode_badge}
      </div>
    </div>
    <div style="text-align:right;flex-shrink:0;">
      <div style="color:#34d399;font-weight:700;font-size:.95rem;">
        P {prob*100:.0f}% · EV {ev*100:+.1f}%
      </div>
      <div style="color:#c9d2e3;font-size:.85rem;margin-top:4px;">
        Кэф <b style="color:#fbbf24;font-family:JetBrains Mono,monospace;">{odds:.2f}</b>
        · <b style="font-family:JetBrains Mono,monospace;">{stake:.2f}</b> у.е.
      </div>
      <div style="margin-top:6px;font-size:.85rem;">{pnl_html}</div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    if b.get("status") == "pending":
        cc = st.columns([1, 1, 1, 1])
        sin = cc[0].text_input("Счёт", key=f"sc_{index}",
                                label_visibility="collapsed",
                                placeholder="2:1")
        sc = sin.strip() if re.match(r"^\d+\s*:\s*\d+$", sin.strip()) else None
        if cc[1].button("✅", key=f"w_{index}", help="Выиграла"):
            st.session_state.data = _apply_settle(D, index, "won", score=sc)
            _persist(st.session_state.data)
            st.rerun()
        if cc[2].button("❌", key=f"l_{index}", help="Проиграла"):
            st.session_state.data = _apply_settle(D, index, "lost", score=sc)
            _persist(st.session_state.data)
            st.rerun()
        if cc[3].button("🚫", key=f"c_{index}", help="Отменить (вернуть stake)"):
            st.session_state.data = _cancel_bet(D, index)
            _persist(st.session_state.data)
            st.toast("Ставка отменена, stake возвращён")
            st.rerun()


def render():
    D = st.session_state.data
    st.header("💼 Портфель")
    if not D["bets"]:
        st.warning("Пусто. После СКАНа ставки появятся здесь.")
        return

    # ==================== ФИЛЬТРЫ ====================
    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

    status_filter = c1.selectbox(
        "Статус",
        ["Все", "⏳ В работе", "🟢 Выигранные", "🔴 Проигранные",
         "⚪ Возврат", "⬜ Void"],
        key="pf_status")

    sort_mode = c2.selectbox(
        "Сортировка",
        ["По дате матча (сначала ближайшие)",
         "По дате матча (сначала поздние)",
         "По дате добавления (новые)",
         "По вероятности",
         "По EV",
         "По кэфу"],
        key="pf_sort")

    group_by_day = c3.checkbox("Группировать по дням", True, key="pf_group")

    mode_filter = c4.selectbox(
        "Режим",
        ["Все", "🎯 Real (реальные)", "📐 Paper (оценочные)"],
        key="pf_mode")

    # ==================== ФИЛЬТРАЦИЯ ====================
    # Сохраняем (bet_dict, original_index) чтобы знать оригинальный idx
    indexed = []
    for idx, b in enumerate(D["bets"]):
        if not isinstance(b, dict):
            continue
        st_ = b.get("status", "pending")
        mode = b.get("mode", "paper")

        if status_filter == "⏳ В работе" and st_ != "pending":
            continue
        if status_filter == "🟢 Выигранные" and st_ != "won":
            continue
        if status_filter == "🔴 Проигранные" and st_ != "lost":
            continue
        if status_filter == "⚪ Возврат" and st_ != "push":
            continue
        if status_filter == "⬜ Void" and st_ != "void":
            continue

        if mode_filter == "🎯 Real (реальные)" and mode != "real":
            continue
        if mode_filter == "📐 Paper (оценочные)" and mode != "paper":
            continue

        indexed.append((b, idx))

    if not indexed:
        st.info("Нет ставок по выбранным фильтрам.")
        return

    # ==================== СОРТИРОВКА ====================
    if sort_mode.startswith("По дате матча (сначала ближайшие)"):
        indexed.sort(key=lambda t: _bet_date(t[0]))
    elif sort_mode.startswith("По дате матча (сначала поздние)"):
        indexed.sort(key=lambda t: _bet_date(t[0]), reverse=True)
    elif sort_mode.startswith("По дате добавления"):
        indexed.sort(key=lambda t: t[0].get("date_time") or "", reverse=True)
    elif sort_mode == "По вероятности":
        indexed.sort(key=lambda t: -(t[0].get("prob") or 0))
    elif sort_mode == "По EV":
        indexed.sort(key=lambda t: -(t[0].get("ev") or 0))
    elif sort_mode == "По кэфу":
        indexed.sort(key=lambda t: -(t[0].get("odds") or 0))

    # ==================== KPI ====================
    total_stake = sum(float(b.get("stake") or 0) for b, _ in indexed)
    pending_cnt = sum(1 for b, _ in indexed if b.get("status") == "pending")
    won_cnt = sum(1 for b, _ in indexed if b.get("status") == "won")
    lost_cnt = sum(1 for b, _ in indexed if b.get("status") == "lost")
    profit = sum(
        (float(b.get("stake") or 0) * (float(b.get("odds") or 1) - 1))
        if b.get("status") == "won"
        else (-float(b.get("stake") or 0) if b.get("status") == "lost" else 0)
        for b, _ in indexed
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Всего", len(indexed))
    k2.metric("В работе", pending_cnt)
    k3.metric("Won / Lost", f"{won_cnt} / {lost_cnt}")
    k4.metric("Profit", f"{profit:+.2f}")

    # ==================== ЭКСПОРТ CSV ====================
    if st.button("📥 Экспорт CSV"):
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Дата", "Лига", "Матч", "Рынок", "Ставка",
                    "Кэф", "Стейк", "P", "EV", "Режим", "Статус", "Счёт"])
        for b, _ in indexed:
            md = b.get("match_ru") or translate_match(b.get("match", ""))
            w.writerow([
                b.get("date_iso", ""), b.get("league", ""), md,
                b.get("market", ""), _bet_pick_label(b),
                b.get("odds", ""), b.get("stake", ""),
                f"{float(b.get('prob') or 0)*100:.0f}%",
                f"{float(b.get('ev') or 0)*100:+.1f}%",
                b.get("mode", ""), b.get("status", ""), b.get("score", ""),
            ])
        st.download_button(
            "⬇️ Скачать neuro_bets.csv",
            buf.getvalue(),
            file_name=f"neuro_bets_{datetime.now():%Y%m%d_%H%M}.csv",
            mime="text/csv")

    st.divider()

    # ==================== РЕНДЕР С ГРУППИРОВКОЙ ====================
    if group_by_day:
        grouped = {}
        for b, idx in indexed:
            dt = _bet_date(b)
            label = day_label(dt)
            grouped.setdefault(label, []).append((b, idx))

        sorted_groups = sorted(
            grouped.items(),
            key=lambda kv: _bet_date(kv[1][0][0]))

        for label, items in sorted_groups:
            g_pending = sum(1 for b, _ in items if b.get("status") == "pending")
            g_won = sum(1 for b, _ in items if b.get("status") == "won")
            g_lost = sum(1 for b, _ in items if b.get("status") == "lost")
            g_stake = sum(float(b.get("stake") or 0) for b, _ in items)

            st.markdown(
                f"<div style='margin-top:24px;margin-bottom:14px;"
                f"padding:12px 20px;"
                f"background:linear-gradient(90deg,"
                f"rgba(34,211,238,.14),rgba(139,92,246,.08));"
                f"border-left:4px solid #22d3ee;border-radius:10px;'>"
                f"<span style='color:#fff;font-weight:900;font-size:1.1rem;"
                f"letter-spacing:1px;'>{label}</span>"
                f"<span style='color:#8b93a7;font-size:.85rem;margin-left:12px;'>"
                f"{len(items)} ставок · ⏳ {g_pending} · "
                f"🟢 {g_won} · 🔴 {g_lost} · 💰 {g_stake:.0f} у.е."
                f"</span></div>",
                unsafe_allow_html=True)

            for b, original_idx in items:
                _render_bet(b, D, original_idx)
    else:
        for b, original_idx in indexed:
            _render_bet(b, D, original_idx)
