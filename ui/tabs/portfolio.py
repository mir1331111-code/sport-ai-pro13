"""ui/tabs/portfolio.py — Портфель с сортировкой по дням."""
from __future__ import annotations
import csv, io, re, hashlib
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


def _bet_uid(b: dict) -> str:
    """Уникальный стабильный ID ставки для widget keys."""
    base = (f"{b.get('match','')}|{b.get('pick','')}|"
            f"{b.get('date_iso','')}|{b.get('market','')}|"
            f"{b.get('stake','')}|{b.get('odds','')}")
    return hashlib.md5(base.encode()).hexdigest()[:10]


def _safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


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
        D2["bank"] = _safe_float(D2.get("bank"), 0.0) + _safe_float(b.get("stake"), 0.0)
        stats["push"] = stats.get("push", 0) + 1
    elif outcome == "void":
        b["status"] = "void"
        D2["bank"] = _safe_float(D2.get("bank"), 0.0) + _safe_float(b.get("stake"), 0.0)
        stats["void"] = stats.get("void", 0) + 1
    elif outcome == "won":
        b["status"] = "won"
        stake = _safe_float(b.get("stake"), 0.0)
        odds = _safe_float(b.get("odds"), 0.0)
        D2["bank"] = _safe_float(D2.get("bank"), 0.0) + stake * odds
        stats["won"] = stats.get("won", 0) + 1
        stats["profit"] = stats.get("profit", 0) + stake * (odds - 1)
    elif outcome == "lost":
        b["status"] = "lost"
        stats["lost"] = stats.get("lost", 0) + 1
        stats["profit"] = stats.get("profit", 0) - _safe_float(b.get("stake"))
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
    D2["bank"] = _safe_float(D2.get("bank"), 0.0) + _safe_float(b.get("stake"), 0.0)
    bets.pop(idx)
    D2["bets"] = bets
    return D2


def _persist(D):
    usage.set_local_data(D)
    if db.SQLITE_BOOT_OK:
        for bet in D.get("bets", []):
            if not isinstance(bet, dict):
                continue
            bet_id = bet.get("id")
            if bet_id is None:
                db.update_bet_by_identity(
                    bet.get("match"),
                    bet.get("market"),
                    bet.get("pick"),
                    bet.get("date_iso"),
                    status=bet.get("status", "pending"),
                    score=bet.get("score"),
                    settled_at=bet.get("settled_at"),
                )
                continue
            db.update_bet(
                int(bet_id),
                status=bet.get("status", "pending"),
                score=bet.get("score"),
                settled_at=bet.get("settled_at"),
            )
        db.log_bank(_safe_float(D.get("bank"), 10000.0), event="manual_settle")
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
        # Уникальный key через хэш ставки (не зависит от index)
        _ukey = _bet_uid(b)
        cc = st.columns([1, 1, 1, 1])
        sin = cc[0].text_input("Счёт", key=f"sc_{_ukey}",
                                label_visibility="collapsed",
                                placeholder="2:1")
        sc = sin.strip() if re.match(r"^\d+\s*:\s*\d+$", sin.strip()) else None
        if cc[1].button("✅", key=f"w_{_ukey}", help="Выиграла"):
            st.session_state.data = _apply_settle(D, index, "won", score=sc)
            _persist(st.session_state.data)
            st.rerun()
        if cc[2].button("❌", key=f"l_{_ukey}", help="Проиграла"):
            st.session_state.data = _apply_settle(D, index, "lost", score=sc)
            _persist(st.session_state.data)
            st.rerun()
        if cc[3].button("🚫", key=f"c_{_ukey}", help="Отменить (вернуть stake)"):
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

    # ==================== RISK MONITOR ====================
    bank_now = float(D.get("bank") or 0.0)
    pending_exposure = sum(
        float(b.get("stake") or 0)
        for b, _ in indexed
        if b.get("status") == "pending"
    )
    exposure_pct = (pending_exposure / bank_now * 100) if bank_now > 0 else 0.0
    league_pending = {}
    for b, _ in indexed:
        if b.get("status") != "pending":
            continue
        league = str(b.get("league") or b.get("div") or "OTHER")
        league_pending[league] = league_pending.get(league, 0.0) + float(b.get("stake") or 0)
    largest_league = max(league_pending.items(), key=lambda x: x[1], default=("—", 0.0))
    largest_league_pct = (
        largest_league[1] / bank_now * 100 if bank_now > 0 else 0.0
    )

    st.markdown("#### 🛡️ Risk Monitor")
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Экспозиция", f"{pending_exposure:.0f}", f"{exposure_pct:.1f}% банка")
    r2.metric("Открытых ставок", pending_cnt)
    r3.metric("Макс. лига", f"{largest_league[1]:.0f}", largest_league[0])
    r4.metric("Доля макс. лиги", f"{largest_league_pct:.1f}%")
    if exposure_pct >= 20:
        st.warning("⚠️ Текущая открытая экспозиция достигла 20% банка. Новые сигналы Risk Guard может отклонять.")

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

    # ==================== КАЛЕНДАРЬ ПОРТФЕЛЯ ====================
    from calendar import Calendar
    from datetime import date as _date

    dated = []
    undated = []
    for b, original_idx in indexed:
        iso = str(b.get("date_iso") or "").strip()
        try:
            day = datetime.strptime(iso[:10], "%Y-%m-%d").date()
            dated.append((day, b, original_idx))
        except (TypeError, ValueError):
            undated.append((b, original_idx))

    if "pf_calendar_month" not in st.session_state:
        today = _date.today()
        st.session_state.pf_calendar_month = (today.year, today.month)
    year, month = st.session_state.pf_calendar_month

    nav_l, nav_title, nav_r = st.columns([1, 4, 1])
    if nav_l.button("◀ Предыдущий", key="pf_cal_prev"):
        month -= 1
        if month < 1:
            year, month = year - 1, 12
        st.session_state.pf_calendar_month = (year, month)
        st.rerun()
    nav_title.markdown(f"### 🗓️ {['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Ноябрь','Декабрь'][month-1]} {year}")
    if nav_r.button("Следующий ▶", key="pf_cal_next"):
        month += 1
        if month > 12:
            year, month = year + 1, 1
        st.session_state.pf_calendar_month = (year, month)
        st.rerun()

    by_day = {}
    for day, b, original_idx in dated:
        by_day.setdefault(day, []).append((b, original_idx))
    weekdays = st.columns(7)
    for col, name in zip(weekdays, ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]):
        col.markdown(f"**{name}**")

    cal = Calendar(firstweekday=0)
    for week in cal.monthdayscalendar(year, month):
        cols = st.columns(7)
        for col, day_num in zip(cols, week):
            if not day_num:
                col.markdown("&nbsp;", unsafe_allow_html=True)
                continue
            day = _date(year, month, day_num)
            day_items = by_day.get(day, [])
            if day_items:
                pending = sum(1 for b, _ in day_items if b.get("status") == "pending")
                won = sum(1 for b, _ in day_items if b.get("status") == "won")
                lost = sum(1 for b, _ in day_items if b.get("status") == "lost")
                label = f"{day_num} · {len(day_items)}\n⏳{pending} 🟢{won} 🔴{lost}"
            else:
                label = str(day_num)
            if col.button(label, key=f"pf_cal_day_{year}_{month}_{day_num}",
                           use_container_width=True,
                           type="primary" if st.session_state.get("pf_selected_day") == day.isoformat() else "secondary"):
                st.session_state.pf_selected_day = day.isoformat()
                st.rerun()

    selected = st.session_state.get("pf_selected_day")
    if selected:
        try:
            selected_day = datetime.strptime(selected, "%Y-%m-%d").date()
        except ValueError:
            selected_day = _date.today()
    else:
        today = _date.today()
        selected_day = today if today in by_day else (min(by_day) if by_day else today)
        st.session_state.pf_selected_day = selected_day.isoformat()

    st.divider()
    st.subheader(f"📌 События за {selected_day.strftime('%d.%m.%Y')}")
    day_items = by_day.get(selected_day, [])
    if day_items:
        day_items = sorted(day_items, key=lambda item: str(item[0].get("match_time") or "99:99"))
        d_pending = sum(1 for b, _ in day_items if b.get("status") == "pending")
        d_won = sum(1 for b, _ in day_items if b.get("status") == "won")
        d_lost = sum(1 for b, _ in day_items if b.get("status") == "lost")
        d_stake = sum(_safe_float(b.get("stake")) for b, _ in day_items)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Событий", len(day_items))
        m2.metric("В работе", d_pending)
        m3.metric("Выигрыш / проигрыш", f"{d_won} / {d_lost}")
        m4.metric("Сумма ставок", f"{d_stake:.2f}")
        for b, original_idx in day_items:
            _render_bet(b, D, original_idx)
    else:
        st.info("На выбранную дату ставок нет. Нажми на другой день в календаре.")

    if undated:
        with st.expander(f"⚠️ Ставки без корректной даты ({len(undated)})", expanded=False):
            st.caption("Эти записи нельзя разместить в календаре, пока у них нет даты матча.")
            for b, original_idx in undated:
                _render_bet(b, D, original_idx)

    with st.expander("📋 Все ставки по выбранным фильтрам", expanded=False):
        for b, original_idx in sorted(indexed, key=lambda t: _bet_date(t[0])):
            _render_bet(b, D, original_idx)
