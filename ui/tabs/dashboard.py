"""Premium data-first overview dashboard for NEURO BET PRO."""
from __future__ import annotations

import html
import math
from collections import Counter
from datetime import datetime

import streamlit as st
from betting.ranking import model_favorite
from betting.risk import risk_snapshot, MAX_MARKET_STAKE_PCT, MAX_MATCH_EXPOSURE_PCT, MAX_DAILY_EXPOSURE_PCT
from data.fixture_time import app_timezone


def esc(value):
    return html.escape(str(value or ""), quote=True)


def num(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def favorite(card):
    try:
        return model_favorite(card)
    except Exception:
        return None


def has_prices(card):
    prices = card.get("market_odds") or {}
    return (
        isinstance(prices, dict)
        and str(card.get("odds_source") or "").casefold() not in {"", "no_market", "model_only"}
        and all(1.01 < (num(prices.get(k)) or 0) <= 100 for k in ("П1", "X", "П2"))
    )


def context_summary(card):
    context = card.get("deep_analysis_context")
    if not isinstance(context, dict):
        context = card.get("api_context")
    context = context if isinstance(context, dict) else {}
    lineup = context.get("lineup") if isinstance(context.get("lineup"), dict) else {}
    home = lineup.get("home") if isinstance(lineup.get("home"), dict) else {}
    away = lineup.get("away") if isinstance(lineup.get("away"), dict) else {}
    referee = context.get("referee") if isinstance(context.get("referee"), dict) else {}
    return {
        "xi": bool(home.get("confirmed") and away.get("confirmed")),
        "referee": bool(referee.get("is_confirmed") and referee.get("name")),
        "injuries": bool(context.get("injuries_is_confirmed")),
    }


def match_name(card):
    return str(card.get("match_ru") or card.get("match") or "Матч не указан").replace(" vs ", " — ")


def panel(kicker, title, detail=""):
    return (
        "<section class='sp-panel'>"
        f"<div class='sp-panel-kicker'>{esc(kicker)}</div>"
        f"<div class='sp-panel-title'>{esc(title)}</div>"
        + (f"<div class='sp-panel-detail'>{esc(detail)}</div>" if detail else "")
    )


def odds_html(card):
    if not has_prices(card):
        return "<div class='sp-empty-inline'>Полная реальная линия 1X2 не подтверждена.</div>"
    return (
        "<div class='sp-odds'>"
        + "".join(
            f"<span><small>{k}</small><b>{num(card['market_odds'][k]):.2f}</b></span>"
            for k in ("П1", "X", "П2")
        )
        + "</div>"
        + f"<div class='sp-source'>КОТИРОВКИ · {esc(card.get('odds_source') or 'SStats')}</div>"
    )


def probability_rows(card):
    rows = []
    for label, key, tone in (("П1", "p1", "home"), ("X", "px", "draw"), ("П2", "p2", "away")):
        value = min(1.0, max(0.0, num(card.get(key)) or 0.0))
        rows.append(
            "<div class='sp-prob-row'>"
            f"<span>{label}</span><div class='sp-track'><i class='{tone}' style='width:{value * 100:.1f}%'></i></div>"
            f"<b>{value:.0%}</b></div>"
        )
    return "".join(rows)


def team_badge(url, name):
    label = esc(name)
    letter = esc((str(name or "?").strip()[:1] or "?").upper())
    fallback = f"<span class='sp-team-fallback'>{letter}</span>"
    if not url:
        return fallback
    return (
        f"<span class='sp-team-emblem'>{fallback}"
        f"<img src='{esc(url)}' alt='{label}' loading='lazy' "
        """referrerpolicy='no-referrer' onerror="this.style.display='none'"></span>"""
    )


def league_badge(url, name):
    label = esc(name)
    if url:
        return (
            f"<span class='sp-league-emblem'><img src='{esc(url)}' alt='{label}' "
            "loading='lazy' referrerpolicy='no-referrer' "
            "onerror=\"this.style.display='none'\"></span>"
        )
    return "<span class='sp-league-emblem fallback'>⚽</span>"


def match_tile(card, rank):
    pred = favorite(card) or {}
    deep = card.get("deep_analysis") if isinstance(card.get("deep_analysis"), dict) else {}
    groq = str(deep.get("independent_pick") or "")
    groq_tag = (
        f"<span class='sp-ai-tag'>GROQ · {esc(groq)}</span>"
        if groq else "<span class='sp-ai-tag muted'>GROQ · ОЖИДАЕТ</span>"
    )
    return (
        "<div class='sp-match-tile'>"
        "<div class='sp-match-meta'>"
        f"<span class='sp-league-meta'>{league_badge(card.get('league_badge'), card.get('league') or card.get('div') or 'ФУТБОЛ')}{esc(card.get('league') or card.get('div') or 'ФУТБОЛ')}</span>"
        f"<span>{esc(card.get('date') or card.get('date_iso') or '')} {esc(card.get('time') or '')}</span>"
        "</div>"
        "<div class='sp-team-line'>"
        f"<div class='sp-team-side'>{team_badge(card.get('home_badge'), card.get('home_team') or 'Хозяева')}"
        f"<b>{esc(card.get('home_team') or 'Хозяева')}</b></div>"
        "<span class='sp-vs'>VS</span>"
        f"<div class='sp-team-side away'><b>{esc(card.get('away_team') or 'Гости')}</b>"
        f"{team_badge(card.get('away_badge'), card.get('away_team') or 'Гости')}</div></div>"
        f"<div class='sp-match-bottom'><span class='sp-pick'>{esc(pred.get('pick') or '—')} · {float(pred.get('probability') or 0):.1%}</span>{groq_tag}</div>"
        + odds_html(card)
        + "</div>"
    )


def render():
    data = st.session_state.get("data") or {}
    cards = [c for c in (data.get("cards") or []) if isinstance(c, dict)]
    cards.sort(key=lambda c: (str(c.get("date_iso") or "9999"), str(c.get("time") or "23:59")))

    ranked = []
    for card in cards:
        pred = favorite(card)
        if pred:
            ranked.append((pred["probability"], card, pred))
    ranked.sort(key=lambda x: x[0], reverse=True)
    groq_matches = [
        c for c in cards
        if isinstance(c.get("deep_analysis"), dict) and c.get("deep_analysis_fixture_id")
    ]
    groq_matches.sort(
        key=lambda c: (favorite(c) or {}).get("probability", 0.0),
        reverse=True,
    )

    bets = [b for b in (data.get("bets") or []) if isinstance(b, dict)]
    pending = sum(b.get("status") == "pending" for b in bets)
    closed = [b for b in bets if b.get("status") in {"won", "lost"}]
    wins = sum(b.get("status") == "won" for b in closed)
    losses = sum(b.get("status") == "lost" for b in closed)
    turnover = sum(num(b.get("stake")) or 0 for b in closed)
    profit = sum(
        (num(b.get("stake")) or 0) * ((num(b.get("odds")) or 1) - 1)
        * (1 if b.get("status") == "won" else -1)
        for b in closed
    )
    roi = profit / turnover * 100 if turnover else 0.0
    risk = risk_snapshot(data.get("bank"), bets, now=datetime.now(app_timezone()))
    priced = sum(has_prices(c) for c in cards)
    value = sum(
        bool(c.get("best") is not None
             and (c.get("verdict") or {}).get("real_odds")
             and (c.get("verdict") or {}).get("value_passed"))
        for c in cards
    )
    groq_count = sum(bool(c.get("deep_analysis")) for c in cards)
    ctx = [context_summary(c) for c in cards]
    xi_count = sum(x["xi"] for x in ctx)
    referee_count = sum(x["referee"] for x in ctx)
    injury_count = sum(x["injuries"] for x in ctx)

    st.markdown(
        "<div class='sp-dashboard-hero'><div class='sp-hero-copy'>"
        "<div class='sp-eyebrow'>OVERVIEW / MATCH INTELLIGENCE</div>"
        "<h1>Футбол. Данные. Решения.</h1>"
        "<p>Единая панель предматчевой аналитики: основная модель, реальные котировки SStats и независимый разбор Groq.</p>"
        "<div class='sp-hero-tags'><span>01 · MATCHES</span><span>02 · ODDS</span><span>03 · AI DOSSIER</span><span>04 · RISK CONTROL</span></div>"
        "</div><div class='sp-hero-mark'><div class='sp-pitch'><i></i><b></b><em></em></div>"
        "<div class='sp-hero-mark-caption'>NEURO<br/>INTELLIGENCE</div></div></div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(5)
    for col, label, metric in zip(
        cols,
        ("МАТЧЕЙ В СКАНЕ", "С РЕАЛЬНЫМИ 1X2", "VALUE ПРОШЛИ", "GROQ РАЗБОРОВ", "АКТИВНЫЕ СТАВКИ"),
        (len(cards), priced, value, groq_count, pending),
    ):
        col.metric(label, metric)

    if not cards:
        st.markdown(
            panel("FIRST RUN", "Рабочая панель готова", "После первого сканирования здесь появятся реальные матчи и аналитика.")
            + "<div class='sp-empty-state'><div class='sp-empty-icon'>⚽</div>"
            "<b>Пока нет данных сканирования</b>"
            "<span>Открой LIVE SCANNER и запусти анализ. Здесь не показываются демонстрационные или выдуманные матчи.</span>"
            "</div></section>",
            unsafe_allow_html=True,
        )
        return

    top = groq_matches[0] if groq_matches else (ranked[0][1] if ranked else cards[0])
    pred = favorite(top) or {}
    deep = top.get("deep_analysis") if isinstance(top.get("deep_analysis"), dict) else {}
    ai_probs = deep.get("probabilities") or {}
    ai_pick = deep.get("independent_pick")
    ai_prob = num(ai_probs.get(ai_pick)) if ai_pick else None
    ai_text = str(deep.get("match_assessment") or top.get("llm_opinion") or "")[:340]
    next_matches = cards[:6]
    focus_matches = groq_matches[:7] if groq_matches else ([item[1] for item in ranked[:5]] or cards[:5])
    focus_title = "TOP MATCHES · Groq + основная модель" if groq_matches else "Матчи в фокусе · основная модель"

    left, mid, right = st.columns([1.08, 0.96, 0.88], gap="medium")
    with left:
        st.markdown(
            panel("UPCOMING MATCHES", focus_title, "До 7 матчей; слабые кандидаты не добавляются ради заполнения списка")
            + "".join(match_tile(card, i + 1) for i, card in enumerate(focus_matches))
            + "</section>",
            unsafe_allow_html=True,
        )

    with mid:
        ai_prob_text = f"{ai_prob:.1f}%" if ai_prob is not None else "—"
        ai_caption = f"Независимый Groq · {ai_pick}" if ai_pick else "Независимый разбор пока не получен"
        st.markdown(
            panel("AI PREDICTION", "Главный матч", match_name(top))
            + "<div class='sp-ai-score'>"
            f"<div><small>ОСНОВНАЯ МОДЕЛЬ</small><strong>{float(pred.get('probability') or 0):.0%}</strong><span>{esc(pred.get('pick') or '—')}</span></div>"
            f"<div><small>GROQ · НЕЗАВИСИМО</small><strong>{esc(ai_prob_text)}</strong><span>{esc(ai_caption)}</span></div></div>"
            + probability_rows(top)
            + (
                f"<div class='sp-ai-summary'><b>GROQ INSIGHT</b><p>{esc(ai_text)}</p></div>"
                if ai_text else
                "<div class='sp-ai-summary muted'><b>GROQ INSIGHT</b><p>Разбор появится после пакетного анализа TOP MATCHES.</p></div>"
            )
            + "</section>",
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            panel("DATA QUALITY", "Контекст матчей", "Подтверждение источником SStats")
            + f"<div class='sp-context-score'><strong>{xi_count}/{len(cards)}</strong><span>полных стартовых XI</span></div>"
            + "".join(
                "<div class='sp-context-row'>"
                f"<span>{label}</span><b>{count}/{len(cards)}</b></div>"
                f"<div class='sp-context-track'><i style='width:{count / max(1, len(cards)) * 100:.1f}%'></i></div>"
                for label, count in (
                    ("Стартовые XI", xi_count),
                    ("Назначен судья", referee_count),
                    ("Фид травм подтверждён", injury_count),
                )
            )
            + "<div class='sp-context-note'>Неизвестные поля остаются неизвестными. Данные не заменяются предположениями.</div></section>",
            unsafe_allow_html=True,
        )

    bottom_left, bottom_right = st.columns([1.12, 0.88], gap="medium")
    with bottom_left:
        league_counts = Counter(str(c.get("league") or c.get("div") or "Другое") for c in cards)
        total = max(1, len(cards))
        rows = sorted(league_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        league_html = "".join(
            "<div class='sp-league-row'>"
            f"<span>{esc(name)}</span><div class='sp-league-track'><i style='width:{count / total * 100:.1f}%'></i></div><b>{count}</b></div>"
            for name, count in rows
        )
        st.markdown(
            panel("PREDICTIVE ANALYTICS", "Охват турниров", "Матчи из последнего сканирования")
            + league_html
            + "<div class='sp-divider'></div><div class='sp-analysis-kpis'>"
            f"<div><small>ЗАКРЫТО</small><b>{wins + losses}</b><span>{wins} побед · {losses} поражений</span></div>"
            f"<div><small>ROI ОТ ОБОРОТА</small><b class='{'positive' if roi >= 0 else 'negative'}'>{roi:+.1f}%</b><span>Только закрытые ставки</span></div>"
            f"<div><small>БАНКРОЛЛ</small><b>{(num(data.get('bank')) or 0):,.0f}</b><span>Виртуальный режим</span></div>"
            "</div></section>",
            unsafe_allow_html=True,
        )

    with bottom_right:
        priced_cards = [c for c in next_matches if has_prices(c)]
        if priced_cards:
            chosen = priced_cards[0]
            market_html = (
                "<div class='sp-market-match'>"
                f"<b>{esc(match_name(chosen))}</b><span>{esc(chosen.get('league') or 'Футбол')}</span></div>"
                + odds_html(chosen)
            )
        else:
            market_html = "<div class='sp-empty-inline'>В текущем скане нет подтверждённой полной линии 1X2.</div>"
        home_lambdas = [num(c.get("lam_h")) for c in cards if num(c.get("lam_h")) is not None]
        away_lambdas = [num(c.get("lam_a")) for c in cards if num(c.get("lam_a")) is not None]
        home_avg = sum(home_lambdas) / len(home_lambdas) if home_lambdas else None
        away_avg = sum(away_lambdas) / len(away_lambdas) if away_lambdas else None
        home_text = f"{home_avg:.2f}" if home_avg is not None else "—"
        away_text = f"{away_avg:.2f}" if away_avg is not None else "—"
        st.markdown(
            panel("ODDS OVERVIEW", "Рыночная линия", "Фактическая котировка из источника")
            + market_html
            + "<div class='sp-divider'></div><div class='sp-xg-row'>"
            f"<div><small>СРЕДНЯЯ λ ХОЗЯЕВ</small><b>{home_text}</b></div>"
            "<div class='sp-xg-sep'>VS</div>"
            f"<div><small>СРЕДНЯЯ λ ГОСТЕЙ</small><b>{away_text}</b></div></div>"
            "<div class='sp-context-note'>λ — ожидаемые голы модели, не подтверждённый итог матча.</div></section>",
            unsafe_allow_html=True,
        )

    daily_risk_pct = float(risk.get("daily_exposure_pct") or 0.0)
    daily_risk_amount = float(risk.get("daily_exposure") or 0.0)
    daily_remaining = float(risk.get("daily_remaining") or 0.0)
    max_match_amount = float(risk.get("max_match_exposure") or 0.0)
    risk_state = "Дневной лимит достигнут — новые ставки заблокированы" if daily_risk_pct >= MAX_DAILY_EXPOSURE_PCT else "Лимит в норме"
    risk_state_class = "blocked" if daily_risk_pct >= MAX_DAILY_EXPOSURE_PCT else "ok"
    st.markdown(
        panel("RISK CONTROL", "Контроль совокупного риска", "Лимиты применяются до каждой ставки, включая ставки из разных рынков одного матча")
        + "<div class='sp-risk-strip'>"
        + f"<div><small>РИСК СЕГОДНЯ</small><b>{daily_risk_amount:,.2f}</b><span>{daily_risk_pct:.1%} от банка · предел {MAX_DAILY_EXPOSURE_PCT:.0%}</span></div>"
        + f"<div><small>ОСТАТОК ДНЕВНОГО ЛИМИТА</small><b>{daily_remaining:,.2f}</b><span>с учётом рассчитанных сегодня ставок</span></div>"
        + f"<div><small>ОТКРЫТАЯ ЭКСПОЗИЦИЯ</small><b>{float(risk.get('pending_exposure') or 0):,.2f}</b><span>по всем незакрытым рынкам</span></div>"
        + f"<div><small>МАКСИМУМ НА МАТЧ</small><b>{max_match_amount:,.2f}</b><span>предел {MAX_MATCH_EXPOSURE_PCT:.0%} банка · рынок {MAX_MARKET_STAKE_PCT:.0%}</span></div>"
        + f"<div class='sp-risk-state {risk_state_class}'>{esc(risk_state)}</div></div></section>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='sp-footer-note'><span>NEURO BET PRO</span> · данные последнего сканирования · "
        "Прогнозы не гарантируют результат. Value допускается только после независимой рыночной проверки.</div>",
        unsafe_allow_html=True,
    )
