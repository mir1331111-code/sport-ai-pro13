"""ui/cards.py — карточки с логотипами + контекст + rank badge + сортировка."""
from __future__ import annotations
import html as _html
from datetime import datetime

from config import STADIUM_WALLS, TEAM_TRANSLATIONS, DIV_NAMES

esc = _html.escape


# ============ ХЕЛПЕРЫ ДЛЯ СОРТИРОВКИ ============
def parse_card_date(c: dict) -> datetime:
    """Возвращает datetime карточки для сортировки по дате."""
    try:
        iso = (c.get("date_iso") or "")[:10]
        if iso:
            return datetime.strptime(iso, "%Y-%m-%d")
    except Exception:
        pass
    return datetime(2099, 1, 1)


def day_label(dt) -> str:
    """Сегодня / Завтра / Послезавтра / Пт 20.09 / Прошлое."""
    if not dt:
        return "—"
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    d = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    diff = (d - today).days
    if diff == 0:
        return "📅 СЕГОДНЯ"
    if diff == 1:
        return "📅 ЗАВТРА"
    if diff == 2:
        return "📅 ПОСЛЕЗАВТРА"
    if diff < 0:
        return f"📅 ПРОШЛОЕ ({abs(diff)}д назад)"
    weekdays = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    return f"📅 {weekdays[dt.weekday()]} {dt.strftime('%d.%m')}"


# ============ ПЕРЕВОДЫ ============
def translate_team(name: str) -> str:
    if not name:
        return name
    name = str(name).strip()
    if name in TEAM_TRANSLATIONS:
        return TEAM_TRANSLATIONS[name]
    low = name.lower()
    for eng, rus in TEAM_TRANSLATIONS.items():
        if low == eng.lower():
            return rus
    for eng, rus in TEAM_TRANSLATIONS.items():
        e = eng.lower()
        if low.startswith(e + " ") or low.endswith(" " + e) or low == e:
            return rus
    return name


def translate_match(match_str: str) -> str:
    if not match_str or " vs " not in match_str:
        return match_str or "—"
    parts = match_str.split(" vs ")
    if len(parts) == 2:
        return f"{translate_team(parts[0])} — {translate_team(parts[1])}"
    return match_str


def stadium_bg(div: str) -> str:
    return STADIUM_WALLS.get(div or "", STADIUM_WALLS["DEFAULT"])


# ============ ЛОГОТИПЫ ============
def _team_badge_html(team_en: str, team_ru: str, badge_from_card: str = "") -> str:
    url = badge_from_card or ""
    letter = esc((team_ru or "?")[:1].upper())
    if url:
        return (
            '<img class="team-badge" src="' + esc(url) + '" alt="' + esc(team_ru) + '" '
            'onerror="this.style.display=&#39;none&#39;;'
            'this.insertAdjacentHTML(&#39;afterend&#39;,'
            '&#39;&lt;div class=team-badge-fallback&gt;' + letter + '&lt;/div&gt;&#39;);"/>'
        )
    return '<div class="team-badge-fallback">' + letter + '</div>'


def _league_badge_html(div: str, fallback_name: str, badge_from_card: str = "") -> str:
    url = badge_from_card or ""
    icon = '<img src="' + esc(url) + '" alt=""/>' if url else ""
    name = DIV_NAMES.get(div) or fallback_name or "—"
    return '<span class="league-badge">' + icon + esc(name) + '</span>'


# ============ ГЛАВНЫЙ РЕНДЕР ============
def render_verdict_card(c: dict, thr: float) -> str:
    v = c.get("verdict") or {}
    if not v:
        return ""
    pick = v.get("label", "—")
    prob = v.get("prob", 0)
    odd = v.get("odd")
    has_real = v.get("real_odds", False) and odd is not None
    odds_source = c.get("odds_source", "")
    source_label = {"market": "BOOKMAKER API", "grok_web": "GROK WEB", "manual": "MANUAL", "fdorg": "FOOTBALL-DATA"}.get(odds_source, "MARKET")
    conf = v.get("confidence", "средняя")
    cc = v.get("conf_color", "#fbbf24")
    reasons = v.get("reasons", [])
    is_action = v.get("is_action", False)
    alt = v.get("alternatives", [])

    parts = c.get("match", "— vs —").split(" vs ")
    h_en = parts[0] if len(parts) > 0 else "—"
    a_en = parts[1] if len(parts) > 1 else "—"
    h_ru = translate_team(h_en)
    a_ru = translate_team(a_en)

    bg = stadium_bg(c.get("div", ""))
    div_code = c.get("div", "")
    league_html = _league_badge_html(div_code, c.get("league", ""),
                                     c.get("league_badge", "") or "")
    h_badge = _team_badge_html(h_en, h_ru, c.get("home_badge", "") or "")
    a_badge = _team_badge_html(a_en, a_ru, c.get("away_badge", "") or "")

    # ============ SIGNAL RANK BADGE ============
    rank_html = ""
    try:
        from betting.ranking import (
            rank_score, rank_label, rank_color, signal_score, value_score,
        )
        _rs = rank_score(c)
        _rl = rank_label(_rs)
        _rc = rank_color(_rs)
        _ss = signal_score(c)
        _vs = value_score(c)
        adaptive_html = ""
        try:
            _af = float(c.get("adaptive_factor") or 1.0)
            if abs(_af - 1.0) > 0.001:
                _ar = esc(str(c.get("adaptive_reason") or "Adaptive priority"))
                adaptive_html = (
                    f'<span title="{_ar}" style="background:rgba(167,139,250,.10);color:#c4b5fd;'
                    f'padding:3px 8px;border-radius:999px;font-size:.68rem;'
                    f'font-weight:700;margin-left:5px;">ADAPTIVE x{_af:.2f}</span>'
                )
        except Exception:
            adaptive_html = ""
        rank_html = (
            f'<span style="background:{_rc}22;color:{_rc};'
            f'padding:3px 10px;border-radius:999px;font-size:.7rem;'
            f'font-weight:800;margin-right:6px;border:1px solid {_rc}55;">'
            f'<span title="Ранг матча: общий приоритет сигнала по совокупности факторов. Не вероятность победы." '
            f'style="background:{_rc}22;color:{_rc};padding:3px 10px;border-radius:999px;'
            f'font-size:.7rem;font-weight:800;margin-right:6px;border:1px solid {_rc}55;">'
            f'{_rl} · {_rs*100:.0f}</span>'
            f'<span title="SIGNAL — оценка силы сигнала модели от 0 до 100; это не вероятность исхода."
            f'style="background:rgba(96,165,250,.10);color:#93c5fd;'
            f'padding:3px 8px;border-radius:999px;font-size:.68rem;'
            f'font-weight:700;margin-right:5px;">'
            f'SIGNAL {_ss*100:.0f}</span>'
            f'<span title="VALUE — оценка ценности прогноза/рынка по внутренней формуле от 0 до 100; не гарантия прибыли."
            f'style="background:rgba(52,211,153,.10);color:#6ee7b7;'
            f'padding:3px 8px;border-radius:999px;font-size:.68rem;'
            f'font-weight:700;">'
            f'VALUE {_vs*100:.0f}</span>'
            f'{adaptive_html}'
        )
    except Exception:
        rank_html = ""

    if is_action and has_real:
        mb = "linear-gradient(135deg,rgba(52,211,153,.22),rgba(16,185,129,.08))"
        mbd = "rgba(52,211,153,.65)"
        mt = "🎯 СТАВЬ: " + esc(pick)
    elif is_action:
        mb = "linear-gradient(135deg,rgba(251,191,36,.20),rgba(202,138,4,.08))"
        mbd = "rgba(251,191,36,.55)"
        mt = "🤔 ВЫСОКАЯ P: " + esc(pick)
    else:
        mb = "linear-gradient(135deg,rgba(148,163,184,.15),rgba(100,116,139,.08))"
        mbd = "rgba(148,163,184,.4)"
        mt = "👀 ФОН: " + esc(pick)

    fh = c.get("fh", "—")
    fa = c.get("fa", "—")

    alt_html = ""
    for i, t in enumerate(alt):
        icon = "🥈" if i == 0 else "🥉"
        alt_html += (
            '<div style="display:flex;justify-content:space-between;padding:8px 0;'
            'border-top:1px solid rgba(255,255,255,.06);font-size:.87rem;">'
            '<span>' + icon + ' ' + esc(t.get('label', '—')) + '</span>'
            '<span><b style="color:#34d399;font-family:JetBrains Mono,monospace">'
            + str(round(t.get('prob', 0) * 100, 1)) + '%</b> '
            '<span style="color:#8b93a7">· fair ~'
            + str(round(t.get('fair_odd', 1), 2)) + '</span></span></div>')

    reasons_html = "".join("<li>" + esc(r) + "</li>" for r in reasons)

    warn = ""
    if is_action and not has_real:
        warn = ('<div style="color:#fde68a;font-size:.78rem;margin-top:10px;'
                'padding:8px 12px;background:rgba(251,191,36,.08);'
                'border-radius:10px;border:1px solid rgba(251,191,36,.25);">'
                '⚠️ Реального кэфа нет — введи его вручную ниже, чтобы проверить value.</div>')

    # ============ КОНТЕКСТ ============
    context_html = ""
    ctx = c.get("context")
    if ctx:
        try:
            from context_football import render_context_html
            context_html = render_context_html(ctx)
        except Exception:
            context_html = ""

    grok_comment = c.get("grok_comment") or ""
    grok_html = ""
    if grok_comment:
        bookmaker = esc(c.get("grok_bookmaker") or "источник не указан")
        grok_html = (
            '<div style="background:rgba(14,165,233,.08);'
            'border:1px solid rgba(14,165,233,.25);border-radius:14px;'
            'padding:12px 16px;margin-top:12px;">'
            '<div style="color:#7dd3fc;font-size:.72rem;text-transform:uppercase;'
            'font-weight:700;margin-bottom:6px;letter-spacing:1.4px;">'
            '🌐 GROK · РАЗБОР ЦЕНЫ</div>'
            '<div style="color:#dbeafe;font-size:.88rem;line-height:1.55;">'
            + esc(grok_comment) + '</div>'
            '<div style="color:#64748b;font-size:.68rem;margin-top:6px;">'
            + bookmaker + '</div></div>'
        )

    llm_opinion = c.get("llm_opinion") or ""
    llm_html = ""
    if llm_opinion:
        llm_html = (
            '<div style="background:rgba(139,92,246,.10);'
            'border:1px solid rgba(139,92,246,.30);border-radius:14px;'
            'padding:12px 16px;margin-top:12px;">'
            '<div style="color:#c4b5fd;font-size:.72rem;text-transform:uppercase;'
            'font-weight:700;margin-bottom:6px;letter-spacing:1.4px;">'
            '🤖 ИИ-аналитик</div>'
            '<div style="color:#e9d5ff;font-size:.88rem;line-height:1.55;">'
            + esc(llm_opinion) + '</div></div>')

    odd_display = str(round(odd, 2)) if has_real else "—"
    prob_display = str(round(prob * 100))

    # ============ BET / WATCH / SKIP ============
    market_html = ""
    if has_real:
        fair = float(v.get("fair_odd") or 0.0)
        ev = v.get("ev")
        edge = v.get("edge")
        kelly_pct = float(v.get("kelly_pct") or 0.0)
        explicit_decision = str(
            v.get("decision") or c.get("decision") or ""
        ).upper().strip()
        if explicit_decision not in ("BET", "WATCH", "SKIP"):
            # Backward-compatible fallback for old saved cards.
            value_passed = bool(v.get("value_passed", False))
            is_bet = bool(v.get("is_bet", False))
            if is_bet and value_passed:
                explicit_decision = "BET"
            elif is_bet:
                explicit_decision = "WATCH"
            else:
                explicit_decision = "SKIP"

        # Decision layer is authoritative when scanner has already classified
        # the real market price.
        if explicit_decision == "BET":
            status, color, label = "BET", "#34d399", "В ПОРТФЕЛЬ"
            rationale = (
                "реальный кэф проходит Edge ≥ 3% и EV ≥ 3%; "
                "Kelly положительный — сигнал можно использовать как вход."
            )
        elif explicit_decision == "WATCH":
            status, color, label = "WATCH", "#fbbf24", "ЖДАТЬ ЛУЧШУЮ ЦЕНУ"
            rationale = (
                "преимущество ещё есть, но текущая цена не проходит сильный "
                "value-порог; ждём улучшения кэфа."
            )
        else:
            status, color, label = "SKIP", "#f87171", "НЕ ВХОДИТЬ"
            rationale = (
                "текущая реальная цена не даёт положительного преимущества; "
                "ставку не добавляем в портфель."
            )

        decision_reason = str(
            v.get("decision_reason") or c.get("decision_reason") or rationale
        ).strip()

        market_html = (
            '<div style="margin-top:12px;padding:12px 14px;border-radius:12px;'
            'background:rgba(15,23,42,.48);border:1px solid ' + color + '55;">'
            '<div style="display:flex;justify-content:space-between;align-items:center;gap:10px;'
            'font-size:.76rem;font-weight:800;letter-spacing:1px;">'
            '<span style="color:' + color + ';">● ' + status + ' · ' + label + '</span>'
            '<span style="color:#94a3b8;">MARKET PRICE · ' + source_label + '</span></div>'
            '<div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:9px;'
            'font-size:.78rem;color:#cbd5e1;">'
            '<span>Fair <b style="color:#e2e8f0;">' + (f"{fair:.2f}" if fair else "—") + '</b></span>'
            '<span>Edge <b style="color:#e2e8f0;">' + (f"{float(edge):.1%}" if edge is not None else "—") + '</b></span>'
            '<span>EV <b style="color:#e2e8f0;">' + (f"{float(ev):.1%}" if ev is not None else "—") + '</b></span>'
            '<span>Kelly <b style="color:#e2e8f0;">' + f"{kelly_pct:.1%}" + '</b></span>'
            '</div>'
            '<div style="margin-top:8px;color:#94a3b8;font-size:.76rem;">'
            'Решение: ' + rationale +
            '</div>' +
            '<div style="margin-top:7px;color:' + color + ';font-size:.73rem;font-weight:700;">' +
            'Причина: ' + esc(decision_reason) +
            '</div>' +
            '<div style="margin-top:8px;font-size:.7rem;color:#64748b;">' +
            ('BET → в автоматический портфель' if explicit_decision == "BET" else
             'WATCH → следим за ценой' if explicit_decision == "WATCH" else
             'SKIP → не замораживаем банк') +
            '</div></div>'
        )

    return (
        '<div class="vcard" style="background:' + bg + ';">'
        '<div style="padding:20px 24px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">'
        '<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">'
        + rank_html + league_html +
        '<span class="date-badge">📅 ' + esc(c.get('date', '—')) + '</span>'
        '</div>'
        '<div style="font-size:.75rem;color:#8b93a7;font-family:JetBrains Mono,monospace;">'
        + str(c.get('games', 0)) + ' игр'
        '</div></div>'

        '<div class="team-row">'
        + h_badge +
        '<span class="team-name">' + esc(h_ru) + '</span>'
        '<span style="color:#8b93a7;font-weight:300;font-size:1.1rem;margin:0 4px;">vs</span>'
        + a_badge +
        '<span class="team-name">' + esc(a_ru) + '</span>'
        '</div>'

        '<div style="font-size:.78rem;color:#8b93a7;margin-top:12px;margin-bottom:6px;">'
        'Форма: <b style="color:#34d399;">' + esc(fh) + '</b> · '
        '<b style="color:#f87171;">' + esc(fa) + '</b>'
        '</div></div>'

        '<div style="background:' + mb + ';border-top:1px solid ' + mbd +
        ';border-bottom:1px solid ' + mbd + ';padding:16px 24px;">'
        '<div style="font-size:1.2rem;font-weight:900;color:#fff;margin-bottom:10px;'
        'letter-spacing:-.3px;">' + mt + '</div>'
        '<div style="display:flex;gap:24px;font-size:.9rem;color:#e6eaf2;flex-wrap:wrap;">'
        '<div>Вероятность: <b style="color:#34d399;font-size:1.15rem;'
        'font-family:JetBrains Mono,monospace;">' + prob_display + '%</b></div>'
        '<div>Кэф: <b style="color:#a5f3fc;font-size:1.15rem;'
        'font-family:JetBrains Mono,monospace;">' + odd_display + '</b></div>'
        '<div>Уверенность: <b style="color:' + cc + ';">' + esc(conf) + '</b></div>'
        '</div>' + warn + market_html + '</div>'

        '<div style="padding:16px 24px;">'
        '<div style="color:#7dd3fc;font-size:.72rem;text-transform:uppercase;'
        'font-weight:700;margin-bottom:10px;letter-spacing:1.4px;">Почему</div>'
        '<ul style="margin:0 0 16px 0;padding-left:20px;color:#c9d2e3;'
        'font-size:.86rem;line-height:1.65;">' + reasons_html + '</ul>'
        '<div style="color:#7dd3fc;font-size:.72rem;text-transform:uppercase;'
        'font-weight:700;margin-bottom:8px;letter-spacing:1.4px;">Альтернативы</div>'
        + alt_html + context_html + grok_html + llm_html +
        '</div></div>'
    )
