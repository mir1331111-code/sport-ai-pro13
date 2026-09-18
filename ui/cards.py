"""ui/cards.py — карточки с логотипами команд и лиг."""
from __future__ import annotations
import html as _html

from config import STADIUM_WALLS, TEAM_TRANSLATIONS, DIV_NAMES

esc = _html.escape


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


def _team_badge_html(team_en: str, team_ru: str, badge_from_card: str = "") -> str:
    """Возвращает HTML бейджа. Приоритет: card badge → API → fallback."""
    url = badge_from_card or ""
    if not url:
        try:
            from data.sources import team_logo_url
            url = team_logo_url(team_en) or ""
        except Exception:
            url = ""
    if url:
        letter = esc((team_ru or "?")[:1].upper())
        return (
            f'<img class="team-badge" src="{esc(url)}" alt="{esc(team_ru)}" '
            f'onerror="this.outerHTML='
            f'\'<div class=&quot;team-badge-fallback&quot;>{letter}</div>\'"/>'
        )
    letter = esc((team_ru or "?")[:1].upper())
    return f'<div class="team-badge-fallback">{letter}</div>'


def _league_badge_html(div: str, fallback_name: str,
                       badge_from_card: str = "") -> str:
    url = badge_from_card or ""
    if not url:
        try:
            from data.sources import league_logo_url
            url = league_logo_url(div) or ""
        except Exception:
            url = ""
    icon = f'<img src="{esc(url)}" alt=""/>' if url else ""
    name = DIV_NAMES.get(div) or fallback_name or "—"
    return f'<span class="league-badge">{icon}{esc(name)}</span>'


def render_verdict_card(c: dict, thr: float) -> str:
    v = c.get("verdict") or {}
    if not v:
        return ""
    pick = v.get("label", "—")
    prob = v.get("prob", 0)
    odd = v.get("odd")
    has_real = v.get("real_odds", False) and odd is not None
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

    if is_action and has_real:
        mb = "linear-gradient(135deg,rgba(52,211,153,.22),rgba(16,185,129,.08))"
        mbd = "rgba(52,211,153,.65)"
        mt = esc(f"🎯 СТАВЬ: {pick}")
    elif is_action:
        mb = "linear-gradient(135deg,rgba(251,191,36,.20),rgba(202,138,4,.08))"
        mbd = "rgba(251,191,36,.55)"
        mt = esc(f"🤔 ВЫСОКАЯ P: {pick}")
    else:
        mb = "linear-gradient(135deg,rgba(148,163,184,.15),rgba(100,116,139,.08))"
        mbd = "rgba(148,163,184,.4)"
        mt = esc(f"👀 ФОН: {pick}")

    fh = c.get("fh", "—")
    fa = c.get("fa", "—")

    alt_html = ""
    for i, t in enumerate(alt):
        icon = "🥈" if i == 0 else "🥉"
        alt_html += (
            f"<div style='display:flex;justify-content:space-between;padding:8px 0;"
            f"border-top:1px solid rgba(255,255,255,.06);font-size:.87rem;'>"
            f"<span>{icon} {esc(t.get('label','—'))}</span>"
            f"<span><b style='color:#34d399;font-family:JetBrains Mono,monospace'>"
            f"{t.get('prob',0)*100:.1f}%</b> "
            f"<span style='color:#8b93a7'>· fair ~{t.get('fair_odd',1):.2f}</span>"
            f"</span></div>")

    reasons_html = "".join(f"<li>{esc(r)}</li>" for r in reasons)

    warn = ""
    if is_action and not has_real:
        warn = ("<div style='color:#fde68a;font-size:.78rem;margin-top:10px;"
                "padding:8px 12px;background:rgba(251,191,36,.08);"
                "border-radius:10px;border:1px solid rgba(251,191,36,.25);'>"
                "⚠️ Реального кэфа нет — paper-режим.</div>")

    llm_opinion = c.get("llm_opinion") or ""
    llm_html = ""
    if llm_opinion:
        llm_html = (
            f"<div style='background:rgba(139,92,246,.10);"
            f"border:1px solid rgba(139,92,246,.30);border-radius:14px;"
            f"padding:12px 16px;margin-top:12px;'>"
            f"<div style='color:#c4b5fd;font-size:.72rem;text-transform:uppercase;"
            f"font-weight:700;margin-bottom:6px;letter-spacing:1.4px;'>"
            f"🤖 ИИ-аналитик</div>"
            f"<div style='color:#e9d5ff;font-size:.88rem;line-height:1.55;'>"
            f"{esc(llm_opinion)}</div></div>")

    return f"""
<div class="vcard" style="background:{bg};">
  <div style="padding:20px 24px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
        {league_html}
        <span class="date-badge">📅 {esc(c.get('date','—'))}</span>
      </div>
      <div style="font-size:.75rem;color:#8b93a7;font-family:JetBrains Mono,monospace;">
        {c.get('games',0)} игр
      </div>
    </div>

    <div class="team-row">
      {h_badge}
      <span class="team-name">{esc(h_ru)}</span>
      <span style="color:#8b93a7;font-weight:300;font-size:1.1rem;margin:0 4px;">vs</span>
      {a_badge}
      <span class="team-name">{esc(a_ru)}</span>
    </div>

    <div style="font-size:.78rem;color:#8b93a7;margin-top:12px;margin-bottom:6px;">
      Форма: <b style="color:#34d399;">{esc(fh)}</b> ·
      <b style="color:#f87171;">{esc(fa)}</b>
    </div>
  </div>

  <div style="background:{mb};border-top:1px solid {mbd};border-bottom:1px solid {mbd};
   padding:16px 24px;">
    <div style="font-size:1.2rem;font-weight:900;color:#fff;margin-bottom:10px;
     letter-spacing:-.3px;">{mt}</div>
    <div style="display:flex;gap:24px;font-size:.9rem;color:#e6eaf2;flex-wrap:wrap;">
      <div>Вероятность: <b style="color:#34d399;font-size:1.15rem;
       font-family:JetBrains Mono,monospace;">{prob*100:.0f}%</b></div>
      <div>Кэф: <b style="color:#a5f3fc;font-size:1.15rem;
       font-family:JetBrains Mono,monospace;">{f"{odd:.2f}" if has_real else "—"}</b></div>
      <div>Уверенность: <b style="color:{cc};">{esc(conf)}</b></div>
    </div>{warn}
  </div>

  <div style="padding:16px 24px;">
    <div style="color:#7dd3fc;font-size:.72rem;text-transform:uppercase;
     font-weight:700;margin-bottom:10px;letter-spacing:1.4px;">Почему</div>
    <ul style="margin:0 0 16px 0;padding-left:20px;color:#c9d2e3;
     font-size:.86rem;line-height:1.65;">{reasons_html}</ul>

    <div style="color:#7dd3fc;font-size:.72rem;text-transform:uppercase;
     font-weight:700;margin-bottom:8px;letter-spacing:1.4px;">Альтернативы</div>
    {alt_html}{llm_html}
  </div>
</div>"""
