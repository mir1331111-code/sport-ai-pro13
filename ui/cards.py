"""ui/cards.py — рендер карточек матчей."""
from __future__ import annotations
import html as _html

from config import STADIUM_WALLS, TEAM_TRANSLATIONS

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
    h = translate_team(parts[0]) if len(parts) > 0 else "—"
    a = translate_team(parts[1]) if len(parts) > 1 else "—"
    bg = stadium_bg(c.get("div", ""))

    if is_action and has_real:
        mb = "linear-gradient(135deg,rgba(52,211,153,.25),rgba(16,185,129,.10))"
        mbd = "rgba(52,211,153,.7)"
        mt = esc(f"СТАВЬ: {pick}")
    elif is_action:
        mb = "linear-gradient(135deg,rgba(251,191,36,.20),rgba(202,138,4,.08))"
        mbd = "rgba(251,191,36,.6)"
        mt = esc(f"ВЫСОКАЯ P, БЕЗ РЕАЛЬНОГО КЭФА: {pick}")
    else:
        mb = "linear-gradient(135deg,rgba(148,163,184,.15),rgba(100,116,139,.08))"
        mbd = "rgba(148,163,184,.4)"
        mt = esc(f"ФОН: {pick}")

    fh = c.get("fh", "—")
    fa = c.get("fa", "—")

    alt_html = ""
    for i, t in enumerate(alt):
        icon = "🥈" if i == 0 else "🥉"
        alt_html += (
            f"<div style='display:flex;justify-content:space-between;padding:6px 0;"
            f"border-top:1px solid rgba(255,255,255,.06);font-size:.85rem;'>"
            f"<span>{icon} {esc(t.get('label','—'))}</span>"
            f"<span><b style='color:#34d399'>{t.get('prob',0)*100:.1f}%</b> "
            f"<span style='color:#8b93a7'>· fair ~{t.get('fair_odd',1):.2f}</span>"
            f"</span></div>")

    reasons_html = "".join(f"<li>{esc(r)}</li>" for r in reasons)

    warn = ""
    if is_action and not has_real:
        warn = ("<div style='color:#fde68a;font-size:.78rem;margin-top:8px;'>"
                "Реального кэфа нет — ставка пойдёт в paper-режим.</div>")

    llm_opinion = c.get("llm_opinion") or ""
    llm_html = ""
    if llm_opinion:
        llm_html = (
            f"<div style='background:rgba(139,92,246,.12);"
            f"border:1px solid rgba(139,92,246,.35);border-radius:10px;"
            f"padding:10px 14px;margin-top:10px;'>"
            f"<div style='color:#c4b5fd;font-size:.72rem;text-transform:uppercase;"
            f"font-weight:700;margin-bottom:4px;letter-spacing:1px;'>"
            f"Мнение ИИ-аналитика</div>"
            f"<div style='color:#e9d5ff;font-size:.88rem;line-height:1.5;'>"
            f"{esc(llm_opinion)}</div></div>")

    return f"""
<div class="vcard" style="background:{bg};border:1px solid rgba(255,255,255,.10);
 border-radius:20px;padding:0;margin-bottom:14px;overflow:hidden;">
  <div style="padding:16px 20px;">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
      <div><span style="background:rgba(34,211,238,.14);color:#a5f3fc;padding:3px 10px;
       border-radius:999px;font-size:.7rem;font-weight:700;">{esc(c.get('league','—'))}</span>
        <span style="background:rgba(251,191,36,.14);color:#fde68a;padding:3px 10px;
         border-radius:999px;font-size:.7rem;font-weight:700;margin-left:6px;">{esc(c.get('date','—'))}</span></div>
      <div style="font-size:.75rem;color:#8b93a7;">{c.get('games',0)} игр</div></div>
    <div style="font-size:1.4rem;font-weight:800;color:#fff;margin-bottom:6px;
     text-shadow:0 2px 8px rgba(0,0,0,.8);">{esc(h)} <span style="color:#8b93a7;
     font-weight:400;">—</span> {esc(a)}</div>
    <div style="font-size:.78rem;color:#8b93a7;margin-bottom:14px;">Форма:
      <b style="color:#34d399;">{esc(fh)}</b> ·
      <b style="color:#f87171;">{esc(fa)}</b></div></div>
  <div style="background:{mb};border-top:1px solid {mbd};border-bottom:1px solid {mbd};
   padding:14px 20px;">
    <div style="font-size:1.15rem;font-weight:900;color:#fff;margin-bottom:8px;">{mt}</div>
    <div style="display:flex;gap:20px;font-size:.9rem;color:#e6eaf2;">
      <div>Вероятность: <b style="color:#34d399;font-size:1.1rem;">{prob*100:.0f}%</b></div>
      <div>Кэф: <b style="color:#a5f3fc;font-size:1.1rem;">{f"{odd:.2f}" if has_real else "нет"}</b></div>
      <div>Уверенность: <b style="color:{cc};">{esc(conf)}</b></div></div>{warn}</div>
  <div style="padding:14px 20px;">
    <div style="color:#7dd3fc;font-size:.72rem;text-transform:uppercase;font-weight:700;
     margin-bottom:8px;letter-spacing:1px;">Почему</div>
    <ul style="margin:0 0 14px 0;padding-left:18px;color:#c9d2e3;font-size:.85rem;
     line-height:1.6;">{reasons_html}</ul>
    <div style="color:#7dd3fc;font-size:.72rem;text-transform:uppercase;font-weight:700;
     margin-bottom:6px;letter-spacing:1px;">Альтернативы</div>{alt_html}{llm_html}</div></div>"""
