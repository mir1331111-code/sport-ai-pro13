"""context_football.py — автономная версия без внешних зависимостей."""
from __future__ import annotations
import csv, io, re, os, hashlib, gzip, pickle, time
import requests
from collections import defaultdict
from datetime import datetime

CACHE_DIR = "neuro_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.bin")

def _cache_get(key: str, max_age: int):
    try:
        p = _cache_path(key)
        if not os.path.exists(p): return None
        if time.time() - os.path.getmtime(p) > max_age: return None
        with open(p, "rb") as f: raw = f.read()
        try: return pickle.loads(gzip.decompress(raw))
        except: return pickle.loads(raw)
    except: return None

def _cache_put(key: str, value):
    try:
        p = _cache_path(key); tmp = p + ".tmp"
        with open(tmp, "wb") as f: f.write(gzip.compress(pickle.dumps(value)))
        os.replace(tmp, p)
    except: pass

def _safe_float(v):
    try: return float(v)
    except: return None

def _parse_date(s):
    if not s: return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        try: return datetime.strptime(str(s).strip()[:10], fmt)
        except: continue
    return None

def load_seasonal_with_stats(div: str, season: str) -> list:
    ck = f"fd_ctx_{div}_{season}"
    cached = _cache_get(ck, 86400 * 3)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 NeuroBet"})
        if r.status_code != 200 or not r.content: return []
        text = r.content.decode("utf-8", errors="ignore").lstrip("\ufeff")
        out = [row for row in csv.DictReader(io.StringIO(text))
               if isinstance(row, dict) and row.get("HomeTeam") and row.get("AwayTeam")]
        _cache_put(ck, out)
        return out
    except: return []

def analyze_corners(rows: list, team_name: str, last_n: int = 10):
    team_matches = []
    for r in rows:
        if r.get("HomeTeam") == team_name:
            hc = _safe_float(r.get("HC")); ac = _safe_float(r.get("AC"))
            if hc is not None: team_matches.append({"for": hc, "against": ac})
        elif r.get("AwayTeam") == team_name:
            hc = _safe_float(r.get("HC")); ac = _safe_float(r.get("AC"))
            if ac is not None: team_matches.append({"for": ac, "against": hc})
    if len(team_matches) < 5: return None
    recent = team_matches[-last_n:]
    avg_for = sum(m["for"] for m in recent) / len(recent)
    avg_against = sum(m["against"] for m in recent) / len(recent)
    avg_total = avg_for + avg_against
    league_totals = [(_safe_float(r.get("HC")) or 0) + (_safe_float(r.get("AC")) or 0)
                     for r in rows if _safe_float(r.get("HC")) is not None]
    league_avg = sum(league_totals) / len(league_totals) if league_totals else 10.0
    flags = []
    if avg_total > league_avg * 1.25:
        flags.append({"type": "corners_high", "severity": "medium",
                      "message": f"🚩 {team_name}: {avg_total:.1f} угловых/матч (лига: {league_avg:.1f}) — ТБ угловых"})
    elif avg_total < league_avg * 0.75:
        flags.append({"type": "corners_low", "severity": "low",
                      "message": f"🚩 {team_name}: мало угловых ({avg_total:.1f} vs {league_avg:.1f})"})
    return {"team": team_name, "avg_total": avg_total, "league_avg": league_avg,
            "sample": len(recent), "flags": flags}

def analyze_cards(rows: list, team_name: str, last_n: int = 10):
    team_matches = []
    for r in rows:
        if r.get("HomeTeam") == team_name:
            hy = _safe_float(r.get("HY")); hr = _safe_float(r.get("HR"))
            if hy is not None: team_matches.append({"yellow": hy, "red": hr or 0})
        elif r.get("AwayTeam") == team_name:
            ay = _safe_float(r.get("AY")); ar = _safe_float(r.get("AR"))
            if ay is not None: team_matches.append({"yellow": ay, "red": ar or 0})
    if len(team_matches) < 5: return None
    recent = team_matches[-last_n:]
    avg_yellow = sum(m["yellow"] for m in recent) / len(recent)
    avg_red = sum(m["red"] for m in recent) / len(recent)
    league_yellows = [(_safe_float(r.get("HY")) or 0) + (_safe_float(r.get("AY")) or 0)
                      for r in rows if _safe_float(r.get("HY")) is not None]
    league_avg = sum(league_yellows) / len(league_yellows) if league_yellows else 4.0
    flags = []
    if avg_yellow > league_avg * 1.3:
        flags.append({"type": "cards_high", "severity": "medium",
                      "message": f"🟨 {team_name}: {avg_yellow:.1f} жёлтых/матч (лига: {league_avg:.1f}) — ТБ карточек"})
    if avg_red > 0.15:
        flags.append({"type": "red_cards", "severity": "high",
                      "message": f"🟥 {team_name}: {avg_red:.2f} красных/матч — риск удаления"})
    return {"team": team_name, "avg_yellow": avg_yellow, "avg_red": avg_red,
            "league_avg_yellow": league_avg, "sample": len(recent), "flags": flags}

def analyze_referee(rows: list, referee_name: str | None = None):
    ref_stats: dict = defaultdict(lambda: {"matches": 0, "yellows": 0, "reds": 0})
    for r in rows:
        ref = r.get("Referee") or ""
        if not ref: continue
        hy = _safe_float(r.get("HY")) or 0; ay = _safe_float(r.get("AY")) or 0
        hr = _safe_float(r.get("HR")) or 0; ar = _safe_float(r.get("AR")) or 0
        ref_stats[ref]["matches"] += 1
        ref_stats[ref]["yellows"] += hy + ay
        ref_stats[ref]["reds"] += hr + ar
    if not ref_stats: return None
    result = {}
    for ref, s in ref_stats.items():
        if s["matches"] < 5: continue
        result[ref] = {"matches": s["matches"],
                       "avg_yellows": s["yellows"] / s["matches"],
                       "avg_reds": s["reds"] / s["matches"]}
    if not result: return None
    all_y = [r["avg_yellows"] for r in result.values()]
    league_avg = sum(all_y) / len(all_y) if all_y else 4.0
    if referee_name and referee_name in result:
        rd = result[referee_name]; flags = []
        if rd["avg_yellows"] > league_avg * 1.3:
            flags.append({"type": "referee_strict", "severity": "high",
                          "message": f"👨‍⚖️ {referee_name}: {rd['avg_yellows']:.1f} жёлтых/матч (лига: {league_avg:.1f}) — СТРОГИЙ → ТБ карточек"})
        if rd["avg_reds"] > 0.2:
            flags.append({"type": "referee_reds", "severity": "high",
                          "message": f"🟥 {referee_name}: {rd['avg_reds']:.2f} красных/матч — часто удаляет"})
        return {"referee": referee_name, "stats": rd, "flags": flags, "league_avg": league_avg}
    sorted_refs = sorted(result.items(), key=lambda x: x[1]["avg_yellows"], reverse=True)
    return {"top_strict": sorted_refs[:5], "all": result, "league_avg": league_avg}

def analyze_form(rows: list, team_name: str, last_n: int = 5):
    team_matches = []
    for r in rows:
        if r.get("HomeTeam") == team_name:
            hg = _safe_float(r.get("FTHG")); ag = _safe_float(r.get("FTAG"))
            if hg is not None:
                res = "W" if hg > ag else ("D" if hg == ag else "L")
                team_matches.append({"date": r.get("Date",""), "result": res, "gf": hg, "ga": ag})
        elif r.get("AwayTeam") == team_name:
            hg = _safe_float(r.get("FTHG")); ag = _safe_float(r.get("FTAG"))
            if ag is not None:
                res = "W" if ag > hg else ("D" if ag == hg else "L")
                team_matches.append({"date": r.get("Date",""), "result": res, "gf": ag, "ga": hg})
    if len(team_matches) < 3: return None
    team_matches.sort(key=lambda x: _parse_date(x["date"]) or datetime.min)
    recent = team_matches[-last_n:]
    wins = sum(1 for m in recent if m["result"] == "W")
    draws = sum(1 for m in recent if m["result"] == "D")
    losses = sum(1 for m in recent if m["result"] == "L")
    gf = sum(m["gf"] for m in recent)
    streak = recent[-1]["result"]; sc = 0
    for m in reversed(recent):
        if m["result"] == streak: sc += 1
        else: break
    flags = []
    if wins >= 4:
        flags.append({"type": "form_hot", "severity": "medium",
                      "message": f"🔥 {team_name}: {wins}W-{draws}D-{losses}L — горячая форма"})
    elif losses >= 4:
        flags.append({"type": "form_cold", "severity": "medium",
                      "message": f"❄️ {team_name}: {wins}W-{draws}D-{losses}L — холодная форма"})
    if streak == "W" and sc >= 3:
        flags.append({"type": "win_streak", "severity": "low",
                      "message": f"✅ {team_name}: серия из {sc} побед"})
    elif streak == "L" and sc >= 3:
        flags.append({"type": "lose_streak", "severity": "medium",
                      "message": f"⚠️ {team_name}: серия из {sc} поражений"})
    if gf / last_n > 2.0:
        flags.append({"type": "scoring_high", "severity": "low",
                      "message": f"⚽ {team_name}: {gf/last_n:.1f} голов/матч"})
    return {"team": team_name, "recent_form": "".join(m["result"] for m in recent),
            "wins": wins, "draws": draws, "losses": losses, "goals_for": gf,
            "streak": streak, "streak_count": sc, "flags": flags}

def detect_combo_signals(context: dict, home: str, away: str) -> list:
    flags = []
    hc = context.get("home_corners"); ac = context.get("away_corners")
    if hc and ac:
        combined = hc["avg_total"] + ac["avg_total"]
        league = (hc["league_avg"] + ac["league_avg"]) / 2
        if combined > league * 1.3:
            flags.append({"type": "combo_corners", "severity": "high",
                          "message": f"💎 КОМБО: обе команды дают много угловых ({combined:.1f} vs лига {league:.1f}) → ТБ угловых"})
    ref = context.get("referee", {})
    ref_flags = ref.get("flags", []) if isinstance(ref, dict) else []
    is_strict = any(f["type"] == "referee_strict" for f in ref_flags)
    h_cards = context.get("home_cards"); a_cards = context.get("away_cards")
    if is_strict and h_cards and a_cards:
        if (h_cards["avg_yellow"] > h_cards["league_avg_yellow"] and
                a_cards["avg_yellow"] > a_cards["league_avg_yellow"]):
            flags.append({"type": "combo_cards", "severity": "high",
                          "message": "💎 КОМБО: строгий судья + обе команды грубые → ТБ карточек"})
    hf = context.get("home_form"); af = context.get("away_form")
    if hf and af:
        if hf["wins"] >= 4 and af["losses"] >= 3:
            flags.append({"type": "combo_form", "severity": "medium",
                          "message": f"💎 {home} в огне ({hf['recent_form']}), {away} в кризисе ({af['recent_form']})"})
        elif af["wins"] >= 4 and hf["losses"] >= 3:
            flags.append({"type": "combo_form", "severity": "medium",
                          "message": f"💎 {away} в огне ({af['recent_form']}), {home} в кризисе ({hf['recent_form']})"})
    return flags

def analyze_match_context(home_team: str, away_team: str, div: str,
                          season: str, referee_name: str | None = None) -> dict:
    rows = load_seasonal_with_stats(div, season)
    if not rows:
        return {"error": "Нет данных архива", "flags": []}
    all_flags: list = []; context: dict = {}
    for fn, key in [(analyze_corners, "corners"), (analyze_cards, "cards"), (analyze_form, "form")]:
        h_res = fn(rows, home_team); a_res = fn(rows, away_team)
        if h_res: context[f"home_{key}"] = h_res; all_flags.extend(h_res["flags"])
        if a_res: context[f"away_{key}"] = a_res; all_flags.extend(a_res["flags"])
    if referee_name:
        ref_res = analyze_referee(rows, referee_name)
        if ref_res and "flags" in ref_res:
            context["referee"] = ref_res; all_flags.extend(ref_res["flags"])
    else:
        ref_all = analyze_referee(rows)
        if ref_all: context["referee_league"] = ref_all
    combo = detect_combo_signals(context, home_team, away_team)
    all_flags.extend(combo)
    severity = (sum(1 for f in all_flags if f["severity"] == "high") * 3 +
                sum(1 for f in all_flags if f["severity"] == "medium") * 2 +
                sum(1 for f in all_flags if f["severity"] == "low"))
    return {"home": home_team, "away": away_team, "div": div, "season": season,
            "flags": all_flags, "context": context, "severity_score": severity,
            "has_high_severity": any(f["severity"] == "high" for f in all_flags),
            "total_flags": len(all_flags)}

def render_context_flags(flags: list) -> str:
    if not flags:
        return "<div style='color:#64748b;font-size:.85rem;'>🔍 Нет контекстных сигналов</div>"
    order = {"high": 0, "medium": 1, "low": 2}
    sorted_f = sorted(flags, key=lambda f: order.get(f["severity"], 3))
    html = ""
    for f in sorted_f:
        color = {"high": "#f87171", "medium": "#fbbf24", "low": "#94a3b8"}.get(f["severity"], "#94a3b8")
        bg = {"high": "rgba(248,113,113,.12)", "medium": "rgba(251,191,36,.10)",
              "low": "rgba(148,163,184,.08)"}.get(f["severity"], "rgba(148,163,184,.08)")
        border = {"high": "#f87171", "medium": "#fbbf24", "low": "#64748b"}.get(f["severity"], "#64748b")
        html += (f'<div style="background:{bg};border-left:3px solid {border};'
                 f'padding:8px 12px;margin-bottom:6px;border-radius:6px;">'
                 f'<div style="color:{color};font-size:.85rem;font-weight:600;'
                 f'line-height:1.4;">{f["message"]}</div></div>')
    return html
