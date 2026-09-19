"""Контекстный анализатор для футбола: угловые, судьи, карточки, форма.
НЕ ломает существующий App.py — работает как отдельный модуль."""
import csv, io, re, time, hashlib, os, pickle
import requests
from collections import defaultdict
from datetime import datetime, timedelta

CACHE_DIR = "neuro_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(key):
    return os.path.join(CACHE_DIR, f"{hashlib.md5(key.encode()).hexdigest()}.bin")

def _cache_get(key, max_age):
    try:
        p = _cache_path(key)
        if not os.path.exists(p): return None
        if time.time() - os.path.getmtime(p) > max_age: return None
        with open(p, "rb") as f: raw = f.read()
        try: return pickle.loads(gzip.decompress(raw))
        except: return pickle.loads(raw)
    except: return None

def _cache_put(key, value):
    try:
        import gzip
        p = _cache_path(key); tmp = p + ".tmp"
        with open(tmp, "wb") as f: f.write(gzip.compress(pickle.dumps(value)))
        os.replace(tmp, p)
    except: pass


# ============================================================
# 1. ЗАГРУЗКА АРХИВОВ С УГЛОВЫМИ И КАРТОЧКАМИ
# ============================================================
def load_seasonal_with_stats(div, season):
    """Загружает CSV с football-data.co.uk включая угловые (HC/AC) и карточки (HY/AY)."""
    ck = f"fd_stats_{div}_{season}"
    cached = _cache_get(ck, 86400 * 7)
    if cached is not None:
        return cached if isinstance(cached, list) else []
    
    url = f"https://www.football-data.co.uk/mmz4281/{season}/{div}.csv"
    try:
        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0 NeuroBet"})
        if r.status_code != 200 or not r.content:
            return []
        text = r.content.decode("utf-8", errors="ignore").lstrip("\ufeff")
        out = []
        for row in csv.DictReader(io.StringIO(text)):
            if not isinstance(row, dict): continue
            if not row.get("HomeTeam") or not row.get("AwayTeam"): continue
            out.append(row)
        _cache_put(ck, out)
        return out
    except Exception:
        return []


# ============================================================
# 2. АНАЛИЗ УГЛОВЫХ
# ============================================================
def analyze_corners(rows, team_name, last_n=10):
    """Анализирует угловые команды за последние N матчей."""
    team_matches = []
    for r in rows:
        if r.get("HomeTeam") == team_name:
            hc = _safe_float(r.get("HC"))
            ac = _safe_float(r.get("AC"))
            if hc is not None:
                team_matches.append({"corners_for": hc, "corners_against": ac, "home": True})
        elif r.get("AwayTeam") == team_name:
            hc = _safe_float(r.get("HC"))
            ac = _safe_float(r.get("AC"))
            if ac is not None:
                team_matches.append({"corners_for": ac, "corners_against": hc, "home": False})
    
    if len(team_matches) < 5:
        return None
    
    recent = team_matches[-last_n:]
    avg_for = sum(m["corners_for"] for m in recent) / len(recent)
    avg_against = sum(m["corners_against"] for m in recent) / len(recent)
    avg_total = avg_for + avg_against
    
    # Сравнение с лигой
    league_totals = []
    for r in rows:
        hc = _safe_float(r.get("HC"))
        ac = _safe_float(r.get("AC"))
        if hc is not None and ac is not None:
            league_totals.append(hc + ac)
    league_avg = sum(league_totals) / len(league_totals) if league_totals else 10.0
    
    flags = []
    if avg_total > league_avg * 1.25:
        flags.append({
            "type": "corners_high",
            "severity": "medium",
            "message": f"🚩 {team_name}: {avg_total:.1f} угловых/матч (лига: {league_avg:.1f})",
            "value": avg_total,
            "league_avg": league_avg,
        })
    elif avg_total < league_avg * 0.75:
        flags.append({
            "type": "corners_low",
            "severity": "low",
            "message": f"🚩 {team_name}: мало угловых ({avg_total:.1f} vs {league_avg:.1f})",
            "value": avg_total,
            "league_avg": league_avg,
        })
    
    return {
        "team": team_name,
        "avg_for": avg_for,
        "avg_against": avg_against,
        "avg_total": avg_total,
        "league_avg": league_avg,
        "sample": len(recent),
        "flags": flags,
    }


# ============================================================
# 3. АНАЛИЗ КАРТОЧЕК
# ============================================================
def analyze_cards(rows, team_name, last_n=10):
    """Анализирует жёлтые/красные карточки команды."""
    team_matches = []
    for r in rows:
        if r.get("HomeTeam") == team_name:
            hy = _safe_float(r.get("HY"))
            ay = _safe_float(r.get("AY"))
            hr = _safe_float(r.get("HR"))
            ar = _safe_float(r.get("AR"))
            if hy is not None:
                team_matches.append({"yellow": hy, "red": hr or 0, "home": True})
        elif r.get("AwayTeam") == team_name:
            hy = _safe_float(r.get("HY"))
            ay = _safe_float(r.get("AY"))
            hr = _safe_float(r.get("HR"))
            ar = _safe_float(r.get("AR"))
            if ay is not None:
                team_matches.append({"yellow": ay, "red": ar or 0, "home": False})
    
    if len(team_matches) < 5:
        return None
    
    recent = team_matches[-last_n:]
    avg_yellow = sum(m["yellow"] for m in recent) / len(recent)
    avg_red = sum(m["red"] for m in recent) / len(recent)
    
    # Среднее по лиге
    league_yellows = []
    for r in rows:
        hy = _safe_float(r.get("HY"))
        ay = _safe_float(r.get("AY"))
        if hy is not None and ay is not None:
            league_yellows.append(hy + ay)
    league_avg = sum(league_yellows) / len(league_yellows) if league_yellows else 4.0
    
    flags = []
    if avg_yellow > league_avg * 1.3:
        flags.append({
            "type": "cards_high",
            "severity": "medium",
            "message": f"🟨 {team_name}: {avg_yellow:.1f} жёлтых/матч (лига: {league_avg:.1f})",
            "value": avg_yellow,
        })
    if avg_red > 0.15:
        flags.append({
            "type": "red_cards",
            "severity": "high",
            "message": f"🟥 {team_name}: {avg_red:.2f} красных/матч — высокий риск удаления",
            "value": avg_red,
        })
    
    return {
        "team": team_name,
        "avg_yellow": avg_yellow,
        "avg_red": avg_red,
        "league_avg_yellow": league_avg,
        "sample": len(recent),
        "flags": flags,
    }


# ============================================================
# 4. АНАЛИЗ СУДЕЙ (через историю матчей)
# ============================================================
def analyze_referee(rows, referee_name=None):
    """Анализирует статистику судьи. Если referee_name не задан — возвращает топ-судей лиги."""
    # В football-data.co.uk нет поля Referee во всех лигах, но есть в некоторых
    # Альтернатива: используем косвенные признаки (среднее карточек в матчах)
    
    referee_stats = defaultdict(lambda: {"matches": 0, "yellows": 0, "reds": 0, "corners": 0})
    
    for r in rows:
        ref = r.get("Referee") or r.get("referee") or "Unknown"
        if ref == "Unknown": continue
        
        hy = _safe_float(r.get("HY")) or 0
        ay = _safe_float(r.get("AY")) or 0
        hr = _safe_float(r.get("HR")) or 0
        ar = _safe_float(r.get("AR")) or 0
        hc = _safe_float(r.get("HC")) or 0
        ac = _safe_float(r.get("AC")) or 0
        
        referee_stats[ref]["matches"] += 1
        referee_stats[ref]["yellows"] += hy + ay
        referee_stats[ref]["reds"] += hr + ar
        referee_stats[ref]["corners"] += hc + ac
    
    if not referee_stats:
        return None
    
    # Считаем средние
    result = {}
    for ref, stats in referee_stats.items():
        if stats["matches"] < 5: continue
        result[ref] = {
            "matches": stats["matches"],
            "avg_yellows": stats["yellows"] / stats["matches"],
            "avg_reds": stats["reds"] / stats["matches"],
            "avg_corners": stats["corners"] / stats["matches"],
        }
    
    if referee_name and referee_name in result:
        ref_data = result[referee_name]
        flags = []
        
        # Среднее по лиге
        all_yellows = [r["avg_yellows"] for r in result.values()]
        league_avg = sum(all_yellows) / len(all_yellows) if all_yellows else 4.0
        
        if ref_data["avg_yellows"] > league_avg * 1.3:
            flags.append({
                "type": "referee_strict",
                "severity": "high",
                "message": f"👨‍⚖️ {referee_name}: {ref_data['avg_yellows']:.1f} жёлтых/матч (лига: {league_avg:.1f}) — строгий судья",
                "value": ref_data["avg_yellows"],
            })
        elif ref_data["avg_yellows"] < league_avg * 0.7:
            flags.append({
                "type": "referee_lenient",
                "severity": "medium",
                "message": f"👨⚖️ {referee_name}: {ref_data['avg_yellows']:.1f} жёлтых/матч — мягкий судья",
                "value": ref_data["avg_yellows"],
            })
        
        if ref_data["avg_reds"] > 0.2:
            flags.append({
                "type": "referee_reds",
                "severity": "high",
                "message": f"🟥 {referee_name}: {ref_data['avg_reds']:.2f} красных/матч — часто удаляет",
                "value": ref_data["avg_reds"],
            })
        
        return {"referee": referee_name, "stats": ref_data, "flags": flags}
    
    # Возвращаем топ-5 самых строгих
    sorted_refs = sorted(result.items(), key=lambda x: x[1]["avg_yellows"], reverse=True)
    return {"top_strict": sorted_refs[:5], "all": result}


# ============================================================
# 5. ФОРМА КОМАНДЫ (последние N матчей)
# ============================================================
def analyze_form(rows, team_name, last_n=5):
    """Анализирует форму команды: победы, голы, серия."""
    team_matches = []
    for r in rows:
        date = r.get("Date", "")
        if r.get("HomeTeam") == team_name:
            hg = _safe_float(r.get("FTHG"))
            ag = _safe_float(r.get("FTAG"))
            if hg is not None:
                result = "W" if hg > ag else ("D" if hg == ag else "L")
                team_matches.append({"date": date, "result": result, "goals_for": hg, "goals_against": ag})
        elif r.get("AwayTeam") == team_name:
            hg = _safe_float(r.get("FTHG"))
            ag = _safe_float(r.get("FTAG"))
            if ag is not None:
                result = "W" if ag > hg else ("D" if ag == hg else "L")
                team_matches.append({"date": date, "result": result, "goals_for": ag, "goals_against": hg})
    
    if len(team_matches) < 3:
        return None
    
    # Сортируем по дате
    team_matches.sort(key=lambda x: parse_date(x["date"]) or datetime.min)
    recent = team_matches[-last_n:]
    
    wins = sum(1 for m in recent if m["result"] == "W")
    draws = sum(1 for m in recent if m["result"] == "D")
    losses = sum(1 for m in recent if m["result"] == "L")
    goals_for = sum(m["goals_for"] for m in recent)
    goals_against = sum(m["goals_against"] for m in recent)
    
    # Серия
    streak = recent[-1]["result"]
    streak_count = 0
    for m in reversed(recent):
        if m["result"] == streak:
            streak_count += 1
        else:
            break
    
    flags = []
    if wins >= 4:
        flags.append({
            "type": "form_hot",
            "severity": "medium",
            "message": f"🔥 {team_name}: {wins}W-{draws}D-{losses}L в последних {last_n} — горячая форма",
        })
    elif losses >= 4:
        flags.append({
            "type": "form_cold",
            "severity": "medium",
            "message": f"❄️ {team_name}: {wins}W-{draws}D-{losses}L в последних {last_n} — холодная форма",
        })
    
    if streak == "W" and streak_count >= 3:
        flags.append({
            "type": "win_streak",
            "severity": "low",
            "message": f"✅ {team_name}: серия из {streak_count} побед",
        })
    elif streak == "L" and streak_count >= 3:
        flags.append({
            "type": "lose_streak",
            "severity": "medium",
            "message": f"⚠️ {team_name}: серия из {streak_count} поражений",
        })
    
    if goals_for / last_n > 2.0:
        flags.append({
            "type": "scoring_high",
            "severity": "low",
            "message": f"⚽ {team_name}: {goals_for/last_n:.1f} голов/матч в последних {last_n}",
        })
    
    return {
        "team": team_name,
        "recent_form": "".join(m["result"] for m in recent),
        "wins": wins, "draws": draws, "losses": losses,
        "goals_for": goals_for, "goals_against": goals_against,
        "streak": streak, "streak_count": streak_count,
        "flags": flags,
    }


# ============================================================
# 6. ГЛАВНАЯ ФУНКЦИЯ: КОНТЕКСТНЫЙ АНАЛИЗ МАТЧА
# ============================================================
def analyze_match_context(home_team, away_team, div, season, referee_name=None):
    """Полный контекстный анализ матча. Возвращает все флаги."""
    rows = load_seasonal_with_stats(div, season)
    if not rows:
        return {"error": "Нет данных архива"}
    
    all_flags = []
    context = {}
    
    # Угловые
    home_corners = analyze_corners(rows, home_team)
    away_corners = analyze_corners(rows, away_team)
    if home_corners:
        context["home_corners"] = home_corners
        all_flags.extend(home_corners["flags"])
    if away_corners:
        context["away_corners"] = away_corners
        all_flags.extend(away_corners["flags"])
    
    # Карточки
    home_cards = analyze_cards(rows, home_team)
    away_cards = analyze_cards(rows, away_team)
    if home_cards:
        context["home_cards"] = home_cards
        all_flags.extend(home_cards["flags"])
    if away_cards:
        context["away_cards"] = away_cards
        all_flags.extend(away_cards["flags"])
    
    # Судья
    if referee_name:
        ref_analysis = analyze_referee(rows, referee_name)
        if ref_analysis and "flags" in ref_analysis:
            context["referee"] = ref_analysis
            all_flags.extend(ref_analysis["flags"])
    
    # Форма
    home_form = analyze_form(rows, home_team)
    away_form = analyze_form(rows, away_team)
    if home_form:
        context["home_form"] = home_form
        all_flags.extend(home_form["flags"])
    if away_form:
        context["away_form"] = away_form
        all_flags.extend(away_form["flags"])
    
    # Суммарный скоринг
    severity_score = sum(1 for f in all_flags if f["severity"] == "high") * 3 + \
                     sum(1 for f in all_flags if f["severity"] == "medium") * 2 + \
                     sum(1 for f in all_flags if f["severity"] == "low")
    
    return {
        "home": home_team,
        "away": away_team,
        "div": div,
        "season": season,
        "flags": all_flags,
        "context": context,
        "severity_score": severity_score,
        "has_high_severity": any(f["severity"] == "high" for f in all_flags),
    }


def _safe_float(v):
    try: return float(v)
    except: return None

def parse_date(s):
    if not s: return None
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        try: return datetime.strptime(str(s).strip(), fmt)
        except: continue
    return None


# ============================================================
# 7. РЕНДЕРИНГ ДЛЯ STREAMLIT
# ============================================================
def render_context_flags(flags):
    """Рендерит флаги в HTML для Streamlit."""
    if not flags:
        return "<div style='color:#64748b;font-size:.85rem;'>Нет контекстных сигналов</div>"
    
    html = ""
    for f in flags:
        color = {"high": "#f87171", "medium": "#fbbf24", "low": "#94a3b8"}.get(f["severity"], "#94a3b8")
        bg = {"high": "rgba(248,113,113,.1)", "medium": "rgba(251,191,36,.1)", "low": "rgba(148,163,184,.08)"}.get(f["severity"], "rgba(148,163,184,.08)")
        html += f"""
<div style="background:{bg};border-left:3px solid {color};padding:8px 12px;margin-bottom:6px;border-radius:6px;">
  <div style="color:{color};font-size:.85rem;font-weight:600;">{f['message']}</div>
</div>"""
    return html
