"""betting/verdict.py — построение вердикта и альтернатив."""
from __future__ import annotations
from betting.kelly import kelly, market_type


def build_alternatives(rows: list, main_pick: str) -> list:
    alt = []
    if main_pick in ("П1", "X", "П2"):
        for r in rows:
            if r["pick"] in ("П1", "X", "П2") and r["pick"] != main_pick:
                alt.append(r)
            if len(alt) == 2:
                break
    elif main_pick in ("ТБ 2.5", "ТМ 2.5"):
        for r in rows:
            if r["pick"] in ("ТБ 2.5", "ТМ 2.5") and r["pick"] != main_pick:
                alt.append(r)
            elif r["pick"] in ("BTTS да", "BTTS нет"):
                alt.append(r)
            if len(alt) == 2:
                break
    elif main_pick in ("BTTS да", "BTTS нет"):
        for r in rows:
            if r["pick"] in ("BTTS да", "BTTS нет") and r["pick"] != main_pick:
                alt.append(r)
            elif r["pick"] in ("ТБ 2.5", "ТМ 2.5"):
                alt.append(r)
            if len(alt) == 2:
                break
    return alt[:2]


def build_verdict(P: dict, thr: float, bank: float, kelly_frac: float,
                  h_name: str, a_name: str, fh: str, fa: str,
                  h2h_n: int):
    probs = {
        "П1": P["p1"], "X": P["x"], "П2": P["p2"],
        "ТБ 2.5": P["over"], "ТМ 2.5": 1 - P["over"],
        "BTTS да": P["btts"], "BTTS нет": 1 - P["btts"],
    }
    names = {
        "П1": f"Победа {h_name}", "X": "Ничья", "П2": f"Победа {a_name}",
        "ТБ 2.5": "Тотал Больше 2.5", "ТМ 2.5": "Тотал Меньше 2.5",
        "BTTS да": "Обе забьют — Да", "BTTS нет": "Обе забьют — Нет",
    }
    rows = []
    for pick, prob in probs.items():
        prob = min(max(float(prob or 0.0), 0.01), 0.99)
        fair = 1.0 / prob
        rows.append({"pick": pick, "label": names[pick], "prob": prob,
                     "odd": fair * 0.94, "fair_odd": fair})
    rows.sort(key=lambda r: -r["prob"])
    top = rows[0]
    alt = build_alternatives(rows, top["pick"])

    reasons = []
    lh, la = P["lams"]
    tg = lh + la
    reasons.append(f"Модель ожидает {tg:.2f} голов (xG {lh:.2f}-{la:.2f})")
    if P["p1"] > 0.45:
        reasons.append(f"{h_name} сильнее дома — форма: {fh}")
    elif P["p2"] > 0.45:
        reasons.append(f"{a_name} сильнее на выезде — форма: {fa}")
    else:
        reasons.append(f"Команды близки — форма: {fh} vs {fa}")
    if tg > 3.0:
        reasons.append("Результативный матч — ТБ 2.5 фаворит")
    elif tg < 2.3:
        reasons.append("Низовой матч — ТМ 2.5 фаворит")
    if h2h_n >= 6:
        reasons.append(f"Учтены {h2h_n} личных встреч")

    if top["prob"] >= 0.80:
        conf, cc = "очень высокая", "#34d399"
    elif top["prob"] >= 0.72:
        conf, cc = "высокая", "#34d399"
    elif top["prob"] >= 0.65:
        conf, cc = "хорошая", "#fbbf24"
    elif top["prob"] >= 0.55:
        conf, cc = "средняя", "#fbbf24"
    else:
        conf, cc = "низкая", "#f87171"

    verdict = {
        "pick": top["pick"], "label": top["label"],
        "prob": top["prob"], "odd": None, "fair_odd": top["fair_odd"],
        "confidence": conf, "conf_color": cc,
        "reasons": reasons, "alternatives": alt,
        "is_action": top["prob"] >= thr,
        "real_odds": False, "ev": None,
    }
    return verdict, rows, None


def refine_with_real_odds(verdict: dict, rows: list,
                          real_odds,
                          bank: float, kelly_frac: float):
    for r in rows:
        ro = (real_odds or {}).get(r["pick"])
        if ro:
            r["odd"] = ro
            r["real"] = True
        else:
            r["real"] = False
    pick = verdict["pick"]
    real_odd = (real_odds or {}).get(pick)
    if not real_odd:
        verdict["real_odds"] = False
        verdict["odd"] = None
        verdict["ev"] = None
        return verdict, None
    prob = verdict["prob"]
    ev = prob * real_odd - 1
    verdict["odd"] = real_odd
    verdict["real_odds"] = True
    verdict["ev"] = ev
    best = None
    if verdict.get("is_action"):
        stake = kelly(prob, real_odd, bank, kelly_frac)
        if stake > 0:
            best = (market_type(pick), pick, real_odd, ev, prob, stake)
    return verdict, best
