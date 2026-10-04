"""betting/verdict.py — модельный verdict + проверка реального рынка."""
from __future__ import annotations

from betting.kelly import (
    edge,
    expected_value,
    kelly,
    market_type,
)


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


def build_verdict(
    P: dict,
    thr: float,
    bank: float,
    kelly_frac: float,
    h_name: str,
    a_name: str,
    fh: str,
    fa: str,
    h2h_n: int,
):
    """
    Создаёт MODEL verdict.

    ВАЖНО:
    Здесь fair_odd используется только как математическая
    оценка модели и НЕ является основанием для ставки.

    Ставка будет разрешена только после получения реального
    bookmaker market odd через refine_with_real_odds().
    """

    probs = {
        "П1": P["p1"],
        "X": P["x"],
        "П2": P["p2"],
        "ТБ 2.5": P["over"],
        "ТМ 2.5": 1.0 - P["over"],
        "BTTS да": P["btts"],
        "BTTS нет": 1.0 - P["btts"],
    }

    names = {
        "П1": f"Победа {h_name}",
        "X": "Ничья",
        "П2": f"Победа {a_name}",
        "ТБ 2.5": "Тотал Больше 2.5",
        "ТМ 2.5": "Тотал Меньше 2.5",
        "BTTS да": "Обе забьют — Да",
        "BTTS нет": "Обе забьют — Нет",
    }

    rows = []

    for pick, prob in probs.items():
        prob = min(max(float(prob or 0.0), 0.01), 0.99)

        fair = 1.0 / prob

        rows.append(
            {
                "pick": pick,
                "label": names[pick],
                "prob": prob,

                # Только reference/model fair odds.
                "fair_odd": fair,

                # До получения market odds здесь ставки нет.
                "odd": None,
                "real": False,
            }
        )

    rows.sort(key=lambda r: -r["prob"])

    top = rows[0]
    alt = build_alternatives(rows, top["pick"])

    reasons = []

    lh, la = P["lams"]
    tg = lh + la

    reasons.append(
        f"Модель ожидает {tg:.2f} голов "
        f"(xG {lh:.2f}-{la:.2f})"
    )

    if P["p1"] > 0.45:
        reasons.append(
            f"{h_name} сильнее дома — форма: {fh}"
        )
    elif P["p2"] > 0.45:
        reasons.append(
            f"{a_name} сильнее на выезде — форма: {fa}"
        )
    else:
        reasons.append(
            f"Команды близки — форма: {fh} vs {fa}"
        )

    if tg > 3.0:
        reasons.append(
            "Результативный матч — ТБ 2.5 фаворит"
        )
    elif tg < 2.3:
        reasons.append(
            "Низовой матч — ТМ 2.5 фаворит"
        )

    if h2h_n >= 6:
        reasons.append(
            f"Учтены {h2h_n} личных встреч"
        )

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

    # Прозрачный confidence score: не новая модель, а нормализация
    # уже рассчитанных model probability / edge / EV.
    confidence_score = round(min(100.0, max(0.0, (
        0.55 * top["prob"] * 100.0
        + 0.45 * min(1.0, max(0.0, (top["prob"] - 0.50) / 0.35)) * 100.0
    ))), 1)

    verdict = {
        "pick": top["pick"],
        "label": top["label"],
        "prob": top["prob"],

        # До market odds ничего здесь нет.
        "odd": None,

        # Reference only.
        "fair_odd": top["fair_odd"],

        "confidence": conf,
        "conf_color": cc,
        "confidence_score": confidence_score,
        "confidence_label": (
            "A" if confidence_score >= 80
            else "B" if confidence_score >= 65
            else "C" if confidence_score >= 50
            else "D"
        ),

        "reasons": reasons,
        "alternatives": alt,

        # Это означает только "модельный кандидат".
        # Это НЕ означает, что ставку можно делать.
        "is_action": top["prob"] >= thr,

        "real_odds": False,

        "ev": None,
        "edge": None,

        "stake": 0.0,

        "market_required": True,
    }

    return verdict, rows, None


def refine_with_manual_odd(
    verdict: dict,
    rows: list,
    manual_odd: float,
    bank: float,
    kelly_frac: float,
    min_edge: float = 0.03,
    min_ev: float = 0.03,
):
    """Apply one manually entered bookmaker odd to the model pick.

    Manual input is treated exactly like a real market quote: it can create
    a bet only when the model candidate, edge, EV and Kelly checks pass.
    The model fair_odd is never used as a market price.
    """
    try:
        odd = float(manual_odd)
    except (TypeError, ValueError):
        odd = 0.0

    if odd <= 1.01:
        verdict["real_odds"] = False
        verdict["odd"] = None
        verdict["ev"] = None
        verdict["edge"] = None
        verdict["stake"] = 0.0
        verdict["kelly_pct"] = 0.0
        verdict["is_bet"] = False
        return verdict, None

    return refine_with_real_odds(
        verdict,
        rows,
        {verdict["pick"]: odd},
        bank,
        kelly_frac,
        min_edge=min_edge,
        min_ev=min_ev,
    )


def refine_with_real_odds(
    verdict: dict,
    rows: list,
    real_odds,
    bank: float,
    kelly_frac: float,
    min_edge: float = 0.03,
    min_ev: float = 0.03,
):
    """
    Превращает model candidate в реальный betting signal.

    Требования:
    1. Есть реальный bookmaker odd.
    2. Probability модели выше implied probability рынка.
    3. Edge >= min_edge.
    4. EV >= min_ev.
    5. Kelly > 0.

    Никаких ставок по estimated/fair odds.
    """

    real_odds = real_odds or {}

    for r in rows:
        ro = real_odds.get(r["pick"])

        if ro is None:
            r["odd"] = None
            r["real"] = False
            continue

        try:
            ro = float(ro)
        except (TypeError, ValueError):
            r["odd"] = None
            r["real"] = False
            continue

        if ro <= 1.01:
            r["odd"] = None
            r["real"] = False
            continue

        r["odd"] = ro
        r["real"] = True

    pick = verdict["pick"]
    real_odd = real_odds.get(pick)

    if real_odd is None:
        verdict["real_odds"] = False
        verdict["odd"] = None
        verdict["ev"] = None
        verdict["edge"] = None
        verdict["stake"] = 0.0
        verdict["is_bet"] = False

        return verdict, None

    try:
        real_odd = float(real_odd)
    except (TypeError, ValueError):
        verdict["real_odds"] = False
        verdict["odd"] = None
        verdict["ev"] = None
        verdict["edge"] = None
        verdict["stake"] = 0.0
        verdict["is_bet"] = False

        return verdict, None

    if real_odd <= 1.01:
        verdict["real_odds"] = False
        verdict["odd"] = None
        verdict["ev"] = None
        verdict["edge"] = None
        verdict["stake"] = 0.0
        verdict["is_bet"] = False

        return verdict, None

    prob = float(verdict["prob"])

    market_edge = edge(prob, real_odd)
    ev = expected_value(prob, real_odd)

    verdict["odd"] = real_odd
    verdict["real_odds"] = True
    verdict["edge"] = market_edge
    verdict["ev"] = ev

    # Модельный probability threshold.
    if not verdict.get("is_action", False):
        verdict["stake"] = 0.0
        verdict["is_bet"] = False
        return verdict, None

    # Реальный market edge.
    if market_edge < min_edge:
        verdict["stake"] = 0.0
        verdict["is_bet"] = False
        return verdict, None

    # Реальный EV.
    if ev < min_ev:
        verdict["stake"] = 0.0
        verdict["is_bet"] = False
        return verdict, None

    stake = kelly(
        prob=prob,
        odds=real_odd,
        bank=bank,
        frac=kelly_frac,
        cap=0.05,
    )

    if stake <= 0:
        verdict["stake"] = 0.0
        verdict["is_bet"] = False
        return verdict, None

    verdict["stake"] = stake
    verdict["kelly_pct"] = (stake / float(bank)) if float(bank) > 0 else 0.0
    verdict["is_bet"] = True

    best = (
        market_type(pick),
        pick,
        real_odd,
        ev,
        prob,
        stake,
    )

    return verdict, best
