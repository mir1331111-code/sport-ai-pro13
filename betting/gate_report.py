"""Human-readable forecast and Value gates for dashboard diagnostics.

This module explains scanner decisions; it never changes or weakens eligibility.
"""
from __future__ import annotations

from dataclasses import dataclass
import html
import math

from betting.market_ranking import market_forecast_summary, rank_market_forecasts
from betting.risk import MIN_BET_ODDS
from config import MIN_FAVORITE_HISTORY, STRONG_FORECAST_MIN_PROB


@dataclass(frozen=True)
class GateResult:
    name: str
    passed: bool
    detail: str


STATUS_LEGEND = {
    "approved": (
        "✅", "Одобрено к ставке",
        "Все проверки прогноза, контекста, рынка, линии и EV/edge/Kelly пройдены.",
    ),
    "forecast": (
        "🟡", "Сильный прогноз — не ставка",
        "Истории и вероятности достаточно, но один или несколько Value-гейтов не пройдены.",
    ),
    "near_strong": (
        "🟠", "Почти сильный прогноз",
        "Истории достаточно, но вероятность немного ниже сильного порога.",
    ),
    "provisional": (
        "🟣", "Предварительный прогноз",
        "Недостаточно домашней/выездной истории или модель пока не поддержана командами.",
    ),
    "background": (
        "⚪", "Фоновый прогноз",
        "Порог сильного прогноза не достигнут; это не рекомендация к ставке.",
    ),
}


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _is_sstats(value):
    return "sstats" in str(value or "").casefold()


def _source_for_pick(card, pick, fallback=""):
    sources = card.get("market_odds_source_by_pick")
    if isinstance(sources, dict) and sources.get(pick):
        return sources[pick]
    return fallback


def _same_bookmaker(card, picks):
    books = card.get("bookmaker_by_pick")
    if not isinstance(books, dict):
        return False
    values = [str(books.get(pick) or "").strip().casefold() for pick in picks]
    return bool(values) and all(values) and len(set(values)) == 1


def _market_validation_gate(card, family):
    if family == "winner":
        report = card.get("winner_market_validation_report")
        report = report if isinstance(report, dict) else {}
        count = int(_number(report.get("n")) or 0)
        passed = bool(card.get("winner_market_validation"))
        detail = (
            f"walk-forward против de-vig рынка пройден, парных матчей {count}."
            if passed else
            f"walk-forward против de-vig рынка не пройден или не завершён: пар {count}/500."
        )
        return GateResult("Валидация рынка", passed, detail)

    if family == "goals":
        report = card.get("total_market_validation_report")
        report = report if isinstance(report, dict) else {}
        line_reports = card.get("total_market_validation_by_line")
        line_reports = line_reports if isinstance(line_reports, dict) else {}
        count = int(_number(report.get("n")) or 0)
        benchmark, exact_line = (
            bool(card.get("total_market_validation")),
            bool(line_reports.get("2.5")),
        )
        passed = benchmark and exact_line
        if passed:
            detail = f"de-vig benchmark и точная линия 2.5 пройдены, пар {count}."
        elif not benchmark:
            detail = f"нет доказанного улучшения над de-vig рынком для тотала 2.5: пар {count}/500."
        else:
            detail = "точная линия 2.5 не прошла отдельную историческую проверку."
        return GateResult("Валидация рынка", passed, detail)

    return GateResult(
        "Валидация рынка", False,
        "для этого рынка нет подтверждённого walk-forward улучшения над de-vig котировками букмекеров; доступен только прогноз.",
    )


def _coherent_line_gate(card, family):
    books = card.get("bookmaker_by_pick") or {}
    if family == "winner":
        picks = ("П1", "X", "П2")
        odds = card.get("market_odds") or {}
        missing = [pick for pick in picks if (_number(odds.get(pick)) or 0) <= 1]
        source_ok = _is_sstats(card.get("odds_source")) and all(
            _is_sstats(_source_for_pick(card, pick, card.get("odds_source")))
            for pick in picks
        )
        same_book = _same_bookmaker(card, picks)
        passed = not missing and source_ok and same_book
        if missing:
            detail = "неполная линия 1X2: отсутствуют цены " + ", ".join(missing) + "."
        elif not source_ok:
            detail = "не все три котировки подтверждены как цены SStats."
        elif not same_book:
            detail = "П1/X/П2 не принадлежат одному идентифицированному букмекеру."
        else:
            detail = f"П1/X/П2 получены от одного букмекера: {str(books.get('П1'))}."
        return GateResult("Полная линия одного букмекера", passed, detail)

    if family == "goals":
        picks = ("ТБ 2.5", "ТМ 2.5")
        odds = card.get("total_market_odds") or {}
        missing = [pick for pick in picks if (_number(odds.get(pick)) or 0) <= 1]
        source_ok = _is_sstats(card.get("total_odds_source")) and all(
            _is_sstats(_source_for_pick(card, pick, card.get("total_odds_source")))
            for pick in picks
        )
        same_book = _same_bookmaker(card, picks)
        confirmed = card.get("total_market_confirmed") or {}
        counts = card.get("total_market_book_count") or {}
        corroborated = all(
            bool(confirmed.get(pick))
            and int(_number(counts.get(pick)) or 0) >= 2
            for pick in picks
        )
        passed = not missing and source_ok and same_book and corroborated
        if missing:
            detail = "неполная линия тотала 2.5: отсутствуют " + ", ".join(missing) + "."
        elif not source_ok:
            detail = "котировки тотала не подтверждены источником SStats."
        elif not same_book:
            detail = "ТБ и ТМ 2.5 не принадлежат одному букмекеру."
        elif not corroborated:
            detail = "для обеих сторон нет подтверждения минимум от двух букмекеров."
        else:
            detail = f"обе стороны линии есть у {str(books.get(picks[0]))}; цены подтверждены."
        return GateResult("Полная линия одного букмекера", passed, detail)

    return GateResult(
        "Полная линия одного букмекера", False,
        "автоматический Value-допуск для этого рынка не включён.",
    )


def _context_gate(card):
    context = card.get("context_quality")
    context = context if isinstance(context, dict) else {}
    passed = bool(context.get("strong_value_eligible"))
    if passed:
        detail = "контекстный допуск сканера подтверждён."
    else:
        reason = str(context.get("reason") or "").strip()
        detail = (
            "контекстный допуск не пройден"
            + (f": {reason}." if reason else "; нет достаточного подтверждения состава/контекста.")
        )
    return GateResult("Контекст матча", passed, detail)


def _recommendation(card, family):
    mapping = card.get("best_by_market")
    recommendation = mapping.get(family) if isinstance(mapping, dict) else None
    if not isinstance(recommendation, dict):
        return None, None
    best, verdict = recommendation.get("best"), recommendation.get("verdict")
    return (
        best if isinstance(best, (tuple, list)) else None,
        verdict if isinstance(verdict, dict) else None,
    )


def _ev_edge_gate(card, family, market_gate, line_gate):
    best, verdict = _recommendation(card, family)
    if best is None or len(best) < 6 or verdict is None:
        if not market_gate.passed:
            detail = "EV/edge/Kelly не допускаются к расчёту до прохождения исторической проверки рынка."
        elif not line_gate.passed:
            detail = "EV/edge/Kelly не допускаются к ставке без полной подтверждённой линии."
        else:
            detail = (
                f"нет итоговой рекомендации: требуется EV ≥5%, Edge ≥5%, "
                f"коэффициент ≥{MIN_BET_ODDS:.2f} и положительный Kelly."
            )
        return GateResult("EV / edge / Kelly", False, detail)

    odds = _number(best[2])
    ev = _number(best[3])
    probability = _number(best[4])
    stake = _number(best[5])
    edge = _number(verdict.get("edge"))
    passed = bool(
        verdict.get("real_odds") and verdict.get("value_passed")
        and verdict.get("is_bet")
        and odds is not None and odds >= MIN_BET_ODDS
        and ev is not None and ev >= 0.05
        and edge is not None and edge >= 0.05
        and probability is not None and 0 < probability < 1
        and stake is not None and stake > 0
    )
    if passed:
        detail = f"EV={ev:+.1%}, Edge={edge:+.1%}, коэффициент {odds:.2f}, Kelly положительный."
    else:
        detail = (
            f"порог не пройден: EV={ev if ev is not None else '—'}, "
            f"Edge={edge if edge is not None else '—'}, "
            f"коэф.={odds if odds is not None else '—'} (минимум {MIN_BET_ODDS:.2f}), "
            f"Kelly={stake if stake is not None else '—'}."
        )
    return GateResult("EV / edge / Kelly", passed, detail)


def forecast_gates(card: dict, family: str):
    summary = market_forecast_summary(card, family)
    if summary is None:
        return [GateResult("Модельный прогноз", False, "нет корректной вероятности для выбранного рынка.")], None
    history_ok = bool(summary.get("history_sufficient"))
    probability = _number(summary.get("probability")) or 0.0
    gates = [
        GateResult(
            "История команд", history_ok,
            f"дома {int(summary.get('home_team_sample') or 0)}/{MIN_FAVORITE_HISTORY}, "
            f"в гостях {int(summary.get('away_team_sample') or 0)}/{MIN_FAVORITE_HISTORY} матчей.",
        ),
        GateResult(
            "Порог сильного прогноза", bool(summary.get("strong_forecast")),
            f"P={probability:.1%}, требуется ≥{STRONG_FORECAST_MIN_PROB:.0%}.",
        ),
    ]
    return gates, summary


def value_gates(card: dict, family: str):
    market = _market_validation_gate(card, family)
    line = _coherent_line_gate(card, family)
    context = _context_gate(card)
    value = _ev_edge_gate(card, family, market, line)
    return [market, line, context, value]


def classify_card(card: dict, family: str):
    forecast, summary = forecast_gates(card, family)
    values = value_gates(card, family)
    support = str(card.get("model_support_status") or "").strip().casefold()
    # Fail closed: only an explicit team-supported model state may enter one
    # of the three recommendation categories. Missing/unknown states are not
    # silently treated as evidence-backed forecasts.
    team_supported = support == "team_supported"
    history_ok = bool(summary and summary.get("history_sufficient"))
    probability = _number(summary.get("probability")) if summary else None

    if not summary or not team_supported or not history_ok:
        status = "provisional"
    elif (
        summary.get("strong_forecast")
        and all(item.passed for item in values)
    ):
        status = "approved"
    elif summary.get("strong_forecast"):
        status = "forecast"
    elif (
        probability is not None
        and probability >= float(STRONG_FORECAST_MIN_PROB) * 0.9
    ):
        status = "near_strong"
    else:
        status = "background"
    return status, forecast + values


def status_text(status: str):
    emoji, label, _hint = STATUS_LEGEND.get(status, STATUS_LEGEND["background"])
    return emoji, label


def near_misses(cards: list, family: str, limit: int = 3):
    result = []
    for item in rank_market_forecasts(cards, family):
        card = item["card"]
        status, gates = classify_card(card, family)
        failed = [gate for gate in gates if not gate.passed]
        if status == "approved" or not failed:
            continue
        result.append({
            "card": card,
            "match": card.get("match_ru") or card.get("match") or "Матч",
            "date": card.get("date_iso") or card.get("date") or "",
            "label": item.get("label"), "pick": item.get("pick"),
            "probability": float(item.get("probability") or 0.0),
            "status": status, "family": family, "failed_gates": failed,
            "_score": (len(failed), -float(item.get("probability") or 0.0)),
        })
    result.sort(key=lambda item: item["_score"])
    for item in result:
        item.pop("_score", None)
    return result[:max(0, int(limit))]


def render_status_legend():
    lines = ["| Статус | Значение |", "|---|---|"]
    for emoji, label, hint in STATUS_LEGEND.values():
        lines.append(f"| {emoji} **{label}** | {hint} |")
    return "\n".join(lines)


def render_gate_details_html(status: str, gates: list[GateResult]):
    emoji, label = status_text(status)
    rows = []
    for gate in gates:
        mark = "✅" if gate.passed else "❌"
        rows.append(
            f"<li>{mark} <b>{html.escape(gate.name)}</b> — "
            f"{html.escape(gate.detail)}</li>"
        )
    return (
        "<details class='neo-gate-details'>"
        f"<summary>{emoji} Почему такой статус · {html.escape(label)}</summary>"
        "<ul>" + "".join(rows) + "</ul></details>"
    )


def render_near_misses_html(items: list):
    if not items:
        return "<div class='neo-empty-cell'>Нет кандидатов для пояснения гейтов в выбранной выборке.</div>"
    blocks = []
    for item in items:
        emoji, label = status_text(item.get("status") or "background")
        missing = "; ".join(
            f"{gate.name}: {gate.detail}" for gate in item.get("failed_gates") or []
        )
        probability = _number(item.get("probability")) or 0.0
        blocks.append(
            "<div class='neo-near-miss'>"
            f"<b>{html.escape(str(item.get('match') or 'Матч'))}</b> · "
            f"{html.escape(str(item.get('date') or ''))} · "
            f"{html.escape(str(item.get('label') or item.get('pick') or '—'))} · "
            f"P={probability:.1%} · {emoji} {html.escape(label)}"
            f"<small>Не пройдены: {html.escape(missing)}</small>"
            "</div>"
        )
    return "<div class='neo-near-miss-list'>" + "".join(blocks) + "</div>"
