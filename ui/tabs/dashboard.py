"""Neural Terminal overview dashboard for NEURO BET PRO."""
from __future__ import annotations

import html
import math
from datetime import datetime, timedelta
import re

import streamlit as st

from betting.ranking import model_favorite, rank_best_fixtures
from betting.event_shortlist import (
    build_high_probability_market_events,
    _market_quote as _event_market_quote,
    _source as _event_market_source,
)
from betting.gate_report import (
    classify_card, near_misses, render_gate_details_html,
    render_near_misses_html, render_status_legend,
)
from betting.market_ranking import rank_market_forecasts
from data.fixture_time import app_timezone, fixture_kickoff_local
from config import (
    APP_VERSION, DB_FILE, LLM_PROVIDERS, MIN_FAVORITE_HISTORY,
    STRONG_FORECAST_MIN_PROB,
)
from betting.risk import MIN_BET_ODDS
from ui.cards import _LEAGUE_FLAGS
from model.forecast_ledger import ForecastLedger
from model.independent_forecasts import compare_forecast_rows


OUTCOMES = ("H", "D", "A")
OUTCOME_LABELS = {"H": "П1", "D": "X", "A": "П2"}

DASHBOARD_PERIOD_OPTIONS = (
    "Сегодня",
    "Ближайшие 24 часа",
    "Весь горизонт сканирования",
)
DASHBOARD_DEFAULT_PERIOD = "Весь горизонт сканирования"
DASHBOARD_FORECAST_VISIBLE_LIMIT = 50
STRONG_EVENT_MIN_EDGE = 0.05
STRONG_EVENT_MIN_EV = 0.05


def dashboard_strong_events(
    cards,
    *,
    min_probability=STRONG_FORECAST_MIN_PROB,
    min_odds=MIN_BET_ODDS,
    min_team_history=MIN_FAVORITE_HISTORY,
    min_edge=STRONG_EVENT_MIN_EDGE,
    min_ev=STRONG_EVENT_MIN_EV,
):
    """Collect only independently validated, executable strong market signals.

    The scanner's best_by_market selector has already enforced exact coherent
    same-bookmaker lines and market walk-forward validation. This final
    dashboard gate applies the user-facing probability/price/value floor.
    """
    if not isinstance(cards, list):
        return []
    selected = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        context = card.get("context_quality")
        if not isinstance(context, dict) or not context.get("strong_value_eligible"):
            continue
        features = card.get("independent_features")
        features = features if isinstance(features, dict) else {}
        home_n = num(features.get("home_home_sample"))
        away_n = num(features.get("away_away_sample"))
        if home_n is None or away_n is None:
            continue
        if min(home_n, away_n) < min_team_history:
            continue
        recommendations = card.get("best_by_market")
        if not isinstance(recommendations, dict):
            continue
        for family, recommendation in recommendations.items():
            if not isinstance(recommendation, dict):
                continue
            verdict = recommendation.get("verdict")
            best = recommendation.get("best")
            if not isinstance(verdict, dict) or not isinstance(best, (tuple, list)) or len(best) < 6:
                continue
            try:
                pick = str(best[1])
                odds = float(best[2])
                ev = float(verdict.get("ev"))
                edge = float(verdict.get("edge"))
                probability = float(verdict.get("prob"))
                stake = float(verdict.get("stake"))
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(value) for value in (odds, ev, edge, probability, stake)):
                continue
            if (
                not verdict.get("real_odds")
                or not verdict.get("value_passed")
                or not verdict.get("is_bet")
                or odds < min_odds
                or probability < min_probability
                or ev < min_ev
                or edge < min_edge
                or stake <= 0
            ):
                continue
            # Every actionable recommendation must carry explicit SStats
            # quote provenance; a fair/model price is never an executable odd.
            sources = card.get("market_odds_source_by_pick")
            source = str((sources or {}).get(pick) or "").casefold() if isinstance(sources, dict) else ""
            if "sstats" not in source:
                continue
            fixture_id = str(card.get("fixture_id") or "")
            fixture_key = fixture_id or "|".join((
                str(card.get("date_iso") or ""),
                str(card.get("home_team") or ""),
                str(card.get("away_team") or ""),
            ))
            key = (fixture_key, str(family))
            item = {
                "card": card,
                "family": str(family),
                "pick": pick,
                "odds": odds,
                "probability": probability,
                "ev": ev,
                "edge": edge,
                "stake": stake,
                "home_sample": int(home_n),
                "away_sample": int(away_n),
                "bookmaker": str((card.get("bookmaker_by_pick") or {}).get(pick) or ""),
            }
            previous = selected.get(key)
            if previous is None or (
                item["probability"], item["ev"], item["edge"]
            ) > (
                previous["probability"], previous["ev"], previous["edge"]
            ):
                selected[key] = item
    return sorted(
        selected.values(),
        key=lambda item: (
            item["probability"], item["ev"], item["edge"], -item["odds"]
        ),
        reverse=True,
    )


def dashboard_all_market_candidates(
    cards, *, min_probability=STRONG_FORECAST_MIN_PROB,
    min_odds=MIN_BET_ODDS, limit=30,
):
    """Return only benchmark-validated cross-market Value selections."""
    return build_high_probability_market_events(
        cards,
        min_probability=min_probability,
        min_odds=min_odds,
        limit=limit,
    )


def all_market_candidates_html(events):
    """Render only benchmark-validated cross-market Value selections."""
    events = [
        item for item in (events or [])
        if isinstance(item, dict) and bool(item.get("value_verified"))
    ]
    if not events:
        return (
            "<div class='neo-empty-cell'>"
            "Пока нет исходов, прошедших независимую проверку именно этого рынка, "
            "подтверждение линии одного букмекера и все Value-гейты. "
            "Непроверенные вероятности не выдаются за рекомендации."
            "</div>"
        )
    rows = []
    verified = len(events)
    for item in events:
        probability = num(item.get("probability"))
        odds = num(item.get("odds"))
        if probability is None or odds is None:
            continue
        status = "<span class='neo-bet-status approved'>ПРОВЕРЕННЫЙ VALUE</span>"
        sample = (
            f"{int(item.get('history_matches') or 0)} матчей лиги"
            if int(item.get("history_matches") or 0) > 0
            else f"история команд {int(item.get('team_samples') or 0)}"
        )
        rows.append(
            "<tr>"
            f"<td>{esc(item.get('date') or '—')} {esc(item.get('time') or '')}</td>"
            f"<td>{esc(item.get('league') or '—')}</td>"
            f"<td><b>{esc(item.get('match') or '—')}</b></td>"
            f"<td>{esc(item.get('family_label') or '—')}</td>"
            f"<td><b>{esc(item.get('pick') or '—')}</b></td>"
            f"<td><b>{probability:.1%}</b></td>"
            f"<td><b>{odds:.2f}</b></td>"
            f"<td>{esc(item.get('bookmaker') or '—')}</td>"
            f"<td>{status}</td>"
            f"<td>{esc(sample)}</td>"
            "</tr>"
        )
    return (
        "<div class='neo-table-scroll'><table class='neo-match-table'>"
        "<thead><tr><th>Время</th><th>Лига</th><th>Матч</th><th>Рынок</th>"
        "<th>Исход</th><th>Вероятность</th><th>Коэф.</th><th>Букмекер</th>"
        "<th>Статус</th><th>История</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
        f"<div class='neo-table-footer'>Показано {len(rows)} проверенных сигналов. "
        f"Проверенных Value-сигналов: {verified}. Независимый benchmark, "
        "полная линия одного букмекера, EV/Edge и проверка Kelly пройдены.</div>"
    )


def strong_events_html(events):
    """Render the actionable strong-event shortlist with transparent metrics."""
    if not events:
        return (
            "<div class='neo-empty-cell'>"
            "Пока нет событий, одновременно прошедших все фильтры: "
            f"вероятность ≥{STRONG_FORECAST_MIN_PROB:.0%}, коэффициент ≥{MIN_BET_ODDS:.2f}, "
            "EV ≥5%, Edge ≥5%, история обеих команд ≥10 и проверенная линия SStats. "
            "Неподходящие матчи не добавляются искусственно."
            "</div>"
        )
    rows = []
    for item in events:
        card = item["card"]
        rows.append(
            "<tr>"
            f"<td>{esc(card.get('date') or card.get('date_iso') or '—')}</td>"
            f"<td>{esc(card.get('league') or card.get('div') or '—')}</td>"
            f"<td>{esc(match_name(card))}</td>"
            f"<td><b>{esc(item['pick'])}</b></td>"
            f"<td><b>{item['odds']:.2f}</b></td>"
            f"<td>{item['probability']:.1%}</td>"
            f"<td>{item['ev']:+.1%}</td>"
            f"<td>{item['edge']:+.1%}</td>"
            f"<td>{item['home_sample']}/{item['away_sample']}</td>"
            "</tr>"
        )
    return (
        "<div class='neo-table-scroll'><table class='neo-match-table'>"
        "<thead><tr><th>Время</th><th>Лига</th><th>Матч</th><th>Событие</th>"
        "<th>Коэф.</th><th>Вероятность</th><th>EV</th><th>Edge</th>"
        "<th>История Д/Г</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def best_matches_html(items):
    """Render the highest-quality team-supported forecasts independently of bet approval."""
    if not items:
        return (
            "<div class='neo-empty-cell'>"
            "В выбранном периоде нет матчей, одновременно имеющих подтверждённую "
            f"историю дом/выезд не менее {MIN_FAVORITE_HISTORY} игр и фаворита от 52%. "
            "Слабые и prior-only прогнозы в этот рейтинг не добавляются."
            "</div>"
        )
    rows = []
    for number, item in enumerate(items, start=1):
        card = item.get("card") if isinstance(item, dict) else {}
        card = card if isinstance(card, dict) else {}
        calibration = (
            f"Да · n={int(item.get('calibration_matches') or 0)}"
            if item.get("calibration_ready")
            else "нет · сырая вероятность"
        )
        probability = num(item.get("probability"))
        margin = num(item.get("probability_margin"))
        quality = num(item.get("quality_score"))
        probability_text = f"{probability:.1%}" if probability is not None else "—"
        margin_text = f"{margin:.1%}" if margin is not None else "—"
        quality_text = f"{quality:.0f}/100" if quality is not None else "—"
        rows.append(
            "<tr>"
            f"<td>{number}</td>"
            f"<td>{esc(card.get('date') or card.get('date_iso') or '—')} "
            f"{esc(card.get('time') or '')}</td>"
            f"<td>{esc(card.get('league') or card.get('div') or '—')}</td>"
            f"<td><b>{esc(match_name(card))}</b></td>"
            f"<td><b>{esc(item.get('label') or item.get('pick') or '—')}</b></td>"
            f"<td>{probability_text}</td>"
            f"<td>{margin_text}</td>"
            f"<td>{int(item.get('home_sample') or 0)}/"
            f"{int(item.get('away_sample') or 0)}</td>"
            f"<td>{quality_text}</td>"
            f"<td>{esc(calibration)}</td>"
            "</tr>"
        )
    return (
        "<div class='neo-table-scroll'><table class='neo-match-table'>"
        "<thead><tr><th>№</th><th>Время</th><th>Лига</th><th>Матч</th>"
        "<th>Наиболее вероятный исход</th><th>Вероятность</th><th>Отрыв от 2-го</th>"
        "<th>История Д/Г</th><th>Качество</th><th>Калибровка</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


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


def ai_provider_label(data):
    provider = str((data.get("meta") or {}).get("llm_provider") or "Groq GPT-OSS 120B")
    base = str(LLM_PROVIDERS.get(provider, {}).get("base") or "").casefold()
    if "api.x.ai" in base:
        return "Grok"
    if "groq" in base:
        return "Groq"
    if "generativelanguage.googleapis.com" in base:
        return "Gemini"
    return provider.split(" (", 1)[0] or "AI"


def has_prices(card):
    prices = card.get("market_odds") or {}
    return (
        isinstance(prices, dict)
        and str(card.get("odds_source") or "").casefold() not in {"", "no_market", "model_only"}
        and all(1.01 < (num(prices.get(k)) or 0) <= 100 for k in ("П1", "X", "П2"))
    )

def dashboard_bet_approved(card):
    """Only admit a match to the dashboard bet list after every gate passes."""
    if not isinstance(card, dict) or card.get("_daily_strict_eligible") is not True:
        return False
    verdict = card.get("verdict") if isinstance(card.get("verdict"), dict) else {}
    pick = str(verdict.get("pick") or "").strip()
    return bool(
        has_prices(card)
        and verdict.get("real_odds")
        and verdict.get("value_passed")
        and verdict.get("is_bet")
        and pick in {"П1", "X", "П2"}
    )


def _provider_is_grok(provider):
    name = str(provider or "").strip()
    details = LLM_PROVIDERS.get(name, {})
    base = str(details.get("base") or "").casefold()
    return "api.x.ai" in base or "grok" in name.casefold()


def _probability_health_note(cards):
    """Detect a repeated probability vector that usually signals prior-only forecasts."""
    counts = {}
    valid = 0
    for card in cards or []:
        values = [num(card.get(key)) for key in ("p1", "px", "p2")]
        if any(value is None or not 0 <= value <= 1 for value in values):
            continue
        if not 0.90 <= sum(values) <= 1.10:
            continue
        valid += 1
        signature = tuple(round(value, 3) for value in values)
        counts[signature] = counts.get(signature, 0) + 1
    if valid < 5 or not counts:
        return ""
    signature, repeated = max(counts.items(), key=lambda item: item[1])
    if repeated < max(5, math.ceil(valid * 0.60)):
        return ""
    return (
        f"⚠️ Повторяется один и тот же вектор 1X2 у {repeated} из {valid} матчей "
        f"(П1 {signature[0]:.0%} · X {signature[1]:.0%} · П2 {signature[2]:.0%}). "
        "Это похоже на прогноз по нейтральным priors при недостатке истории команд. "
        "Такие матчи не должны попадать в список допущенных ставок."
    )


def _grok_panel(cards, meta):
    """Render independent AI analyses only when they match the currently selected provider."""
    meta = meta if isinstance(meta, dict) else {}
    selected_provider = str(meta.get("llm_provider") or "Groq GPT-OSS 120B")
    selected_is_grok = _provider_is_grok(selected_provider)
    selected_label = ai_provider_label({"meta": {"llm_provider": selected_provider}})
    rows = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        analysis = card.get("deep_analysis")
        analysis = analysis if isinstance(analysis, dict) else {}
        analysis_provider = str(card.get("deep_analysis_provider") or "")
        matching_analysis = bool(analysis and analysis_provider == selected_provider)
        grok_comment = str(card.get("grok_comment") or "").strip()
        legacy_grok_comment = bool(selected_is_grok and grok_comment)
        if not matching_analysis and not legacy_grok_comment:
            continue
        title = match_name(card)
        body = []
        if matching_analysis:
            probabilities = analysis.get("probabilities")
            probabilities = probabilities if isinstance(probabilities, dict) else {}
            values = {key: num(probabilities.get(key)) for key in ("П1", "X", "П2")}
            if all(value is not None and 0 <= value <= 100 for value in values.values()):
                total = sum(values.values())
                if total > 0 and abs(total - 100) <= 5:
                    body.append(
                        "<div class='neo-grok-probs'>"
                        + "".join(
                            f"<span>{key}<b>{value:.1f}%</b></span>"
                            for key, value in values.items()
                        )
                        + "</div>"
                    )
            independent_pick = str(analysis.get("independent_pick") or "").strip()
            model_prediction = favorite(card) or {}
            model_pick = str(model_prediction.get("pick") or "")
            if independent_pick in {"П1", "X", "П2"}:
                comparison = (
                    "совпадает с основной моделью"
                    if independent_pick == model_pick
                    else f"расходится с основной моделью ({model_pick or 'нет прогноза'})"
                )
                body.append(
                    f"<div class='neo-grok-meta'>AI · {esc(analysis_provider)}: "
                    f"<b>{esc(independent_pick)}</b> · {esc(comparison)} · "
                    f"модель {esc(card.get('deep_analysis_model') or 'не указана')}</div>"
                )
            crosscheck = card.get("ai_crosscheck")
            if isinstance(crosscheck, dict):
                value_pick = str(crosscheck.get("validated_value_pick") or "")
                value_odds = num(crosscheck.get("validated_value_odds"))
                value_text = (
                    f"Проверенный Value-селектор: {esc(value_pick)} @ {value_odds:.2f}. "
                    if value_pick in {"П1", "X", "П2"} and value_odds is not None
                    else "Проверенный 1X2 Value-сигнал отсутствует. "
                )
                ai_value_state = (
                    "AI совпадает с Value-выбором."
                    if crosscheck.get("ai_agrees_with_value_pick") is True
                    else "AI расходится с Value-выбором."
                    if crosscheck.get("ai_agrees_with_value_pick") is False
                    else "AI не сравнивался с 1X2 Value-выбором."
                )
                body.append(
                    f"<div class='neo-grok-meta'>{value_text}{esc(ai_value_state)} "
                    "Диагностика не меняет вероятность, EV, Edge или размер ставки.</div>"
                )
            for field, label in (
                ("lineup_assessment", "Составы"),
                ("injury_assessment", "Травмы / дисквалификации"),
                ("referee_assessment", "Судья"),
                ("form_assessment", "Форма"),
            ):
                detail = str(analysis.get(field) or "").strip()
                if detail:
                    body.append(
                        f"<p class='neo-grok-copy'><b>{esc(label)}:</b> {esc(detail)}</p>"
                    )
            assessment = str(
                analysis.get("match_assessment") or analysis.get("opinion") or ""
            ).strip()
            if assessment:
                body.append(f"<p class='neo-grok-copy'>{esc(assessment)}</p>")
            risks = str(analysis.get("risks") or "").strip()
            if risks:
                body.append(f"<p class='neo-grok-risk'><b>Риски:</b> {esc(risks)}</p>")
            quality = str(analysis.get("data_quality") or "").strip()
            if quality:
                body.append(f"<small class='neo-grok-meta'>Качество данных: {esc(quality)}</small>")
        if legacy_grok_comment:
            bookmaker = str(card.get("grok_bookmaker") or "источник не указан")
            body.append(
                f"<p class='neo-grok-copy'>{esc(grok_comment)}</p>"
                f"<small class='neo-grok-meta'>Источник: {esc(bookmaker)}</small>"
            )
        rows.append(
            "<article class='neo-grok-row'><div class='neo-grok-match'>"
            + esc(title) + "</div>" + "".join(body) + "</article>"
        )
        if len(rows) >= 4:
            break

    if not rows:
        if selected_is_grok:
            message = (
                "Выбран Grok (x.ai), но сохранённого разбора пока нет. "
                "Укажите действующий XAI_API_KEY, проверьте подключение и обновите TOP MATCHES."
            )
        else:
            message = (
                f"Выбран {selected_label} · модель "
                f"{LLM_PROVIDERS.get(selected_provider, {}).get('model') or 'не указана'}. "
                "После успешной проверки API запустите обновление SStats-досье и TOP MATCHES."
            )
        content = (
            "<div class='neo-grok-empty'><b>Независимый AI-разбор пока не сохранён</b>"
            f"<span>{esc(message)}</span></div>"
        )
    else:
        content = "".join(rows)
    return (
        "<section class='neo-panel neo-grok-panel'>"
        "<div class='neo-panel-heading'><div>"
        f"<span class='neo-panel-kicker'>INDEPENDENT AI · {esc(selected_label)}</span>"
        "<h3>Независимый AI-анализ</h3></div>"
        f"<span class='neo-count-pill'>{len(rows)} разборов</span></div>"
        + content + "</section>"
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
    return str(card.get("match_ru") or card.get("match") or (
        f"{card.get('home_team') or 'Хозяева'} — {card.get('away_team') or 'Гости'}"
    )).replace(" vs ", " — ")


def _rolling_series(values, window=12):
    values = [float(value) for value in values if num(value) is not None]
    if len(values) < 2:
        return []
    result = []
    for index in range(len(values)):
        sample = values[max(0, index - window + 1):index + 1]
        result.append(sum(sample) / len(sample))
    return result


def _forecast_metrics(rows):
    settled = sorted(
        [
            row for row in rows
            if row.get("result") in OUTCOMES
            and isinstance(row.get("probabilities"), dict)
            and all(num((row.get("probabilities") or {}).get(key)) is not None for key in OUTCOMES)
        ],
        key=lambda row: str(row.get("forecast_at") or ""),
    )
    if not settled:
        return {
            "n": 0, "brier": None, "log_loss": None, "accuracy": None,
            "market_delta_brier": None, "market_n": 0,
            "brier_series": [], "log_loss_series": [], "accuracy_series": [],
        }

    briers, losses, correct = [], [], 0
    market_differences = []
    brier_values, loss_values, accuracy_values = [], [], []
    for row in settled:
        probabilities = row["probabilities"]
        actual = row["result"]
        row_brier = sum(
            (float(probabilities[key]) - float(key == actual)) ** 2
            for key in OUTCOMES
        )
        row_loss = -math.log(max(float(probabilities[actual]), 1e-15))
        briers.append(row_brier)
        losses.append(row_loss)
        predicted = max(OUTCOMES, key=lambda key: float(probabilities[key]))
        correct += int(predicted == actual)
        brier_values.append(row_brier)
        loss_values.append(row_loss)
        accuracy_values.append(float(predicted == actual))

        prices = row.get("entry_odds")
        if isinstance(prices, dict) and all(
            num(prices.get(key)) and num(prices.get(key)) > 1 for key in OUTCOMES
        ):
            implied = {key: 1.0 / float(prices[key]) for key in OUTCOMES}
            total = sum(implied.values())
            market = {key: implied[key] / total for key in OUTCOMES}
            market_brier = sum(
                (market[key] - float(key == actual)) ** 2 for key in OUTCOMES
            )
            market_differences.append(row_brier - market_brier)

    n = len(settled)
    return {
        "n": n,
        "brier": sum(briers) / n,
        "log_loss": sum(losses) / n,
        "accuracy": correct / n,
        "market_delta_brier": (
            sum(market_differences) / len(market_differences)
            if market_differences else None
        ),
        "market_n": len(market_differences),
        "brier_series": _rolling_series(brier_values),
        "log_loss_series": _rolling_series(loss_values),
        "accuracy_series": _rolling_series(accuracy_values),
    }

def _ai_comparison_html(report):
    """Show paired settled results for the main model and each independent AI version."""
    report = report if isinstance(report, dict) else {}
    models = [
        item for item in (report.get("models") or [])
        if isinstance(item, dict) and int(item.get("n") or 0) > 0
    ]
    if not report.get("ai_forecasts"):
        detail = "Независимые AI-прогнозы ещё не попали в журнал. Выполни скан с успешным TOP MATCHES."
    elif not models:
        detail = (
            "AI-прогнозы уже записываются отдельно, но пока нет матчей с подтверждённым FT, "
            "которые присутствуют одновременно в журнале основной модели и AI."
        )
    else:
        detail = (
            "Сравнение только на одинаковых матчах с подтверждённым результатом FT. "
            "Brier и Log Loss: меньше — лучше. Это диагностическая оценка, не автоматическое "
            "смешивание вероятностей и не разрешение на ставку."
        )
    rows = []
    for item in sorted(
        models,
        key=lambda value: (
            int(value.get("n") or 0),
            str(value.get("model_version") or ""),
        ),
        reverse=True,
    )[:5]:
        main = item.get("main") or {}
        ai = item.get("ai") or {}
        metric_values = (
            num(main.get("brier")),
            num(ai.get("brier")),
            num(item.get("brier_delta")),
            num(item.get("log_loss_delta")),
            num(item.get("accuracy_delta")),
        )
        formats = (".4f", ".4f", "+.4f", "+.4f", "+.1%")
        cells = "".join(
            f"<td>{format(value, spec) if value is not None else '—'}</td>"
            for value, spec in zip(metric_values, formats)
        )
        rows.append(
            "<tr>"
            f"<td>{esc(item.get('model_version') or 'AI')}</td>"
            f"<td>{int(item.get('n') or 0)}</td>"
            + cells
            + "</tr>"
        )
    table = (
        "<div class='neo-table-scroll'><table class='neo-match-table'>"
        "<thead><tr><th>AI-версия</th><th>Парных FT</th><th>Brier · модель</th>"
        "<th>Brier · AI</th><th>Δ Brier</th><th>Δ Log Loss</th><th>Δ Accuracy</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table></div>"
        if rows else
        "<div class='neo-chart-empty'><strong>Пока нет парных результатов</strong>"
        f"<span>{esc(detail)}</span></div>"
    )
    return (
        "<section class='neo-panel neo-ai-comparison-panel'>"
        "<div class='neo-panel-heading'><div>"
        "<span class='neo-panel-kicker'>MODEL ↔ AI · VERIFIED FT COMPARISON</span>"
        "<h3>Сверка основной модели и независимого AI</h3>"
        "</div>"
        f"<span class='neo-count-pill'>{int(report.get('paired_n') or 0)} парных матчей</span></div>"
        f"<div class='neo-ranking-disclaimer'>{esc(detail)}</div>"
        + table
        + "</section>"
    )


def _ledger_snapshot():
    try:
        ledger = ForecastLedger(DB_FILE)
        all_rows = ledger.rows()
        main_history = [
            row for row in all_rows
            if str(row.get("model_version") or "").startswith("main-model-")
        ]
        active_version = f"main-model-v{APP_VERSION}"
        active_rows = [
            row for row in main_history
            if str(row.get("model_version") or "") == active_version
        ]
        settled = [row for row in active_rows if row.get("result") in OUTCOMES]
        metrics = _forecast_metrics(active_rows)
        reliability = ledger.reliability_bins(
            bins=10, model_version=active_version
        )
        ai_comparison = compare_forecast_rows(all_rows)
        return {
            "ok": True,
            "rows": active_rows,
            "all_rows": main_history,
            "comparison_rows": all_rows,
            "ai_comparison": ai_comparison,
            "active_version": active_version,
            "settled": settled,
            "pending": max(0, len(active_rows) - len(settled)),
            "metrics": metrics,
            "reliability": reliability,
        }
    except Exception as exc:
        return {
            "ok": False, "error": type(exc).__name__, "rows": [],
            "settled": [], "pending": 0,
            "metrics": _forecast_metrics([]), "reliability": {"rows": [], "n": 0},
        }


def _reliability_svg(report):
    """Pooled one-vs-rest reliability bars, built only from settled 1X2 rows."""
    source_rows = report.get("rows") or []
    bins = 10
    buckets = [
        {"n": 0, "probability_sum": 0.0, "observed_sum": 0.0}
        for _ in range(bins)
    ]
    for row in source_rows:
        probability = num(row.get("mean_probability"))
        observed = num(row.get("observed_frequency"))
        n = int(row.get("n", 0) or 0)
        index = int(row.get("bin_index", -1))
        if probability is None or observed is None or n <= 0 or not 0 <= index < bins:
            continue
        bucket = buckets[index]
        bucket["n"] += n
        bucket["probability_sum"] += probability * n
        bucket["observed_sum"] += observed * n

    width, height = 420, 238
    left, right, top, bottom = 40, 400, 16, 190
    plot_w, plot_h = right - left, bottom - top
    grid = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - tick * plot_h
        x = left + tick * plot_w
        grid.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{right}' y2='{y:.1f}' class='neo-gridline'/>"
            f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{bottom}' class='neo-gridline'/>"
            f"<text x='{left-8}' y='{y+4:.1f}' text-anchor='end' class='neo-axis'>{tick:.2f}</text>"
            f"<text x='{x:.1f}' y='{bottom+18}' text-anchor='middle' class='neo-axis'>{tick:.1f}</text>"
        )

    bar_width = plot_w / bins * 0.58
    bars = []
    for index, bucket in enumerate(buckets):
        n = bucket["n"]
        if n < 5:
            continue
        mean_probability = bucket["probability_sum"] / n
        observed = bucket["observed_sum"] / n
        x_center = left + (index + 0.5) * plot_w / bins
        bar_height = max(0.0, min(1.0, observed)) * plot_h
        y = bottom - bar_height
        title = (
            f"Бин {index + 1}: прогноз {mean_probability:.1%}, "
            f"факт {observed:.1%}, наблюдений {n}"
        )
        bars.append(
            f"<rect x='{x_center - bar_width/2:.1f}' y='{y:.1f}' "
            f"width='{bar_width:.1f}' height='{bar_height:.1f}' rx='3' "
            f"class='neo-calibration-bar'><title>{esc(title)}</title></rect>"
        )

    paths = (
        f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{top}' class='neo-ideal'/>"
    )
    legend = (
        "<div class='neo-chart-legend'>"
        "<span><i style='background:#35d39a'></i>Фактическая частота</span>"
        "<span><i class='neo-ideal-key'></i>Идеальная калибровка</span>"
        "</div>"
    )
    return (
        "<div class='neo-chart-wrap'>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Калибровка вероятностей по бинам'>"
        + "".join(grid) + paths + "".join(bars)
        + f"<text x='{(left+right)/2}' y='{height-4}' text-anchor='middle' class='neo-axis-title'>Средняя прогнозная вероятность</text>"
        + f"<text x='10' y='{(top+bottom)/2}' text-anchor='middle' transform='rotate(-90 10 {(top+bottom)/2})' class='neo-axis-title'>Фактическая частота</text>"
        + "</svg>" + legend + "</div>"
    )

def _team_badge_html(name, url="", size=24):
    safe_url = _https_image_url(url)
    initial = esc((str(name or "?").strip()[:1] or "?").upper())
    inner = initial
    if safe_url:
        inner += (
            f"<img src='{esc(safe_url)}' alt='{esc(name)}' loading='lazy' "
            "decoding='async' referrerpolicy='no-referrer' "
            "onerror=\"this.style.display='none'\"/>"
        )
    return (
        f"<span class='neo-team-badge' style='--badge-size:{int(size)}px' "
        f"aria-label='{esc(name)}'>{inner}</span>"
    )


def _https_image_url(value):
    """Normalize safe provider image URLs, including legacy HTTP and // URLs."""
    source = str(value or "").strip()
    if source.startswith("//"):
        return "https:" + source
    if source.casefold().startswith("http://"):
        return "https://" + source[7:]
    if source.casefold().startswith("https://"):
        return source
    return ""


def _forecast_rejection_reason(card):
    """Explain why a visible forecast is not an approved bet."""
    reasons = []
    if card.get("model_support_status") in {"prior_only", "league_only"}:
        return (
            "нет истории завершённых матчей по этому турниру: показанный общий prior "
            "не является прогнозом по данным команд"
        )
    selection_note = str(card.get("_daily_selection_note") or "").strip()
    if selection_note:
        selection_note = selection_note.replace("Ознакомительный прогноз · ", "", 1)
        if selection_note and selection_note not in reasons:
            reasons.append(selection_note)

    context = card.get("context_quality")
    context = context if isinstance(context, dict) else {}
    if not context.get("strong_value_eligible"):
        reason = str(context.get("reason") or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)

    verdict = card.get("verdict")
    verdict = verdict if isinstance(verdict, dict) else {}
    verdict_reasons = verdict.get("reasons")
    if isinstance(verdict_reasons, list):
        for item in verdict_reasons:
            text = str(item or "").strip()
            if text and ("Strong/Value" in text or "Value" in text or "Kelly" in text):
                if text not in reasons:
                    reasons.append(text)

    if not has_prices(card):
        reasons.append("нет полной подтверждённой линии П1/X/П2 одного букмекера")
    elif not verdict.get("real_odds"):
        reasons.append("рыночная линия не подтверждена для расчёта Value")
    elif not verdict.get("value_passed"):
        reasons.append("не пройдены Value / Edge / EV")
    elif not verdict.get("is_bet"):
        reasons.append("ставка не прошла итоговый допуск / Kelly")

    # Keep a concise explanation in the narrow table cell.
    return "; ".join(dict.fromkeys(reasons)) or "прогноз информационный; допуска к ставке нет"


def _match_table_row(card, *, preview_only=False):
    unsupported = card.get("model_support_status") in {"prior_only", "league_only"}
    prediction = {} if unsupported else (favorite(card) or {})
    values = []
    for key in ("p1", "px", "p2"):
        value = None if unsupported else num(card.get(key))
        values.append(
            f"<span class='neo-prob-pill'>{f'{value:.0%}' if value is not None and 0 <= value <= 1 else '—'}</span>"
        )

    odds = card.get("market_odds") or {}
    odds_cells = []
    for key in ("П1", "X", "П2"):
        price = num(odds.get(key)) if has_prices(card) else None
        odds_cells.append(
            f"<td class='neo-odds'>{f'{price:.2f}' if price is not None else '—'}</td>"
        )

    pick = "НЕТ ПРОГНОЗА · НЕТ ИСТОРИИ КОМАНД" if unsupported else str(prediction.get("pick") or "—")
    probability = None if unsupported else num(prediction.get("probability"))
    pick_text = f"{pick} · {probability:.0%}" if probability is not None else pick
    verdict = card.get("verdict") if isinstance(card.get("verdict"), dict) else {}
    value_passed = bool(
        not preview_only and verdict.get("real_odds")
        and verdict.get("value_passed") and verdict.get("is_bet")
    )
    has_confirmed_line = has_prices(card)
    if preview_only:
        reason_text = _forecast_rejection_reason(card)
        bet_status_html = (
            "<small class='neo-bet-status blocked'>ПРОГНОЗ · НЕ ДОПУЩЕН К СТАВКЕ</small>"
            f"<small class='neo-pick-note provisional'>{esc(reason_text)}</small>"
        )
    elif value_passed and verdict.get("is_bet"):
        bet_status_html = (
            "<small class='neo-bet-status approved'>VALUE · СТАВКА ДОПУЩЕНА</small>"
        )
    elif has_confirmed_line:
        bet_status_html = (
            "<small class='neo-bet-status blocked'>НЕ ДОПУЩЕНО К СТАВКЕ · "
            "строгий Value-фильтр не пройден</small>"
        )
    else:
        bet_status_html = (
            "<small class='neo-bet-status forecast-only'>ТОЛЬКО ПРОГНОЗ · "
            "нет подтверждённой полной линии 1X2</small>"
        )
    strict_daily = card.get("_daily_strict_eligible")
    daily_note = str(card.get("_daily_selection_note") or "").strip()
    quality_score = num(card.get("_daily_quality_score"))
    score_html = (
        f"<small class='neo-pick-note rank-score'>БАЛЛ СОРТИРОВКИ · {quality_score:.0f}/100</small>"
        if quality_score is not None else ""
    )
    if strict_daily is True:
        selection_html = (
            "<small class='neo-pick-note proven'>ИСТОРИЯ И ФАВОРИТ ПОДТВЕРЖДЕНЫ</small>"
            + score_html
        )
    elif strict_daily is False:
        note_text = daily_note.replace("Ознакомительный прогноз · ", "", 1)
        sample_text = (
            f"Дом/гость: {int(card.get('_daily_home_sample') or 0)}/"
            f"{int(card.get('_daily_away_sample') or 0)}"
        )
        selection_html = (
            f"<small class='neo-pick-note provisional'>ОЗНАКОМИТЕЛЬНО · "
            f"{esc(note_text or 'ограниченная доказательная база')} · "
            f"{esc(sample_text)}</small>"
            + score_html
        )
    else:
        selection_html = score_html

    league = str(card.get("league") or card.get("div") or "Футбол")
    league_code = str(card.get("div") or "")
    league_logo = str(card.get("league_badge") or "")
    safe_league_logo = _https_image_url(league_logo)
    league_mark = (
        f"<img src='{esc(safe_league_logo)}' alt='' loading='lazy' decoding='async' "
        "referrerpolicy='no-referrer' onerror=\"this.style.display='none'\"/>"
        if safe_league_logo
        else f"<span class='neo-league-flag'>{esc(_LEAGUE_FLAGS.get(league_code, '⚽'))}</span>"
    )
    time_text = str(card.get("time") or "").strip()
    date_text = str(card.get("date") or card.get("date_iso") or "").strip()
    if time_text and time_text in date_text:
        kickoff_text = date_text
    else:
        kickoff_text = " · ".join(part for part in (time_text, date_text) if part)

    home_name = str(card.get("home_team") or "Хозяева")
    away_name = str(card.get("away_team") or "Гости")
    chosen = card.get("best")
    market_code = (
        str(chosen[0]).upper()
        if isinstance(chosen, (tuple, list)) and chosen else ""
    )
    gate_family = "goals" if market_code == "OU" else "winner"
    gate_status, gate_results = classify_card(card, gate_family)
    gate_details = render_gate_details_html(gate_status, gate_results)
    return (
        "<tr>"
        f"<td class='neo-time'>{esc(kickoff_text or '—')}</td>"
        f"<td><span class='neo-league'>{league_mark}{esc(league)}</span></td>"
        "<td class='neo-fixture'><div class='neo-fixture-line'>"
        "<span class='neo-team'>"
        + _team_badge_html(home_name, card.get("home_badge"), 21)
        + f"<b>{esc(home_name)}</b></span><span class='neo-vs'>—</span>"
        "<span class='neo-team'>"
        + _team_badge_html(away_name, card.get("away_badge"), 21)
        + f"<b>{esc(away_name)}</b></span>"
        "</div></td>"
        + "".join(f"<td class='neo-prob'>{value}</td>" for value in values)
        + "".join(odds_cells)
        + f"<td class='neo-pick-cell'><span class='neo-pick {'value' if value_passed else ''}'>{esc(pick_text)}</span>{bet_status_html}{selection_html}{gate_details}</td>"
        + "</tr>"
    )


def _sparkline_svg(values, color="#39d49b"):
    cleaned = [num(value) for value in (values or [])]
    cleaned = [value for value in cleaned if value is not None]
    if len(cleaned) < 2:
        return ""
    width, height, pad = 118, 34, 3
    low, high = min(cleaned), max(cleaned)
    span = high - low or max(abs(high) * 0.08, 0.01)
    points = []
    for index, value in enumerate(cleaned):
        x = pad + index * (width - pad * 2) / (len(cleaned) - 1)
        y = height - pad - ((value - low) / span) * (height - pad * 2)
        points.append((x, y))
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    area = (
        f"M {points[0][0]:.1f},{height-pad:.1f} "
        + " ".join(f"L {x:.1f},{y:.1f}" for x, y in points)
        + f" L {points[-1][0]:.1f},{height-pad:.1f} Z"
    )
    return (
        f"<svg class='neo-sparkline' viewBox='0 0 {width} {height}' "
        "preserveAspectRatio='none' aria-hidden='true'>"
        f"<path d='{area}' fill='{color}' opacity='.12'/>"
        f"<polyline points='{line}' fill='none' stroke='{color}' "
        "stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/>"
        "</svg>"
    )


def _kpi(label, value, note="", tone="", series=None, color="#39d49b"):
    sparkline = _sparkline_svg(series, color=color)
    trend_ready = len([item for item in (series or []) if num(item) is not None]) >= 2
    if not sparkline:
        sparkline = (
            "<div class='neo-spark-empty' aria-label='Недостаточно истории для динамики'>"
            "<span>НЕТ ИСТОРИИ</span><b>—</b></div>"
        )
    status = "" if trend_ready else (
        "НЕТ ОБОРОТА" if "ROI" in str(label).upper() else "НЕТ ВЫБОРКИ"
    )
    status_html = (
        f"<span class='neo-kpi-status'>{esc(status)}</span>"
        if status else "<span class='neo-kpi-mark' aria-hidden='true'></span>"
    )
    return (
        f"<div class='neo-kpi {esc(tone)}'>"
        "<div class='neo-kpi-label-row'>"
        f"<div class='neo-kpi-label'>{esc(label)}</div>"
        + status_html
        + "</div>"
        "<div class='neo-kpi-main'>"
        f"<div class='neo-kpi-value'>{esc(value)}</div>"
        f"<div class='neo-kpi-spark'>{sparkline}</div>"
        "</div>"
        f"<div class='neo-kpi-note'>{esc(note)}</div>"
        "</div>"
    )


def _quality_panel(cards):
    total = len(cards)
    summaries = [context_summary(card) for card in cards]
    metrics = [
        ("Коэффициенты 1X2", sum(has_prices(card) for card in cards)),
        ("Подтверждённые составы XI", sum(item["xi"] for item in summaries)),
        ("Назначенный судья", sum(item["referee"] for item in summaries)),
        ("Подтверждённый фид травм", sum(item["injuries"] for item in summaries)),
    ]
    tracked = len(metrics) * total
    quality_count = sum(count for _label, count in metrics)
    quality_pct = (quality_count / tracked * 100) if tracked else None
    quality_text = f"{quality_pct:.0f}%" if quality_pct is not None else "—"
    circumference = 2 * math.pi * 39
    dash = circumference * quality_pct / 100 if quality_pct is not None else 0
    gauge = (
        "<svg class='neo-gauge-svg' viewBox='0 0 100 100' role='img' "
        f"aria-label='Покрытие проверяемых источников {esc(quality_text)}'>"
        f"<circle cx='50' cy='50' r='39' fill='none' stroke='#263d39' stroke-width='8'/>"
        f"<circle cx='50' cy='50' r='39' fill='none' stroke='#39d49b' stroke-width='8' "
        f"stroke-linecap='round' stroke-dasharray='{dash:.1f} {circumference:.1f}' "
        "transform='rotate(-90 50 50)'/>"
        "</svg>"
    )
    rows = []
    for label, count in metrics:
        pct = count / max(1, total) * 100
        rows.append(
            "<div class='neo-quality-row'>"
            f"<div><span>{esc(label)}</span><b>{count}/{total}</b></div>"
            f"<div class='neo-progress'><i style='width:{pct:.1f}%'></i></div>"
            "</div>"
        )
    return (
        "<section class='neo-panel neo-bottom-panel'>"
        "<div class='neo-panel-heading'><div><span class='neo-panel-kicker'>DATA QUALITY</span>"
        "<h3>Качество данных</h3></div><span class='neo-panel-icon'>◈</span></div>"
        "<div class='neo-quality-layout'><div class='neo-gauge'>"
        + gauge
        + f"<div class='neo-gauge-center'><strong>{esc(quality_text)}</strong><span>КАЧЕСТВО</span></div>"
        "</div><div class='neo-quality-list'>"
        + "".join(rows)
        + "</div></div>"
        "<p class='neo-panel-footnote'>Доля подтверждённых полей в четырёх проверках по матчам текущего скана; это не вероятность исхода и не оценка уверенности модели.</p>"
        "</section>"
    )

def _backup_panel():
    backup = st.session_state.get("_sqlite_backup_status") or {}
    checked = st.session_state.get("_sqlite_remote_status_check") or {}
    restore = st.session_state.get("_sqlite_restore_status") or {}
    ok = bool(backup.get("ok") or checked.get("ok"))
    restored = bool(restore.get("restored"))
    error = backup.get("state") == "backup_error" or restore.get("state") == "restore_error"
    if ok:
        state, state_class = "ПОДТВЕРЖДЕНО", "ok"
        detail = str(
            backup.get("message") or checked.get("message")
            or "Удалённая зашифрованная копия доступна."
        )
        last_time = backup.get("backup_at") or backup.get("checked_at") or checked.get("checked_at")
    elif restored:
        state, state_class = "ВОССТАНОВЛЕНО", "ok"
        detail = str(restore.get("message") or "SQLite успешно восстановлена.")
        last_time = restore.get("checked_at")
    elif error:
        state, state_class = "ОШИБКА", "error"
        detail = str(backup.get("message") or restore.get("message") or "Не удалось проверить резервную копию.")
        last_time = backup.get("checked_at") or restore.get("checked_at")
    else:
        state, state_class = "ОЖИДАЕТ ПРОВЕРКИ", "pending"
        detail = "После первого успешного сохранения здесь появится подтверждённое состояние копии."
        last_time = None
    time_text = str(last_time or "—").replace("T", " ").replace("+00:00", " UTC")
    return (
        "<section class='neo-panel neo-bottom-panel'>"
        "<div class='neo-panel-heading'><div><span class='neo-panel-kicker'>REMOTE STORAGE</span>"
        "<h3>Резервное копирование</h3></div><span class='neo-panel-icon'>⌁</span></div>"
        f"<div class='neo-status-line'><span class='neo-status-dot {state_class}'></span>"
        f"<b>{esc(state)}</b><span>GitHub Secret Gist</span></div>"
        f"<div class='neo-backup-time'>Последняя отметка <b>{esc(time_text)}</b></div>"
        f"<p class='neo-backup-detail'>{esc(detail)}</p>"
        "<div class='neo-panel-footnote'>Проверка статуса и ручное сохранение доступны в разделе «Настройки и обслуживание» слева.</div>"
        "</section>"
    )


def _history_panel(snapshot, roi, profit, turnover):
    rows = snapshot.get("rows") or []
    all_rows = snapshot.get("all_rows") or rows
    settled = snapshot.get("settled") or []
    dates = [str(row.get("forecast_at") or "")[:10] for row in all_rows if row.get("forecast_at")]
    first_date = min(dates) if dates else "—"
    last_date = max(dates) if dates else "—"
    leagues = len({str(row.get("league") or "").strip() for row in rows if str(row.get("league") or "").strip()})
    xg_rows = [row for row in rows if row.get("xg_used") is not None]
    xg_used = sum(bool(row.get("xg_used")) for row in xg_rows)
    xg_text = f"{xg_used}/{len(xg_rows)}" if xg_rows else "—"
    metrics = snapshot.get("metrics") or {}
    brier_text = f"{metrics['brier']:.4f}" if metrics.get("brier") is not None else "—"
    log_text = f"{metrics['log_loss']:.4f}" if metrics.get("log_loss") is not None else "—"
    roi_text = f"{roi:+.1f}%" if turnover > 0 else "—"
    profit_text = f"{profit:+.2f}" if turnover > 0 else "—"
    return (
        "<section class='neo-panel neo-bottom-panel'>"
        "<div class='neo-panel-heading'><div><span class='neo-panel-kicker'>HISTORY & COVERAGE</span>"
        "<h3>История и покрытие</h3></div><span class='neo-panel-icon'>▤</span></div>"
        "<div class='neo-history-row'><span>Прогнозов основной модели · все версии</span><b>"
        f"{len(all_rows):,}</b></div>"
        "<div class='neo-history-row'><span>Прогнозов текущей версии</span><b>"
        f"{len(rows):,}</b></div>"
        "<div class='neo-history-row'><span>Подтверждено FT · текущая версия</span><b>"
        f"{len(settled):,}</b></div>"
        "<div class='neo-history-row'><span>Ожидают результата</span><b>"
        f"{snapshot.get('pending', 0):,}</b></div>"
        "<div class='neo-history-row'><span>Покрытие турниров</span><b>"
        f"{leagues}</b></div>"
        "<div class='neo-history-row'><span>xG реально использован</span><b>"
        f"{esc(xg_text)}</b></div>"
        "<div class='neo-history-divider'></div>"
        "<div class='neo-history-foot'>"
        f"<span>Период {esc(first_date)} — {esc(last_date)}</span>"
        f"<span>Brier {esc(brier_text)} · Log Loss {esc(log_text)}</span>"
        f"<span>ROI виртуального портфеля {esc(roi_text)} · прибыль {esc(profit_text)}</span>"
        "<b class='neo-baseline-note'>Baseline: отдельный эталонный снимок ещё не зафиксирован</b>"
        "</div></section>"
    )


_DASHBOARD_STATUS_PRIORITY = {
    "approved": 3,
    "forecast": 2,
    "near_strong": 1,
}
_DASHBOARD_STATUS_LABELS = {
    "approved": "Одобрено к ставке",
    "forecast": "Сильный прогноз — не ставка",
    "near_strong": "Почти сильный прогноз",
}
_DASHBOARD_STATUS_HINTS = {
    "approved": "Все проверки прогноза, контекста, рынка, линии и EV/Edge/Kelly пройдены.",
    "forecast": "Истории и вероятности достаточно, но один или несколько Value-гейтов не пройдены.",
    "near_strong": "Истории достаточно, но вероятность немного ниже сильного порога.",
}
_DASHBOARD_FAMILIES = ("winner", "goals", "btts", "corners", "cards")
_DASHBOARD_FAMILY_LABELS = {
    "winner": "Исход 1X2",
    "goals": "Тотал голов",
    "btts": "Обе забьют",
    "corners": "Угловые",
    "cards": "Карточки",
}


def _dashboard_real_1x2_quotes(card):
    """Return only provider-backed 1X2 prices; never derive a bookmaker price."""
    if not isinstance(card, dict):
        return []
    prices = card.get("market_odds")
    prices = prices if isinstance(prices, dict) else {}
    sources = card.get("market_odds_source_by_pick")
    sources = sources if isinstance(sources, dict) else {}
    bookmakers = card.get("bookmaker_by_pick")
    bookmakers = bookmakers if isinstance(bookmakers, dict) else {}
    fallback_source = str(card.get("odds_source") or "").strip()
    result = []
    for pick in ("П1", "X", "П2"):
        price = num(prices.get(pick))
        source = str(sources.get(pick) or fallback_source).strip()
        if (
            price is None or not 1.0 < price <= 100.0
            or source.casefold() in {"", "model_only", "no_market", "unknown"}
        ):
            continue
        result.append({
            "pick": pick,
            "odds": price,
            "bookmaker": str(bookmakers.get(pick) or "").strip(),
            "source": source,
        })
    return result


def _dashboard_ai_details(card):
    """Compact per-fixture AI/context evidence, available on demand."""
    analysis = card.get("deep_analysis")
    analysis = analysis if isinstance(analysis, dict) else {}
    provider = str(card.get("deep_analysis_provider") or "").strip()
    model = str(card.get("deep_analysis_model") or "").strip()
    if not analysis:
        return (
            "<span class='neo-ai-status'>AI: анализ для этого матча не сохранён</span>"
            if not provider else
            f"<span class='neo-ai-status'>AI: {esc(provider)} · нет сохранённого разбора</span>"
        )

    parts = []
    probabilities = analysis.get("probabilities")
    if isinstance(probabilities, dict):
        normalized = []
        for label in ("П1", "X", "П2"):
            value = num(probabilities.get(label))
            if value is not None and 0 <= value <= 100:
                normalized.append(f"{label} {value:.0f}%")
        if normalized:
            parts.append("Вероятности AI: " + " · ".join(normalized))
    independent_pick = str(analysis.get("independent_pick") or "").strip()
    if independent_pick:
        parts.append(f"Выбор AI: {independent_pick}")
    for field, label in (
        ("lineup_assessment", "Составы"),
        ("injury_assessment", "Травмы / дисквалификации"),
        ("referee_assessment", "Судья"),
        ("form_assessment", "Форма"),
        ("match_assessment", "Анализ матча"),
        ("risks", "Риски"),
        ("data_quality", "Качество данных"),
    ):
        value = str(analysis.get(field) or "").strip()
        if value:
            parts.append(f"<b>{esc(label)}:</b> {esc(value)}")
    if not parts:
        parts.append("AI вернул пустой или неполный разбор.")
    heading = " · ".join(part for part in (provider, model) if part) or "Независимый AI"
    return (
        "<details class='neo-ai-details'><summary>"
        f"AI / составы / риски · {esc(heading)}</summary>"
        "<div>" + "<br>".join(parts) + "</div></details>"
    )


def _dashboard_fixture_html(card):
    """Render team crests and competition marker in the compact match cell."""
    home = str(card.get("home_team") or "Хозяева")
    away = str(card.get("away_team") or "Гости")
    league = str(card.get("league") or card.get("div") or "Футбол")
    code = str(card.get("div") or "").strip()
    league_logo = _https_image_url(card.get("league_badge"))
    if league_logo:
        league_marker = (
            f"<img class='neo-dashboard-league-logo' src='{esc(league_logo)}' "
            "alt='' loading='lazy' referrerpolicy='no-referrer' "
            "onerror=\"this.style.display='none'\"/>"
        )
    else:
        league_marker = f"<span class='neo-dashboard-league-flag'>{esc(_LEAGUE_FLAGS.get(code, '⚽'))}</span>"
    return (
        "<div class='neo-dashboard-fixture'>"
        f"<small>{league_marker}{esc(league)}</small>"
        "<div class='neo-fixture-line'>"
        f"<span class='neo-team'>{_team_badge_html(home, card.get('home_badge'), 22)}<b>{esc(home)}</b></span>"
        "<span class='neo-vs'>—</span>"
        f"<span class='neo-team'>{_team_badge_html(away, card.get('away_badge'), 22)}<b>{esc(away)}</b></span>"
        "</div></div>"
    )


def dashboard_recommendation_groups(
    cards, *, now=None, period_end=None, selected_league=None,
):
    """Return one highest-status recommendation per fixture; omit all other matches."""
    now = now or datetime.now(app_timezone())
    candidates = {}
    for card in cards if isinstance(cards, list) else []:
        if not isinstance(card, dict):
            continue
        league = str(card.get("league") or card.get("div") or "").strip()
        if selected_league and selected_league != "Все турниры" and league != selected_league:
            continue
        kickoff = fixture_kickoff_local(card)
        if kickoff is None or kickoff < now:
            continue
        if period_end is not None and kickoff > period_end:
            continue

        for family in _DASHBOARD_FAMILIES:
            ranked = rank_market_forecasts([card], family, limit=1)
            if not ranked:
                continue
            status, gates = classify_card(card, family)
            if status not in _DASHBOARD_STATUS_PRIORITY:
                continue

            summary = ranked[0]
            probability = num(summary.get("probability"))
            if probability is None:
                continue
            recommendation = (card.get("best_by_market") or {}).get(family)
            recommendation = recommendation if isinstance(recommendation, dict) else {}
            best = recommendation.get("best")
            best = best if isinstance(best, (tuple, list)) else None
            verdict = recommendation.get("verdict")
            verdict = verdict if isinstance(verdict, dict) else {}

            # For approved picks use the exact price that passed Value/Kelly.
            # For forecast-only statuses show the real current quote for the
            # model-selected outcome, even though Value gates did not pass.
            pick = str(summary.get("pick") or "—")
            probability_for_display = probability
            odds = _event_market_quote(card, family, pick)
            odds_source = str(_event_market_source(card, family) or "").strip()
            if odds_source.casefold() in {"", "model_only", "no_market", "unknown"}:
                odds = None
                odds_source = ""
            if (
                status == "approved"
                and verdict.get("real_odds")
                and verdict.get("value_passed")
                and verdict.get("is_bet")
                and best
                and len(best) >= 6
            ):
                verified_pick = str(best[1] or "").strip()
                verified_price = num(best[2])
                verified_probability = num(verdict.get("prob"))
                if verified_pick:
                    pick = verified_pick
                if verified_price is not None and verified_price > 1:
                    odds = verified_price
                if verified_probability is not None and 0 < verified_probability < 1:
                    probability_for_display = verified_probability
                odds_source = odds_source or "Подтверждённая линия"

            bookmaker = str(
                (card.get("bookmaker_by_pick") or {}).get(pick) or ""
            ).strip()
            if status == "approved" and not odds_source and odds is not None:
                odds_source = "Подтверждённая линия"

            failed = [gate.name for gate in gates if not gate.passed]
            if status == "forecast":
                reason = "Не пройдены: " + ", ".join(failed[:3]) if failed else (
                    "Не все Value-гейты подтверждены."
                )
            elif status == "near_strong":
                gap = max(0.0, float(STRONG_FORECAST_MIN_PROB) - probability)
                reason = (
                    f"До порога {STRONG_FORECAST_MIN_PROB:.0%} не хватает "
                    f"{gap:.1%}."
                )
            else:
                reason = "Все обязательные проверки пройдены."

            fixture_key = str(card.get("fixture_id") or "").strip()
            if not fixture_key:
                fixture_key = "|".join((
                    str(card.get("date_iso") or card.get("date") or "")[:10],
                    str(card.get("home_team") or card.get("home") or "").casefold(),
                    str(card.get("away_team") or card.get("away") or "").casefold(),
                ))
            if not fixture_key.strip("|"):
                fixture_key = match_name(card).casefold()

            item = {
                "card": card,
                "kickoff": kickoff,
                "status": status,
                "family": family,
                "family_label": _DASHBOARD_FAMILY_LABELS.get(family, family),
                "pick": pick,
                "probability": probability_for_display,
                "home_sample": int(summary.get("home_team_sample") or 0),
                "away_sample": int(summary.get("away_team_sample") or 0),
                "odds": odds,
                "one_x_two_quotes": _dashboard_real_1x2_quotes(card),
                "probabilities_1x2": {
                    "П1": num(card.get("p1")),
                    "X": num(card.get("px")),
                    "П2": num(card.get("p2")),
                },
                "ev": num(verdict.get("ev")) if status == "approved" else None,
                "edge": num(verdict.get("edge")) if status == "approved" else None,
                "stake": num(verdict.get("stake")) if status == "approved" else None,
                "bookmaker": bookmaker,
                "odds_source": odds_source,
                "reason": reason,
            }
            # One fixture appears once: approved beats strong forecast, which
            # beats near-strong. Within the same status choose the clearest pick.
            rank = (
                _DASHBOARD_STATUS_PRIORITY[status],
                probability,
                float(item.get("ev") or 0.0),
                item["home_sample"] + item["away_sample"],
            )
            previous = candidates.get(fixture_key)
            if previous is None or rank > previous["_rank"]:
                item["_rank"] = rank
                candidates[fixture_key] = item

    groups = {status: [] for status in _DASHBOARD_STATUS_PRIORITY}
    for item in candidates.values():
        item.pop("_rank", None)
        groups[item["status"]].append(item)
    for status in groups:
        groups[status].sort(key=lambda item: (
            item["kickoff"], -item["probability"], item["family_label"],
        ))
    return groups


def recommendation_group_html(status, items):
    """Render one compact recommendation table with full prices and AI evidence."""
    if status not in _DASHBOARD_STATUS_LABELS:
        return ""
    items = [item for item in (items or []) if isinstance(item, dict)]
    label = _DASHBOARD_STATUS_LABELS[status]
    hint = _DASHBOARD_STATUS_HINTS[status]
    if not items:
        empty = {
            "approved": "Сейчас нет матчей, прошедших все проверки.",
            "forecast": "Сейчас нет сильных прогнозов, заблокированных Value-гейтами.",
            "near_strong": "Сейчас нет прогнозов рядом с сильным порогом.",
        }[status]
        body = f"<div class='neo-empty-cell'>{esc(empty)}</div>"
    else:
        rows = []
        for item in items:
            kickoff = item.get("kickoff")
            time_text = (
                kickoff.strftime("%d.%m %H:%M")
                if isinstance(kickoff, datetime) else "—"
            )
            card = item.get("card") if isinstance(item.get("card"), dict) else {}
            odds = num(item.get("odds"))
            selected_price = (
                f"{esc(str(item.get('pick') or ''))} @ {odds:.2f}"
                if odds is not None else "нет подтверждённой цены"
            )
            selected_book = str(item.get("bookmaker") or "").strip()
            selected_source = str(item.get("odds_source") or "").strip()
            selected_quote_source = " · ".join(
                value for value in (selected_book, selected_source) if value
            )
            if status == "approved":
                ev = num(item.get("ev"))
                edge = num(item.get("edge"))
                stake = num(item.get("stake"))
                details = []
                if ev is not None:
                    details.append(f"EV {ev:+.1%}")
                if edge is not None:
                    details.append(f"Edge {edge:+.1%}")
                if stake is not None:
                    details.append(f"Kelly {stake:.2f}")
                gate_text = " · ".join(details) or "Все Value-гейты пройдены"
            else:
                gate_text = str(item.get("reason") or "")

            probabilities = item.get("probabilities_1x2")
            probabilities = probabilities if isinstance(probabilities, dict) else {}
            prob_text = " · ".join(
                f"{pick} {value:.0%}"
                for pick in ("П1", "X", "П2")
                if (value := num(probabilities.get(pick))) is not None
                and 0 <= value <= 1
            ) or "—"

            quote_rows = []
            for quote in item.get("one_x_two_quotes") or []:
                if not isinstance(quote, dict):
                    continue
                price = num(quote.get("odds"))
                if price is None:
                    continue
                pick = str(quote.get("pick") or "")
                book = str(quote.get("bookmaker") or "").strip()
                source = str(quote.get("source") or "").strip()
                quote_rows.append(
                    f"{esc(pick)} <b>{price:.2f}</b>"
                    + (f" <small>{esc(book)}</small>" if book else "")
                    + (f" <small>{esc(source)}</small>" if source else "")
                )
            one_x_two_text = (
                "<div class='neo-dashboard-odds-list'>"
                + "<br>".join(quote_rows)
                + "</div>"
                if quote_rows else
                "<span class='neo-ai-status'>Нет реальных котировок 1X2</span>"
            )
            history_text = (
                f"{int(item.get('home_sample') or 0)}/"
                f"{int(item.get('away_sample') or 0)}"
            )
            ai_html = _dashboard_ai_details(card)
            rows.append(
                "<tr>"
                f"<td class='neo-time'>{esc(time_text)}</td>"
                f"<td>{_dashboard_fixture_html(card)}</td>"
                f"<td>{esc(item.get('family_label') or '—')}<br><b>{esc(item.get('pick') or '—')}</b></td>"
                f"<td>{esc(prob_text)}</td>"
                f"<td>{one_x_two_text}</td>"
                f"<td><b>{selected_price}</b><small>{esc(selected_quote_source)}</small></td>"
                f"<td>{esc(history_text)}</td>"
                f"<td>{esc(gate_text)}<br>{ai_html}</td>"
                "</tr>"
            )
        body = (
            "<div class='neo-table-scroll'><table class='neo-match-table neo-recommendation-table'>"
            "<thead><tr><th>Время</th><th>Матч / турнир</th><th>Рынок / выбор</th>"
            "<th>Вероятности 1X2</th><th>Реальные коэффициенты 1X2</th>"
            "<th>Цена выбранного рынка</th><th>История Д/Г</th><th>Допуск / AI</th></tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody></table></div>"
        )
    return (
        "<section class='neo-panel neo-matches-panel'>"
        "<div class='neo-panel-heading'><div>"
        f"<h3>{esc(label)}</h3><div class='neo-ranking-disclaimer'>{esc(hint)}</div>"
        "</div>"
        f"<span class='neo-count-pill'>{len(items)} матчей</span></div>"
        + body
        + "</section>"
    )


def render():
    data = st.session_state.get("data") or {}
    cards = [
        card for card in (data.get("cards") or [])
        if isinstance(card, dict)
    ]
    now = datetime.now(app_timezone())

    st.markdown(
        "<div class='neo-dashboard-header'>"
        "<div><div class='neo-dashboard-eyebrow'>SPORT-AI-PRO13</div>"
        "<h1>Рекомендации по матчам</h1>"
        "<p>Только три статуса: одобрено к ставке, сильный прогноз и почти сильный прогноз.</p></div>"
        "<div class='neo-header-meta'><span class='neo-live-pill'><i></i> ПАНЕЛЬ АКТИВНА</span>"
        f"<span>v{esc(APP_VERSION)}</span><span>{esc(now.strftime('%d.%m.%Y %H:%M'))}</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    leagues = sorted({
        str(card.get("league") or card.get("div") or "").strip()
        for card in cards
        if str(card.get("league") or card.get("div") or "").strip()
    })
    filter_league, filter_period, filter_refresh = st.columns(
        [1.3, 1.2, 0.25], gap="small"
    )
    with filter_league:
        selected_league = st.selectbox(
            "Турнир",
            ["Все турниры"] + leagues,
            index=0,
            label_visibility="collapsed",
            key="dashboard_recommendation_league_v1",
        )
    with filter_period:
        selected_period = st.selectbox(
            "Период",
            DASHBOARD_PERIOD_OPTIONS,
            index=DASHBOARD_PERIOD_OPTIONS.index(DASHBOARD_DEFAULT_PERIOD)
            if DASHBOARD_DEFAULT_PERIOD in DASHBOARD_PERIOD_OPTIONS else 2,
            label_visibility="collapsed",
            key="dashboard_recommendation_period_v1",
        )
    with filter_refresh:
        if st.button(
            "↻",
            key="dashboard_recommendation_refresh_v1",
            use_container_width=True,
            help="Обновить рекомендации",
        ):
            st.rerun()

    if selected_period == "Сегодня":
        period_end = datetime.combine(
            now.date() + timedelta(days=1),
            datetime.min.time(),
            tzinfo=now.tzinfo,
        )
    elif selected_period == "Ближайшие 24 часа":
        period_end = now + timedelta(hours=24)
    else:
        period_end = None

    groups = dashboard_recommendation_groups(
        cards,
        now=now,
        period_end=period_end,
        selected_league=selected_league,
    )
    counts = [
        (
            "Одобрено к ставке",
            len(groups["approved"]),
            "approved",
        ),
        (
            "Сильный прогноз — не ставка",
            len(groups["forecast"]),
            "forecast",
        ),
        (
            "Почти сильный прогноз",
            len(groups["near_strong"]),
            "near_strong",
        ),
    ]
    st.markdown(
        "<div class='neo-recommendation-summary'>"
        + "".join(
            "<span class='neo-count-pill'>"
            f"{esc(label)} · {count}</span>"
            for label, count, _status in counts
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    for _label, _count, status in counts:
        st.markdown(
            recommendation_group_html(status, groups[status]),
            unsafe_allow_html=True,
        )

    st.caption(
        f"Показано {sum(len(value) for value in groups.values())} матчей. "
        "Каждый матч отображается один раз — в самой высокой пройденной категории. "
        "Остальные матчи остаются в Сканере и не засоряют Dashboard."
    )
