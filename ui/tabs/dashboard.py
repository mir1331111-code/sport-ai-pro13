"""Neural Terminal overview dashboard for NEURO BET PRO."""
from __future__ import annotations

import html
import math
from datetime import datetime, timedelta
import re

import streamlit as st

from betting.ranking import model_favorite, rank_best_fixtures
from betting.market_math import proportional_devig, competition_market_is_validated
from data.fixture_time import app_timezone, fixture_kickoff_local
from config import APP_VERSION, DB_FILE, LLM_PROVIDERS, llm_provider_key_config
from llm.deep_analysis import format_top_match_opinion, last_top_matches_status
from ui.cards import _LEAGUE_FLAGS
from model.forecast_ledger import ForecastLedger


OUTCOMES = ("H", "D", "A")
OUTCOME_LABELS = {"H": "П1", "D": "X", "A": "П2"}


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
    provider = str((data.get("meta") or {}).get("llm_provider") or "Groq (бесплатно, быстро)")
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
        and str(card.get("odds_source") or "").strip().casefold() == "sstats"
        and proportional_devig(
            prices,
            ("П1", "X", "П2"),
            bookmaker_by_pick=card.get("bookmaker_by_pick") or {},
        ) is not None
    )


def approved_1x2_recommendation(card):
    """Return an actual 1X2 bet approval, never a forecast or market-only signal."""
    if not isinstance(card, dict):
        return None
    if not bool((card.get("context_quality") or {}).get("strong_value_eligible")):
        return None
    if not competition_market_is_validated(card, "winner"):
        return None
    if not has_prices(card):
        return None
    recommendation = (card.get("best_by_market") or {}).get("winner")
    if not isinstance(recommendation, dict):
        return None
    best = recommendation.get("best")
    verdict = recommendation.get("verdict")
    if (
        not isinstance(best, (list, tuple))
        or len(best) < 6
        or not isinstance(verdict, dict)
        or str(best[0] or "").upper() != "1X2"
        or str(best[1] or "") not in {"П1", "X", "П2"}
        or not verdict.get("real_odds")
        or not verdict.get("value_passed")
        or not verdict.get("is_bet")
    ):
        return None
    try:
        odd, ev, probability, stake = map(float, (best[2], best[3], best[4], best[5]))
        edge_value = float(verdict.get("edge"))
    except (TypeError, ValueError):
        return None
    bookmaker = str((card.get("bookmaker_by_pick") or {}).get(best[1]) or "").strip()
    if (
        not bookmaker
        or odd < 1.70
        or ev < 0.05
        or edge_value < 0.05
        or not 0.0 < probability < 1.0
        or stake <= 0.0
    ):
        return None
    return recommendation


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
        return {
            "ok": True,
            "rows": active_rows,
            "all_rows": main_history,
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
    safe_url = str(url or "").strip()
    initial = esc((str(name or "?").strip()[:1] or "?").upper())
    inner = initial
    if safe_url.startswith("https://"):
        inner += (
            f"<img src='{esc(safe_url)}' alt='{esc(name)}' loading='lazy' "
            "decoding='async' referrerpolicy='no-referrer' "
            "onerror=\"this.style.display='none'\"/>"
        )
    return (
        f"<span class='neo-team-badge' style='--badge-size:{int(size)}px' "
        f"aria-label='{esc(name)}'>{inner}</span>"
    )


def _match_table_row(card):
    prediction = favorite(card) or {}
    values = []
    for key in ("p1", "px", "p2"):
        value = num(card.get(key))
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

    model_pick = str(prediction.get("pick") or "—")
    model_probability = num(prediction.get("probability"))
    pass_pick = str(card.get("_daily_pass_pick") or model_pick)
    pass_probability = num(card.get("_daily_pass_probability"))
    pass_source = str(card.get("_daily_pass_source") or "model_supported")
    if pass_probability is None:
        pass_probability = model_probability
    pick_text = (
        f"{pass_pick} · {pass_probability:.0%}"
        if pass_probability is not None else pass_pick
    )
    verdict = card.get("verdict") if isinstance(card.get("verdict"), dict) else {}
    single_book_line = proportional_devig(
        card.get("market_odds") or {},
        ("П1", "X", "П2"),
        bookmaker_by_pick=card.get("bookmaker_by_pick") or {},
    )
    context_ok = bool(
        (card.get("context_quality") or {}).get("strong_value_eligible")
    )
    value_passed = bool(
        verdict.get("real_odds")
        and verdict.get("value_passed")
        and verdict.get("is_bet")
        and single_book_line is not None
        and context_ok
        and competition_market_is_validated(card, "winner")
    )
    has_confirmed_line = has_prices(card)
    if value_passed:
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
    daily_note = str(card.get("_daily_selection_note") or "").strip()
    home_sample = int(card.get("_daily_home_sample") or 0)
    away_sample = int(card.get("_daily_away_sample") or 0)
    league_sample = int(card.get("_daily_league_sample") or 0)
    sample_text = (
        f"история: лига {league_sample}/30 · дома {home_sample}/5 · в гостях {away_sample}/5"
    )
    model_market_gap = num(card.get("_daily_model_market_gap"))
    model_pass_probability = num(card.get("_daily_model_probability_for_pass"))
    if pass_source == "market_no_vig":
        disagreement = ""
        if model_market_gap is not None and abs(model_market_gap) >= 0.10:
            disagreement = (
                f" · модель: {model_pass_probability:.0%} "
                f"({model_market_gap:+.0%} к рынку)"
                if model_pass_probability is not None else ""
            )
        selection_html = (
            "<small class='neo-pick-note proven'>РЫНОК БЕЗ МАРЖИ · ОЦЕНКА ПРОХОДА</small>"
            f"<small class='neo-pick-note rank-score'>{esc(sample_text)}{esc(disagreement)}</small>"
            "<small class='neo-pick-note provisional'>Это рыночный ориентир, не модельная вероятность и не допуск к ставке.</small>"
        )
    else:
        selection_html = (
            "<small class='neo-pick-note provisional'>МОДЕЛЬНЫЙ ПРОГНОЗ · "
            "линия рынка без маржи отсутствует</small>"
            f"<small class='neo-pick-note rank-score'>{esc(sample_text)}</small>"
            "<small class='neo-pick-note provisional'>Не подтверждён рынком; не является допуском к ставке.</small>"
        )

    league = str(card.get("league") or card.get("div") or "Футбол")
    league_code = str(card.get("div") or "")
    league_logo = str(card.get("league_badge") or "")
    league_mark = (
        f"<img src='{esc(league_logo)}' alt='' loading='lazy' decoding='async' "
        "referrerpolicy='no-referrer' onerror=\"this.style.display='none'\"/>"
        if league_logo.startswith("https://")
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
        + f"<td class='neo-pick-cell'><span class='neo-pick {'value' if value_passed else ''}'>{esc(pick_text)}</span>{bet_status_html}{selection_html}</td>"
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


def render():
    data = st.session_state.get("data") or {}
    cards = [card for card in (data.get("cards") or []) if isinstance(card, dict)]
    cards.sort(key=lambda card: (str(card.get("date_iso") or "9999"), str(card.get("time") or "23:59")))
    ai_name = ai_provider_label(data)
    priced = sum(has_prices(card) for card in cards)

    snapshot = _ledger_snapshot()
    metrics = snapshot["metrics"]
    bets = [bet for bet in (data.get("bets") or []) if isinstance(bet, dict)]
    closed_bets = [bet for bet in bets if bet.get("status") in {"won", "lost"}]
    closed_bets.sort(key=lambda bet: str(
        bet.get("settled_at") or bet.get("updated_at") or bet.get("date_iso") or ""
    ))
    turnover = sum(num(bet.get("stake")) or 0.0 for bet in closed_bets)
    profit = sum(
        (num(bet.get("stake")) or 0.0) * ((num(bet.get("odds")) or 1.0) - 1.0)
        * (1.0 if bet.get("status") == "won" else -1.0)
        for bet in closed_bets
    )
    roi = profit / turnover * 100 if turnover else None
    roi_series = []
    running_turnover = 0.0
    running_profit = 0.0
    for bet in closed_bets:
        stake = num(bet.get("stake")) or 0.0
        odds = num(bet.get("odds")) or 1.0
        running_turnover += stake
        running_profit += stake * ((odds - 1.0) if bet.get("status") == "won" else -1.0)
        if running_turnover > 0:
            roi_series.append(running_profit / running_turnover * 100)

    now = datetime.now(app_timezone())
    st.markdown(
        "<div class='neo-dashboard-header'>"
        "<div><div class='neo-dashboard-eyebrow'>SPORT-AI-PRO13 · FOOTBALL INTELLIGENCE</div>"
        "<h1>Футбол. Прогнозы на матчи</h1>"
        "<p>Основная модель · реальные котировки · проверенное качество прогнозов</p></div>"
        "<div class='neo-header-meta'><span class='neo-live-pill'><i></i> ПАНЕЛЬ АКТИВНА</span>"
        f"<span>v{esc(APP_VERSION)}</span><span>Обновлено {esc(now.strftime('%d.%m.%Y %H:%M'))}</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    leagues = sorted({
        str(card.get("league") or card.get("div") or "").strip()
        for card in cards if str(card.get("league") or card.get("div") or "").strip()
    })
    filter_sport, filter_league, filter_period, filter_refresh = st.columns(
        [0.82, 1.25, 1.12, 0.28], gap="small"
    )
    with filter_sport:
        st.selectbox(
            "Вид спорта", ["⚽ Футбол"], index=0,
            label_visibility="collapsed", disabled=True,
            key="dashboard_sport_filter",
        )
    with filter_league:
        league_options = ["Все турниры"] + leagues
        selected_league = st.selectbox(
            "Турнир", league_options, index=0,
            label_visibility="collapsed", key="dashboard_league_filter",
        )
    with filter_period:
        selected_period = st.selectbox(
            "Период",
            [
                "Сегодня · лучшие события",
                "Ближайшие 24 часа · лучшие события",
                "Весь горизонт скана · лучшие события",
            ],
            index=0,
            label_visibility="collapsed",
            key="dashboard_period_filter_v2",
        )
    with filter_refresh:
        refresh_dashboard = st.button(
            "↻", key="dashboard_refresh_button", use_container_width=True,
            help="Обновить панель и перечитать текущие данные",
        )
        if refresh_dashboard:
            st.rerun()

    filtered_cards = cards
    if selected_league != "Все турниры":
        filtered_cards = [
            card for card in filtered_cards
            if str(card.get("league") or card.get("div") or "").strip() == selected_league
        ]

    # The default view evaluates the full remaining local calendar day. It no
    # longer takes the first five fixtures merely because they kick off first.
    if selected_period == "Сегодня · лучшие события":
        period_end = datetime.combine(
            now.date() + timedelta(days=1),
            datetime.min.time(),
            tzinfo=now.tzinfo,
        )
        table_title = "Лучшие события сегодня · 1X2"
    elif selected_period == "Ближайшие 24 часа · лучшие события":
        period_end = now + timedelta(hours=24)
        table_title = "Лучшие события ближайших 24 часов · 1X2"
    else:
        period_end = None
        table_title = "Лучшие события всего горизонта · 1X2"

    selection_diagnostics = {}
    ranked_events = rank_best_fixtures(
        filtered_cards,
        now=now,
        period_end=period_end,
        limit=5,
        min_sample_games=5,
        min_probability=0.52,
        include_limited_history=True,
        diagnostics=selection_diagnostics,
    )
    selected_event_cards = []
    for item in ranked_events:
        card = dict(item["card"])
        card["_daily_pass_pick"] = str(item.get("pass_pick") or "")
        card["_daily_pass_probability"] = float(item.get("pass_probability") or 0.0)
        card["_daily_pass_source"] = str(item.get("pass_probability_source") or "")
        card["_daily_market_probabilities"] = item.get("market_probabilities") or {}
        card["_daily_model_probability_for_pass"] = item.get("model_probability_for_pass")
        card["_daily_model_market_gap"] = item.get("model_market_gap")
        card["_daily_model_supported"] = bool(item.get("model_supported"))
        card["_daily_selection_note"] = str(item.get("selection_note") or "")
        card["_daily_home_sample"] = int(item.get("home_sample") or 0)
        card["_daily_away_sample"] = int(item.get("away_sample") or 0)
        card["_daily_league_sample"] = int(item.get("league_history_matches") or 0)
        selected_event_cards.append(card)
    market_count = int(selection_diagnostics.get("market_unique_candidates", 0) or 0)
    model_only_count = int(selection_diagnostics.get("model_only_unique_candidates", 0) or 0)
    available_event_count = int(selection_diagnostics.get("unique_candidates", 0) or 0)
    scan_diagnostics_text = (
        f"Не распознано время: {selection_diagnostics.get('invalid_kickoff', 0)} · "
        f"уже начались: {selection_diagnostics.get('already_started', 0)} · "
        f"вне периода: {selection_diagnostics.get('outside_window', 0)} · "
        f"нет полной линии одного букмекера: {selection_diagnostics.get('incoherent_market_line', 0)} · "
        f"нет рынка и достаточной истории модели: {selection_diagnostics.get('no_market_or_supported_model', 0)}"
    )

    brier = f"{metrics['brier']:.3f}" if metrics.get("brier") is not None else "—"
    log_loss = f"{metrics['log_loss']:.3f}" if metrics.get("log_loss") is not None else "—"
    accuracy = f"{metrics['accuracy']:.1%}" if metrics.get("accuracy") is not None else "—"
    roi_text = f"{roi:+.1f}%" if roi is not None else "—"
    market_delta = metrics.get("market_delta_brier")
    market_delta_text = f"{market_delta:+.4f}" if market_delta is not None else "—"
    market_note = (
        f"парных матчей: {metrics.get('market_n', 0)} · отрицательное значение лучше"
        if market_delta is not None else "нет полной парной линии и результата"
    )
    sample_note = f"{metrics.get('n', 0):,} завершённых прогнозов"
    roi_note = (
        f"виртуальный портфель · {len(closed_bets)} закрытых ставок"
        if turnover > 0 else "нет закрытых ставок с оборотом"
    )
    kpis = [
        _kpi("Brier Score", brier, f"{sample_note} · ниже — лучше",
             series=metrics.get("brier_series"), color="#39d49b"),
        _kpi("Log Loss", log_loss, f"{sample_note} · ниже — лучше",
             series=metrics.get("log_loss_series"), color="#4ec8d7"),
        _kpi("Accuracy (1X2)", accuracy, f"{sample_note} · доля верных исходов",
             series=metrics.get("accuracy_series"), color="#39d49b"),
        _kpi("ROI (ставки)", roi_text, roi_note,
             series=roi_series, color="#39d49b" if (roi or 0) >= 0 else "#f17b83"),
    ]
    st.markdown(
        "<div class='neo-kpi-grid'>" + "".join(kpis) + "</div>"
        + "<div class='neo-market-delta'>"
        "<span>СРАВНЕНИЕ С РЫНКОМ · Δ BRIER</span>"
        f"<b>{esc(market_delta_text)}</b>"
        f"<small>{esc(market_note)}</small>"
        f"<small class='neo-meta-spacer'>В скане: {len(cards):,} · полных линий 1X2: {priced:,} · FT: {metrics.get('n', 0):,}</small>"
        "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.68, 1.0], gap="medium")
    with left:
        table_rows = "".join(_match_table_row(card) for card in selected_event_cards)
        if not table_rows:
            table_rows = (
                "<tr><td colspan='10' class='neo-empty-cell'>"
                "В выбранном периоде нет будущих матчей с корректным прогнозом 1X2. "
                "Проверь время начала, выбранный турнир и актуальность сканирования."
                "</td></tr>"
            )
        st.markdown(
            "<section class='neo-panel neo-matches-panel'>"
            "<div class='neo-panel-heading'><div><span class='neo-panel-kicker'>DAILY PASS-PROBABILITY RANKING · НЕ СПИСОК СТАВОК</span>"
            f"<h3>{esc(table_title)}</h3></div>"
            f"<span class='neo-count-pill'>{available_event_count} событий</span></div>"
            "<div class='neo-ranking-disclaimer'>"
            "<b>Место в списке = качество прогноза, а не допуск к ставке.</b> "
            "Ставка разрешена только при отдельной зелёной метке "
            "<strong>VALUE · СТАВКА ДОПУЩЕНА</strong>."
            "</div>"
            "<div class='neo-table-scroll'><table class='neo-match-table'><thead><tr>"
            "<th>Время</th><th>Лига</th><th>Матч</th><th>Модель П1</th><th>Модель X</th><th>Модель П2</th>"
            "<th>Коэф. 1</th><th>Коэф. X</th><th>Коэф. 2</th><th>Оценка прохода / статус ставки</th>"
            "</tr></thead><tbody>"
            + table_rows
            + "</tbody></table></div>"
            f"<div class='neo-table-footer'>Показано {len(selected_event_cards)} из {available_event_count} предстоящих событий за выбранный период · оценка по рынку без маржи: {market_count} · модельный прогноз с достаточной историей: {model_only_count}. При наличии полной линии одного букмекера порядок строится по no-vig вероятности рынка; иначе показывается модель только при истории лиги ≥30 и выборке хозяев дома/гостей в гостях ≥5. Колонки П1/X/П2 — вероятности модели; последняя колонка отдельно указывает источник оценки прохода. Ни рыночный ориентир, ни место в списке не являются разрешением на ставку. Зелёная VALUE-метка возможна только после отдельного league holdout, полного рынка, контекста и риск-фильтров.<br>{esc(scan_diagnostics_text)}</div>"
            "</section>",
            unsafe_allow_html=True,
        )

    with right:
        reliability = snapshot.get("reliability") or {"rows": [], "n": 0}
        reliability_n = int(reliability.get("n", 0) or 0)
        chart = (
            _reliability_svg(reliability)
            if reliability_n >= 30 else
            "<div class='neo-chart-empty'><strong>Недостаточно истории для графика</strong>"
            f"<span>Закрытых прогнозов основной модели: {reliability_n}. "
            "Калибровка появится после накопления более широкой выборки.</span></div>"
        )
        st.markdown(
            "<section class='neo-panel neo-reliability-panel'>"
            "<div class='neo-panel-heading'><div><span class='neo-panel-kicker'>MODEL QUALITY</span>"
            "<h3>Reliability · калибровка</h3></div>"
            f"<span class='neo-count-pill'>{reliability_n:,} FT</span></div>"
            + chart
            + "<div class='neo-panel-footnote'>Вероятность модели и наблюдаемая частота объединены по классам П1 / X / П2; пунктир — идеальная калибровка.</div>"
            "</section>",
            unsafe_allow_html=True,
        )

    bottom_left, bottom_mid, bottom_right = st.columns(3, gap="medium")
    with bottom_left:
        st.markdown(_quality_panel(cards), unsafe_allow_html=True)
    with bottom_mid:
        st.markdown(_backup_panel(), unsafe_allow_html=True)
    with bottom_right:
        st.markdown(
            _history_panel(snapshot, roi or 0.0, profit, turnover),
            unsafe_allow_html=True,
        )

    if not snapshot.get("ok"):
        st.caption(
            "Журнал прогнозов временно недоступен: "
            + esc(snapshot.get("error") or "ошибка SQLite")
        )
    st.markdown(
        "<div class='neo-dashboard-footer'>"
        f"<span>SPORT-AI-PRO13</span> · {esc(ai_name)} анализатор · "
        "Показатели качества рассчитаны только по подтверждённым FT основной модели. "
        "ROI относится только к виртуальному портфелю и не является метрикой точности прогнозов."
        "</div>",
        unsafe_allow_html=True,
    )
