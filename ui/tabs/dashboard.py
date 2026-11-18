"""Neural Terminal overview dashboard for NEURO BET PRO."""
from __future__ import annotations

import html
import math
from datetime import datetime

import streamlit as st

from betting.ranking import model_favorite
from data.fixture_time import app_timezone
from config import APP_VERSION, DB_FILE, LLM_PROVIDERS
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
        and str(card.get("odds_source") or "").casefold() not in {"", "no_market", "model_only"}
        and all(1.01 < (num(prices.get(k)) or 0) <= 100 for k in ("П1", "X", "П2"))
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


def _forecast_metrics(rows):
    settled = [
        row for row in rows
        if row.get("result") in OUTCOMES
        and isinstance(row.get("probabilities"), dict)
        and all(num((row.get("probabilities") or {}).get(key)) is not None for key in OUTCOMES)
    ]
    if not settled:
        return {
            "n": 0, "brier": None, "log_loss": None, "accuracy": None,
            "market_delta_brier": None, "market_n": 0,
        }

    briers, losses, correct = [], [], 0
    market_differences = []
    for row in settled:
        probabilities = row["probabilities"]
        actual = row["result"]
        briers.append(sum((float(probabilities[key]) - float(key == actual)) ** 2 for key in OUTCOMES))
        losses.append(-math.log(max(float(probabilities[actual]), 1e-15)))
        predicted = max(OUTCOMES, key=lambda key: float(probabilities[key]))
        correct += int(predicted == actual)

        prices = row.get("entry_odds")
        if isinstance(prices, dict) and all(num(prices.get(key)) and num(prices.get(key)) > 1 for key in OUTCOMES):
            implied = {key: 1.0 / float(prices[key]) for key in OUTCOMES}
            total = sum(implied.values())
            market = {key: implied[key] / total for key in OUTCOMES}
            market_brier = sum((market[key] - float(key == actual)) ** 2 for key in OUTCOMES)
            market_differences.append(briers[-1] - market_brier)

    n = len(settled)
    return {
        "n": n,
        "brier": sum(briers) / n,
        "log_loss": sum(losses) / n,
        "accuracy": correct / n,
        "market_delta_brier": (
            sum(market_differences) / len(market_differences) if market_differences else None
        ),
        "market_n": len(market_differences),
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
    source_rows = report.get("rows") or []
    colors = {"H": "#46d6a2", "D": "#55b9d9", "A": "#b19aff"}
    labels = {"H": "П1 · хозяева", "D": "X · ничья", "A": "П2 · гости"}
    series = {}
    for key in OUTCOMES:
        points = []
        for row in source_rows:
            if row.get("outcome") != key:
                continue
            if int(row.get("n", 0) or 0) < 3:
                continue
            x = num(row.get("mean_probability"))
            y = num(row.get("observed_frequency"))
            if x is not None and y is not None:
                points.append((x, y))
        series[key] = sorted(points)

    width, height = 420, 238
    left, right, top, bottom = 42, 395, 18, 193
    plot_w, plot_h = right - left, bottom - top
    grid = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = bottom - tick * plot_h
        x = left + tick * plot_w
        grid.append(
            f"<line x1='{left}' y1='{y:.1f}' x2='{right}' y2='{y:.1f}' class='neo-gridline'/>"
            f"<line x1='{x:.1f}' y1='{top}' x2='{x:.1f}' y2='{bottom}' class='neo-gridline'/>"
            f"<text x='{left-10}' y='{y+4:.1f}' text-anchor='end' class='neo-axis'>{tick:.2f}</text>"
            f"<text x='{x:.1f}' y='{bottom+18}' text-anchor='middle' class='neo-axis'>{tick:.2f}</text>"
        )
    paths = [
        f"<line x1='{left}' y1='{bottom}' x2='{right}' y2='{top}' class='neo-ideal'/>"
    ]
    for key in OUTCOMES:
        points = series[key]
        if len(points) >= 2:
            coords = " ".join(
                f"{left + x * plot_w:.1f},{bottom - y * plot_h:.1f}"
                for x, y in points
            )
            paths.append(
                f"<polyline points='{coords}' fill='none' stroke='{colors[key]}' "
                "stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/>"
            )
        for x, y in points:
            paths.append(
                f"<circle cx='{left + x * plot_w:.1f}' cy='{bottom - y * plot_h:.1f}' "
                f"r='4' fill='{colors[key]}' stroke='#0d1a20' stroke-width='1.5'/>"
            )
    legend = "".join(
        f"<span><i style='background:{colors[key]}'></i>{labels[key]}</span>"
        for key in OUTCOMES
    )
    return (
        "<div class='neo-chart-wrap'>"
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Калибровка вероятностей'>"
        + "".join(grid) + "".join(paths)
        + f"<text x='{(left+right)/2}' y='{height-4}' text-anchor='middle' class='neo-axis-title'>Прогнозная вероятность</text>"
        + f"<text x='11' y='{(top+bottom)/2}' text-anchor='middle' transform='rotate(-90 11 {(top+bottom)/2})' class='neo-axis-title'>Фактическая частота</text>"
        + "</svg><div class='neo-chart-legend'>"
        + legend
        + "<span><i class='neo-ideal-key'></i>Идеальная калибровка</span>"
        + "</div></div>"
    )


def _match_table_row(card):
    prediction = favorite(card) or {}
    values = []
    for key in ("p1", "px", "p2"):
        value = num(card.get(key))
        values.append(f"{value:.0%}" if value is not None and 0 <= value <= 1 else "—")
    odds = card.get("market_odds") or {}
    odds_text = (
        " / ".join(f"{num(odds[key]):.2f}" for key in ("П1", "X", "П2"))
        if has_prices(card) else "—"
    )
    pick = str(prediction.get("pick") or "—")
    probability = num(prediction.get("probability"))
    pick_text = f"{pick} · {probability:.0%}" if probability is not None else pick
    verdict = card.get("verdict") if isinstance(card.get("verdict"), dict) else {}
    value_passed = bool(verdict.get("real_odds") and verdict.get("value_passed"))
    league = str(card.get("league") or card.get("div") or "Футбол")
    time_text = " ".join(
        part for part in (str(card.get("time") or "").strip(), str(card.get("date") or card.get("date_iso") or "").strip())
        if part
    )
    return (
        "<tr>"
        f"<td class='neo-time'>{esc(time_text or '—')}</td>"
        f"<td><span class='neo-league'>{esc(league)}</span></td>"
        f"<td class='neo-fixture'><b>{esc(card.get('home_team') or 'Хозяева')}</b>"
        f"<span> — </span><b>{esc(card.get('away_team') or 'Гости')}</b></td>"
        + "".join(f"<td class='neo-prob'>{esc(value)}</td>" for value in values)
        + f"<td class='neo-odds'>{esc(odds_text)}</td>"
        + f"<td><span class='neo-pick {'value' if value_passed else ''}'>{esc(pick_text)}</span></td>"
        + "</tr>"
    )


def _kpi(label, value, note="", tone=""):
    return (
        f"<div class='neo-kpi {esc(tone)}'>"
        f"<div class='neo-kpi-label'>{esc(label)}</div>"
        f"<div class='neo-kpi-value'>{esc(value)}</div>"
        f"<div class='neo-kpi-note'>{esc(note)}</div>"
        "</div>"
    )


def _quality_panel(cards):
    total = len(cards)
    summaries = [context_summary(card) for card in cards]
    metrics = [
        ("Реальные коэффициенты 1X2", sum(has_prices(card) for card in cards)),
        ("Подтверждённые стартовые XI", sum(item["xi"] for item in summaries)),
        ("Назначенный судья", sum(item["referee"] for item in summaries)),
        ("Подтверждённый фид травм", sum(item["injuries"] for item in summaries)),
    ]
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
        "<div class='neo-panel-heading'><div><span class='neo-panel-kicker'>SOURCE MONITOR</span>"
        "<h3>Качество данных</h3></div><span class='neo-panel-icon'>◈</span></div>"
        f"<div class='neo-quality-summary'><strong>{sum(has_prices(c) for c in cards)}/{total}</strong>"
        "<span>полных линий 1X2 в текущем скане</span></div>"
        + "".join(rows)
        + "<p class='neo-panel-footnote'>Нет подтверждения — это неизвестное значение, а не доказательство отсутствия события.</p>"
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
        f"<span>ROI виртуального портфеля {esc(roi_text)} · прибыль {profit:+.2f}</span>"
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
    turnover = sum(num(bet.get("stake")) or 0.0 for bet in closed_bets)
    profit = sum(
        (num(bet.get("stake")) or 0.0) * ((num(bet.get("odds")) or 1.0) - 1.0)
        * (1.0 if bet.get("status") == "won" else -1.0)
        for bet in closed_bets
    )
    roi = profit / turnover * 100 if turnover else 0.0

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

    brier = f"{metrics['brier']:.4f}" if metrics.get("brier") is not None else "—"
    log_loss = f"{metrics['log_loss']:.4f}" if metrics.get("log_loss") is not None else "—"
    accuracy = f"{metrics['accuracy']:.1%}" if metrics.get("accuracy") is not None else "—"
    market_delta = metrics.get("market_delta_brier")
    market_delta_text = (
        f"{market_delta:+.4f}" if market_delta is not None else "—"
    )
    market_note = (
        f"совпадающих матчей: {metrics.get('market_n', 0)} · отрицательное лучше"
        if market_delta is not None else "нет достаточной парной линии и результата"
    )
    kpis = [
        _kpi("Матчей в скане", f"{len(cards):,}", "текущий набор матчей"),
        _kpi("Реальная линия 1X2", f"{priced:,}", "полная подтверждённая линия"),
        _kpi("Settled · FT", f"{metrics.get('n', 0):,}", "только основная модель"),
        _kpi("Brier Score", brier, "ниже — лучше"),
        _kpi("Log Loss", log_loss, "ниже — лучше"),
        _kpi("Accuracy · 1X2", accuracy, "доля угаданных исходов"),
    ]
    st.markdown(
        "<div class='neo-kpi-grid'>" + "".join(kpis) + "</div>"
        + "<div class='neo-market-delta'>"
        "<span>СРАВНЕНИЕ С РЫНКОМ · Δ BRIER</span>"
        f"<b>{esc(market_delta_text)}</b><small>{esc(market_note)}</small>"
        "</div>",
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.68, 1.0], gap="medium")
    with left:
        table_rows = "".join(_match_table_row(card) for card in cards[:7])
        if not table_rows:
            table_rows = (
                "<tr><td colspan='8' class='neo-empty-cell'>"
                "Пока нет матчей для отображения. Открой сканер матчей и запусти обновление."
                "</td></tr>"
            )
        st.markdown(
            "<section class='neo-panel neo-matches-panel'>"
            "<div class='neo-panel-heading'><div><span class='neo-panel-kicker'>MATCH INTELLIGENCE</span>"
            "<h3>Ближайшие матчи — прогнозы модели</h3></div>"
            f"<span class='neo-count-pill'>{len(cards)} матчей</span></div>"
            "<div class='neo-table-scroll'><table class='neo-match-table'><thead><tr>"
            "<th>Время</th><th>Лига</th><th>Матч</th><th>П1</th><th>X</th><th>П2</th>"
            "<th>Коэффициенты 1X2</th><th>Прогноз</th>"
            "</tr></thead><tbody>"
            + table_rows
            + "</tbody></table></div>"
            f"<div class='neo-table-footer'>Показано {min(7, len(cards))} из {len(cards)} матчей · котировки отображаются только при полной реальной линии.</div>"
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
            "Калибровка будет показана после накопления более широкой выборки.</span></div>"
        )
        st.markdown(
            "<section class='neo-panel neo-reliability-panel'>"
            "<div class='neo-panel-heading'><div><span class='neo-panel-kicker'>MODEL QUALITY</span>"
            "<h3>Reliability · калибровка</h3></div>"
            f"<span class='neo-count-pill'>{reliability_n:,} FT</span></div>"
            + chart
            + "<div class='neo-panel-footnote'>Сопоставление прогнозной вероятности и фактической частоты по исходам П1 / X / П2.</div>"
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
            _history_panel(snapshot, roi, profit, turnover),
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
