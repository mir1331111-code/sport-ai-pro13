"""ui/tabs/scanner.py — Сканер с ручным вводом кэфов."""
from __future__ import annotations
import os
import re
import json
import csv
import io
import html
import math
import time
import hashlib
import threading
import requests
from datetime import datetime, timedelta

import streamlit as st

from config import (
    APP_VERSION, DIV_NAMES, DIV_TO_ODDS, DIV_TO_TSDB, DB_FILE,
    DISK_CACHE_DIR, STRONG_FORECAST_MIN_PROB,
)
from storage import usage
from storage import sqlite_store as db
from data.sources import (load_seasonal, season_str, fdorg_matches, fdorg_match_result,
                          parse_date, tsdb_matches, espn_matches, highlightly_odds_fixture,
                          sstats_upcoming_matches, sstats_odds_fixture,
                          footballapi_xg_history)
from data.fixture_merge import merge_fixture_sources, fixture_key
from data.fixture_time import app_timezone, fixture_kickoff_local
from data.context_data import (
    ApiFootballClient, ApiFootballContextProvider, fixture_row_key,
)
from data.context_history import parse_context_csv
from data.history_import import parse_historical_matches_csv
from model.context_model import (
    assess_context_quality, context_probabilities, walk_forward_context_ablation,
)
from model.market_models import (
    asian_handicap_probabilities, asian_handicap_expected_value,
    adjust_corners_for_game_state,
)
from storage.context_store import ContextStore
from model.engine import Engine, _norm
from model.validation_cache import select_validation_rows, validation_cache_key
from model.training import prepare_completed_training_rows, train_engine_chronologically
from model.market_forecasts import count_over_probability, is_half_line
from betting.verdict import build_verdict
from betting.best_picks import best_value_market, best_value_markets
from betting.ranking import model_favorite, rank_model_favorites, rank_score
from betting.market_ranking import market_forecast_summary, rank_market_forecasts
from betting.kelly import kelly, market_type
from betting.market_math import blend_probability, proportional_devig
from llm.analyst import LLMRateLimitError, analyze_match, has_sufficient_history
from model.independent_forecasts import historical_features, record_scan_forecasts
from ui.cards import (render_verdict_card, translate_team,
                      parse_card_date, day_label, render_fixture_header)
from ui.scanner_engine import _train_engine, _shared_engine_cache, _load_engine

LLM_TOP_N = 8
MAX_STAKE_PCT = 0.05
AUTO_PORTFOLIO_MIN_PROB = STRONG_FORECAST_MIN_PROB  # Shared 65% gate; separate from Value proof.
MARKET_FORECAST_DISPLAY_LIMIT = 20


def _runtime_secret(session_key: str, *names: str) -> str:
    """Read credentials from the current password widget, environment, or Streamlit secrets."""
    value = str(st.session_state.get(session_key, "") or "").strip()
    if value:
        return value
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    try:
        for name in names:
            value = str(st.secrets.get(name, "") or "").strip()
            if value:
                return value
        # Also accept common nested layouts such as [api] or [highlightly].
        for section_name in ("api", "apis", "secrets", "highlightly", "sstats"):
            section = st.secrets.get(section_name, {})
            if hasattr(section, "get"):
                for name in names:
                    value = str(section.get(name, "") or "").strip()
                    if value:
                        return value
                if section_name == "sstats":
                    for alias in ("api_key", "key"):
                        value = str(section.get(alias, "") or "").strip()
                        if value:
                            return value
    except Exception:
        pass
    return ""


def _runtime_int(session_key: str, *names: str, default: int, minimum: int = 0) -> int:
    """Read an integer deployment setting from session, environment or Streamlit Secrets."""
    raw = _runtime_secret(session_key, *names)
    try:
        return max(int(minimum), int(raw)) if raw else int(default)
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _safe_filter(rows):
    if not isinstance(rows, list):
        return []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if not r.get("HomeTeam") or not r.get("AwayTeam"):
            continue
        out.append(r)
    return out


def _render_context_comparison(card: dict) -> None:
    """Show Base/Context side-by-side and make missing mandatory data explicit."""
    quality = card.get("context_quality") or {}
    pair = card.get("context_forecasts") or {}
    base = pair.get("Base") or {}
    contextual = pair.get("Context") or {}
    api_context = card.get("api_context") or {}
    lineup = api_context.get("lineup") or {}
    referee = api_context.get("referee") or {}
    home_xi = (lineup.get("home") or {}).get("confirmed", False)
    away_xi = (lineup.get("away") or {}).get("confirmed", False)
    referee_name = str(referee.get("name") or "не подтверждён")
    q_label = str(quality.get("level") or "Low")
    gate = "разрешён" if quality.get("strong_value_eligible") else "ЗАБЛОКИРОВАН"
    st.caption(
        f"Контекст {q_label} · стартовый XI: хозяева {'✓' if home_xi else 'нет'}, "
        f"гости {'✓' if away_xi else 'нет'} · судья: {referee_name} · "
        f"Strong/Value: {gate}"
    )
    if base and contextual:
        st.caption(
            "Base → Context · "
            f"П1 {base.get('H', 0):.1%} → {contextual.get('H', 0):.1%} "
            f"(Δ {(contextual.get('H', 0) - base.get('H', 0)) * 100:+.2f} п.п.) · "
            f"X {base.get('D', 0):.1%} → {contextual.get('D', 0):.1%} · "
            f"П2 {base.get('A', 0):.1%} → {contextual.get('A', 0):.1%}"
        )
    if not quality.get("strong_value_eligible"):
        st.warning(
            str(quality.get("reason") or "Контекст не подтверждён; Strong/Value запрещены."),
            icon="🔒",
        )

    # Show the source records, not just the yes/no confirmation flags.
    home_players = (lineup.get("home") or {}).get("starting_xi") or []
    away_players = (lineup.get("away") or {}).get("starting_xi") or []
    injuries = api_context.get("injuries") or []
    if home_players or away_players or injuries:
        with st.expander("Составы и сведения о травмах", expanded=False):
            if home_players or away_players:
                col_home, col_away = st.columns(2)
                with col_home:
                    st.markdown("**Стартовый состав хозяев**")
                    if home_players:
                        st.write(", ".join(
                            str(player.get("name") or player.get("id") or "Игрок")
                            for player in home_players
                        ))
                    else:
                        st.caption("Стартовый состав не опубликован.")
                with col_away:
                    st.markdown("**Стартовый состав гостей**")
                    if away_players:
                        st.write(", ".join(
                            str(player.get("name") or player.get("id") or "Игрок")
                            for player in away_players
                        ))
                    else:
                        st.caption("Стартовый состав не опубликован.")
            st.markdown("**Травмы / дисквалификации по фиду API**")
            if injuries:
                for entry in injuries:
                    player_name = str(entry.get("player_name") or "Игрок без имени")
                    team_name = str(entry.get("team_name") or "Команда не указана")
                    reason = str(entry.get("reason") or entry.get("type") or "Причина не указана")
                    st.caption(f"{team_name}: {player_name} · {reason}")
            else:
                st.caption(
                    "Ответ API получен без записей. Это не доказывает, что в турнире "
                    "отсутствуют травмированные игроки."
                )

    reliability = referee.get("source_reliability")
    lineup_reliability = lineup.get("source_reliability")
    st.caption(
        "Надёжность источников · составы: "
        + (f"{float(lineup_reliability):.0%}" if lineup_reliability is not None else "не откалибрована")
        + " · судья: "
        + (f"{float(reliability):.0%}" if reliability is not None else "не откалибрована")
        + " · травматический фид: "
        + ("ответ получен" if api_context.get("injuries_is_confirmed") else "нет подтверждения")
    )

    historical = card.get("historical_stats") or {}
    card_stats = historical.get("cards") or {}
    ref_stats = card_stats.get("referee") or {}
    if ref_stats.get("available"):
        st.caption(
            f"Судья в исторической выборке карточек: n={ref_stats['n']} · "
            f"среднее {ref_stats['mean']:.2f} · "
            f"дисперсия {ref_stats['variance']:.2f} · "
            f"вес shrinkage {ref_stats['blend_weight']:.0%}. "
            "Это информационная поправка, не проверенный букмекерский edge."
        )

    corner_stats = historical.get("corners") or {}
    state_splits = corner_stats.get("game_state_splits") or {}
    level_stats = state_splits.get("level") or {}
    if level_stats.get("n", 0) > 0:
        means = {
            state: value.get("mean")
            for state, value in state_splits.items()
            if isinstance(value, dict) and value.get("mean") is not None
        }
        state_labels = {
            "home_trailing": "хозяева проигрывают к перерыву",
            "level": "ничья к перерыву",
            "away_trailing": "гости проигрывают к перерыву",
        }
        scenario_text = []
        for state in ("home_trailing", "level", "away_trailing"):
            sample = state_splits.get(state) or {}
            if sample.get("n", 0) <= 0:
                continue
            adjusted = adjust_corners_for_game_state(
                float(level_stats["mean"]),
                state=state,
                historical_state_means=means,
            )
            scenario_text.append(
                f"{state_labels[state]}: {adjusted['mean']:.1f} угловых во 2-м тайме "
                f"(n={sample['n']})"
            )
        if scenario_text:
            st.caption(
                "Исторические угловые по состоянию счёта на перерыв · "
                + " · ".join(scenario_text)
                + ". Требуются HTHG/HTAG и HTHC/HTAC; сценарии не подменяют предматчевую базу."
            )


def _portfolio_auto_recommendation(card: object) -> dict | None:
    """Return one conservative virtual-portfolio candidate for a fixture.

    1X2 requires a complete, named single-bookmaker 1X2 line. Goals O/U is
    allowed only on 2.5 and only after the separate paired bookmaker benchmark
    passed. BTTS, corners and cards remain forecast-only until separately
    validated against an untouched market benchmark.
    """
    if not isinstance(card, dict):
        return None
    # Legacy cached cards without a confirmed XI and referee are never
    # grandfathered into the portfolio after the context gate is introduced.
    if not bool((card.get("context_quality") or {}).get("strong_value_eligible")):
        return None
    recommendations = card.get("best_by_market") or {}
    books = card.get("bookmaker_by_pick") or {}
    candidates = []

    for family in ("winner", "goals"):
        recommendation = recommendations.get(family)
        if not isinstance(recommendation, dict):
            continue
        best = recommendation.get("best")
        verdict = recommendation.get("verdict")
        if (
            not isinstance(best, (tuple, list))
            or len(best) < 6
            or not isinstance(verdict, dict)
            or not verdict.get("real_odds")
            or not verdict.get("value_passed")
            or not verdict.get("is_bet")
        ):
            continue

        market_code = str(best[0] or "").upper()
        pick = str(best[1] or "")
        if not pick or not str(books.get(pick) or "").strip():
            continue

        if family == "winner":
            if market_code != "1X2":
                continue
            coherent_line = proportional_devig(
                card.get("market_odds") or {},
                ("П1", "X", "П2"),
                bookmaker_by_pick=books,
            )
            if coherent_line is None:
                continue
        else:
            if market_code != "OU" or not bool(card.get("total_market_validation")):
                continue
            line_match = re.fullmatch(r"Т[БМ]\s+(\d+(?:[.,]\d+)?)", pick, re.I)
            if not line_match:
                continue
            try:
                line = float(line_match.group(1).replace(",", "."))
            except ValueError:
                continue
            if abs(line - 2.5) > 1e-9:
                continue
            coherent_line = proportional_devig(
                card.get("total_market_odds") or {},
                ("ТБ 2.5", "ТМ 2.5"),
                bookmaker_by_pick=books,
            )
            if coherent_line is None:
                continue

        try:
            odd, ev, probability, stake = map(
                float, (best[2], best[3], best[4], best[5])
            )
            edge_value = float(verdict.get("edge"))
        except (TypeError, ValueError):
            continue
        if (
            odd <= 1.01
            or not 0.0 < probability < 1.0
            or ev < 0.05
            or edge_value < 0.05
            or stake <= 0.0
        ):
            continue
        candidates.append({
            "family": family,
            "best": best,
            "verdict": verdict,
            "score": float(recommendation.get("score") or 0.0),
        })

    return max(candidates, key=lambda item: item["score"]) if candidates else None


def _is_actionable_recommendation(card: object) -> bool:
    """True only for a conservatively eligible, fully priced portfolio candidate."""
    if not isinstance(card, dict):
        return False
    choice = _portfolio_auto_recommendation(card)
    if not isinstance(choice, dict):
        return False
    verdict = choice.get("verdict") or {}
    best = choice.get("best")
    if not isinstance(best, (tuple, list)) or len(best) < 6:
        return False
    try:
        odd, ev, probability, stake = map(float, (best[2], best[3], best[4], best[5]))
        edge_value = float(verdict.get("edge"))
    except (TypeError, ValueError):
        return False
    return (
        bool(verdict.get("real_odds"))
        and bool(verdict.get("value_passed"))
        and bool(verdict.get("is_bet"))
        and odd > 1.01
        and 0.0 < probability < 1.0
        and ev >= 0.05
        and edge_value >= 0.05
        and stake > 0.0
    )


def _qualified_bet_cards(cards: object) -> list[dict]:
    """Build one conservative 1X2/validated O/U 2.5 pick per fixture.

    The dedicated BTTS/corners/cards tabs still show their own independent
    forecast signals. They are not labelled as portfolio-ready bets here.
    """
    if not isinstance(cards, list):
        return []
    qualified = []
    for card in cards:
        choice = _portfolio_auto_recommendation(card)
        if not choice or not _is_actionable_recommendation(card):
            continue
        item = dict(card)
        item["best"] = choice["best"]
        item["verdict"] = choice["verdict"]
        pick = str(choice["best"][1])
        item["best_odds_source"] = (
            (card.get("market_odds_source_by_pick") or {}).get(pick)
            or card.get("best_odds_source")
            or card.get("odds_source", "unknown")
        )
        qualified.append(item)
    return qualified


def _stat_market_lines(odds: dict, prefix: str) -> list[float]:
    """Return the exact half/integer lines present in an SStats corner/card quote map."""
    if not isinstance(odds, dict):
        return []
    pattern = re.compile(
        rf"^{re.escape(str(prefix))}\s+(?:ТБ|ТМ)\s+(\d+(?:[.,]\d+)?)$",
        re.IGNORECASE,
    )
    lines = set()
    for selection in odds:
        match = pattern.fullmatch(str(selection or "").strip())
        if not match:
            continue
        try:
            line = float(match.group(1).replace(",", "."))
        except ValueError:
            continue
        if 0.0 < line <= 30.0 and is_half_line(line):
            lines.add(line)
    return sorted(lines)


def _print_log(msg: str):
    try:
        import sys
        print(f"[SCAN] {msg}", file=sys.stdout, flush=True)
    except Exception:
        pass


def _render_manual_odds_block(
    c, D, kelly_frac, min_prob=0.40, min_edge=0.03, min_ev=0.03,
    market_filter=None, market_recommendation=None,
):
    """Понятный ввод букмекерского коэффициента для выбранного рынка."""
    v = c.get("verdict", {})
    stat_preds = c.get("historical_stats") or {}
    from model.historical_match_stats import poisson_over_probability
    market_options = {
        "Победа хозяев (П1)": ("П1", float(c.get("p1") or 0)),
        "Ничья (X)": ("X", float(c.get("px") or 0)),
        "Победа гостей (П2)": ("П2", float(c.get("p2") or 0)),
        "Обе команды забьют — да": ("BTTS Да", float(c.get("btts") or 0)),
        "Обе команды забьют — нет": ("BTTS Нет", 1.0 - float(c.get("btts") or 0)),
    }
    # Evaluate every common full-match goals line, not only 2.5.
    try:
        total_mean = max(0.05, float(c.get("lam_h") or 0) + float(c.get("lam_a") or 0))
    except (TypeError, ValueError):
        total_mean = 0.0
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        over_p = (float(c.get("over") or 0) if line == 2.5 else
                  poisson_over_probability(total_mean, line) if total_mean else None)
        if over_p is None or not 0.0 < over_p < 1.0:
            continue
        line_label = f"{line:.1f}"
        market_options[f"Тотал голов больше {line_label}"] = (f"ТБ {line_label}", over_p)
        market_options[f"Тотал голов меньше {line_label}"] = (f"ТМ {line_label}", 1.0 - over_p)
    # Build statistical-market forecasts for the exact lines returned by SStats.
    # Previously the UI knew only corners 8.5/9.5/10.5 and cards 3.5/4.5/5.5;
    # a real quote at another line caused the fixture to appear in the tab but
    # its price was then hidden from the card.
    stat_odds = c.get("stat_market_odds") or {}
    stat_specs = (
        ("corners", "Угл", "Угловые", (8.5, 9.5, 10.5)),
        ("cards", "Карты", "Карточки", (3.5, 4.5, 5.5)),
    )
    for stat_market, market_code, label_prefix, default_lines in stat_specs:
        pred = stat_preds.get(stat_market) or {}
        quoted_lines = _stat_market_lines(stat_odds, market_code)
        # Keep the stable reference line available even when the bookmaker only
        # publishes alternative lines; the card will label it as model-only.
        lines = sorted(set(quoted_lines) | set(default_lines))
        if pred.get("total") is not None and pred.get("history_matches", 0) >= 30:
            for line in lines:
                over_p = count_over_probability(
                    pred["total"], line, pred.get("total_variance")
                )
                if over_p is not None and 0.0 < over_p < 1.0:
                    line_label = f"{line:g}"
                    market_options[f"{label_prefix}: больше {line_label}"] = (
                        f"{market_code} ТБ {line_label}", over_p
                    )
                    market_options[f"{label_prefix}: меньше {line_label}"] = (
                        f"{market_code} ТМ {line_label}", 1.0 - over_p
                    )

    # Automatically select the strongest forecast; no manual market cycling.
    source_odds = c.get("market_odds") or {}
    total_odds = c.get("total_market_odds") or {}
    btts_odds = c.get("btts_market_odds") or {}
    all_market_odds = {}
    for price_group in (source_odds, total_odds, btts_odds, stat_odds):
        if isinstance(price_group, dict):
            all_market_odds.update(price_group)
    bookmaker_by_pick = c.get("bookmaker_by_pick") or {}

    def has_coherent_market_line(candidate_pick):
        kind = market_type(candidate_pick)
        if kind == "1X2":
            paired = ("П1", "X", "П2")
        elif kind == "OU":
            matched = re.fullmatch(
                r"Т[БМ]\s+(\d+(?:[.,]\d+)?)", str(candidate_pick), re.I
            )
            if not matched:
                return False
            line = matched.group(1).replace(",", ".")
            paired = (f"ТБ {line}", f"ТМ {line}")
        elif kind == "BTTS":
            paired = ("BTTS Да", "BTTS Нет")
        elif kind == "STAT":
            matched = re.fullmatch(
                r"(Угл|Карты)\s+Т[БМ]\s+(\d+(?:[.,]\d+)?)",
                str(candidate_pick), re.I,
            )
            if not matched:
                return False
            prefix = "Угл" if matched.group(1).casefold() == "угл" else "Карты"
            line = matched.group(2).replace(",", ".")
            paired = (f"{prefix} ТБ {line}", f"{prefix} ТМ {line}")
        else:
            return False
        return proportional_devig(
            all_market_odds, paired, bookmaker_by_pick=bookmaker_by_pick
        ) is not None

    def quote_for(candidate_pick):
        kind = market_type(candidate_pick)
        # Extra markets are usable only when the detail page explicitly labels
        # a quote belongs to the exact market. A page-wide bookmaker mention is not proof
        # that every parsed price belongs to that bookmaker.
        extra_source = str(c.get("extra_market_source") or "").casefold()
        # Secondary markets have independent provenance. A verified SStats
        # BTTS quote must not accidentally bless a stale totals quote from
        # another provider merely because both live on the same fixture card.
        if kind == "OU":
            source = str(c.get("total_odds_source") or "").casefold()
            if source not in {"highlightly", "manual_import", "sstats"}:
                return None
        elif kind == "BTTS":
            source = str(c.get("btts_odds_source") or "").casefold()
            if source not in {"highlightly", "manual_import", "sstats"}:
                return None
        elif kind == "STAT" and extra_source not in {"highlightly", "manual_import", "sstats"}:
            return None
        if kind == "1X2":
            raw = source_odds.get(candidate_pick)
        elif kind == "OU":
            raw = total_odds.get(candidate_pick)
        elif kind == "BTTS":
            raw = btts_odds.get(candidate_pick)
        elif kind == "STAT":
            raw = stat_odds.get(candidate_pick)
        else:
            raw = None
        try:
            value = float(raw or 0)
            return value if 1.01 < value <= 100 else None
        except (TypeError, ValueError):
            return None

    filter_types = {
        "winner": {"1X2"},
        "goals": {"OU"},
        "btts": {"BTTS"},
        "corners": {"STAT"},
        "cards": {"STAT"},
    }
    # In "all matches" mode, compare every modeled market. Dedicated tabs
    # restrict the list to their own market type.
    # The default "all matches" view is a betting shortlist, not a list of
    # every visible quote. Keep unvalidated BTTS/stat markets out of the
    # headline selection; they remain available in their dedicated info tabs.
    allowed_types = filter_types.get(market_filter, {"1X2", "OU"} if market_filter is None else {"1X2", "OU", "BTTS", "STAT"})
    if market_filter is None and not c.get("total_market_validation"):
        allowed_types = {"1X2"}
    candidate_options = {
        label: item for label, item in market_options.items()
        if market_type(item[0]) in allowed_types
    }
    if market_filter == "corners":
        candidate_options = {label: item for label, item in candidate_options.items()
                             if item[0].startswith("Угл ")}
    elif market_filter == "cards":
        candidate_options = {label: item for label, item in candidate_options.items()
                             if item[0].startswith("Карты ")}

    # Show all available exact-market prices in the current market view.
    displayed_quotes = [
        (item[0], quote_for(item[0]))
        for item in candidate_options.values()
        if quote_for(item[0]) is not None
    ]
    if displayed_quotes:
        quote_text = " · ".join(f"{pick}: **{odd:.2f}**" for pick, odd in displayed_quotes)
        with st.expander("Другие коэффициенты", expanded=False):
            st.caption("Котировки SStats по точным рынкам: " + quote_text)
    elif market_filter in {"goals", "btts"}:
        market_names = {"goals": "тотала голов", "btts": "рынка «Обе забьют»"}
        st.caption(
            f"Нет подтверждённой котировки SStats на точный исход {market_names.get(market_filter, 'рынка')}. "
            "Коэффициенты 1X2 не подставляются вместо неё."
        )
    asian_quotes = c.get("asian_handicap_market_odds") or {}
    if asian_quotes:
        asian_text = " · ".join(
            f"{pick}: **{float(odd):.2f}**"
            for pick, odd in sorted(asian_quotes.items())
        )
        st.caption("Азиатские форы SStats (информационно): " + asian_text)

    quoted_candidates = [
        (label, item[0], item[1], quote_for(item[0]))
        for label, item in candidate_options.items()
        if quote_for(item[0]) is not None and 0 < item[1] < 1
    ]
    # Prefer the strongest model forecast among markets with a real,
    # exact-market quote. If none is priced, keep the best model forecast
    # visible but mark it as unpriced and do not allow value/portfolio actions.
    if not candidate_options:
        if market_filter in {"corners", "cards"}:
            market_code = "Угл" if market_filter == "corners" else "Карты"
            actual_quotes = []
            for selection, raw_price in stat_odds.items():
                if not str(selection).startswith(market_code + " "):
                    continue
                try:
                    price = float(raw_price)
                    if 1.01 < price <= 100:
                        actual_quotes.append((str(selection), price))
                except (TypeError, ValueError):
                    continue
            if actual_quotes:
                quote_text = " · ".join(
                    f"{selection}: **{price:.2f}**"
                    for selection, price in sorted(actual_quotes)
                )
                st.caption("Реальные котировки SStats по доступным линиям: " + quote_text)
                stat_name = "corners" if market_filter == "corners" else "cards"
                stat_history = stat_preds.get(stat_name) or {}
                history_n = int(stat_history.get("history_matches") or 0)
                home_n = int(stat_history.get("home_team_games") or 0)
                away_n = int(stat_history.get("away_team_games") or 0)
                league_name = str(c.get("league") or "неизвестная лига")
                div_code = str(c.get("div") or "G")
                if history_n == 0:
                    reason = (
                        f"Для лиги «{league_name}» (код {div_code}) в обучающей выборке нет "
                        f"подтверждённых исторических данных {('HC/AC' if stat_name == 'corners' else 'HY/AY/HR/AR')}."
                    )
                elif history_n < 30:
                    reason = (
                        f"По лиге «{league_name}» (код {div_code}) найдено только {history_n} "
                        "матчей с полной статистикой; требуется минимум 30."
                    )
                else:
                    reason = (
                        f"История лиги: {history_n} матчей; домашних матчей хозяев: {home_n}; "
                        f"выездных матчей гостей: {away_n}. Для оценки пары нужно минимум "
                        "4 домашних и 4 выездных наблюдения."
                    )
                st.info(reason + " Коэффициенты получены, но вероятность не рассчитывается; ставку не рекомендую.")
            else:
                st.info("Для этого матча SStats не вернул котировки этого статистического рынка.")
        else:
            st.info("Для этого рынка модель пока не сформировала прогноз с достаточными данными.")
        return
    # Rank priced picks by expected value, not raw probability. A 38% outcome
    # at 2.36 is not a recommendation when its EV is negative. Keep unpriced
    # model forecasts visible in market-specific views, but never present them
    # as a bet recommendation.
    priced_metrics = [
        (label, pick_name, probability, odd, probability * odd - 1,
         probability - (1.0 / odd))
        for label, pick_name, probability, odd in quoted_candidates
    ]
    positive_value = [row for row in priced_metrics if row[4] > 0 and row[5] > 0]
    if positive_value:
        selected_label, pick, model_prob, selected_odd, selected_ev, selected_edge = max(
            positive_value, key=lambda row: (row[4], row[5], row[2])
        )
        auto_quote_exists = True
        has_positive_value = True
    elif priced_metrics:
        selected_label, pick, model_prob, selected_odd, selected_ev, selected_edge = max(
            priced_metrics, key=lambda row: (row[4], row[5], row[2])
        )
        auto_quote_exists = True
        has_positive_value = False
    else:
        summary = market_forecast_summary(c, market_filter) if market_filter else None
        preferred_pick = str((summary or {}).get("pick") or "")
        preferred = next(
            (
                (label, item) for label, item in candidate_options.items()
                if str(item[0]) == preferred_pick
            ),
            None,
        )
        if preferred is None:
            selected_label, (pick, model_prob) = max(
                candidate_options.items(), key=lambda row: row[1][1]
            )
        else:
            selected_label, (pick, model_prob) = preferred
        selected_odd = None
        selected_ev = None
        selected_edge = None
        auto_quote_exists = False
        has_positive_value = False

    # Dedicated market tabs render their own independently selected best pick.
    rec_verdict = None
    if isinstance(market_recommendation, dict):
        candidate_verdict = market_recommendation.get("verdict")
        candidate_best = market_recommendation.get("best")
        if isinstance(candidate_verdict, dict) and isinstance(candidate_best, (tuple, list)) and len(candidate_best) >= 6:
            try:
                rec_verdict = candidate_verdict
                pick = str(candidate_best[1])
                selected_label = str(candidate_verdict.get("label") or candidate_best[1])
                selected_odd = float(candidate_best[2])
                selected_ev = float(candidate_verdict.get("ev"))
                selected_edge = float(candidate_verdict.get("edge"))
                model_prob = float(candidate_verdict.get("model_prob") or candidate_verdict.get("prob"))
                auto_quote_exists = True
                has_positive_value = bool(candidate_verdict.get("value_passed"))
            except (TypeError, ValueError):
                rec_verdict = None

    selected_market_type = market_type(pick)
    if market_filter is None:
        try:
            selected_probability = float(model_prob)
        except (TypeError, ValueError):
            selected_probability = None
        if (
            selected_probability is not None
            and math.isfinite(selected_probability)
            and 0.0 < selected_probability < 1.0
        ):
            context_eligible = bool(
                (c.get("context_quality") or {}).get("strong_value_eligible")
            )
            strength_label = (
                f"СИЛЬНЫЙ ПРОГНОЗ ≥{STRONG_FORECAST_MIN_PROB:.0%}"
                if selected_probability >= STRONG_FORECAST_MIN_PROB and context_eligible else
                "Strong заблокирован: нет подтверждённого XI и/или судьи"
                if selected_probability >= STRONG_FORECAST_MIN_PROB else
                f"прогноз ниже порога сильного {STRONG_FORECAST_MIN_PROB:.0%}"
            )
            st.caption(
                f"{strength_label} · {selected_label} "
                f"{selected_probability:.1%}; это вероятность модели, не гарантия."
            )
    market_pair_validated = has_coherent_market_line(pick)
    if selected_market_type == "1X2":
        market_is_validated = (
            bool(c.get("winner_market_validation")) and market_pair_validated
        )
        validation_parts = []
        if not c.get("winner_market_validation"):
            validation_parts.append(
                "Модель пока не доказала статистически значимое улучшение над de-vig букмекерским рынком "
                "на ≥500 парных исторических матчах. Это прогноз, не Value-ставка."
            )
        if not market_pair_validated:
            validation_parts.append(
                "Нет полной линии П1/X/П2 у одного идентифицированного букмекера."
            )
        validation_note = " ".join(validation_parts)
    elif selected_market_type == "OU":
        # Validate the exact goal-total line, not whichever line happened to
        # make the fixture's global shortlist.
        line_match = re.fullmatch(r"Т[БМ]\s+(\d+(?:[.,]\d+)?)", str(pick), re.I)
        line_key = (
            f"{float(line_match.group(1).replace(',', '.')):g}"
            if line_match else ""
        )
        line_reports = c.get("total_market_validation_by_line") or {}
        # Only an explicitly tested exact line is eligible. A passing 2.5
        # line must never validate 1.5, 3.5, or another untested line.
        history_validated = bool(line_reports.get(line_key, False))
        source = str(c.get("total_odds_source") or "").casefold()
        book_count = int((c.get("total_market_book_count") or {}).get(pick) or 0)
        quote_corroborated = (
            source != "sstats"
            or bool((c.get("total_market_confirmed") or {}).get(pick))
        )
        bookmaker_benchmark_validated = (
            abs(float(line_key or 0.0) - 2.5) < 1e-9
            and bool(c.get("total_market_validation"))
        )
        market_is_validated = (
            history_validated and bookmaker_benchmark_validated
            and quote_corroborated and market_pair_validated
        )
        validation_notes = []
        if not history_validated:
            validation_notes.append(
                f"Линия тотала {line_key or 'выбранная'} не прошла отдельную историческую проверку."
            )
        if not bookmaker_benchmark_validated:
            validation_notes.append(
                "Для этой линии нет подтверждённого улучшения модели над de-vig букмекерским рынком; "
                "проверка против средней лиги не даёт права на Value."
            )
        if not quote_corroborated:
            validation_notes.append(
                f"Котировка тотала получена только от {book_count or 1} букмекера. "
                "Для допуска нужна вторая независимая котировка."
            )
        if not market_pair_validated:
            validation_notes.append("Нет обеих сторон линии у одного и того же букмекера.")
        validation_note = " ".join(validation_notes)
    elif selected_market_type == "BTTS":
        # Existing BTTS walk-forward compares against the league base rate,
        # not an untouched bookmaker benchmark. Keep the forecast visible only.
        market_is_validated = False
        validation_notes = [
            "BTTS пока не прошёл отдельный out-of-sample тест против de-vig букмекерского рынка."
        ]
        if not market_pair_validated:
            validation_notes.append("Нет обеих сторон BTTS у одного и того же букмекера.")
        validation_note = " ".join(validation_notes)
    elif selected_market_type == "STAT":
        stat_match = re.fullmatch(
            r"(Угл|Карты)\s+Т[БМ]\s+(\d+(?:[.,]\d+)?)",
            str(pick), re.I,
        )
        if stat_match:
            stat_name = (
                "corners" if stat_match.group(1).casefold() == "угл" else "cards"
            )
            stat_line = f"{float(stat_match.group(2).replace(',', '.')):g}"
            line_validated = bool(
                ((c.get("stat_market_validation") or {}).get(stat_name) or {}).get(stat_line)
            )
            stat_prediction = stat_preds.get(stat_name) or {}
            history_sufficient = (
                int(stat_prediction.get("history_matches") or 0) >= 30
                and int(stat_prediction.get("home_team_games") or 0) >= 4
                and int(stat_prediction.get("away_team_games") or 0) >= 4
            )
            source = str(c.get("extra_market_source") or "").casefold()
            quote_source_valid = source in {"highlightly", "manual_import", "sstats"}
            # The current statistical walk-forward compares with a league
            # baseline, not bookmaker prices. It is not an actionable Value gate.
            market_is_validated = False
            validation_parts = [
                "Модель угловых/карточек пока не прошла парный исторический тест против de-vig букмекерского рынка."
            ]
            if not line_validated:
                validation_parts.append(
                    f"Линия {stat_line} рынка {stat_name} не прошла собственный walk-forward тест."
                )
            if not history_sufficient:
                validation_parts.append(
                    "Недостаточно фактической истории: "
                    f"лига {stat_prediction.get('history_matches', 0)}/30, "
                    f"хозяева дома {stat_prediction.get('home_team_games', 0)}/4, "
                    f"гости на выезде {stat_prediction.get('away_team_games', 0)}/4."
                )
            if not quote_source_valid:
                validation_parts.append("Нет подтверждённой котировки на этот рынок.")
            if not market_pair_validated:
                validation_parts.append("Нет обеих сторон линии у одного и того же букмекера.")
            validation_note = " ".join(validation_parts)
        else:
            market_is_validated = False
            validation_note = "Не удалось определить точный статистический рынок и линию."
    else:
        market_is_validated = False
        validation_note = "Для этого рынка нет отдельной исторической проверки."

    if rec_verdict is not None and has_positive_value and market_is_validated:
        st.success("Рынок прошёл независимый отбор: полная линия, историческая проверка и положительный Value.")
    elif has_positive_value and not market_is_validated:
        st.warning((validation_note or "Рынок не прошёл подтверждение.") + " Ставку не рекомендовать.")
    elif auto_quote_exists and rec_verdict is None:
        st.caption("Котировка и модельная оценка есть, но этот исход не прошёл независимый отбор рынка.")
    elif auto_quote_exists and not market_is_validated:
        st.caption(validation_note or "Историческая надёжность или линия букмекера не подтверждена — не ставить.")
    elif auto_quote_exists:
        st.caption("Коэффициент есть, но надёжного преимущества не подтверждено — не ставить.")
    else:
        st.caption("Нет подтверждённого коэффициента — рекомендации к ставке нет.")

    # Only markets with an actual bookmaker quote can be auto-priced.
    # Model fair odds are not a substitute for a real quote.
    if selected_odd is not None:
        st.markdown(
            f"<div style='display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 16px;margin:8px 0;border:1px solid #285f51;border-radius:12px;background:linear-gradient(120deg,#102923,#101b22)'>"
            f"<div><div style='font-size:11px;letter-spacing:.12em;color:#8fd8bd;font-weight:700'>{'НАШ ВЫБОР' if rec_verdict is not None and market_is_validated else 'КОТИРОВКА · НЕТ ОТБОРА' if rec_verdict is None else 'КОТИРОВКА · РЫНОК НЕ ПРОВЕРЕН'}</div>"
            f"<strong style='font-size:20px;color:#f1f7f5'>{html.escape(selected_label)}</strong></div>"
            f"<div style='font-size:22px;font-weight:800;color:#8fe2c1;white-space:nowrap'>КФ {selected_odd:.2f}</div></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='padding:12px 16px;margin:8px 0;border:1px solid #46515a;border-radius:12px;background:#121a20'>"
            f"<div style='font-size:11px;letter-spacing:.12em;color:#9aa9b2;font-weight:700'>НЕТ ПОДТВЕРЖДЁННОГО КЭФА</div>"
            f"<strong style='font-size:18px;color:#d9e1e5'>{html.escape(selected_label)}</strong></div>",
            unsafe_allow_html=True,
        )

    key_base = f"manual_{c.get('match', '')}_{pick}".replace(" ", "_").replace("/", "_")
    input_key = f"odd_{key_base}"
    btn_key = f"btn_{key_base}"

    col_input, col_metrics, col_btn = st.columns([2, 2, 1])

    # Reuse the exact-market quote from any accepted real-odds source, including imported CSV/JSON.
    # Never treat the model's fair price as a bookmaker quote.
    source_odd = quote_for(pick)

    if source_odd is not None:
        manual_odd = source_odd
        quote_is_confirmed = True
        selected_kind = market_type(pick)
        source_label = str(
            c.get("total_odds_source") if selected_kind == "OU"
            else c.get("btts_odds_source") if selected_kind == "BTTS"
            else c.get("extra_market_source") if selected_kind == "STAT"
            else c.get("odds_source")
            or "SStats"
        )
        if selected_kind == "1X2" and not market_pair_validated:
            quote_is_confirmed = False
            source_label += " · нет полной линии одного букмекера"
        elif selected_kind != "1X2" and not market_pair_validated:
            quote_is_confirmed = False
            source_label += " · нет полной пары одного букмекера"
        if market_type(pick) == "OU" and str(c.get("total_odds_source") or "").casefold() == "sstats":
            book_name = (c.get("bookmaker_by_pick") or {}).get(pick)
            book_count = int((c.get("total_market_book_count") or {}).get(pick) or 0)
            quote_is_confirmed = bool((c.get("total_market_confirmed") or {}).get(pick))
            source_label = (
                f"SStats · {book_name or 'букмекер не указан'} · "
                f"{book_count} {'букмекера' if book_count in (2, 3, 4) else 'букмекеров' if book_count > 4 else 'букмекер'}"
            )
            if not quote_is_confirmed:
                source_label += " · не подтверждена второй книгой"
        if quote_is_confirmed:
            col_input.success(f"Коэффициент: **{manual_odd:.2f}** · {source_label}")
        else:
            col_input.warning(f"Коэффициент: **{manual_odd:.2f}** · {source_label}")
    else:
        manual_odd = None
        col_input.warning(
            "Нет реального коэффициента именно на этот рынок и линию из доступных источников. "
            "Расчёт value и добавление в портфель отключены — коэффициент 1X2 нельзя "
            "использовать для «Обе забьют», тоталов, угловых или карточек."
        )

    if manual_odd is not None and manual_odd > 1.01 and 0.0 < model_prob < 1.0:
        market_implied = 1.0 / manual_odd
        if rec_verdict is not None:
            # Preserve the blended probability and risk metrics selected for this
            # exact market using a complete same-bookmaker line.
            prob = float(rec_verdict.get("prob"))
            ev = float(rec_verdict.get("ev"))
            edge = float(rec_verdict.get("edge"))
            stake = float(rec_verdict.get("stake"))
            market_fair_probability = rec_verdict.get("market_fair_prob")
        else:
            # Informational fallback; a two-way market without a complete
            # same-bookmaker line cannot pass the validation gate below.
            prob = model_prob
            ev = prob * manual_odd - 1
            edge = prob - market_implied
            stake = kelly(prob, manual_odd, D["bank"], kelly_frac, cap=0.02)
            market_fair_probability = None

        with col_metrics.expander("Детали (по желанию)", expanded=False):
            st.write(f"Вероятность модели: {prob * 100:.1f}%")
            st.write(f"Вероятность из коэффициента до снятия маржи: {market_implied * 100:.1f}%")
            if market_fair_probability is not None:
                st.write(f"Вероятность рынка без маржи (один букмекер): {float(market_fair_probability) * 100:.1f}%")
            st.write(f"Преимущество модели: {edge * 100:+.1f}%")
            st.write(f"Ожидаемая доходность: {ev * 100:+.1f}%")
            if abs(model_prob - market_implied) >= 0.20:
                st.warning("Модель сильно расходится с котировкой. Проверьте матч, исход и точную линию.")
            if market_is_validated:
                st.caption(f"Расчётный размер виртуальной ставки: {stake:.2f} у.е.")

        context_eligible = bool(
            (c.get("context_quality") or {}).get("strong_value_eligible")
        )
        passed = (
            context_eligible
            and rec_verdict is not None
            and market_is_validated
            and prob >= max(float(min_prob), AUTO_PORTFOLIO_MIN_PROB)
            and edge >= min_edge
            and ev >= min_ev
            and stake > 0
        )
        if not context_eligible:
            col_btn.info(
                "Добавление в портфель отключено: требуется подтверждённый стартовый XI "
                "обеих команд и подтверждённый судья."
            )
        elif rec_verdict is None:
            col_btn.info(
                "Справочная котировка: исход не прошёл независимый отбор этого рынка. "
                "Добавление в портфель отключено."
            )
        elif not market_is_validated:
            col_btn.info("Автодобавление отключено: для этого рынка не пройдена историческая проверка или не подтверждена полная линия одного букмекера.")
        elif passed:
            existing = any(
                f"{b.get('match', '')}|{b.get('pick', '')}" == f"{c['match']}|{pick}"
                and b.get("status") == "pending"
                for b in (D.get("bets") or [])
            )
            if existing:
                col_btn.caption("Этот исход уже находится в портфеле.")
            else:
                D2 = dict(D)
                new_bet = {
                    "match": c["match"], "match_ru": c["match_ru"],
                    "div": c["div"], "league": c["league"],
                    "market": market_type(pick), "pick": pick,
                    "odds": manual_odd, "stake": stake,
                    "prob": prob, "model_prob": model_prob,
                    "market_implied_raw": market_implied,
                    "status": "pending", "strat": "scanner_auto_recommendation",
                    "odds_source": str(
                        (c.get("market_odds_source_by_pick") or {}).get(pick)
                        or (
                            c.get("total_odds_source") if selected_kind == "OU"
                            else c.get("btts_odds_source") if selected_kind == "BTTS"
                            else c.get("extra_market_source") if selected_kind == "STAT"
                            else c.get("odds_source")
                        )
                        or "unknown"
                    ),
                    "mode": "real",
                    "date": (parse_date(c.get("date_iso", "")) or datetime.now()).strftime("%d.%m.%Y"),
                    "date_iso": c.get("date_iso"),
                    "match_time": c.get("time", ""),
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "fixture_id": c.get("fixture_id"),
                    "sstats_game_id": c.get("sstats_game_id"),
                    "market_forecast_key": (
                        (c.get("market_forecast_keys") or {}).get(market_filter)
                        if market_filter else c.get("market_forecast_key")
                    ),
                    "probability_method": (
                        rec_verdict.get("probability_method")
                        if rec_verdict is not None
                        else "market_model_raw_price_informational"
                    ),
                    "score": None, "ev": ev,
                }
                D2["bets"] = list(D.get("bets") or []) + [new_bet]
                D2["bank"] = max(0.0, float(D.get("bank") or 0.0) - stake)
                st.session_state.data = D2
                if db.SQLITE_BOOT_OK:
                    db.insert_bets_batch([new_bet])
                    db.log_bank(D2["bank"], event="scanner_auto_recommendation")
                    db.invalidate_caches()
                usage.set_local_data(D2)
                st.toast("✅ Прогноз прошёл фильтры и добавлен в виртуальный портфель.", icon="✅")
                st.rerun()
        else:
            col_btn.caption("Автодобавление не сработало: не пройдены пороги вероятности, Edge, EV или Kelly.")
    elif source_odd is None:
        col_metrics.caption("Нет подтверждённого коэффициента для этого исхода.")

def render(min_prob, kelly_frac, matrix_n, min_edge=0.03):
    D = st.session_state.data
    # Modern scanner header and clear workflow: fixtures -> odds -> value -> portfolio.
    st.markdown(
        "<div class='scanner-hero'>"
        "<div class='scanner-kicker'>NEURO BET PRO · MATCH INTELLIGENCE</div>"
        "<div class='scanner-title'>Футбольный сканер</div>"
        "<div class='scanner-subtitle'>Матчи, рыночные коэффициенты и value-сигналы — в одном рабочем пространстве.</div>"
        "<div class='scanner-steps'><span>01 Матчи</span><i>→</i><span>02 Коэффициенты</span><i>→</i><span>03 Проверка value</span><i>→</i><span>04 Виртуальный портфель</span></div>"
        "</div>",
        unsafe_allow_html=True,
    )
    # Compact dashboard summary keeps the first screen useful before and after a scan.
    previous_cards = [c for c in (D.get("cards") or []) if isinstance(c, dict)]
    priced_count = sum(bool(c.get("market_odds")) for c in previous_cards)
    qualified_count = sum(
        bool(c.get("best") is not None
             and (c.get("verdict") or {}).get("real_odds")
             and (c.get("verdict") or {}).get("value_passed"))
        for c in previous_cards
    )
    pending_count = sum(1 for b in (D.get("bets") or []) if b.get("status") == "pending")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Матчи в последнем скане", len(previous_cards))
    k2.metric("С реальными кэфами", priced_count)
    k3.metric("Value-проходы", qualified_count)
    k4.metric("Виртуальный портфель", pending_count)

    st.markdown("<div class='scanner-section-label'>БЫСТРЫЙ ЗАПУСК</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.25, 1], vertical_alignment="bottom")
    days = c1.slider(
        "Горизонт матчей", 1, 14, 7, key="scanner_horizon_days_v4",
        help="Ищет матчи на ближайшие 7 дней по умолчанию; можно расширить до 14."
    )
    scan = c2.button("⚡ Запустить сканирование", type="primary",
                     use_container_width=True,
                     disabled=st.session_state.get("_scan_in_progress", False))

    with st.expander("📥 Импорт коэффициентов", expanded=False):
        st.markdown(
            "Вставь котировки, полученные вручную или из разрешённого API. "
            "Они сопоставляются с найденными матчами по командам и дате. "
            "Импорт не размещает ставки у букмекера."
        )
        odds_file = st.file_uploader(
            "Загрузить файл коэффициентов (.csv или .json)",
            type=["csv", "json"],
            key="scanner_odds_file",
            help="Принимает обычный CSV и Football-Data: Date, HomeTeam, AwayTeam, AvgH/AvgD/AvgA, B365H/B365D/B365A, PSH/PSD/PSA. Даты dd/mm/yy распознаются автоматически.",
        )
        download_odds = st.button(
            "🌐 Скачать и применить доступные коэффициенты",
            key="scanner_fetch_football_data",
            help="Загружает актуальный CSV Football-Data при нажатии. Покрытие ограничено доступными лигами и матчами источника.",
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Скачать шаблон CSV",
            data="date,home,away,П1,X,П2,source\\n,,,,,,\\n",
            file_name="neuro_bet_odds_template.csv",
            mime="text/csv",
            use_container_width=False,
        )
        odds_json = st.text_area(
            "Или вставь JSON",
            key="scanner_manual_odds_json",
            height=100,
            placeholder='[{"date":"2026-10-26","home":"Santa Clara","away":"Benfica","П1":9.0,"X":5.0,"П2":1.3,"ТБ 2.5":1.9,"ТМ 2.5":1.9,"BTTS Да":1.8,"BTTS Нет":2.0,"source":"manual"}]',
        )

        st.markdown("**Контекст матча · API-Football + исторический CSV**")
        st.caption("🔑 API-Football key берётся из боковой панели «API ключи»; отдельный повторный ввод здесь не требуется.")
        context_csv_file = st.file_uploader(
            "Исторический архив составов/судей (.csv)",
            type=["csv"],
            key="scanner_context_history_csv",
            help="Нужны fixture_id, home_team, away_team, kickoff_at, observed_at. observed_at должен быть строго раньше kickoff_at.",
        )
        st.download_button(
            "⬇️ Скачать шаблон контекстного CSV",
            data=(
                "fixture_id,home_team,away_team,kickoff_at,observed_at,is_confirmed,"
                "lineup_is_confirmed,referee_is_confirmed,injuries_is_confirmed,"
                "source_reliability,lineup_source_reliability,referee_source_reliability,"
                "injuries_source_reliability,data_source,lineup_strength_home,lineup_strength_away,"
                "key_players_missing_home,key_players_missing_away,referee_id,"
                "referee_yellow_mean,referee_yellow_variance,corners_home_trailing_mean,"
                "corners_away_trailing_mean,corners_home_level_mean,corners_away_level_mean,"
                "opponent_corner_style_home,opponent_corner_style_away,"
                "base_p_home,base_p_draw,base_p_away,result\\n"
            ),
            file_name="neuro_bet_context_history_template.csv",
            mime="text/csv",
        )
        if context_csv_file is not None:
            try:
                context_payload = parse_context_csv(context_csv_file.getvalue())
                st.session_state["_historical_context_rows"] = context_payload["rows"]
                st.success(
                    f"Контекстный архив: принято {context_payload['accepted']} строк, "
                    f"отклонено {len(context_payload['rejected'])} (post-kickoff данные исключены)."
                )
                if context_payload["rejected"]:
                    st.caption("Причины отклонения: " + "; ".join(
                        str(item.get("reason")) for item in context_payload["rejected"][:5]
                    ))
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(f"Не удалось прочитать контекстный CSV: {exc}")
        historical_match_file = st.file_uploader(
            "Архив исторических матчей для женских лиг / сборных (.csv)",
            type=["csv"],
            key="scanner_historical_match_csv",
            help=(
                "Обязательные колонки: Date, HomeTeam, AwayTeam, FTHG, FTAG, Div. "
                "Для женщин/сборных используйте отдельные коды, например W_E0, W_SP1, "
                "NT_EURO, NT_WC. Не загружайте odds/будущие результаты."
            ),
        )
        st.download_button(
            "⬇️ Скачать шаблон архива исторических матчей",
            data=(
                "Date,HomeTeam,AwayTeam,FTHG,FTAG,Div,League,HTHG,HTAG,HTHC,HTAC,"
                "HC,AC,HY,AY,HR,AR,Referee,HxG,AxG\\n"
                "2025-04-01,Home, Away,2,1,W_E0,Women's league,1,0,3,2,7,4,2,1,0,0,Referee Name,1.4,0.8\\n"
            ),
            file_name="neuro_bet_historical_matches_template.csv",
            mime="text/csv",
        )
        if historical_match_file is not None:
            try:
                history_payload = parse_historical_matches_csv(
                    historical_match_file.getvalue()
                )
                st.session_state["_custom_historical_match_rows"] = history_payload["rows"]
                st.success(
                    f"Исторический архив: принято {history_payload['accepted']} матчей "
                    f"из {len(history_payload['divisions'])} соревнований; "
                    f"отклонено {len(history_payload['rejected'])}."
                )
                st.caption(
                    "Коды: " + ", ".join(history_payload["divisions"])
                    + ". Эти соревнования обучают Base-модель отдельно и пока не проходят "
                    "Value-валидацию крупных лиг."
                )
            except (UnicodeDecodeError, ValueError) as exc:
                st.error(f"Не удалось прочитать исторический CSV: {exc}")
        custom_historical_rows = st.session_state.get("_custom_historical_match_rows") or []

        historical_context_rows = st.session_state.get("_historical_context_rows") or []
        if st.button(
            "🧪 Проверить Base vs Context (walk-forward ablation)",
            key="scanner_context_ablation",
            disabled=not historical_context_rows,
        ):
            report = walk_forward_context_ablation(
                historical_context_rows, min_train=200, min_validation=500
            )
            st.session_state["_context_ablation_report"] = report
            st.session_state["_context_blender"] = report.get("promoted_model") or {}
        context_report = st.session_state.get("_context_ablation_report") or {}
        if context_report:
            delta = context_report.get("delta_context_minus_base") or {}
            st.caption(
                f"Ablation на одинаковых матчах: n={context_report.get('n', 0)} · "
                f"Δ Brier={delta.get('brier')} · Δ Log Loss={delta.get('log_loss')}."
            )
            if (context_report.get("promoted_model") or {}).get("walk_forward_validated"):
                st.success("Контекстная модель прошла порог абляции и может использоваться при наличии всех live-признаков.")
            else:
                st.warning("Context не активирован: не хватает 500 парных матчей или нет улучшения одновременно по Brier и Log Loss.")
        try:
            parsed_quotes = []
            import_source_label = "JSON"
            if odds_file is not None or download_odds or st.session_state.get("_scanner_downloaded_odds_csv"):
                if download_odds:
                    @st.cache_data(ttl=4 * 24 * 60 * 60, show_spinner=False)
                    def _fetch_public_fixture_csv():
                        response = requests.get(
                            "https://www.football-data.co.uk/fixtures.csv",
                            timeout=(5, 20),
                            headers={"User-Agent": "Sport-AI-Pro13/1.0 (user-triggered fixture import)"},
                        )
                        response.raise_for_status()
                        if len(response.content) > 8_000_000:
                            raise ValueError("Файл источника неожиданно большой; импорт остановлен.")
                        text_body = response.content.decode("utf-8-sig", errors="replace")
                        # Never treat an HTML landing page / anti-bot response as a CSV.
                        sample = text_body.lstrip()[:500].casefold()
                        if sample.startswith("<!doctype html") or sample.startswith("<html") or "<head>" in sample:
                            raise ValueError(
                                "Источник вернул HTML-страницу вместо CSV. Проверьте доступность "
                                "https://www.football-data.co.uk/fixtures.csv и попробуйте позже."
                            )
                        if not text_body.strip():
                            raise ValueError("Источник вернул пустой файл.")
                        return text_body
                    try:
                        raw_file = _fetch_public_fixture_csv()
                        st.session_state["_scanner_downloaded_odds_csv"] = raw_file
                    except Exception as exc:
                        st.error(f"Не удалось скачать коэффициенты: {exc}")
                        raw_file = ""
                    import_source_label = "Football-Data (CSV, кэш до 4 дней)"
                elif odds_file is not None and odds_file.name.lower().endswith(".json"):
                    raw_file = odds_file.getvalue().decode("utf-8-sig", errors="replace")
                    parsed_quotes = json.loads(raw_file)
                    import_source_label = odds_file.name
                elif odds_file is not None:
                    raw_file = odds_file.getvalue().decode("utf-8-sig", errors="replace")
                    import_source_label = odds_file.name
                else:
                    raw_file = st.session_state.get("_scanner_downloaded_odds_csv", "")
                    import_source_label = "Football-Data (CSV, сохранён в сессии)"
                if odds_file is not None and odds_file.name.lower().endswith(".json"):
                    pass
                elif raw_file:
                    # Accept generic odds CSVs plus Football-Data.co.uk fixtures/results
                    # exports (Date/HomeTeam/AwayTeam/B365H/B365D/B365A, AvgH/AvgD/AvgA, PSH/PSD/PSA).
                    # Parse explicitly instead of relying on DictReader.fieldnames timing.
                    # Some feeds prepend blank lines or use semicolons despite a CSV extension.
                    csv_text = raw_file.lstrip("\ufeff")
                    nonempty_lines = [line for line in csv_text.splitlines() if line.strip()]
                    try:
                        dialect = csv.Sniffer().sniff("\n".join(nonempty_lines[:20]), delimiters=",;\t|")
                    except csv.Error:
                        dialect = csv.excel
                    csv_reader = csv.reader(nonempty_lines, dialect=dialect)
                    parsed_table = list(csv_reader)
                    csv_headers = [str(h).strip().lstrip("\ufeff") for h in parsed_table[0]] if parsed_table else []
                    csv_rows = []
                    if csv_headers:
                        for values in parsed_table[1:]:
                            # Ignore fully blank lines and preserve short/long rows safely.
                            if not values or not any(str(v).strip() for v in values):
                                continue
                            row = {header: values[i] if i < len(values) else "" for i, header in enumerate(csv_headers) if header}
                            if len(values) > len(csv_headers):
                                row["_extra_columns"] = values[len(csv_headers):]
                            csv_rows.append(row)
                    csv_rows_seen = len(csv_rows)
                    def _clean_date(value):
                        value = str(value or "").strip()
                        if not value:
                            return ""
                        value = value[:19].replace("Z", "")
                        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%d/%m/%Y", "%d/%m/%y", "%m/%d/%Y", "%m/%d/%y"):
                            try:
                                return datetime.strptime(value[:len(datetime.now().strftime(fmt))], fmt).strftime("%Y-%m-%d")
                            except ValueError:
                                continue
                        return value[:10] if len(value) >= 10 and value[4:5] == "-" else ""
                    for row in csv_rows:
                        lower = {str(k or "").strip().lstrip("\ufeff").casefold(): v for k, v in row.items()}
                        def _csv_value(*names):
                            for name in names:
                                value = lower.get(name.casefold())
                                if value is not None and str(value).strip():
                                    return str(value).strip()
                            return ""
                        # Prefer market-average prices; fall back to a named bookmaker.
                        home_price = _csv_value("П1", "1", "home_odds", "home_win", "odds_home", "odds1", "oddsh", "avgh", "b365h", "psh", "maxh", "bwh")
                        draw_price = _csv_value("X", "draw", "draw_odds", "odds_draw", "oddsx", "oddsd", "avgd", "b365d", "psd", "maxd", "bwd")
                        away_price = _csv_value("П2", "2", "away_odds", "away_win", "odds_away", "odds2", "oddsa", "avga", "b365a", "psa", "maxa", "bwa")
                        parsed_quotes.append({
                            "date": _clean_date(_csv_value("date", "match_date", "commence_time", "datetime", "kickoff", "matchdate")),
                            "home": _csv_value("home", "home_team", "hometeam", "host", "team1"),
                            "away": _csv_value("away", "away_team", "awayteam", "visitor", "team2"),
                            "П1": home_price,
                            "X": draw_price,
                            "П2": away_price,
                            "ТБ 0.5": _csv_value("ТБ 0.5", "over 0.5", "over0.5", "o0.5"),
                            "ТМ 0.5": _csv_value("ТМ 0.5", "under 0.5", "under0.5", "u0.5"),
                            "ТБ 1.5": _csv_value("ТБ 1.5", "over 1.5", "over1.5", "o1.5"),
                            "ТМ 1.5": _csv_value("ТМ 1.5", "under 1.5", "under1.5", "u1.5"),
                            "ТБ 2.5": _csv_value("ТБ 2.5", "over 2.5", "over2.5", "o2.5"),
                            "ТМ 2.5": _csv_value("ТМ 2.5", "under 2.5", "under2.5", "u2.5"),
                            "ТБ 3.5": _csv_value("ТБ 3.5", "over 3.5", "over3.5", "o3.5"),
                            "ТМ 3.5": _csv_value("ТМ 3.5", "under 3.5", "under3.5", "u3.5"),
                            "BTTS Да": _csv_value("BTTS Да", "btts yes", "both teams to score yes"),
                            "BTTS Нет": _csv_value("BTTS Нет", "btts no", "both teams to score no"),
                            "Угл ТБ 9.5": _csv_value("Угл ТБ 9.5", "corners over 9.5"),
                            "Угл ТМ 9.5": _csv_value("Угл ТМ 9.5", "corners under 9.5"),
                            "source": _csv_value("source", "bookmaker") or import_source_label,
                        })
                    import_source_label = import_source_label or (odds_file.name if odds_file is not None else "CSV import")
            elif odds_json.strip():
                parsed_quotes = json.loads(odds_json)
            else:
                parsed_quotes = []

            if isinstance(parsed_quotes, dict):
                parsed_quotes = [parsed_quotes]
            if not isinstance(parsed_quotes, list):
                raise ValueError("Нужен JSON-объект, список объектов или CSV-файл.")
            normalized_quotes = []
            for quote in parsed_quotes:
                if not isinstance(quote, dict):
                    continue
                quote_date = str(quote.get("date") or quote.get("commence_time") or "").strip()[:10]
                home_quote = str(quote.get("home") or quote.get("home_team") or "").strip()
                away_quote = str(quote.get("away") or quote.get("away_team") or "").strip()
                raw_prices = quote.get("odds") if isinstance(quote.get("odds"), dict) else quote
                aliases = {
                    "П1": ("П1", "1", "home_odds", "home_win", "odds_home"),
                    "X": ("X", "draw", "draw_odds", "odds_draw"),
                    "П2": ("П2", "2", "away_odds", "away_win", "odds_away"),
                    "ТБ 0.5": ("ТБ 0.5", "over_0_5", "over0.5"),
                    "ТМ 0.5": ("ТМ 0.5", "under_0_5", "under0.5"),
                    "ТБ 1.5": ("ТБ 1.5", "over_1_5", "over1.5"),
                    "ТМ 1.5": ("ТМ 1.5", "under_1_5", "under1.5"),
                    "ТБ 2.5": ("ТБ 2.5", "over_2_5", "over2.5"),
                    "ТМ 2.5": ("ТМ 2.5", "under_2_5", "under2.5"),
                    "ТБ 3.5": ("ТБ 3.5", "over_3_5", "over3.5"),
                    "ТМ 3.5": ("ТМ 3.5", "under_3_5", "under3.5"),
                    "BTTS Да": ("BTTS Да", "btts_yes", "btts yes"),
                    "BTTS Нет": ("BTTS Нет", "btts_no", "btts no"),
                    "Угл ТБ 9.5": ("Угл ТБ 9.5", "corners_over_9_5"),
                    "Угл ТМ 9.5": ("Угл ТМ 9.5", "corners_under_9_5"),
                }
                prices = {}
                for market, keys in aliases.items():
                    value = next((raw_prices[k] for k in keys if raw_prices.get(k) not in (None, "")), None)
                    try:
                        prices[market] = float(str(value).replace(",", ".")) if value is not None else 0.0
                    except (TypeError, ValueError):
                        prices[market] = 0.0
                try:
                    datetime.strptime(quote_date, "%Y-%m-%d")
                except ValueError:
                    continue
                valid_prices = {key: price for key, price in prices.items() if 1.01 < price <= 100}
                if not home_quote or not away_quote or not valid_prices:
                    continue
                normalized_quotes.append({
                    "date": quote_date, "home": home_quote, "away": away_quote,
                    "odds": valid_prices, "source": str(quote.get("source") or quote.get("bookmaker") or "manual_import"),
                })
            st.session_state["_scanner_manual_odds_quotes"] = normalized_quotes
            if normalized_quotes:
                st.success(f"Готово: {len(normalized_quotes)} набор(а) коэффициентов из {import_source_label}.")
                st.caption("Нажми «Запустить скан», чтобы сопоставить котировки с матчами.")
            elif odds_file is not None or odds_json.strip() or download_odds:
                st.warning("Источник не вернул подходящие коэффициенты или файл не содержит валидных записей для импорта.")
                if "csv_headers" in locals():
                    with st.expander("Диагностика импорта CSV", expanded=True):
                        st.write(f"Строк данных в CSV: {csv_rows_seen}; заголовков: {len(csv_headers)}.")
                        st.code(", ".join(csv_headers) if csv_headers else "Заголовки CSV не обнаружены")
                        st.caption("Начало ответа источника (для диагностики):")
                        st.code(str(raw_file[:500]) if raw_file else "Пустой ответ")
                        if csv_rows:
                            st.caption("Первая строка после разбора:")
                            st.write({str(k): str(v)[:120] for k, v in list(csv_rows[0].items())[:12]})
                        if csv_headers:
                            normalized_headers = {str(h).strip().casefold() for h in csv_headers}
                            odds_groups = {
                                "П1": {"п1", "1", "home_odds", "home_win", "odds_home", "odds1", "oddsh", "avgh", "b365h", "psh", "maxh", "bwh"},
                                "X": {"x", "draw", "draw_odds", "odds_draw", "oddsx", "oddsd", "avgd", "b365d", "psd", "maxd", "bwd"},
                                "П2": {"п2", "2", "away_odds", "away_win", "odds_away", "odds2", "oddsa", "avga", "b365a", "psa", "maxa", "bwa"},
                            }
                            missing_groups = [label for label, aliases in odds_groups.items() if not normalized_headers.intersection(aliases)]
                            if missing_groups:
                                st.info("В CSV не найдены колонки коэффициентов: " + ", ".join(missing_groups) + ". Возможно, источник отдал только список матчей без котировок.")
                        st.caption("Пришли скриншот этого блока — по заголовкам будет видно, что именно вернул источник.")
            elif "_scanner_manual_odds_quotes" not in st.session_state:
                st.session_state["_scanner_manual_odds_quotes"] = []
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            st.error(f"Импорт не прочитан: {exc}")

    with st.expander("⚙️ Настройки сканера", expanded=False):
        st.caption("Дополнительные параметры скрыты, чтобы не перегружать основной экран.")
        source_col, portfolio_col = st.columns(2)
        with portfolio_col:
            auto_add = st.checkbox(
                "Автодобавление в виртуальный портфель",
                value=False,
                key="scanner_auto_add_qualified_v4",
                help="Добавляются только прогнозы с реальными коэффициентами, прошедшие value-фильтры. Реальные ставки не размещаются.",
            )
    auto_add = st.session_state.get("scanner_auto_add_qualified_v4", False)

    if not scan:
        fn = D.get("funnel")
        if fn:
            st.success(
                f"🧠 Обучено {fn.get('trained', 0)} · "
                f"📡 Источников {fn.get('src', 0)} · "
                f"🎯 Найдено {fn.get('found', 0)} · "
                f"🤖 Автодобавлено {fn.get('added', 0)} · "
                f"💰 Заморожено {fn.get('frozen', 0):.0f} · "
                f"🤖 LLM: {fn.get('llm', 0)}"
            )
        with st.expander("🔌 Диагностика"):
            for line in D.get("report", []):
                st.text(line)

        all_cards = [c for c in D.get("cards", []) if isinstance(c, dict)]
        if not all_cards:
            st.info("Нажми ⚡ СКАН.")
            return

        qualified = _qualified_bet_cards(all_cards)
        try:
            from betting.ranking import rank_score
            qualified.sort(key=lambda c: (
                -rank_score(c),
                -float((c.get("verdict") or {}).get("ev") or -1),
                -float((c.get("verdict") or {}).get("prob") or 0),
            ))
        except Exception:
            qualified.sort(key=lambda c: -float(
                (c.get("verdict") or {}).get("ev") or -1))

        # Forecast favorites are ranked independently of bookmaker value gates.
        # They are displayed as model probabilities, never as guaranteed winners
        # or automatic betting recommendations.
        favorite_history_floor = 5
        favorite_candidates = rank_model_favorites(all_cards)
        favorite_matches = rank_model_favorites(
            all_cards, limit=10, min_sample_games=favorite_history_floor
        )
        low_history_favorites = sum(
            1 for item in favorite_candidates
            if int(item.get("sample_games") or 0) < favorite_history_floor
        )
        strong_favorites = sum(
            1 for item in favorite_matches
            if float(item.get("probability") or 0.0) >= STRONG_FORECAST_MIN_PROB
        )
        st.markdown(
            "<div style='display:flex;align-items:end;justify-content:space-between;gap:12px;margin:18px 0 4px'>"
            "<div><div class='scanner-kicker'>MODEL FAVORITES · 1X2</div>"
            "<h2 style='margin:0'>Самые вероятные исходы</h2>"
            "<div style='color:#91a4ad;font-size:.88rem;margin-top:5px'>"
            f"Топ-10 уникальных матчей с историей не менее {favorite_history_floor} матчей у каждой команды. "
            f"Сильных прогнозов ≥{STRONG_FORECAST_MIN_PROB:.0%}: {strong_favorites}. "
            "Сначала — самая высокая вероятность; при равенстве учитываются отрыв от второго исхода "
            "и объём истории. Это не гарантия победы.</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if low_history_favorites:
            st.caption(
                f"Не включено в список фаворитов из-за недостаточной истории команд: "
                f"{low_history_favorites}. Такие матчи остаются доступными в общем сканере."
            )
        if favorite_matches:
            favorite_columns = st.columns(2)
            for favorite_rank, favorite_item in enumerate(favorite_matches[:10], start=1):
                favorite_card = favorite_item["card"]
                favorite_match = str(
                    favorite_card.get("match_ru") or favorite_card.get("match") or "Матч"
                )
                favorite_league = str(favorite_card.get("league") or "Футбол")
                favorite_date_text = str(favorite_card.get("date") or "")
                favorite_time_text = str(favorite_card.get("time") or "").strip()
                favorite_kickoff = favorite_date_text
                if favorite_time_text and favorite_time_text not in favorite_date_text:
                    favorite_kickoff = " · ".join(
                        value for value in (favorite_date_text, favorite_time_text) if value
                    )
                favorite_probability = float(favorite_item["probability"])
                favorite_margin = float(favorite_item["probability_margin"])
                favorite_sample = int(favorite_item["sample_games"])
                favorite_strength_label = (
                    f"СИЛЬНЫЙ ПРОГНОЗ ≥{STRONG_FORECAST_MIN_PROB:.0%}"
                    if favorite_probability >= STRONG_FORECAST_MIN_PROB
                    else f"НИЖЕ ПОРОГА СИЛЬНОГО ПРОГНОЗА {STRONG_FORECAST_MIN_PROB:.0%}"
                )
                with favorite_columns[(favorite_rank - 1) % 2]:
                    st.markdown(
                        "<div style='padding:13px 15px;margin:7px 0;border:1px solid rgba(118,159,180,.28);"
                        "border-radius:12px;background:linear-gradient(120deg,rgba(22,42,54,.9),rgba(15,24,32,.94))'>"
                        f"<div style='display:flex;justify-content:space-between;gap:8px;color:#9eb1bd;font-size:11px'>"
                        f"<span>#{favorite_rank} · {html.escape(favorite_league)}</span>"
                        f"<span>{html.escape(favorite_item['strength'])}</span></div>"
                        f"<div style='font-weight:750;font-size:16px;color:#f0f5f7;margin:7px 0 4px'>"
                        f"{html.escape(favorite_match)}</div>"
                        f"<div style='color:#8fd8bd;font-weight:800;font-size:19px'>"
                        f"{html.escape(favorite_item['label'])} · {favorite_probability:.1%}</div>"
                        f"<div style='color:#b7c2c9;font-size:11px;margin-top:2px'>"
                        f"{html.escape(favorite_strength_label)}</div>"
                        f"<div style='color:#9aaab3;font-size:12px;margin-top:4px'>"
                        f"{html.escape(favorite_kickoff or 'Время начала уточняется')} · "
                        f"отрыв от второго исхода {favorite_margin:.1%} · история команд {favorite_sample} матчей"
                        f"</div><div style='color:#8fd8bd;font-size:11px;margin-top:3px'>"
                        f"Сортировка: вероятность → отрыв → история · это не гарантия"
                        f"</div></div>",
                        unsafe_allow_html=True,
                    )
        else:
            st.info(
                f"Нет матчей с достаточной историей команд (не менее {favorite_history_floor} "
                "завершённых матчей у каждой команды). Прогнозы без этой базы не выдаются за фаворитов."
            )

        st.markdown(
            "<div style='display:flex;align-items:end;justify-content:space-between;gap:12px;margin:18px 0 4px'>"
            "<div><div class='scanner-kicker'>VALUE SHORTLIST · BOOKMAKER CHECK</div>"
            "<h2 style='margin:0'>Подтверждённые Value-сигналы</h2>"
            "<div style='color:#91a4ad;font-size:.88rem;margin-top:5px'>"
            "Этот отдельный список намеренно строгий: полная линия одного букмекера, "
            "подтверждённый Edge и EV от 5%, положительный Kelly. Здесь может быть 0–2 матча, "
            "даже когда в расписании намного больше событий.</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if not qualified:
            st.markdown(
                "<div class='pick-story' style='animation:none'><div class='pick-choice'>Пока нет ставки, которую можно подтвердить</div>"
                "<div class='pick-ai'><b>Почему так?</b>Для рекомендации нужны совпавшие реальные коэффициенты и пройденные фильтры. "
                "Сканер не будет придумывать котировки или добавлять слабый вариант ради заполнения списка.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='pick-feed'>", unsafe_allow_html=True)
            for rank, card in enumerate(qualified[:5], start=1):
                verdict = card.get("verdict") or {}
                match_name = str(card.get("match_ru") or card.get("match") or "Матч")
                league_name = str(card.get("league") or "Футбол")
                best_result = card.get("best")
                # best_value_market: (market_type, pick, odd, EV, probability, Kelly stake).
                _market, best_pick, best_odd, best_ev, best_prob, best_stake = best_result[:6]
                pick_name = str(verdict.get("label") or best_pick)
                bookmaker = str((card.get("bookmaker_by_pick") or {}).get(best_pick) or "Букмекер не указан")
                date_text = str(card.get("date") or "")
                time_text = str(card.get("time") or "")
                kickoff = " · ".join(v for v in (date_text, time_text) if v)
                try:
                    edge_value = float(verdict.get("edge"))
                    edge_text = f"Edge +{edge_value:.1%}"
                except (TypeError, ValueError):
                    edge_text = "Edge —"
                try:
                    ev_text = f"EV +{float(best_ev):.1%}"
                except (TypeError, ValueError):
                    ev_text = "EV —"
                prob_text = f"Итоговая оценка {float(best_prob):.1%}"
                model_prob = verdict.get("model_prob")
                model_chip = (
                    f"<span class='pick-chip'>Модель {float(model_prob):.1%}</span>"
                    if model_prob is not None else ""
                )
                xg_chip = (
                    "<span class='pick-chip'>xG учтён</span>"
                    if card.get("xg_used") else
                    "<span class='pick-chip'>без xG</span>"
                )
                stake_text = f"Kelly {float(verdict.get('kelly_pct') or 0):.1%}"
                reason_items = verdict.get("reasons") or []
                reasons_html = (
                    "<ul style='margin:6px 0 0 18px;padding:0'>"
                    + "".join(f"<li>{html.escape(str(reason))}</li>" for reason in reason_items[:4])
                    + "</ul>"
                    if reason_items else ""
                )
                market_analysis = str(verdict.get("market_analysis") or "").strip()
                market_analysis_html = (
                    "<div class='pick-ai'><b>Анализ рынка</b>" + html.escape(market_analysis) + "</div>"
                    if market_analysis else ""
                )
                independent_note = str(card.get("llm_opinion") or "").strip()
                independent_html = (
                    "<div class='pick-ai'><b>Независимый анализ ИИ</b>"
                    + html.escape(independent_note) + "</div>"
                    if independent_note else
                    "<div class='pick-ai pick-ai-empty'><b>Независимый анализ</b>"
                    "Не получен для этого матча; решение основано на основной модели и проверенной линии букмекера.</div>"
                )
                st.markdown(
                    f"<article class='pick-story' style='animation-delay:{min(rank-1,4)*55}ms'>"
                    f"<div class='pick-story-top'><span class='pick-league'>{html.escape(league_name)}</span>"
                    f"<span class='pick-status'>ПРОШЛО ФИЛЬТРЫ</span></div>"
                    f"{render_fixture_header(card)}"
                    f"<div class='pick-match'>{html.escape(match_name)}</div>"
                    f"<div class='pick-time'>{html.escape(kickoff or 'Время начала уточняется')}</div>"
                    f"<div class='pick-divider'></div><div class='pick-choice-label'>КОНКРЕТНАЯ СТАВКА · {html.escape(str(_market))}</div>"
                    f"<div class='pick-choice'>{html.escape(pick_name)}</div>"
                    f"<div class='pick-meta'>"
                    f"<span class='pick-chip'>Коэффициент {float(best_odd):.2f}</span>"
                    f"<span class='pick-chip'>{html.escape(prob_text)}</span>"
                    f"<span class='pick-chip'>{html.escape(edge_text)}</span>"
                    f"<span class='pick-chip'>{html.escape(ev_text)}</span>"
                    f"<span class='pick-chip'>{html.escape(stake_text)}</span>"
                    f"{model_chip}{xg_chip}</div>"
                    f"<div class='pick-ai'><b>Линия букмекера</b>{html.escape(bookmaker)}"
                    f" · Источник: {html.escape(str(card.get('odds_source') or 'SStats'))}</div>"
                    f"<div class='pick-ai'><b>Почему прошла отбор</b>{reasons_html}"
                    f"Только полная линия одного букмекера; Edge и EV не ниже 5%; Kelly положительный.</div>"
                    f"{market_analysis_html}{independent_html}</article>",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 3, 2])
        sort_mode = c1.selectbox(
            "Сортировка",
            ["По сбалансированному рейтингу",
             "По EV (высокие сначала)",
             "По вероятности (высокие сначала)",
             "По дате (ближайшие сначала)"],
            index=0,
            key="scanner_sort_v5")
        display_mode = c2.selectbox(
            "Что показывать",
            ["⭐ Вероятные исходы (все матчи)",
             "🎯 Только подтверждённый Value-shortlist",
             "Все матчи с реальными котировками"],
            index=0,
            key="scanner_display_mode_v4",
        )
        probability_mode = display_mode.startswith("⭐")
        recommendation_mode = display_mode.startswith("🎯")
        group_by_day = c3.checkbox("Группировать по дням", True,
                                    key="scanner_group_by_day")

        # The combined shortlist is intentionally one best pick per fixture.
        # Dedicated market tabs must never inherit that 1X2/global-best filter:
        # each market is recalculated across the complete fixture pool.
        cards_view = qualified if recommendation_mode else all_cards
        market_pool = list(all_cards)
        if recommendation_mode:
            st.caption(
                "Показаны только исходы, прошедшие строгую проверку Value/EV/Edge/Kelly. "
                "Для просмотра остальных матчей переключитесь на прогнозный режим."
            )
        elif probability_mode:
            st.caption(
                "По умолчанию показываются все найденные матчи, отсортированные по вероятнейшему "
                "исходу 1X2. Отдельные вкладки также показывают весь пул событий; отсутствие цены "
                "или Value не скрывает прогноз и не превращает его в ставку."
            )
        else:
            st.caption(
                "Показаны матчи с доступными котировками. Для полного списка событий выберите "
                "«Вероятные исходы (все матчи)»."
            )

        if sort_mode.startswith("По дате"):
            cards_view.sort(key=lambda c: parse_card_date(c))
        elif sort_mode.startswith("По вероятности"):
            cards_view.sort(key=lambda c: (
                -float((model_favorite(c) or {}).get("probability") or 0.0),
                -float((model_favorite(c) or {}).get("probability_margin") or 0.0),
                -int((model_favorite(c) or {}).get("sample_games") or 0),
            ))
        elif sort_mode.startswith("По EV"):
            cards_view.sort(
                key=lambda c: -(c.get("verdict", {}).get("ev") or -1))
        else:
            try:
                from betting.ranking import rank_score
                cards_view.sort(key=lambda c: -rank_score(c))
            except Exception:
                cards_view.sort(key=lambda c: -float(
                    (c.get("verdict", {}).get("prob") or 0)))

        if not market_pool:
            st.warning("Матчи для отображения не найдены.")
            return
        if recommendation_mode and not cards_view:
            st.info(
                "Общий shortlist пуст, но отдельные рынки ниже всё равно проверяются "
                "по всем матчам — хорошие варианты на тоталы, BTTS, угловые или карточки "
                "не зависят от наличия сигнала 1X2."
            )

        quote_counts = {
            "winner": sum(bool(x.get("market_odds")) for x in market_pool),
            "goals": sum(bool(x.get("total_market_odds")) for x in market_pool),
            "btts": sum(bool(x.get("btts_market_odds")) for x in market_pool),
            "corners": sum(
                any(str(k).startswith("Угл ") for k in (x.get("stat_market_odds") or {}))
                for x in market_pool
            ),
            "cards": sum(
                any(str(k).startswith("Карты ") for k in (x.get("stat_market_odds") or {}))
                for x in market_pool
            ),
            "handicap": sum(bool(x.get("asian_handicap_market_odds")) for x in market_pool),
        }
        signal_counts = {
            family: sum(
                isinstance((x.get("best_by_market") or {}).get(family), dict)
                for x in market_pool
            )
            for family in ("winner", "goals", "btts", "corners", "cards")
        }
        ranked_forecast_pool = {
            family: rank_market_forecasts(market_pool, family)
            for family in ("winner", "goals", "btts", "corners", "cards")
        }
        forecast_counts = {
            family: len(items) for family, items in ranked_forecast_pool.items()
        }
        strong_forecast_counts = {
            family: sum(bool(item.get("strong_forecast")) for item in items)
            for family, items in ranked_forecast_pool.items()
        }
        market_tabs = st.tabs([
            f"Тоталы · прогнозы {forecast_counts['goals']} (сильных {strong_forecast_counts['goals']}) / Value {signal_counts['goals']} / цены {quote_counts['goals']}",
            f"{'Общий shortlist' if recommendation_mode else 'Все матчи'} ({len(cards_view)})",
            f"1X2 · прогнозы {forecast_counts['winner']} (сильных {strong_forecast_counts['winner']}) / Value {signal_counts['winner']} / цены {quote_counts['winner']}",
            f"BTTS · прогнозы {forecast_counts['btts']} (сильных {strong_forecast_counts['btts']}) / Value {signal_counts['btts']} / цены {quote_counts['btts']}",
            f"Угловые · прогнозы {forecast_counts['corners']} (сильных {strong_forecast_counts['corners']}) / Value {signal_counts['corners']} / цены {quote_counts['corners']}",
            f"Карточки · прогнозы {forecast_counts['cards']} (сильных {strong_forecast_counts['cards']}) / Value {signal_counts['cards']} / цены {quote_counts['cards']}",
            f"Форы · котировки {quote_counts['handicap']}",
        ])

        def _market_recommendation(card, family):
            return (card.get("best_by_market") or {}).get(family)

        def _global_market_recommendation(card):
            best = card.get("best")
            if not isinstance(best, (tuple, list)) or len(best) < 2:
                return None
            pick = str(best[1])
            kind = market_type(pick)
            family = (
                "winner" if kind == "1X2"
                else "goals" if kind == "OU"
                else "btts" if kind == "BTTS"
                else "corners" if kind == "STAT" and pick.startswith("Угл ")
                else "cards" if kind == "STAT" and pick.startswith("Карты ")
                else None
            )
            recommendation = _market_recommendation(card, family) if family else None
            candidate = (recommendation or {}).get("best")
            return (
                recommendation
                if isinstance(candidate, (tuple, list)) and len(candidate) >= 2
                and str(candidate[1]) == pick
                else None
            )

        def _market_summary_caption(card, family):
            summary = market_forecast_summary(card, family)
            if not summary:
                return None
            details = [
                f"Модельный прогноз: {summary['label']} — {summary['probability']:.1%}",
            ]
            if summary.get("reference_line") is not None:
                details.append(f"контрольная линия {summary['reference_line']:g}")
            if summary.get("expected_total") is not None:
                details.append(f"ожидаемый тотал {summary['expected_total']:.2f}")
            history_n = int(summary.get("history_matches") or 0)
            if family in {"corners", "cards"}:
                details.append(
                    f"история лиги {history_n}; домашняя/выездная выборка "
                    f"{summary.get('home_team_games', 0)}/{summary.get('away_team_games', 0)}"
                )
            elif summary.get("team_samples"):
                details.append(
                    f"минимальная выборка матчей команд: {summary['team_samples']}"
                )
            return " · ".join(details)

        def _render_ranked_market_card(card, family):
            st.markdown(render_fixture_header(card), unsafe_allow_html=True)
            _render_context_comparison(card)
            summary = market_forecast_summary(card, family)
            summary_text = _market_summary_caption(card, family)
            if summary is not None and summary_text:
                strength_threshold = float(
                    summary.get("strength_threshold") or STRONG_FORECAST_MIN_PROB
                )
                if not summary.get("history_sufficient", True):
                    strength_label = "НЕДОСТАТОЧНО ИСТОРИИ КОМАНД (нужно не менее 5 матчей)"
                elif (
                    summary.get("strong_forecast")
                    and bool((card.get("context_quality") or {}).get("strong_value_eligible"))
                ):
                    strength_label = f"СИЛЬНЫЙ ПРОГНОЗ ≥{strength_threshold:.0%}"
                elif summary.get("strong_forecast"):
                    strength_label = "Strong заблокирован: нет подтверждённого XI и/или судьи"
                else:
                    strength_label = f"ниже порога сильного прогноза {strength_threshold:.0%}"
                st.caption(
                    f"{strength_label} · {summary_text} · прогноз модели, не гарантия и не подтверждённая ставка"
                )
            else:
                st.caption(
                    "Для этого матча нет достаточной истории, чтобы рассчитать отдельный прогноз этого рынка."
                )
            _render_manual_odds_block(
                card, D, kelly_frac, min_prob, min_edge,
                market_filter=family,
                market_recommendation=_market_recommendation(card, family),
            )

        def _sort_market_cards(cards, family):
            def _key(card):
                recommendation = _market_recommendation(card, family)
                result = (recommendation or {}).get("verdict") or {}
                summary = market_forecast_summary(card, family) or {}
                market_probability = float(summary.get("probability") or 0.0)
                market_history = int(summary.get("history_matches") or 0)
                market_team_sample = int(summary.get("team_samples") or 0)
                if sort_mode.startswith("По дате"):
                    return (0 if recommendation else 1, parse_card_date(card))
                if sort_mode.startswith("По вероятности"):
                    # Every tab ranks with its own event probability; 1X2 is
                    # never used as a proxy for totals, BTTS, corners or cards.
                    return (
                        -market_probability,
                        -market_history,
                        -market_team_sample,
                    )
                if sort_mode.startswith("По EV"):
                    return (
                        0 if recommendation else 1,
                        -float(result.get("ev") or -1.0),
                        -market_probability,
                    )
                return (
                    0 if recommendation else 1,
                    -float((recommendation or {}).get("score") or -1.0),
                    -market_probability,
                    -market_history,
                    -market_team_sample,
                )
            return sorted(cards, key=_key)

        # Forecast lists are market-specific. They are not derived from the
        # global 1X2 shortlist and do not repeat every fixture on every tab.
        goal_cards = (
            [card for card in market_pool if _market_recommendation(card, "goals")]
            if recommendation_mode else
            [item["card"] for item in rank_market_forecasts(
                market_pool, "goals", limit=MARKET_FORECAST_DISPLAY_LIMIT,
                mode="balanced" if sort_mode.startswith("По сбалансированному") else "probability",
            )] if probability_mode else
            [card for card in market_pool if card.get("total_market_odds")]
        )
        goal_cards = _sort_market_cards(goal_cards, "goals")
        with market_tabs[0]:
            st.caption(
                "Тоталы ранжируются независимо по всем матчам. Проверка конкретной линии "
                "против исторической средней лиги оценивает прогноз модели, но сама по себе "
                "не доказывает преимущество над букмекером. Для портфеля отдельно проверяется "
                "только полный матч ТБ/ТМ 2.5 против букмекерского рынка."
            )
            if not goal_cards:
                st.info(
                    "В этом скане нет тоталов, прошедших собственную проверку, "
                    "согласованную линию одного букмекера и пороги Value/EV/Kelly."
                )
            elif quote_counts["goals"] == 0:
                st.info(
                    "Котировок на тоталы в источниках нет. Модельные оценки не являются "
                    "ставкой без реальной цены."
                )
            st.caption(
                f"Показаны отдельные прогнозы тотала, ранжированные по вероятности события "
                f"(не по рейтингу 1X2); максимум {MARKET_FORECAST_DISPLAY_LIMIT} матчей."
            )
            for card in goal_cards:
                _render_ranked_market_card(card, "goals")

        with market_tabs[1]:
            if not cards_view:
                st.info("В общем shortlist нет ставок, прошедших все пороги. Отдельные вкладки рынков проверяются независимо.")
            elif group_by_day:
                grouped = {}
                for card in cards_view:
                    grouped.setdefault(day_label(parse_card_date(card)), []).append(card)
                sorted_groups = sorted(grouped.items(), key=lambda kv: parse_card_date(kv[1][0]))
                for label, items in sorted_groups:
                    st.markdown(f"### {label} · {len(items)} матч.")
                    for card in items:
                        st.markdown(render_fixture_header(card), unsafe_allow_html=True)
                        _render_context_comparison(card)
                        _render_manual_odds_block(
                            card, D, kelly_frac, min_prob, min_edge,
                            market_filter=None,
                            market_recommendation=_global_market_recommendation(card),
                        )
            else:
                for card in cards_view:
                    st.markdown(render_fixture_header(card), unsafe_allow_html=True)
                    _render_context_comparison(card)
                    _render_manual_odds_block(
                            card, D, kelly_frac, min_prob, min_edge,
                            market_filter=None,
                            market_recommendation=_global_market_recommendation(card),
                        )

        market_specs = [
            (2, "winner", "market_odds", "1X2"),
            (3, "btts", "btts_market_odds", "BTTS"),
            (4, "corners", "stat_market_odds", "угловые"),
            (5, "cards", "stat_market_odds", "карточки"),
        ]
        for tab_index, market_filter, odds_field, market_label in market_specs:
            with market_tabs[tab_index]:
                if market_filter == "btts":
                    st.caption(
                        "BTTS оценивается отдельно от победы и тоталов. Исторический фильтр "
                        "сравнивает прогноз с предыдущей средней лиги; он не подтверждает "
                        "положительный EV против букмекерского рынка. Рынок остаётся прогнозным."
                    )
                elif market_filter in {"corners", "cards"}:
                    st.caption(
                        "Этот рынок рассчитывается и ранжируется независимо. Исторический "
                        "walk-forward сравнивает модель с предыдущей средней лиги, а не доказывает "
                        "прибыльность ставки против букмекера. Поэтому угловые и карточки не "
                        "добавляются в виртуальный портфель автоматически."
                    )
                def has_market_quote(card):
                    quotes = card.get(odds_field) or {}
                    if market_filter == "corners":
                        return any(str(key).startswith("Угл ") for key in quotes)
                    if market_filter == "cards":
                        return any(str(key).startswith("Карты ") for key in quotes)
                    return bool(quotes)

                if recommendation_mode:
                    selected_cards = [
                        card for card in market_pool
                        if _market_recommendation(card, market_filter)
                    ]
                elif probability_mode:
                    ranked_market = rank_market_forecasts(
                        market_pool, market_filter,
                        limit=MARKET_FORECAST_DISPLAY_LIMIT,
                        mode="balanced" if sort_mode.startswith("По сбалансированному") else "probability",
                    )
                    selected_cards = [item["card"] for item in ranked_market]
                    # If a statistical market has quotes but lacks enough team
                    # history for a forecast, retain a small diagnostic sample.
                    if not selected_cards and market_filter in {"corners", "cards"}:
                        selected_cards = [
                            card for card in market_pool if has_market_quote(card)
                        ][:5]
                else:
                    selected_cards = [
                        card for card in market_pool if has_market_quote(card)
                    ]
                selected_cards = _sort_market_cards(selected_cards, market_filter)
                if not selected_cards:
                    if recommendation_mode:
                        st.info(
                            f"Нет {market_label}-сигналов, прошедших собственные критерии "
                            "вероятности, Value, исторической проверки и полного рынка одного букмекера."
                        )
                    elif probability_mode:
                        st.info(
                            f"Для найденных матчей нет рассчитанного прогноза или цены рынка {market_label}. "
                            "Матчи с отсутствующей котировкой не удаляются из остальных разделов."
                        )
                    else:
                        st.info(
                            f"Нет подтверждённых котировок на {market_label} для найденных матчей. "
                            "Коэффициенты 1X2 не подставляются вместо отсутствующего рынка."
                        )
                if probability_mode and selected_cards:
                    st.caption(
                        f"Показаны независимые прогнозы рынка «{market_label}». "
                        f"Сильный прогноз — от {STRONG_FORECAST_MIN_PROB:.0%}. "
                        f"В сбалансированной сортировке 60% веса даёт вероятность; EV и Edge учитываются "
                        f"только у уже прошедших строгий Value-отбор. Максимум "
                        f"{MARKET_FORECAST_DISPLAY_LIMIT} матчей; отсутствие Value не скрывает прогноз."
                    )
                for card in selected_cards:
                    _render_ranked_market_card(card, market_filter)

        with market_tabs[6]:
            handicap_cards = [card for card in market_pool if card.get("asian_handicap_market_odds")]
            if not handicap_cards:
                st.info("SStats не вернул подтверждённых котировок на форы для показанных матчей.")
            for card in handicap_cards:
                st.markdown(render_fixture_header(card), unsafe_allow_html=True)
                _render_context_comparison(card)
                quotes = card.get("asian_handicap_market_odds") or {}
                st.caption("Котировки SStats: " + " · ".join(
                    f"{pick}: **{float(odd):.2f}**" for pick, odd in sorted(quotes.items())
                ))
                for forecast in card.get("asian_handicap_forecasts") or []:
                    probabilities = forecast["probabilities"]
                    st.write(
                        f"**{forecast['pick']} @ {forecast['odds']:.2f}** · "
                        f"победа {probabilities['win']:.1%} · "
                        f"полпобеды {probabilities['half_win']:.1%} · "
                        f"возврат {probabilities['push']:.1%} · "
                        f"полупоражение {probabilities['half_loss']:.1%} · "
                        f"поражение {probabilities['loss']:.1%} · "
                        f"расчётный EV {forecast['expected_value']:+.1%}"
                    )
                st.caption(
                    "EV азиатской форы рассчитан по полной матрице счетов, включая push и split-ставки. "
                    "Пока нет отдельного out-of-sample benchmark на одинаковой выборке, рынок остаётся "
                    "информационным и не может попасть в Value-портфель."
                )

        st.caption(
            "✅ Общий shortlist: " + str(len(cards_view))
            + " матчей · отдельные рынки: весь пул из " + str(len(market_pool)) + " матчей"
        )
        return

    # ==================== СКАН ====================
    st.session_state["_scan_in_progress"] = True
    try:
        loader_ph = st.empty()
        log_ph = st.empty()

        def update_loader(text, pct, logs=None):
            loader_ph.markdown(
                f"<div class='nbr-loader'><div class='nbr-ring'></div>"
                f"<div class='nbr-text'><b>{text}</b>"
                f"<div class='nbr-bar'><div class='nbr-bar-fill' "
                f"style='width:{pct*100:.0f}%'></div></div></div></div>",
                unsafe_allow_html=True)
            if logs:
                log_ph.markdown(
                    "<div style='color:#8b93a7;font-size:.8rem;"
                    "background:rgba(10,14,24,.6);padding:10px;"
                    "border-radius:10px;max-height:180px;overflow-y:auto'>"
                    + "<br>".join(logs[-12:]) + "</div>",
                    unsafe_allow_html=True)

        scan_timezone = app_timezone()
        scan_now = datetime.now(scan_timezone)
        today_date = scan_now.date()
        logs = []

        token = _runtime_secret("fdorg_token_input", "FDORG_TOKEN", "fdorg_token")
        sstats_key = _runtime_secret("sstats_api_key_input", "SSTATS_API_KEY", "sstats_api_key")
        footballapi_key = _runtime_secret(
            "footballapi_key_input", "FOOTBALLAPI_KEY", "footballapi_key"
        )
        api_football_key = _runtime_secret(
            "api_football_key_input", "API_FOOTBALL_KEY", "api_football_key"
        )
        fixture_sources = []
        tsdb_rows = []
        espn_rows = []
        fdorg_rows = []
        sstats_rows = []
        source_counts = {"TheSportsDB": 0, "ESPN": 0, "football-data.org": 0, "SStats": 0}
        try:
            tsdb_rows = tsdb_matches(days, logs)
            source_counts["TheSportsDB"] = len(tsdb_rows)
        except Exception as exc:
            logs.append(f"⚠️ TheSportsDB fixtures: {type(exc).__name__}")
        try:
            espn_rows = espn_matches(days, logs)
            source_counts["ESPN"] = len(espn_rows)
        except Exception as exc:
            logs.append(f"⚠️ ESPN fixtures: {type(exc).__name__}")
        if token:
            try:
                fdorg_rows = fdorg_matches(days, token, logs)
                source_counts["football-data.org"] = len(fdorg_rows)
            except Exception as exc:
                logs.append(f"⚠️ football-data.org fixtures: {type(exc).__name__}")
        if sstats_key:
            try:
                sstats_rows = sstats_upcoming_matches(days, sstats_key, logs)
                source_counts["SStats"] = len(sstats_rows)
            except Exception as exc:
                logs.append(f"⚠️ SStats fixtures/odds: {type(exc).__name__}")
        else:
            logs.append("ℹ️ SSTATS_API_KEY не настроен — бесплатный источник SStats пропущен.")

        # Merge from highest-quality competition metadata to broad schedule feeds.
        # SStats remains last for odds enrichment, while fixture_merge can now
        # upgrade generic Div/League placeholders when it sees a specific league.
        fixture_sources.extend(fdorg_rows)
        fixture_sources.extend(tsdb_rows)
        fixture_sources.extend(espn_rows)
        fixture_sources.extend(sstats_rows)
        logs.append(
            "📡 Источники расписания: " + " · ".join(
                f"{name} {count}" for name, count in source_counts.items()
            )
        )
        try:
            rows = _safe_filter(merge_fixture_sources(fixture_sources))
        except Exception:
            # Keep scanner usable even if cross-provider deduplication fails.
            rows = _safe_filter(fixture_sources)

        # Explicitly merge SStats quotes by canonical fixture key. Some fixture
        # providers populate a partial/estimated odds dictionary; the SStats
        # quote should still be attached to the same scheduled match.
        if sstats_rows:
            sstats_by_key = {
                fixture_key(source_row): source_row
                for source_row in sstats_rows
                if isinstance(source_row, dict) and all(fixture_key(source_row))
            }
            merged_sstats = 0
            for row in rows:
                source_row = sstats_by_key.get(fixture_key(row))
                if not source_row:
                    continue
                for group in (
                    "market_odds", "total_market_odds", "btts_market_odds",
                    "stat_market_odds", "asian_handicap_market_odds",
                ):
                    target = row.setdefault(group, {})
                    incoming = source_row.get(group) or {}
                    if not isinstance(target, dict) or not isinstance(incoming, dict):
                        continue
                    for market_name, raw_price in incoming.items():
                        try:
                            price = float(raw_price)
                            if 1.01 < price <= 100:
                                target[market_name] = max(
                                    float(target.get(market_name) or 0), price
                                )
                        except (TypeError, ValueError):
                            continue
                row["sstats_game_id"] = source_row.get("sstats_game_id")
                row.setdefault("bookmaker_by_pick", {}).update(
                    source_row.get("bookmaker_by_pick") or {}
                )
                if row.get("market_odds") and row.get("odds_source") in (None, "", "no_market"):
                    row["odds_source"] = "sstats"
                if row.get("total_market_odds") and row.get("total_odds_source") in (None, "", "no_market"):
                    row["total_odds_source"] = "sstats"
                if any(row.get(group) for group in (
                    "total_market_odds", "btts_market_odds", "stat_market_odds",
                    "asian_handicap_market_odds",
                )) and not row.get("extra_market_source"):
                    row["extra_market_source"] = "sstats"
                merged_sstats += 1
            if merged_sstats:
                logs.append(f"🔗 SStats: коэффициенты сопоставлены с {merged_sstats} матчами из общего расписания.")

        logs.append(
            f"🔀 Объединение расписаний: получено {len(fixture_sources)} строк, "
            f"после дедупликации {len(rows)} уникальных матчей."
        )
        logs.append(f"📡 Матчи для анализа из футбольных источников: {len(rows)}")
        if not rows:
            logs.append("⚠️ Матчи не найдены. Проверьте доступность источников расписания.")

        # Normalize fixture dates before the API provider performs exact matching.
        for fixture_row in rows:
            # Match the provider's local kickoff date, not a source feed's
            # original UTC/calendar date (which can differ around midnight).
            local_kickoff = fixture_kickoff_local(
                fixture_row,
                timezone_name=getattr(scan_timezone, "key", "UTC"),
            )
            parsed_fixture_date = (
                local_kickoff
                if local_kickoff is not None
                else parse_date(fixture_row.get("Date") or fixture_row.get("date"))
            )
            if parsed_fixture_date is not None:
                fixture_row["date_iso"] = parsed_fixture_date.strftime("%Y-%m-%d")

        context_by_fixture = {}
        context_store = None
        if api_football_key:
            try:
                context_store = ContextStore(DB_FILE)
            except Exception as exc:
                logs.append(f"⚠️ Context DB недоступна ({type(exc).__name__}); live-скан продолжается.")
        if api_football_key:
            try:
                lineup_call_cap = _runtime_int(
                    "api_football_max_lineup_calls",
                    "API_FOOTBALL_MAX_LINEUP_CALLS",
                    default=12,
                )
                api_daily_limit = _runtime_int(
                    "api_football_daily_limit",
                    "API_FOOTBALL_DAILY_LIMIT",
                    default=100,
                    minimum=1,
                )
                api_daily_reserve = _runtime_int(
                    "api_football_daily_reserve",
                    "API_FOOTBALL_DAILY_RESERVE",
                    default=10,
                )
                api_client = ApiFootballClient(
                    api_football_key, cache_dir=DISK_CACHE_DIR,
                    daily_limit=api_daily_limit,
                    reserve=api_daily_reserve,
                )
                context_provider = ApiFootballContextProvider(
                    api_client,
                    timezone_name=getattr(scan_timezone, "key", "UTC"),
                    max_lineup_calls=lineup_call_cap,
                    lineup_window_hours=6,
                )
                context_by_fixture, context_logs = context_provider.enrich(
                    rows,
                    date_from=today_date.isoformat(),
                    date_to=(today_date + timedelta(days=days - 1)).isoformat(),
                    now=scan_now,
                )
                logs.extend(context_logs)
            except Exception as exc:
                logs.append(f"⚠️ API-Football context: {type(exc).__name__}; Base-прогнозы сохраняются.")
        else:
            logs.append(
                "🔒 API-Football не настроен: Base-прогнозы будут показаны, "
                "но Strong/Value заблокированы до подтверждения стартового XI и судьи."
            )

        update_loader("📈 Загрузка xG-истории...", 0.16, logs)
        footballapi_xg_rows = footballapi_xg_history(footballapi_key, logs)

        update_loader("🧠 Загрузка модели...", 0.20, logs)
        engine = _load_engine(
            matrix_n, logs, update_loader,
            footballapi_key=footballapi_key,
            footballapi_xg_rows=footballapi_xg_rows,
            custom_historical_rows=custom_historical_rows,
        )

        update_loader("🧠 Анализ матчей...", 0.80, logs)
        cards = []
        matches_with_best = 0
        xg_used_count = 0
        highlightly_key = _runtime_secret("highlightly_api_key_input", "HIGHLIGHTLY_API_KEY", "highlightly_api_key")
        # The compact SStats fixture feed provides 1X2 and main totals in one request.
        # Full bookmaker markets are requested only for a limited shortlist to respect
        # the provider's free API quota.
        sstats_detail_calls = 0
        # Bound per-fixture requests to avoid exhausting the provider's API quota.
        # Compact odds from Games/list remain available for every fixture.
        SSTATS_DETAIL_MAX_PER_SCAN = 100

        funnel_counts = {"model": 0, "market_req": 0, "market_got": 0,
                          "bet": 0, "watch": 0, "skip": 0,
                          "model_only": 0, "estimated": 0, "real": 0}
        if not highlightly_key:
            logs.append(
                "ℹ️ SStats — основной источник котировок. Если нужного рынка нет в ответе API, "
                "можно загрузить собственные коэффициенты через CSV/JSON."
            )

        date_filter_counts = {
            "invalid_date": 0, "outside_horizon": 0, "past_kickoff": 0,
            "unknown_same_day_kickoff": 0, "missing_teams": 0,
            "inside_horizon": 0,
        }
        for match_idx, r in enumerate(rows, 1):
            if match_idx == 1 or match_idx % 5 == 0:
                update_loader(
                    f"🧠 Анализ матчей... {match_idx}/{len(rows)}",
                    min(0.88, 0.80 + 0.08 * match_idx / max(1, len(rows))),
                    logs,
                )

            # A calendar-date check is not enough: after kickoff, the same-day
            # fixture is no longer a pre-match forecasting opportunity.
            kickoff_local = fixture_kickoff_local(
                r, timezone_name=os.environ.get("APP_TIMEZONE")
            )
            if kickoff_local is not None:
                if kickoff_local <= scan_now:
                    date_filter_counts["past_kickoff"] += 1
                    continue
                d = kickoff_local.replace(
                    hour=0, minute=0, second=0, microsecond=0, tzinfo=None
                )
                kickoff_time_text = kickoff_local.strftime("%H:%M")
            else:
                d = parse_date(r.get("Date", ""))
                if not d:
                    date_filter_counts["invalid_date"] += 1
                    continue
                # Do not invent a same-day kickoff time. Without either an
                # absolute timestamp or a clock time, we cannot know whether
                # the match has already started.
                if d.date() == today_date:
                    date_filter_counts["unknown_same_day_kickoff"] += 1
                    continue
                kickoff_time_text = str(r.get("Time") or "").strip()

            if not (today_date <= d.date() < today_date + timedelta(days=days)):
                date_filter_counts["outside_horizon"] += 1
                continue
            h_en = (r.get("HomeTeam") or "").strip()
            a_en = (r.get("AwayTeam") or "").strip()
            if not h_en or not a_en:
                date_filter_counts["missing_teams"] += 1
                continue
            date_filter_counts["inside_horizon"] += 1
            h_ru = translate_team(h_en)
            a_ru = translate_team(a_en)
            lg = r.get("Div") or "G"

            funnel_counts["model"] += 1

            try:
                P = engine.predict(h_en, a_en, lg, match_date=d,
                                   cup=lg in ("C1", "EL", "EC"))
            except Exception:
                funnel_counts["skip"] += 1
                continue
            if P.get("xg_used"):
                xg_used_count += 1

            fh = engine.form_str(h_en)
            fa = engine.form_str(a_en)
            verdict, rows_, _ = build_verdict(
                P, min_prob, D["bank"], kelly_frac,
                h_ru, a_ru, fh, fa, P.get("h2h_n", 0))

            api_context = context_by_fixture.get(
                fixture_row_key(h_en, a_en, d.strftime("%Y-%m-%d")), {}
            )
            if context_store is not None and api_context:
                try:
                    stable_context_key = fixture_row_key(
                        h_en, a_en, d.strftime("%Y-%m-%d")
                    )
                    context_store.record_snapshot(
                        fixture_id="|".join(stable_context_key),
                        kickoff_at=(
                            kickoff_local.isoformat()
                            if kickoff_local else str(api_context.get("kickoff_at") or "")
                        ),
                        observed_at=str(api_context.get("timestamp") or ""),
                        context=api_context,
                        data_source="api-football",
                    )
                except (TypeError, ValueError, OSError) as exc:
                    logs.append(f"⚠️ Context snapshot не сохранён: {type(exc).__name__}")
            context_quality = assess_context_quality(api_context)
            context_forecasts = context_probabilities(
                {"H": P["p1"], "D": P["x"], "A": P["p2"]},
                api_context,
                model=st.session_state.get("_context_blender"),
            )
            verdict["context_quality"] = context_quality
            if not context_quality["strong_value_eligible"] and verdict.get("is_action"):
                verdict["is_action"] = False
                verdict["strong_blocked"] = True
                verdict["confidence"] = (
                    "низкая" if context_quality["level"] == "Low" else "средняя"
                )
                verdict["conf_color"] = (
                    "#f87171" if context_quality["level"] == "Low" else "#fbbf24"
                )
                verdict.setdefault("reasons", []).append(
                    "Strong/Value заблокированы: " + context_quality["reason"]
                )

            best = None
            market_odds = dict(r.get("market_odds") or {})
            odds_source = str(r.get("odds_source") or ("sstats" if market_odds else "no_market"))
            bookmaker_by_pick = dict(r.get("bookmaker_by_pick") or {})
            if market_odds and odds_source in {"", "no_market", "model_only"}:
                odds_source = "sstats"
            total_market_odds = dict(r.get("total_market_odds") or {})
            total_market_book_count = dict(r.get("total_market_book_count") or {})
            total_market_confirmed = dict(r.get("total_market_confirmed") or {})
            btts_market_odds = dict(r.get("btts_market_odds") or {})
            stat_market_odds = dict(r.get("stat_market_odds") or {})
            asian_handicap_market_odds = dict(r.get("asian_handicap_market_odds") or {})
            total_odds_source = str(
                r.get("total_odds_source") or ("sstats" if total_market_odds else "no_market")
            )
            btts_odds_source = str(
                r.get("btts_odds_source") or ("sstats" if btts_market_odds else "no_market")
            )
            extra_market_source = str(r.get("extra_market_source") or "")
            # Keep secondary-market quotes from verified providers and imported CSV/JSON.
            # Only accept prices explicitly attached to this fixture and market/line.
            api_market_data = None

            # Fetch real 1X2 odds for every displayed fixture, not only when
            # the model already crosses the probability threshold. The strict
            # value gates below decide whether a bet is eligible; odds display
            # and source diagnostics must not depend on model confidence.
            if rows_ is not None:
                try:
                    candidates = []

                    # Detailed SStats bookmaker markets for a bounded shortlist of fixtures.
                    # Do not gate API lookup on model probability: that previously meant
                    # zero real prices were fetched whenever every team probability was < 58%.
                    # The model/value filters decide whether a bet is eligible only AFTER
                    # verified market odds have been fetched.
                    sstats_game_id = r.get("sstats_game_id")
                    if (
                        sstats_key and sstats_game_id not in (None, "")
                        and sstats_detail_calls < SSTATS_DETAIL_MAX_PER_SCAN
                    ):
                        # Keep the funnel's market-request counter in sync with
                        # the detailed SStats lookup attempts shown in diagnostics.
                        funnel_counts["market_req"] += 1
                        detailed = sstats_odds_fixture(sstats_game_id, sstats_key, logs)
                        sstats_detail_calls += 1
                        if isinstance(detailed, dict):
                            detailed_1x2 = {
                                key: detailed[key]
                                for key in ("П1", "X", "П2")
                                if detailed.get(key) is not None
                            }
                            # Full odds are bookmaker-specific; prefer the best
                            # available decimal price over the compact close price,
                            # and keep the bookmaker name paired with that best price.
                            detailed_bookmakers = detailed.get("bookmaker_by_pick") or {}
                            for key, value in detailed_1x2.items():
                                current = float(market_odds.get(key) or 0)
                                if float(value) >= current:
                                    market_odds[key] = float(value)
                                    if detailed_bookmakers.get(key):
                                        bookmaker_by_pick[key] = detailed_bookmakers[key]
                            if detailed_1x2:
                                odds_source = "sstats"
                                # SStats detailed quotes are the selected primary line.
                                # Do not mix in a different provider's line below.
                                candidates = []
                            for group in (
                                "total_market_odds", "btts_market_odds",
                                "stat_market_odds", "asian_handicap_market_odds",
                            ):
                                target = {
                                    "total_market_odds": total_market_odds,
                                    "btts_market_odds": btts_market_odds,
                                    "stat_market_odds": stat_market_odds,
                                    "asian_handicap_market_odds": asian_handicap_market_odds,
                                }[group]
                                verified_quotes = detailed.get(group) or {}

                                # Detailed SStats quotes have explicit
                                # bookmaker provenance. Replace each complete
                                # market map rather than mixing another provider's
                                # stale prices into this line.
                                if group in {
                                    "total_market_odds", "btts_market_odds", "stat_market_odds"
                                } and verified_quotes:
                                    # Remove stale bookmaker provenance alongside stale prices.
                                    # Otherwise devigging can attribute a new provider's quote
                                    # to a bookmaker attached to the replaced quote.
                                    for old_selection in list(target):
                                        bookmaker_by_pick.pop(old_selection, None)
                                    target.clear()
                                    if group == "total_market_odds":
                                        total_market_book_count = dict(
                                            detailed.get("total_market_book_count") or {}
                                        )
                                        total_market_confirmed = dict(
                                            detailed.get("total_market_confirmed") or {}
                                        )
                                    for selection, raw_value in verified_quotes.items():
                                        try:
                                            value = float(raw_value)
                                            if 1.01 < value <= 8.0:
                                                target[selection] = value
                                                if detailed_bookmakers.get(selection):
                                                    bookmaker_by_pick[selection] = detailed_bookmakers[selection]
                                        except (TypeError, ValueError):
                                            continue
                                    continue

                                for selection, raw_value in verified_quotes.items():
                                    try:
                                        value = float(raw_value)
                                        current = float(target.get(selection) or 0)
                                        if value >= current:
                                            target[selection] = value
                                            if detailed_bookmakers.get(selection):
                                                bookmaker_by_pick[selection] = detailed_bookmakers[selection]
                                    except (TypeError, ValueError):
                                        continue
                            if detailed.get("total_market_odds"):
                                total_odds_source = "sstats"
                            if detailed.get("btts_market_odds"):
                                btts_odds_source = "sstats"
                            if detailed.get("stat_market_odds"):
                                extra_market_source = "sstats"

                    # Imported CSV/JSON quote: match by normalized teams and date.
                    if not all(key in market_odds for key in ("П1", "X", "П2")):
                        def _team_key(value):
                            # Ignore punctuation, accents and common club prefixes.
                            import unicodedata
                            raw = unicodedata.normalize("NFKD", str(value or ""))
                            raw = "".join(ch for ch in raw if not unicodedata.combining(ch))
                            key = "".join(ch.lower() for ch in raw if ch.isalnum())
                            for prefix in ("clubde", "cd", "fc", "sc", "cf", "sporting"):
                                if key.startswith(prefix) and len(key) > len(prefix) + 3:
                                    key = key[len(prefix):]
                            return key

                        fixture_home_key, fixture_away_key = _team_key(h_en), _team_key(a_en)
                        fixture_date = d.strftime("%Y-%m-%d")
                        for quote in st.session_state.get("_scanner_manual_odds_quotes", []):
                            quote_home_key = _team_key(quote.get("home"))
                            quote_away_key = _team_key(quote.get("away"))
                            # Exact normalized match first; allow a clear containment
                            # match for harmless prefixes such as "CD Santa Clara".
                            home_match = (
                                fixture_home_key == quote_home_key
                                or (min(len(fixture_home_key), len(quote_home_key)) >= 5
                                    and (fixture_home_key in quote_home_key or quote_home_key in fixture_home_key))
                            )
                            away_match = (
                                fixture_away_key == quote_away_key
                                or (min(len(fixture_away_key), len(quote_away_key)) >= 5
                                    and (fixture_away_key in quote_away_key or quote_away_key in fixture_away_key))
                            )
                            quote_date = str(quote.get("date") or "")
                            try:
                                date_delta = abs((datetime.strptime(quote_date, "%Y-%m-%d").date()
                                                  - d.date()).days)
                            except ValueError:
                                date_delta = 999
                            if date_delta <= 1 and home_match and away_match:
                                imported = dict(quote.get("odds") or {})
                                valid_imported = {}
                                for key, raw_value in imported.items():
                                    try:
                                        value = float(raw_value)
                                        if 1.01 < value <= 100:
                                            valid_imported[str(key)] = value
                                    except (TypeError, ValueError):
                                        continue
                                if valid_imported:
                                    # Imported prices do not inherit bookmaker identity
                                    # from an older SStats/Highlightly quote on the same
                                    # pick. Without an explicit book in the import, the
                                    # two-way line must stay informational.
                                    for imported_pick in valid_imported:
                                        bookmaker_by_pick.pop(imported_pick, None)
                                    if all(key in valid_imported for key in ("П1", "X", "П2")):
                                        market_odds = {key: valid_imported[key] for key in ("П1", "X", "П2")}
                                        odds_source = str(quote.get("source") or "manual_import")
                                    imported_totals = {
                                        k: v for k, v in valid_imported.items()
                                        if k.startswith(("ТБ ", "ТМ "))
                                    }
                                    imported_btts = {
                                        k: v for k, v in valid_imported.items()
                                        if k.startswith("BTTS ")
                                    }
                                    imported_stats = {
                                        k: v for k, v in valid_imported.items()
                                        if k.startswith(("Угл ", "Карты "))
                                    }
                                    if imported_totals:
                                        for old_selection in list(total_market_odds):
                                            bookmaker_by_pick.pop(old_selection, None)
                                        total_market_odds = imported_totals
                                        total_market_book_count = {}
                                        total_market_confirmed = {}
                                        total_odds_source = "manual_import"
                                    if imported_btts:
                                        for old_selection in list(btts_market_odds):
                                            bookmaker_by_pick.pop(old_selection, None)
                                        btts_market_odds = imported_btts
                                        btts_odds_source = "manual_import"
                                    if imported_stats:
                                        for old_selection in list(stat_market_odds):
                                            bookmaker_by_pick.pop(old_selection, None)
                                        stat_market_odds = imported_stats
                                        extra_market_source = "manual_import"
                                    logs.append(
                                        f"📥 Импорт коэффициентов ({quote.get('source') or 'manual_import'}, дата ±{date_delta} дн.): "
                                        f"{h_en} — {a_en} | рынки: {', '.join(valid_imported.keys())}"
                                    )
                                    break

                    # If no provider has a verified line, leave odds empty.

                    for source_name, candidate in candidates:
                        if not isinstance(candidate, dict):
                            continue
                        valid = {}
                        for key in ("П1", "X", "П2"):
                            try:
                                value = float(candidate.get(key))
                                if 1.01 < value <= 100:
                                    valid[key] = value
                            except (TypeError, ValueError):
                                continue
                        if all(key in valid for key in ("П1", "X", "П2")):
                            # Keep the best available decimal quote across providers
                            # instead of replacing a better SStats price with a lower one.
                            improved = False
                            for key, value in valid.items():
                                current = float(market_odds.get(key) or 0)
                                if value > current:
                                    market_odds[key] = value
                                    bookmaker_by_pick.pop(key, None)
                                    improved = True
                            if improved or not odds_source:
                                odds_source = source_name

                    if market_odds:
                        funnel_counts["market_got"] += 1
                        logs.append(
                            f"🏁 Реальные кэфы ({odds_source}): {h_ru} — {a_ru} | "
                            f"П1 {market_odds['П1']:.2f} / X {market_odds['X']:.2f} / "
                            f"П2 {market_odds['П2']:.2f}"
                        )
                except Exception as exc:
                    logs.append(f"⚠️ Ошибка получения линии: {type(exc).__name__}")

            # Build one cross-market candidate set. Secondary markets are
            # admitted only when their exact line passed a leakage-safe historical test.
            referee_name_for_stats = str(
                (api_context.get("referee") or {}).get("name") or ""
            ).strip()
            historical_stats = (
                engine.match_stats.predict(
                    h_en, a_en, lg, referee_id=referee_name_for_stats
                )
                if hasattr(engine, "match_stats") else {}
            )
            asian_handicap_forecasts = []
            for handicap_pick, raw_price in asian_handicap_market_odds.items():
                match_handicap = re.fullmatch(
                    r"Ф([12])\s+([+-]?\d+(?:[.,]\d+)?)",
                    str(handicap_pick).strip(), re.I,
                )
                if not match_handicap:
                    continue
                try:
                    handicap_price = float(raw_price)
                    handicap_line = float(match_handicap.group(2).replace(",", "."))
                    if not 1.01 < handicap_price <= 100:
                        continue
                    handicap_probs = asian_handicap_probabilities(
                        P["M"], handicap_line,
                        side="home" if match_handicap.group(1) == "1" else "away",
                    )
                    asian_handicap_forecasts.append({
                        "pick": handicap_pick,
                        "line": handicap_line,
                        "odds": handicap_price,
                        "probabilities": handicap_probs,
                        "expected_value": asian_handicap_expected_value(
                            handicap_probs, handicap_price
                        ),
                    })
                except (TypeError, ValueError, OverflowError):
                    continue
            selection_odds = {}
            selection_odds.update(market_odds)
            selection_odds.update(total_market_odds)
            selection_odds.update(btts_market_odds)
            selection_odds.update(stat_market_odds)
            market_source_by_pick = {
                **{pick: odds_source for pick in market_odds},
                **{pick: total_odds_source for pick in total_market_odds},
                **{pick: btts_odds_source for pick in btts_market_odds},
                **{pick: extra_market_source for pick in stat_market_odds},
            }
            # Value flags are stricter than forecast flags. A full historical
            # bookmaker benchmark must pass before any market enters the selector.
            winner_market_validation = bool(
                getattr(engine, "winner_market_validation", False)
            )
            market_validation_by_pick = {
                pick: winner_market_validation for pick in ("П1", "X", "П2")
            }
            extra_rows = []

            # Only full-time O/U 2.5 has a paired out-of-sample comparison
            # against de-vig bookmaker prices. Other totals remain forecasts.
            from model.historical_match_stats import poisson_over_probability
            total_mean = float(P["lams"][0] or 0.0) + float(P["lams"][1] or 0.0)
            total_validation = getattr(engine, "total_market_validation_by_line", {}) or {}
            for selection in total_market_odds:
                match = re.fullmatch(r"Т([БМ])\s+(\d+(?:[.,]\d+)?)", str(selection), re.I)
                if not match:
                    continue
                line = float(match.group(2).replace(",", "."))
                # The binary probability/EV calculation has no push handling.
                # Keep integer and quarter lines informational, never actionable.
                if not is_half_line(line):
                    continue
                line_text = f"{line:g}"
                if (
                    abs(line - 2.5) > 1e-9
                    or not bool(getattr(engine, "total_market_validation", False))
                    or not bool(total_validation.get(line_text))
                ):
                    continue
                # SStats totals need both a coherent same-book pair and quotes
                # corroborated by at least two independent bookmakers.
                if (
                    str(total_odds_source or "").casefold() == "sstats"
                    and not bool(total_market_confirmed.get(selection))
                ):
                    continue
                over_probability = poisson_over_probability(total_mean, line)
                if over_probability is None:
                    continue
                is_over = match.group(1).upper() == "Б"
                probability = over_probability if is_over else 1.0 - over_probability
                extra_rows.append({
                    "pick": selection,
                    "label": f"Тотал голов {'больше' if is_over else 'меньше'} {line_text}",
                    "prob": probability,
                    "analysis": (
                        f"Модель голов: ожидаемый тотал {total_mean:.2f}; "
                        f"хронологический тест линии {line_text} пройден."
                    ),
                })
                market_validation_by_pick[selection] = True

            # BTTS, corners and cards still receive independent model forecasts
            # in their own tabs. Their current walk-forward reports compare with
            # a league-rate baseline, not an untouched bookmaker benchmark, so
            # they must not enter the Value/EV selector yet.

            candidate_rows = list(rows_ or []) + extra_rows
            # Evaluate each family independently before computing the one-best
            # cross-market pick. A strong 1X2 result must not suppress a goals,
            # BTTS, corners or cards signal on this fixture.
            best_by_market = best_value_markets(
                verdict, candidate_rows, selection_odds, D["bank"], kelly_frac,
                min_prob, min_edge=max(min_edge, 0.05), min_ev=0.05,
                bookmaker_by_pick=bookmaker_by_pick,
                market_validation_by_pick=market_validation_by_pick,
                context_qualified=context_quality["strong_value_eligible"],
            )
            best_odds_source = "no_market"
            if selection_odds and candidate_rows:
                chosen_verdict, best = best_value_market(
                    verdict,
                    candidate_rows,
                    selection_odds,
                    D["bank"],
                    kelly_frac,
                    min_prob,
                    min_edge=max(min_edge, 0.05),
                    min_ev=0.05,
                    bookmaker_by_pick=bookmaker_by_pick,
                    market_validation_by_pick=market_validation_by_pick,
                    context_qualified=context_quality["strong_value_eligible"],
                )
                if best is not None:
                    verdict = chosen_verdict
                    best_odds_source = market_source_by_pick.get(best[1]) or (
                        odds_source if best[0] == "1X2" else "unknown"
                    )
                    funnel_counts["real"] += 1
                else:
                    funnel_counts["model_only"] += 1
            else:
                best = None
                funnel_counts["model_only"] += 1

            # A rejected market remains visible as a forecast/quote, never as a bet.
            if best is None:
                verdict["real_odds"] = False
                verdict["odd"] = None
                verdict["ev"] = None
                verdict["edge"] = None
                verdict["stake"] = 0.0
                verdict["is_bet"] = False
                if not selection_odds:
                    odds_source = "no_market"

            if best:
                matches_with_best += 1
                funnel_counts["bet"] += 1
                logs.append(
                    f"✅ MARKET SIGNAL: {h_ru} — {a_ru} | {best[1]} @ {best[2]:.2f} "
                    f"({best[0]} · {best_odds_source})"
                )
            else:
                no_best_reason = (
                    "ни один доступный исход не прошёл проверку модели для конкретной линии, "
                    "полную линию одного букмекера, вероятность, Edge >= 5%, "
                    "EV >= 5% и положительный Kelly"
                )
                logs.append(
                    f"❌ NO BEST: {h_ru} — {a_ru} | "
                    f"is_action={verdict.get('is_action')} | reason={no_best_reason}"
                )

            cards.append({
                "div": lg,
                "league": r.get("League") or DIV_NAMES.get(lg, "Лига"),
                "match": f"{h_en} vs {a_en}",
                "match_ru": f"{h_ru} — {a_ru}",
                "date": d.strftime("%d.%m") +
                        (f" {kickoff_time_text}" if kickoff_time_text else ""),
                "verdict": verdict, "best": best,
                "best_by_market": best_by_market,
                "games": P["games"], "fh": fh, "fa": fa,
                "fixture_id": r.get("fixture_id"),
                "sstats_game_id": r.get("sstats_game_id"),
                "historical_stats": historical_stats,
                "winner_market_validation": bool(getattr(engine, "winner_market_validation", False)),
                "winner_market_validation_report": getattr(engine, "winner_market_validation_report", {}),
                "total_market_validation": bool(getattr(engine, "total_market_validation", False)),
                "total_market_validation_by_line": getattr(engine, "total_market_validation_by_line", {}),
                "btts_market_validation": bool(getattr(engine, "btts_market_validation", False)),
                "btts_market_validation_report": getattr(engine, "btts_market_validation_report", {}),
                "stat_market_validation": getattr(engine, "stat_market_validation", {}),
                "stat_market_validation_report": getattr(engine, "stat_market_validation_report", {}),
                "best_odds_source": best_odds_source,
                "market_odds_source_by_pick": dict(market_source_by_pick),
                "date_iso": d.strftime("%Y-%m-%d"),
                "time": kickoff_time_text,
                "kickoff_utc": r.get("kickoff_utc") or "",
                "lam_h": P["lams"][0], "lam_a": P["lams"][1],
                # LLM receives observed venue-specific goal averages, not the primary model's lambdas.
                "independent_features": {
                    **historical_features(engine, h_en, a_en),
                    "markets": historical_stats,
                },
                "xg_used": bool(P.get("xg_used")),
                "total_market_validation_report": getattr(engine, "total_market_validation_report", {}),
                "p1": P["p1"], "px": P["x"], "p2": P["p2"],
                "over": P["over"], "btts": P["btts"],
                "odds_source": odds_source,
                "market_odds": dict(market_odds),
                "bookmaker_by_pick": dict(bookmaker_by_pick),
                "total_market_odds": dict(total_market_odds),
                "total_market_book_count": dict(total_market_book_count),
                "total_market_confirmed": dict(total_market_confirmed),
                "total_odds_source": total_odds_source if total_market_odds else (str(r.get("extra_market_source") or "") if r.get("total_market_odds") else total_odds_source),
                "btts_odds_source": btts_odds_source if btts_market_odds else "no_market",
                "extra_market_source": extra_market_source,
                "btts_market_odds": dict(btts_market_odds),
                "stat_market_odds": dict(stat_market_odds),
                "asian_handicap_market_odds": dict(asian_handicap_market_odds),
                "asian_handicap_forecasts": asian_handicap_forecasts,
                "api_football_fixture_id": api_context.get("api_fixture_id"),
                "api_context": api_context,
                "context_quality": context_quality,
                "context_forecasts": context_forecasts,
                "home_badge": r.get("home_badge") or "",
                "away_badge": r.get("away_badge") or "",
                "league_badge": r.get("league_badge") or "",
                "venue": r.get("venue") or r.get("stadium") or "",
                "stadium_image": r.get("stadium_image") or r.get("venue_image") or "",
                "context": {},
            })

        if sstats_key:
            sstats_id_count = sum(
                1 for row in rows
                if isinstance(row, dict) and row.get("sstats_game_id") not in (None, "")
            )
            sstats_compact_odds_count = sum(
                1 for row in rows
                if isinstance(row, dict) and row.get("sstats_game_id") not in (None, "")
                and row.get("market_odds")
            )
            logs.append(
                f"📚 SStats: матчей с game ID — {sstats_id_count}; "
                f"с компактными 1X2 — {sstats_compact_odds_count}; "
                f"детальных запросов Odds — {sstats_detail_calls} "
                f"(лимит {SSTATS_DETAIL_MAX_PER_SCAN} на скан)."
            )
            if sstats_id_count and not sstats_compact_odds_count and not sstats_detail_calls:
                logs.append(
                    "⚠️ SStats: ID матчей есть, но коэффициенты не загружены. "
                    "Проверьте ответ /Odds/{gameId} и журнал HTTP-статусов."
                )
        logs.append(
            f"🗓️ Фильтр горизонта {days} дн.: в окне {date_filter_counts['inside_horizon']}, "
            f"вне окна {date_filter_counts['outside_horizon']}, "
            f"уже начались/завершились {date_filter_counts['past_kickoff']}, "
            f"сегодня без времени начала {date_filter_counts['unknown_same_day_kickoff']}, "
            f"дата не распознана {date_filter_counts['invalid_date']}, "
            f"без названий команд {date_filter_counts['missing_teams']}."
        )
        logs.append(
            f"🧮 Итог скана: источники {len(fixture_sources)} строк → "
            f"{len(rows)} уникальных → {date_filter_counts['inside_horizon']} в окне → "
            f"{len(cards)} карточек после модели."
        )
        logs.append(f"📈 xG: использован в {xg_used_count} из {funnel_counts['model']} прогнозов")
        logs.append(f"🎯 BET: {matches_with_best}")

        # ============ LLM ============
        llm_key = _runtime_secret("llm_api_key_input", "LLM_API_KEY", "llm_api_key")
        llm_prov = D.get("meta", {}).get("llm_provider",
                                          "Groq (бесплатно, быстро)")
        llm_model = D.get("meta", {}).get("llm_model", "")

        # Review fixtures with the most historical observations first, not only
        # fixtures already selected by the value model. The LLM still receives no
        # model probabilities, pick, confidence, EV, or bookmaker prices.
        eligible_llm_cards = [
            c for c in cards
            if isinstance(c.get("verdict"), dict)
            and c.get("lam_h") is not None and c.get("lam_a") is not None
        ]
        action_cards = [
            c for c in eligible_llm_cards
            if has_sufficient_history(c.get("independent_features"))
        ]
        skipped_low_history = len(eligible_llm_cards) - len(action_cards)
        # De-duplicate before spending the small independent-LLM quota. Keep
        # the strongest supported forecast row for every unique fixture.
        action_cards = [
            item["card"] for item in rank_model_favorites(action_cards)
        ]
        # Give Groq's small TPM budget to the strongest model favourites first.
        # The actual probabilities and favourite are still excluded from the LLM prompt.
        def _llm_priority(card):
            favorite = model_favorite(card) or {}
            probability = float(favorite.get("probability") or 0.0)
            margin = float(favorite.get("probability_margin") or 0.0)
            sample_size = int(favorite.get("sample_games") or 0)
            stable_key = hashlib.sha256(
                f"{card.get('date_iso','')}|{card.get('match','')}".encode("utf-8")
            ).hexdigest()
            return (-probability, -margin, -sample_size, stable_key)

        action_cards.sort(key=_llm_priority)
        llm_done = 0
        # Groq GPT-OSS in the deployed account reported an 8,000 TPM cap.
        # Three independent fixture reviews per scan avoid flooding that shared
        # budget; stop immediately on the first 429 instead of retrying every card.
        llm_batch_limit = min(3, LLM_TOP_N) if "groq" in str(llm_prov).casefold() else LLM_TOP_N

        logs.append(
            f"🤖 LLM подготовка: key={'есть' if llm_key else 'НЕТ'} | "
            f"provider={llm_prov} | "
            f"analysis_cards={len(action_cards)} | "
            f"batch_limit={llm_batch_limit} | "
            f"skipped_low_history={skipped_low_history} | "
            f"remaining={usage.llm_remaining()}")

        if llm_key and usage.llm_remaining() > 0 and action_cards:
            for i_, card in enumerate(action_cards[:llm_batch_limit]):
                if usage.llm_remaining() <= 0:
                    logs.append("🤖 LLM: лимит")
                    break
                if i_ > 0:
                    time.sleep(2)
                update_loader(
                    f"🤖 Независимый ИИ-анализ [{i_+1}/{min(llm_batch_limit, len(action_cards))}]",
                    0.90 + 0.08 * (i_+1) / max(1, llm_batch_limit), logs)
                v_ = card.get("verdict", {})
                # Blind AI context: the statistical model's probabilities, selected outcome,
                # confidence, EV and market prices are deliberately not passed to the LLM.
                # Model outcome is used only after the LLM responds, for a separate comparison.
                ctx_llm = {
                    "api_key": llm_key, "provider": llm_prov, "model": llm_model,
                    "home": card.get("match_ru", "").split(" — ")[0],
                    "away": card.get("match_ru", "").split(" — ")[-1],
                    "league": card.get("league", ""),
                    "date": f"{card.get('date_iso') or card.get('date', '')} {card.get('time') or ''}".strip(),
                    "independent_features": card.get("independent_features") or {},
                    "fh": card.get("fh", "—"), "fa": card.get("fa", "—"),
                    # Main-model label is derived locally after the blind LLM response.
                    "model_pick_code": max(
                        (("П1", float(card.get("p1") or 0)),
                         ("X", float(card.get("px") or 0)),
                         ("П2", float(card.get("p2") or 0))),
                        key=lambda item: item[1],
                    )[0],
                }
                try:
                    opinion = analyze_match(ctx_llm)
                except LLMRateLimitError as exc:
                    retry_after = exc.retry_after_seconds
                    logs.append(
                        "⛔ Groq ограничил запросы по токенам в минуту. "
                        "Остальные LLM-вызовы остановлены, чтобы не усугублять лимит. "
                        + (
                            f"Повторить новый скан примерно через {retry_after:.0f} сек."
                            if retry_after is not None
                            else "Попробуйте новый скан после сброса лимита провайдера."
                        )
                    )
                    break
                except Exception as exc:
                    opinion = None
                    logs.append(
                        f"🤖 LLM ошибка [{i_+1}]: {type(exc).__name__}: {str(exc)[:160]}"
                    )
                if opinion:
                    card["llm_opinion"] = opinion
                    llm_done += 1
                    logs.append(f"🤖 OK [{i_+1}] multimarket: {opinion[:100]}")
                else:
                    logs.append(f"🤖 FAIL [{i_+1}]: ответ не получен")
            logs.append(f"🤖 LLM итог: {llm_done} мнений")
        elif not llm_key:
            logs.append("🤖 LLM: ключ не задан")
        elif not action_cards:
            logs.append(
                "🤖 LLM: независимые вероятности не рассчитаны — "
                f"недостаточно фактической истории (пропущено {skipped_low_history} матчей; "
                "нужно минимум 5 домашних матчей хозяев и 5 выездных матчей гостей)."
            )

        # Persist both forecasters before the fixture starts. The ledger keeps the
        # first prediction immutable, making repeated scans idempotent and auditable.
        try:
            tracked = record_scan_forecasts(
                cards, provider=llm_prov, model=llm_model, db_path=DB_FILE,
            )
            logs.append(
                "📚 Журнал прогнозов: "
                f"модель +{tracked['main_recorded']}, ИИ +{tracked['ai_recorded']}, "
                f"рынки +{tracked['market_recorded']}, "
                f"без времени старта пропущено {tracked['skipped_no_time']}, "
                f"после старта пропущено {tracked['skipped_started']}, "
                f"ошибок {tracked['invalid']}"
            )
        except Exception as exc:
            logs.append(f"⚠️ Журнал прогнозов ИИ недоступен: {type(exc).__name__}: {str(exc)[:120]}")

        # Автодобавление — только в виртуальный портфель, не в аккаунт букмекера.
        # Учитываем лучший по value исход матча, прошедший все фильтры своего рынка.
        new_bets = []
        total_stake = 0.0
        available_bank = float(D.get("bank") or 0.0)
        existing_keys = {
            (str(b.get("match") or ""), str(b.get("pick") or ""))
            for b in (D.get("bets") or [])
            if b.get("status") == "pending"
        }
        if auto_add:
            for card in cards:
                portfolio_choice = _portfolio_auto_recommendation(card)
                if portfolio_choice is None:
                    continue
                verdict_card = portfolio_choice["verdict"]
                best_card = portfolio_choice["best"]
                pick = str(best_card[1])
                market_code = str(best_card[0]).upper()
                market_family = portfolio_choice["family"]
                match_key = (str(card.get("match") or ""), pick)
                if match_key in existing_keys:
                    continue
                try:
                    odds_value = float(best_card[2])
                    probability = float(verdict_card.get("prob") or 0.0)
                    ev_value = float(verdict_card.get("ev") or 0.0)
                    stake_value = kelly(
                        probability, odds_value, available_bank,
                        kelly_frac, cap=0.02,
                    )
                except (TypeError, ValueError, IndexError):
                    continue
                # High confidence is a hard portfolio gate, in addition to value/EV.
                if probability < max(float(min_prob), AUTO_PORTFOLIO_MIN_PROB):
                    continue
                if stake_value <= 0 or stake_value > available_bank:
                    continue
                new_bet = {
                    "match": card.get("match", ""),
                    "match_ru": card.get("match_ru", card.get("match", "")),
                    "div": card.get("div", ""),
                    "league": card.get("league", ""),
                    "market": market_code,
                    "pick": pick,
                    "odds": odds_value,
                    "stake": stake_value,
                    "prob": probability,
                    "model_prob": verdict_card.get("model_prob"),
                    "market_implied_raw": verdict_card.get("market_implied_raw"),
                    "probability_method": verdict_card.get("probability_method", "market_blend_0.4_model_delta"),
                    "status": "pending",
                    "strat": "scanner_auto_" + market_code.casefold(),
                    "odds_source": (
                        (card.get("market_odds_source_by_pick") or {}).get(pick)
                        or card.get("best_odds_source")
                        or card.get("odds_source", "unknown")
                    ),
                    "mode": "real",
                    "date": (parse_date(card.get("date_iso", "")) or datetime.now()).strftime("%d.%m.%Y"),
                    "date_iso": card.get("date_iso"),
                    "match_time": card.get("time", ""),
                    "kickoff_utc": card.get("kickoff_utc") or None,
                    "date_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "fixture_id": card.get("fixture_id"),
                    "sstats_game_id": card.get("sstats_game_id"),
                    "market_forecast_key": (
                        (card.get("market_forecast_keys") or {}).get(market_family)
                        or (
                            card.get("market_forecast_key")
                            if card.get("best") and str(card["best"][1]) == pick
                            else None
                        )
                    ),
                    "score": None,
                    "ev": ev_value,
                }
                new_bets.append(new_bet)
                existing_keys.add(match_key)
                available_bank -= stake_value
                total_stake += stake_value
                logs.append(f"🤖 AUTO-PORTFOLIO: {card.get('match_ru', card.get('match', ''))} | {pick} @ {odds_value:.2f} | stake {stake_value:.2f}")

        # Explain why the portfolio stayed empty instead of silently showing zero additions.
        if auto_add and not new_bets:
            pending_count = len(existing_keys)
            priced_count = sum(1 for card in cards if card.get("market_odds"))
            qualified_count = sum(
                1 for card in cards
                if card.get("best")
                and (card.get("verdict") or {}).get("real_odds")
                and (card.get("verdict") or {}).get("value_passed")
            )
            logs.append(
                "ℹ️ AUTO-PORTFOLIO: новых ставок 0; "
                f"матчей с реальными котировками 1X2={priced_count}/{len(cards)}, "
                f"прошедших value-фильтры={qualified_count}, "
                f"текущих ожидающих ставок в ключах={pending_count}. "
                "Если котировки есть, но value=0, ни один исход не прошёл пороги "
                "вероятности/edge/EV/Kelly; ставки без value не добавляются."
            )
        elif not auto_add:
            logs.append("ℹ️ AUTO-PORTFOLIO выключен переключателем; ставки автоматически не добавлялись.")

        # All supported markets share the same single-best-pick selection above.
        # Separate OU/BTTS loops are intentionally removed to prevent multiple
        # correlated bets from one fixture and to keep one best market per match.

        D2 = dict(D)
        D2["cards"] = cards
        D2["report"] = logs
        D2["bets"] = list(D.get("bets") or []) + new_bets
        D2["bank"] = max(0.0, float(D.get("bank") or 0.0) - total_stake)
        D2["funnel"] = {
            "trained": getattr(engine, "trained_n", 0),
            "src": len(rows), "found": matches_with_best,
            "added": len(new_bets), "frozen": total_stake, "llm": llm_done,
        }
        st.session_state.data = D2
        usage.set_local_data(D2)
        if db.SQLITE_BOOT_OK:
            db.insert_bets_batch(new_bets)
            db.log_bank(D2["bank"], event="scan")
            db.invalidate_caches()

        # Surface source diagnostics in deployment logs, not only the in-app report.
        for source_line in logs:
            if "Highlightly" in source_line or "матчей после фильтрации" in source_line:
                _print_log(f"SOURCE: {source_line}")

        _print_log(f"FUNNEL: {funnel_counts}")

        update_loader(
            f"✅ Готово! автодобавлено {len(new_bets)} ставок · 🤖 {llm_done} LLM",
            1.0, logs)
        time.sleep(1.2)
        loader_ph.empty()
        log_ph.empty()
    except Exception as exc:
        message = f"Сканирование прервано: {type(exc).__name__}: {str(exc)[:240]}"
        _print_log(message)
        try:
            D2 = dict(st.session_state.get("data") or D)
            report = list(D2.get("report") or [])
            report.append(message)
            D2["report"] = report[-100:]
            st.session_state.data = D2
            usage.set_local_data(D2)
        except Exception:
            pass
        st.error(message)
        st.info("Проверь раздел «Диагностика» и последние строки журнала.")
        return
    finally:
        st.session_state["_scan_in_progress"] = False
    st.rerun()
