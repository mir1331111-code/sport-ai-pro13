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
import copy
import threading
import uuid
import requests
from datetime import datetime, timedelta

import streamlit as st

from config import (
    APP_VERSION, APP_BUILD_ID, DEFAULT_SCAN_HORIZON_DAYS,
    DIV_NAMES, DIV_TO_ODDS, DIV_TO_TSDB, DB_FILE,
    STRONG_FORECAST_MIN_PROB, MIN_RECOMMENDED_MATCHES,
    MAX_RECOMMENDED_MATCHES, MIN_FAVORITE_HISTORY, llm_provider_key_config,
)
from storage import usage
from storage import sqlite_store as db
from storage import user_datasets
from model.scan_ledger import ScanLedger
from data.sources import (load_seasonal, season_str, fdorg_matches, fdorg_match_result,
                          parse_date, tsdb_matches, espn_matches,
                          sstats_upcoming_matches, sstats_historical_matches_for_fixtures,
                          sstats_context_enrich, sstats_odds_fixture,
                          footballapi_xg_history, team_badge_url_by_id)
from data.fixture_merge import (
    enforce_sstats_market_card, enforce_sstats_odds_only,
    enrich_fixture_visual_metadata, is_sstats_market_source, merge_fixture_sources,
)
from data.team_names import normalize_team
from data.fixture_time import app_timezone, fixture_kickoff_local
from data.context_data import (
    fixture_row_key,
)
from data.context_history import parse_context_csv
from data.history_import import parse_historical_matches_csv
from model.context_model import (
    assess_context_quality, context_probabilities, walk_forward_context_ablation,
)
from model.forecast_ledger import ForecastLedger
from model.error_learner import OnlineErrorLearner
from model.learning_service import ForecastLearningService
from model.market_models import (
    asian_handicap_probabilities,
    adjust_corners_for_game_state,
)
from storage.context_store import ContextStore
from model.engine import Engine, _norm
from model.validation_cache import select_validation_rows, validation_cache_key
from model.training import prepare_completed_training_rows, train_engine_chronologically
from model.market_forecasts import count_over_probability, is_half_line
from betting.verdict import build_verdict
from betting.best_picks import best_value_market, best_value_markets
from betting.ranking import (
    favorite_display, model_favorite, rank_best_fixtures, rank_model_favorites,
    rank_model_favorites_with_minimum, rank_score,
)
from betting.market_ranking import market_forecast_summary, rank_market_forecasts
from betting.event_shortlist import build_best_market_events, build_market_coverage
from betting.kelly import market_type
from betting.risk import (
    KELLY_FRACTION, MIN_BET_ODDS, MAX_MARKET_STAKE_PCT,
    MAX_MATCH_EXPOSURE_PCT, MAX_DAILY_EXPOSURE_PCT, size_bet,
)
from betting.market_math import blend_probability, proportional_devig
from llm.analyst import LLMRateLimitError
from llm.deep_analysis import (
    MAX_TOP_MATCHES, build_match_dossier, analyze_top_matches,
    format_top_match_opinion, merge_context_snapshots,
    select_top_matches, top_match_shortlist_diagnostics, last_top_matches_status,
)
from model.independent_forecasts import (
    historical_features, historical_xg_features,
    record_scan_forecasts, parse_llm_probabilities, build_ai_crosscheck,
    _fixture_identity as forecast_fixture_identity,
)
from ui.cards import (render_verdict_card, translate_team,
                      parse_card_date, day_label, render_fixture_header)
from ui.scanner_engine import (
    _train_engine, _shared_engine_cache, _load_engine, has_cached_engine,
)

def _recheck_value_after_priority_context(
    card, refreshed_context, *, bank=None, context_store=None,
):
    """Re-run the exact original Value gates after a fresh SStats-only context update.

    Only the SStats context snapshot can unlock this path. LLM/browser-search
    text is intentionally never accepted as official XI/referee proof.
    """
    if not isinstance(card, dict) or not isinstance(refreshed_context, dict):
        return {"updated": False, "reason": "no_priority_context"}
    if str(refreshed_context.get("provider") or "").casefold() != "sstats":
        return {"updated": False, "reason": "untrusted_context_provider"}

    from llm.deep_analysis import merge_context_snapshots

    old_context = card.get("api_context")
    old_context = old_context if isinstance(old_context, dict) else {}
    merged = merge_context_snapshots(old_context, refreshed_context)
    old_quality = card.get("context_quality")
    old_quality = old_quality if isinstance(old_quality, dict) else assess_context_quality(old_context)
    new_quality = assess_context_quality(merged)

    # Save the freshest verified provider snapshot for downstream dossier,
    # training snapshots, Dashboard and later scans.
    card["api_context"] = merged
    card["context_quality"] = new_quality
    if isinstance(card.get("verdict"), dict):
        card["verdict"]["context_quality"] = new_quality

    try:
        kickoff_at = str(merged.get("kickoff_at") or card.get("kickoff_utc") or "")
        observed_at = str(merged.get("timestamp") or "")
        stable_id = "|".join(fixture_row_key(
            str(card.get("home_team") or ""),
            str(card.get("away_team") or ""),
            str(card.get("date_iso") or "")[:10],
        ))
        if context_store is not None and stable_id:
            context_store.record_snapshot(
                fixture_id=stable_id,
                kickoff_at=kickoff_at,
                observed_at=observed_at,
                context=merged,
                data_source="sstats-priority",
            )
    except Exception:
        # The live card remains usable even if optional context persistence fails.
        pass

    payload = card.get("_value_recheck")
    payload = payload if isinstance(payload, dict) else None
    if not payload:
        return {
            "updated": True,
            "old_eligible": bool(old_quality.get("strong_value_eligible")),
            "new_eligible": bool(new_quality.get("strong_value_eligible")),
            "rechecked": False,
            "reason": "no_saved_value_inputs",
        }

    model_history_ready = bool(payload.get("model_bet_history_ready"))
    old_eligible = bool(old_quality.get("strong_value_eligible") and model_history_ready)
    new_eligible = bool(new_quality.get("strong_value_eligible") and model_history_ready)
    if not new_eligible:
        return {
            "updated": True,
            "old_eligible": old_eligible,
            "new_eligible": False,
            "rechecked": False,
            "reason": "official_XI_referee_or_team_history_not_complete",
        }

    # Do not change probabilities or market prices here. Only re-run the same
    # original EV/Edge/Kelly selectors with the newly confirmed SStats context.
    verdict_base = payload.get("base_verdict")
    candidate_rows = payload.get("candidate_rows")
    selection_odds = payload.get("selection_odds")
    bookmaker_by_pick = payload.get("bookmaker_by_pick")
    market_validation_by_pick = payload.get("market_validation_by_pick")
    if not isinstance(verdict_base, dict) or not isinstance(candidate_rows, list):
        return {
            "updated": True, "old_eligible": old_eligible,
            "new_eligible": True, "rechecked": False,
            "reason": "saved_value_inputs_invalid",
        }

    bank_value = payload.get("bank") if bank is None else bank
    kelly_fraction = float(payload.get("kelly_frac") or 0.25)
    min_probability = float(payload.get("min_prob") or 0.0)
    min_edge = max(float(payload.get("min_edge") or 0.05), 0.05)
    min_ev = max(float(payload.get("min_ev") or 0.05), 0.05)
    best_by_market = best_value_markets(
        copy.deepcopy(verdict_base),
        candidate_rows,
        selection_odds,
        float(bank_value or 0.0),
        kelly_fraction,
        min_probability,
        min_edge=min_edge,
        min_ev=min_ev,
        bookmaker_by_pick=bookmaker_by_pick,
        market_validation_by_pick=market_validation_by_pick,
        context_qualified=True,
    )
    chosen_verdict, best = best_value_market(
        copy.deepcopy(verdict_base),
        candidate_rows,
        selection_odds,
        float(bank_value or 0.0),
        kelly_fraction,
        min_probability,
        min_edge=min_edge,
        min_ev=min_ev,
        bookmaker_by_pick=bookmaker_by_pick,
        market_validation_by_pick=market_validation_by_pick,
        context_qualified=True,
    )

    card["best_by_market"] = best_by_market
    card["best"] = best
    if best is not None and isinstance(chosen_verdict, dict):
        card["verdict"] = chosen_verdict
        sources = payload.get("market_source_by_pick")
        sources = sources if isinstance(sources, dict) else {}
        card["best_odds_source"] = str(sources.get(best[1]) or "sstats")
    else:
        rejected = copy.deepcopy(verdict_base)
        rejected.update({
            "context_quality": new_quality,
            "real_odds": False,
            "odd": None,
            "ev": None,
            "edge": None,
            "stake": 0.0,
            "is_bet": False,
            "value_passed": False,
        })
        card["verdict"] = rejected
        card["best"] = None
        card["best_odds_source"] = "no_market"

    # Inputs are a one-shot snapshot, not part of the permanent forecast card.
    card.pop("_value_recheck", None)
    return {
        "updated": True,
        "old_eligible": old_eligible,
        "new_eligible": True,
        "rechecked": True,
        "approved": best is not None,
        "best": best,
        "reason": "value_gates_rechecked_with_priority_SStats_context",
    }


MAX_STAKE_PCT = MAX_DAILY_EXPOSURE_PCT  # Legacy name; enforced in betting.risk.
AUTO_PORTFOLIO_MIN_PROB = STRONG_FORECAST_MIN_PROB  # Shared 65% gate; separate from Value proof.
MARKET_FORECAST_DISPLAY_LIMIT = 20


def _secondary_market_gate_passed(card: object) -> bool:
    """Secondary-market promotion requires the explicit fixed 1X2 holdout PASS."""
    return bool(
        isinstance(card, dict)
        and card.get("fixed_holdout_1x2_passed") is True
    )


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
        # Also accept the common nested [api], [apis], [secrets] or [sstats] layouts.
        for section_name in ("api", "apis", "secrets", "sstats", "xai"):
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


def _provider_api_key(provider: str) -> str:
    """Resolve only the credential assigned to the selected provider."""
    key_config = llm_provider_key_config(provider)
    return _runtime_secret(
        key_config["session"], key_config["env"], key_config["storage"],
        *tuple(key_config.get("aliases") or ()),
    )


def _provider_display_name(provider: str) -> str:
    from config import LLM_PROVIDERS
    base = str(LLM_PROVIDERS.get(provider, {}).get("base") or "").casefold()
    if "api.x.ai" in base:
        return "Grok"
    if "groq" in base:
        return "Groq"
    return str(provider or "AI").split(" (", 1)[0]


def _runtime_int(session_key: str, *names: str, default: int, minimum: int = 0):
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
                            str(
                                player.get("name") or player.get("player_name") or player.get("id") or "Игрок"
                                if isinstance(player, dict) else player
                            )
                            for player in home_players
                        ))
                    else:
                        st.caption("Стартовый состав не опубликован.")
                with col_away:
                    st.markdown("**Стартовый состав гостей**")
                    if away_players:
                        st.write(", ".join(
                            str(
                                player.get("name") or player.get("player_name") or player.get("id") or "Игрок"
                                if isinstance(player, dict) else player
                            )
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


def _portfolio_auto_recommendation(card: object, market_family: str | None = None) -> dict | None:
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

    eligible_families = (
        ("winner", "goals") if _secondary_market_gate_passed(card) else ("winner",)
    )
    if market_family is not None and market_family not in eligible_families:
        return None
    families = (market_family,) if market_family else eligible_families
    for family in families:
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
            odd < MIN_BET_ODDS
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
        and odd >= MIN_BET_ODDS
        and 0.0 < probability < 1.0
        and ev >= 0.05
        and edge_value >= 0.05
        and stake > 0.0
    )


def _qualified_bet_cards(cards: object) -> list[dict]:
    """Build one conservative 1X2 pick per fixture before Model Lab promotion.

    Validated O/U 2.5 may join only after the fixed 1X2 holdout gate is PASS.
    BTTS/corners/cards remain forecast-only.
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
        # The automatic scanner accepts only explicitly SStats-tagged prices.
        # Do not grandfather Highlightly/manual-import quotes on persisted cards.
        if kind == "1X2":
            if not is_sstats_market_source(c.get("odds_source")):
                return None
            raw = source_odds.get(candidate_pick)
        elif kind == "OU":
            if not is_sstats_market_source(c.get("total_odds_source")):
                return None
            raw = total_odds.get(candidate_pick)
        elif kind == "BTTS":
            if not is_sstats_market_source(c.get("btts_odds_source")):
                return None
            raw = btts_odds.get(candidate_pick)
        elif kind == "STAT":
            if not is_sstats_market_source(c.get("extra_market_source")):
                return None
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

    # Hard visibility gate: a fixture is not shown in the scanner unless the
    # exact selected market has a real, SStats-backed bookmaker quote. A price
    # from another market (or a model fair price) must never qualify it.
    candidate_options = {
        label: item for label, item in candidate_options.items()
        if quote_for(item[0]) is not None
    }

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
    if asian_quotes and is_sstats_market_source(c.get("extra_market_source")):
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
            is_sstats_market_source(c.get("odds_source"))
            and bool(c.get("winner_market_validation")) and market_pair_validated
        )
        validation_parts = []
        if not is_sstats_market_source(c.get("odds_source")):
            validation_parts.append("Рыночная котировка отсутствует из SStats; другие источники не допускаются.")
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
            is_sstats_market_source(source)
            and bool((c.get("total_market_confirmed") or {}).get(pick))
        )
        bookmaker_benchmark_validated = (
            abs(float(line_key or 0.0) - 2.5) < 1e-9
            and bool(c.get("total_market_validation"))
        )
        holdout_gate_passed = _secondary_market_gate_passed(c)
        market_is_validated = (
            holdout_gate_passed and history_validated and bookmaker_benchmark_validated
            and quote_corroborated and market_pair_validated
        )
        validation_notes = []
        if not holdout_gate_passed:
            validation_notes.append(
                "O/U forecast-only: сначала должен пройти Fixed Holdout Model Lab v1 на независимом 1X2 holdout."
            )
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
            quote_source_valid = is_sstats_market_source(source)
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
            # Unvalidated fallback prices are informational only; never size a bet.
            stake = 0.0
            market_fair_probability = None

        context_eligible = bool(
            (c.get("context_quality") or {}).get("strong_value_eligible")
        )
        risk_info = {}
        if rec_verdict is not None and market_is_validated and manual_odd >= MIN_BET_ODDS:
            stake, risk_info = size_bet(
                probability=prob, odds=manual_odd, bank=D.get("bank"),
                bets=D.get("bets") or [], fixture_id=c.get("fixture_id"),
                match=c.get("match"), date_iso=c.get("date_iso"),
                market=market_type(pick), pick=pick,
                division=c.get("div"),
                now=datetime.now(app_timezone()),
            )

        with col_metrics.expander("Детали (по желанию)", expanded=False):
            st.write(f"Вероятность модели: {prob * 100:.1f}%")
            st.write(f"Вероятность из коэффициента до снятия маржи: {market_implied * 100:.1f}%")
            if market_fair_probability is not None:
                st.write(f"Вероятность рынка без маржи (один букмекер): {float(market_fair_probability) * 100:.1f}%")
            st.write(f"Преимущество модели: {edge * 100:+.1f}%")
            st.write(f"Ожидаемая доходность: {ev * 100:+.1f}%")
            if abs(model_prob - market_implied) >= 0.20:
                st.warning("Модель сильно расходится с котировкой. Проверьте матч, исход и точную линию.")
            if market_is_validated and stake > 0:
                st.caption(f"Размер после Kelly и всех лимитов риска: {stake:.2f} у.е.")
            elif manual_odd < MIN_BET_ODDS:
                st.caption(f"Не проходит минимальный коэффициент {MIN_BET_ODDS:.2f}.")

        passed = (
            manual_odd >= MIN_BET_ODDS
            and context_eligible
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
        elif manual_odd < MIN_BET_ODDS:
            col_btn.info(f"Минимальный коэффициент для ставки — {MIN_BET_ODDS:.2f}; эта цена остаётся только для просмотра.")
        elif rec_verdict is not None and stake <= 0:
            reason = str(risk_info.get("reason") or "risk_limit")
            col_btn.warning(
                "Ставка заблокирована лимитами риска (¼ Kelly; 1% на рынок; "
                "2% на матч; 5% за день). Причина: " + reason
            )
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
                    "date_time": datetime.now(app_timezone()).strftime("%Y-%m-%d %H:%M"),
                    "fixture_id": c.get("fixture_id"),
                    "scan_run_id": c.get("scan_run_id") or scan_run_id,
                    "forecast_key": c.get("main_forecast_key"),
                    "fixture_source": c.get("source") or c.get("fixture_source") or "",
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
                    "kelly_fraction": KELLY_FRACTION,
                    "risk_policy": {
                        "max_market_pct": MAX_MARKET_STAKE_PCT,
                        "max_match_pct": MAX_MATCH_EXPOSURE_PCT,
                        "max_daily_pct": MAX_DAILY_EXPOSURE_PCT,
                        "stake_reason": str(risk_info.get("reason") or "approved"),
                    },
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

def _backup_imported_archive(reason: str) -> dict:
    """Checkpoint a newly imported dataset to the encrypted remote SQLite backup."""
    if not _runtime_secret("github_token_input", "GITHUB_TOKEN"):
        return {"ok": False, "state": "not_configured"}
    try:
        from storage.remote_backup import backup_database_to_gist
        result = backup_database_to_gist(DB_FILE)
        result["reason"] = str(reason)
        return result
    except Exception as exc:
        return {
            "ok": False, "state": "backup_error",
            "message": f"{type(exc).__name__}: {str(exc)[:120]}",
        }


def _historical_rows_csv(rows: list[dict]) -> bytes:
    """Serialize validated imported history for a portable local copy."""
    valid = [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    if not valid:
        return b""
    columns = sorted({str(key) for row in valid for key in row})
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(valid)
    return ("\\ufeff" + buffer.getvalue()).encode("utf-8")



def _merge_sstats_historical_rows(existing_rows, fetched_rows):
    """Merge verified SStats FT results into the durable custom training archive.

    Identity is competition + calendar date + ordered, normalized team names.
    Existing scores are never overwritten by conflicting provider rows.
    """
    merged = []
    index = {}
    stats = {
        "added": 0, "duplicates": 0, "enriched": 0,
        "conflicts": 0, "invalid": 0,
    }

    def identity(row):
        if not isinstance(row, dict):
            return None
        division = str(row.get("Div") or "").strip().upper()
        stamp = parse_date(row.get("Date") or row.get("date"))
        home = normalize_team(str(row.get("HomeTeam") or row.get("home_team") or ""))
        away = normalize_team(str(row.get("AwayTeam") or row.get("away_team") or ""))
        if not division or stamp is None or not home or not away:
            return None
        return division, stamp.date().isoformat(), home, away

    for raw in existing_rows or []:
        key = identity(raw)
        if key is None:
            stats["invalid"] += 1
            continue
        if key in index:
            stats["duplicates"] += 1
            continue
        item = dict(raw)
        index[key] = item
        merged.append(item)

    score_fields = ("FTHG", "FTAG", "home_goals", "away_goals")
    enrichment_fields = (
        "HxG", "AxG", "HC", "AC", "HY", "AY", "HR", "AR",
        "HTHG", "HTAG", "HTHC", "HTAC", "Referee",
    )
    for raw in fetched_rows or []:
        key = identity(raw)
        if key is None:
            stats["invalid"] += 1
            continue
        current = index.get(key)
        if current is None:
            item = dict(raw)
            index[key] = item
            merged.append(item)
            stats["added"] += 1
            continue

        stats["duplicates"] += 1
        conflict = False
        for field in score_fields:
            old_value, new_value = current.get(field), raw.get(field)
            if old_value not in (None, "") and new_value not in (None, ""):
                try:
                    if int(float(old_value)) != int(float(new_value)):
                        conflict = True
                except (TypeError, ValueError, OverflowError):
                    if str(old_value).strip() != str(new_value).strip():
                        conflict = True
        if conflict:
            stats["conflicts"] += 1
            continue

        changed = False
        for field in score_fields + enrichment_fields:
            if current.get(field) in (None, "") and raw.get(field) not in (None, ""):
                current[field] = raw[field]
                changed = True
        if changed:
            stats["enriched"] += 1

    merged.sort(key=lambda row: (
        str(row.get("Div") or ""),
        (parse_date(row.get("Date") or row.get("date")) or datetime.min).date().isoformat(),
        normalize_team(str(row.get("HomeTeam") or row.get("home_team") or "")),
        normalize_team(str(row.get("AwayTeam") or row.get("away_team") or "")),
    ))
    return merged, stats


def render(min_prob, kelly_frac, matrix_n, min_edge=0.03):
    D = st.session_state.data
    # User-imported history must survive Streamlit session recreation and app reboot.
    for session_key, dataset_key in (
        ("_custom_historical_match_rows", "custom_historical_match_rows"),
        ("_historical_context_rows", "historical_context_rows"),
    ):
        if session_key not in st.session_state:
            try:
                st.session_state[session_key] = user_datasets.load_user_dataset(dataset_key)
            except Exception as dataset_exc:
                st.session_state[session_key] = []
                st.warning(
                    f"Не удалось восстановить архив {dataset_key}: "
                    f"{type(dataset_exc).__name__}. Локальная база не изменена."
                )
    # If a legacy app-state snapshot has no cards, recover the latest scan from
    # the durable SQLite card ledger before presenting an empty scanner.
    if not (D.get("cards") or []):
        try:
            recovered_cards = ScanLedger(DB_FILE).latest_cards(prefer_completed=True)
            if recovered_cards:
                D["cards"] = recovered_cards
                D["report"] = list(D.get("report") or []) + [
                    f"♻️ Восстановлено из SQLite {len(recovered_cards)} карточек последнего сохранённого скана."
                ]
                st.session_state.data = D
                usage.set_local_data(D)
                st.success(
                    f"История восстановлена: {len(recovered_cards)} матчей из постоянного SQLite-архива."
                )
        except Exception as restore_exc:
            st.warning(
                "Не удалось восстановить карточки из реестра сканирований: "
                f"{type(restore_exc).__name__}."
            )

    # Existing saved cards may outlive an odds provider configuration. Strip any
    # legacy non-SStats prices before they can be shown or treated as real quotes.
    saved_cards = D.get("cards") or []
    cleaned_cards = [
        enforce_sstats_market_card(card)
        for card in saved_cards if isinstance(card, dict)
    ]
    if cleaned_cards != saved_cards:
        D["cards"] = cleaned_cards
        st.session_state.data = D
        usage.set_local_data(D)
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
    priced_count = sum(
        bool(c.get("market_odds")) and is_sstats_market_source(c.get("odds_source"))
        for c in previous_cards
    )
    qualified_count = sum(
        bool(c.get("best") is not None
             and is_sstats_market_source(c.get("best_odds_source"))
             and (c.get("verdict") or {}).get("real_odds")
             and (c.get("verdict") or {}).get("value_passed"))
        for c in previous_cards
    )
    model_forecast_count = 0
    ai_forecast_count = 0
    for card in previous_cards:
        try:
            probabilities = [float(card.get(key)) for key in ("p1", "px", "p2")]
            if (
                all(0 <= probability <= 1 for probability in probabilities)
                and 0.90 <= sum(probabilities) <= 1.10
            ):
                model_forecast_count += 1
        except (TypeError, ValueError):
            pass
        if parse_llm_probabilities(card.get("llm_opinion") or "") is not None:
            ai_forecast_count += 1
    pending_count = sum(1 for b in (D.get("bets") or []) if b.get("status") == "pending")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Матчи в последнем скане", len(previous_cards))
    k2.metric("Прогнозы 1X2", model_forecast_count)
    k3.metric("Прогнозы ИИ", ai_forecast_count)
    k4.metric("С реальными кэфами", priced_count)
    k5.metric("Value-проходы", qualified_count)
    k6.metric("Открытые ставки", pending_count)

    last_scan_meta = D.get("scan_meta") if isinstance(D.get("scan_meta"), dict) else {}
    if last_scan_meta:
        last_horizon = max(1, int(last_scan_meta.get("horizon_days") or 1))
        last_from = str(last_scan_meta.get("date_from") or "—")
        last_to = str(last_scan_meta.get("date_to_inclusive") or last_from)
        last_analysed = int(last_scan_meta.get("analysed_matches") or 0)
        st.caption(
            f"Последний завершённый скан: {last_horizon} календарн. дн. "
            f"({last_from} — {last_to}) · в окне проанализировано {last_analysed} матчей. "
            "Изменение ползунка применяется только к следующему запуску; "
            "сохранённые карточки предыдущего скана не меняются."
        )
    else:
        st.caption(
            "Ползунок задаёт горизонт следующего сканирования. "
            "Сохранённые результаты обновятся только после нажатия «Запустить сканирование»."
        )

    st.markdown("<div class='scanner-section-label'>БЫСТРЫЙ ЗАПУСК</div>",
                unsafe_allow_html=True)
    c1, c2 = st.columns([1.25, 1], vertical_alignment="bottom")
    days = c1.slider(
        "Горизонт матчей", 1, 14, DEFAULT_SCAN_HORIZON_DAYS,
        key="scanner_horizon_days_v5",
        help="По умолчанию ищет матчи на ближайшие 3 дня; горизонт можно расширить до 14."
    )
    scan = c2.button("⚡ Запустить сканирование", type="primary",
                     width="stretch",
                     disabled=st.session_state.get("_scan_in_progress", False))

    with st.expander("📚 Данные контекста и исторические архивы", expanded=False):
        st.caption(
            "Коэффициенты автоматического сканирования поступают только из SStats. "
            "Резервные расписания не поставляют котировки."
        )
        st.markdown("**Постоянная история сканирований**")
        try:
            recent_runs = ScanLedger(DB_FILE).runs(limit=10)
            if recent_runs:
                run_choices = {
                    (
                        f"{str(run.get('started_at') or '')[:19]} · "
                        f"{ {'completed': 'завершён', 'running': 'выполняется', 'failed': 'прерван'} .get(str(run.get('status') or ''), str(run.get('status') or 'unknown'))} · "
                        f"{int(run.get('saved_card_count') or 0)} матчей"
                    ): str(run.get("scan_run_id") or "")
                    for run in recent_runs if run.get("scan_run_id")
                }
                for run in recent_runs:
                    state = str(run.get("status") or "unknown")
                    state_label = {
                        "completed": "завершён",
                        "running": "прерван / выполняется",
                        "failed": "прерван с ошибкой",
                    }.get(state, state)
                    st.caption(
                        f"{str(run.get('started_at') or '')[:19]} · {state_label} · "
                        f"{int(run.get('saved_card_count') or 0)} сохранённых матчей · "
                        f"горизонт {int(run.get('horizon_days') or 0)} дн."
                    )
                if run_choices:
                    selected_run_label = st.selectbox(
                        "Выбери сохранённый скан для восстановления",
                        list(run_choices),
                        key="scanner_saved_run_restore_select",
                    )
                    if st.button(
                        "♻️ Восстановить матчи выбранного скана",
                        key="scanner_restore_saved_run",
                        width="stretch",
                    ):
                        selected_run_id = run_choices[selected_run_label]
                        restored_cards = ScanLedger(DB_FILE).cards_for_run(selected_run_id)
                        selected_run = next(
                            (item for item in recent_runs
                             if str(item.get("scan_run_id") or "") == selected_run_id),
                            {},
                        )
                        if restored_cards:
                            restored_data = dict(st.session_state.get("data") or D)
                            restored_data["cards"] = restored_cards
                            restored_data["report"] = list(selected_run.get("report") or [])
                            restored_data["scan_meta"] = {
                                "scan_run_id": selected_run_id,
                                "scan_started_at": selected_run.get("started_at"),
                                "scan_completed_at": selected_run.get("completed_at"),
                                "horizon_days": selected_run.get("horizon_days"),
                                "date_from": selected_run.get("date_from"),
                                "date_to_inclusive": selected_run.get("date_to_inclusive"),
                                "analysed_matches": selected_run.get("analysed_matches"),
                                "forecast_cards": len(restored_cards),
                                "source_counts": selected_run.get("source_counts") or {},
                                "restored_from_scan_ledger": True,
                            }
                            st.session_state.data = restored_data
                            usage.set_local_data(restored_data)
                            st.success(
                                f"Восстановлено {len(restored_cards)} матчей из скана "
                                f"{selected_run_id[:12]}."
                            )
                            st.rerun()
                        else:
                            st.warning("В выбранном скане нет сохранённых карточек.")
            else:
                st.caption("Сканирований в постоянном архиве пока нет.")
        except Exception as archive_exc:
            st.warning(f"Архив сканирований недоступен: {type(archive_exc).__name__}.")

        st.caption(
            "Обычный скан использует сохранённую модель и постоянный архив. "
            "Историю команд для лиг текущего расписания можно обновить отдельно "
            "без запуска тяжёлого market benchmark. Полная рыночная проверка — отдельная операция."
        )
        if st.button(
            "📚 Обновить историю команд текущих лиг (без долгой проверки рынка)",
            key="scanner_team_history_refresh",
            width="stretch",
            help=(
                "Загрузит до 5 лет завершённых матчей SStats только для соревнований "
                "из следующего расписания, сохранит их в SQLite и переобучит базовую модель. "
                "Не запускает walk-forward benchmark по 25 тысячам матчей. "
                "Value-допуск останется закрытым до отдельной полной рыночной проверки."
            ),
        ):
            st.session_state["_scanner_team_history_refresh_next"] = True
        if st.session_state.get("_scanner_team_history_refresh_next"):
            st.info(
                "Обновление истории команд поставлено в очередь. Нажми «Запустить сканирование». "
                "Архив будет сохранён в SQLite и попадёт в зашифрованную резервную копию."
            )
        if st.button(
            "🔄 Полное обновление архивов + market benchmark (долго)",
            key="scanner_force_history_refresh",
            width="stretch",
            help=(
                "Перечитает сезонные источники, обновит SStats и xG, обучит модель "
                "и пересчитает полный walk-forward market benchmark. Используй только "
                "когда нужно заново проверить допуск рынков к Value."
            ),
        ):
            st.session_state["_scanner_force_history_refresh_next"] = True
        if st.session_state.get("_scanner_force_history_refresh_next"):
            st.warning(
                "Полная рыночная проверка поставлена в очередь. Она может быть долгой; "
                "обычные обновления расписания её не запускают."
            )

        st.markdown("**Контекст матча · SStats + исторический CSV**")
        st.caption("Составы, травмы и судьи запрашиваются через SStats для тех матчей, где поставщик отдаёт подтверждённые данные.")
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
                if context_payload["rows"]:
                    saved_context = user_datasets.merge_user_dataset(
                        "historical_context_rows", context_payload["rows"],
                    )
                    merged_context = user_datasets.load_user_dataset(
                        "historical_context_rows",
                    )
                    st.session_state["_historical_context_rows"] = merged_context
                    backup_context = (
                        _backup_imported_archive("historical_context_rows")
                        if saved_context["added_rows"] or saved_context["enriched_rows"]
                        else {"ok": False, "state": "unchanged"}
                    )
                    persistence_note = (
                        f"; всего в SQLite: {saved_context['row_count']} снимков "
                        f"(новых {saved_context['added_rows']}, дополнено {saved_context['enriched_rows']}, "
                        f"дубликатов {saved_context['duplicate_rows']}, конфликтов {saved_context['conflicts']})"
                    )
                    if backup_context.get("ok"):
                        persistence_note += "; зашифрованная копия обновлена"
                    elif backup_context.get("state") == "not_configured":
                        persistence_note += "; удалённая копия не настроена"
                    elif backup_context.get("state") != "unchanged":
                        persistence_note += "; ⚠️ удалённая копия не обновилась"
                else:
                    persistence_note = "; прежний сохранённый архив оставлен без изменений"
                st.success(
                    f"Контекстный архив: принято {context_payload['accepted']} строк, "
                    f"отклонено {len(context_payload['rejected'])} "
                    f"(post-kickoff данные исключены){persistence_note}."
                )
                if context_payload["rejected"]:
                    st.caption("Причины отклонения: " + "; ".join(
                        str(item.get("reason")) for item in context_payload["rejected"][:5]
                    ))
            except Exception as exc:
                st.error(
                    "Контекстный CSV не удалось проверить или сохранить: "
                    + f"{type(exc).__name__}: {str(exc)[:160]}"
                )
        saved_context_rows = st.session_state.get("_historical_context_rows") or []
        if saved_context_rows:
            st.caption(
                f"Постоянно сохранённый контекстный архив: {len(saved_context_rows)} строк."
            )
            st.download_button(
                "⬇️ Скачать сохранённый контекстный архив",
                data=_historical_rows_csv(saved_context_rows),
                file_name="neuro_bet_context_history_saved.csv",
                mime="text/csv",
                key="download_saved_context_history",
            )

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
                if history_payload["rows"]:
                    saved_history = user_datasets.merge_user_dataset(
                        "custom_historical_match_rows", history_payload["rows"],
                    )
                    merged_history = user_datasets.load_user_dataset(
                        "custom_historical_match_rows",
                    )
                    st.session_state["_custom_historical_match_rows"] = merged_history
                    if (
                        st.session_state.get("_custom_historical_match_sha256")
                        != saved_history["sha256"]
                    ):
                        st.session_state["_scanner_team_history_refresh_next"] = True
                        st.session_state["_custom_historical_match_sha256"] = saved_history["sha256"]
                    backup_history = (
                        _backup_imported_archive("custom_historical_match_rows")
                        if saved_history["added_rows"] or saved_history["enriched_rows"]
                        else {"ok": False, "state": "unchanged"}
                    )
                    persistence_note = (
                        f"; в SQLite теперь {saved_history['row_count']} матчей "
                        f"(новых {saved_history['added_rows']}, дополнено {saved_history['enriched_rows']}, "
                        f"дубликатов {saved_history['duplicate_rows']}, конфликтов {saved_history['conflicts']})"
                    )
                    if backup_history.get("ok"):
                        persistence_note += "; зашифрованная копия обновлена"
                    elif backup_history.get("state") == "not_configured":
                        persistence_note += "; удалённая копия не настроена"
                    elif backup_history.get("state") != "unchanged":
                        persistence_note += "; ⚠️ удалённая копия не обновилась"
                else:
                    persistence_note = "; прежний сохранённый архив оставлен без изменений"
                st.success(
                    f"Исторический архив: принято {history_payload['accepted']} матчей "
                    f"из {len(history_payload['divisions'])} соревнований; "
                    f"отклонено {len(history_payload['rejected'])}{persistence_note}."
                )
                st.caption(
                    "Коды: " + ", ".join(history_payload["divisions"])
                    + ". Эти соревнования обучают Base-модель отдельно и пока не проходят "
                    "Value-валидацию крупных лиг."
                )
            except Exception as exc:
                st.error(
                    "Архив исторических матчей не удалось проверить или сохранить: "
                    + f"{type(exc).__name__}: {str(exc)[:160]}"
                )
        custom_historical_rows = st.session_state.get("_custom_historical_match_rows") or []
        if custom_historical_rows:
            st.caption(
                f"Постоянно сохранённый архив обучения: {len(custom_historical_rows)} матчей."
            )
            st.download_button(
                "⬇️ Скачать сохранённый архив женских лиг / сборных",
                data=_historical_rows_csv(custom_historical_rows),
                file_name="neuro_bet_historical_matches_saved.csv",
                mime="text/csv",
                key="download_saved_custom_history",
            )

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
        report_lines = [
            str(line) for line in (D.get("report", []) or [])
        ]
        history_lines = [
            line for line in report_lines
            if "SStats history" in line or "SStats LeagueId" in line
        ]
        history_is_unavailable = (
            not history_lines
            or any(
                "загружено 0 FT-матчей" in line
                or "нет ни одного безопасно определённого LeagueId" in line
                or "API key отсутствует" in line
                for line in history_lines
            )
        )
        if history_is_unavailable:
            st.warning(
                "История команд SStats пока не подтверждена. "
                "Раскрой диагностику ниже: там показаны LeagueId, "
                "загруженные FT-результаты и точная причина отказа."
            )
        else:
            st.caption(" · ".join(history_lines[-2:]))
        with st.expander("🔌 Диагностика", expanded=history_is_unavailable):
            for line in report_lines:
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
        favorite_history_floor = MIN_FAVORITE_HISTORY
        favorite_matches = rank_model_favorites_with_minimum(
            all_cards,
            minimum=MIN_RECOMMENDED_MATCHES,
            limit=MAX_RECOMMENDED_MATCHES,
            min_sample_games=favorite_history_floor,
        )
        low_history_favorites = sum(
            1 for item in favorite_matches if not item.get("history_sufficient", False)
        )
        strong_favorites = sum(
            1 for item in favorite_matches
            if item.get("history_sufficient", False)
            and item.get("calibration_ready", True)
            and float(item.get("probability") or 0.0) >= STRONG_FORECAST_MIN_PROB
        )
        st.markdown(
            "<div style='display:flex;align-items:end;justify-content:space-between;gap:12px;margin:18px 0 4px'>"
            "<div><div class='scanner-kicker'>MODEL FAVORITES · 1X2</div>"
            "<h2 style='margin:0'>Самые вероятные исходы</h2>"
            "<div style='color:#91a4ad;font-size:.88rem;margin-top:5px'>"
            f"Показано {len(favorite_matches)} уникальных матчей: целевой минимум {MIN_RECOMMENDED_MATCHES}, "
            f"максимум {MAX_RECOMMENDED_MATCHES}. Сильных прогнозов ≥{STRONG_FORECAST_MIN_PROB:.0%} "
            f"с историей ≥{favorite_history_floor} и калибровкой лиги: {strong_favorites}. "
            "Сортировка прежде всего по вероятности. Матчи с короткой историей явно помечаются "
            "и не считаются сильными сигналами. Это прогнозы, не гарантии победы.</div></div>"
            "</div>",
            unsafe_allow_html=True,
        )
        if low_history_favorites:
            st.caption(
                f"Матчей с историей менее {favorite_history_floor} игр хотя бы у одной команды: "
                f"{low_history_favorites}. Они остаются в списке как прогнозы с ограниченной историей."
            )
        uncalibrated_count = sum(
            1 for item in favorite_matches
            if item.get("history_sufficient", False)
            and not item.get("calibration_ready", True)
        )
        if uncalibrated_count:
            st.caption(
                f"Матчей с достаточной историей, но без обученного калибратора своей лиги: "
                f"{uncalibrated_count}. Их проценты показываются как сырые вероятности, "
                "без статуса сильного сигнала."
            )
        if len(favorite_matches) < MIN_RECOMMENDED_MATCHES:
            st.caption(
                f"В текущем расписании доступно только {len(favorite_matches)} уникальных "
                "матчей с корректным распределением вероятностей. Недостающие матчи не создаются искусственно."
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
                favorite_history_sufficient = bool(
                    favorite_item.get("history_sufficient", False)
                )
                favorite_calibration_ready = bool(
                    favorite_item.get("calibration_ready", True)
                )
                favorite_strength_label = (
                    "ОГРАНИЧЕННАЯ ИСТОРИЯ · НЕ СИЛЬНЫЙ СИГНАЛ"
                    if not favorite_history_sufficient
                    else "СЫРАЯ ВЕРОЯТНОСТЬ · НЕТ КАЛИБРАТОРА ЛИГИ"
                    if not favorite_calibration_ready
                    else f"СИЛЬНЫЙ ПРОГНОЗ ≥{STRONG_FORECAST_MIN_PROB:.0%} · КАЛИБРОВАН"
                    if favorite_probability >= STRONG_FORECAST_MIN_PROB
                    else f"НИЖЕ ПОРОГА СИЛЬНОГО ПРОГНОЗА {STRONG_FORECAST_MIN_PROB:.0%}"
                )
                favorite_strength_title = (
                    "МАЛО ИСТОРИИ" if not favorite_history_sufficient
                    else favorite_item["strength"] if favorite_calibration_ready
                    else "НЕ КАЛИБРОВАНО"
                )
                display_favorite = favorite_display(favorite_item)
                favorite_outcome_html = (
                    f"{html.escape(display_favorite['label'])} · {favorite_probability:.1%}"
                    if display_favorite.get("calibrated", False)
                    else html.escape(display_favorite["label"])
                )
                model_distribution = " · ".join(
                    f"{label} {float(favorite_card.get(field) or 0):.1%}"
                    for field, label in (("p1", "П1"), ("px", "X"), ("p2", "П2"))
                )
                model_distribution_label = (
                    "Вероятности модели"
                    if favorite_calibration_ready
                    else "СЫРЫЕ вероятности · калибратор этой лиги не обучен"
                )
                market_distribution_html = ""
                if market_line := proportional_devig(
                    favorite_card.get("market_odds") or {},
                    ("П1", "X", "П2"),
                    bookmaker_by_pick=favorite_card.get("bookmaker_by_pick") or {},
                ):
                    market_pick = max(market_line, key=market_line.get)
                    market_distribution = " · ".join(
                        f"{label} {market_line[pick]:.1%}"
                        for pick, label in (("П1", "П1"), ("X", "X"), ("П2", "П2"))
                    )
                    market_distribution_html = (
                        "<div style='color:#82c8e5;font-size:11px;margin-top:4px'>"
                        f"Рынок без маржи: {html.escape(market_distribution)} · "
                        f"фаворит рынка: {html.escape(market_pick)}</div>"
                    )
                with favorite_columns[(favorite_rank - 1) % 2]:
                    st.markdown(
                        "<div style='padding:13px 15px;margin:7px 0;border:1px solid rgba(118,159,180,.28);"
                        "border-radius:12px;background:linear-gradient(120deg,rgba(22,42,54,.9),rgba(15,24,32,.94))'>"
                        f"<div style='display:flex;justify-content:space-between;gap:8px;color:#9eb1bd;font-size:11px'>"
                        f"<span>#{favorite_rank} · {html.escape(favorite_league)}</span>"
                        f"<span>{html.escape(favorite_strength_title)}</span></div>"
                        f"<div style='font-weight:750;font-size:16px;color:#f0f5f7;margin:7px 0 4px'>"
                        f"{html.escape(favorite_match)}</div>"
                        f"<div style='color:#8fd8bd;font-weight:800;font-size:17px'>"
                        f"{favorite_outcome_html}</div>"
                        f"<div style='color:#d4e0e6;font-size:11px;margin-top:4px'>"
                        f"{html.escape(model_distribution_label)}: {html.escape(model_distribution)}</div>"
                        + market_distribution_html
                        + f"<div style='color:#b7c2c9;font-size:11px;margin-top:2px'>"
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
                    "<span class='pick-chip'>xG proxy учтён</span>"
                    if card.get("xg_modelled_used") else
                    "<span class='pick-chip'>xG провайдера учтён</span>"
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
            "winner": sum(
                bool(x.get("market_odds")) and is_sstats_market_source(x.get("odds_source"))
                for x in market_pool
            ),
            "goals": sum(
                bool(x.get("total_market_odds")) and is_sstats_market_source(x.get("total_odds_source"))
                for x in market_pool
            ),
            "btts": sum(
                bool(x.get("btts_market_odds")) and is_sstats_market_source(x.get("btts_odds_source"))
                for x in market_pool
            ),
            "corners": sum(
                is_sstats_market_source(x.get("extra_market_source"))
                and any(str(k).startswith("Угл ") for k in (x.get("stat_market_odds") or {}))
                for x in market_pool
            ),
            "cards": sum(
                is_sstats_market_source(x.get("extra_market_source"))
                and any(str(k).startswith("Карты ") for k in (x.get("stat_market_odds") or {}))
                for x in market_pool
            ),
            "handicap": sum(
                bool(x.get("asian_handicap_market_odds"))
                and is_sstats_market_source(x.get("extra_market_source"))
                for x in market_pool
            ),
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
        top_match_diagnostics = top_match_shortlist_diagnostics(
            market_pool, min_team_history=MIN_FAVORITE_HISTORY,
        )
        top_match_candidates = select_top_matches(
            market_pool, limit=MAX_TOP_MATCHES, min_team_history=MIN_FAVORITE_HISTORY,
        )
        best_match_diagnostics = {}
        best_match_pool = rank_best_fixtures(
            market_pool,
            now=datetime.now(app_timezone()),
            limit=None,
            min_sample_games=MIN_FAVORITE_HISTORY,
            min_probability=0.52,
            include_limited_history=False,
            require_calibration=False,
            diagnostics=best_match_diagnostics,
        )
        best_matches = best_match_pool[:10]

        # A passable BET is not just a high model probability: require the
        # same conservative portfolio gates used by the automatic portfolio.
        passable_bets = _qualified_bet_cards(market_pool)
        try:
            passable_bets.sort(key=lambda c: (
                -rank_score(c),
                -float((c.get("verdict") or {}).get("ev") or -1),
                -float((c.get("verdict") or {}).get("prob") or 0),
            ))
        except Exception:
            passable_bets.sort(
                key=lambda c: -float((c.get("verdict") or {}).get("ev") or -1)
            )
        passable_bets = passable_bets[:10]

        passable_diagnostics = {}
        passable_pool = rank_best_fixtures(
            market_pool,
            now=datetime.now(app_timezone()),
            limit=None,
            min_sample_games=MIN_FAVORITE_HISTORY,
            min_probability=STRONG_FORECAST_MIN_PROB,
            include_limited_history=True,
            require_calibration=True,
            diagnostics=passable_diagnostics,
        )
        passable_watchlist = [
            item for item in passable_pool
            if item.get("history_ready")
            and item.get("probability_ready")
            and not item.get("calibration_ready")
        ][:10]
        forecast_counts = {
            family: len(items) for family, items in ranked_forecast_pool.items()
        }
        strong_forecast_counts = {
            family: sum(bool(item.get("strong_forecast")) for item in items)
            for family, items in ranked_forecast_pool.items()
        }
        market_tabs = st.tabs([
            "🏆 Лучшие события · все рынки",
            f"Тоталы · прогнозы {forecast_counts['goals']} (сильных {strong_forecast_counts['goals']}) / Value {signal_counts['goals']} / цены {quote_counts['goals']}",
            f"{'Общий shortlist' if recommendation_mode else 'Все матчи'} ({len(cards_view)})",
            f"1X2 · прогнозы {forecast_counts['winner']} (сильных {strong_forecast_counts['winner']}) / Value {signal_counts['winner']} / цены {quote_counts['winner']}",
            f"BTTS · прогнозы {forecast_counts['btts']} (сильных {strong_forecast_counts['btts']}) / Value {signal_counts['btts']} / цены {quote_counts['btts']}",
            f"Угловые · прогнозы {forecast_counts['corners']} (сильных {strong_forecast_counts['corners']}) / Value {signal_counts['corners']} / цены {quote_counts['corners']}",
            f"Карточки · прогнозы {forecast_counts['cards']} (сильных {strong_forecast_counts['cards']}) / Value {signal_counts['cards']} / цены {quote_counts['cards']}",
            f"Форы · котировки {quote_counts['handicap']}",
            f"🏆 ЛУЧШИЕ / VALUE · {len(best_matches)}/{len(passable_bets)}",
            f"🔎 AI-ДОСЬЕ · {len(top_match_candidates)}/{MAX_TOP_MATCHES}",
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
                    strength_label = "НЕДОСТАТОЧНО ИСТОРИИ КОМАНД (нужно не менее 10 матчей)"
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
        with market_tabs[1]:
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

        with market_tabs[2]:
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
            (3, "winner", "market_odds", "1X2"),
            (4, "btts", "btts_market_odds", "BTTS"),
            (5, "corners", "stat_market_odds", "угловые"),
            (6, "cards", "stat_market_odds", "карточки"),
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

        with market_tabs[7]:
            handicap_cards = [
                card for card in market_pool
                if card.get("asian_handicap_market_odds")
                and is_sstats_market_source(card.get("extra_market_source"))
            ]
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
                        f"поражение {probabilities['loss']:.1%}"
                    )
                st.caption(
                    "EV азиатской форы рассчитан по полной матрице счетов, включая push и split-ставки. "
                    "Пока нет отдельного out-of-sample benchmark на одинаковой выборке, рынок остаётся "
                    "информационным и не может попасть в Value-портфель."
                )

        with market_tabs[8]:
            st.subheader("Лучшие матчи и проходные ставки")
            st.caption(
                "Лучшие матчи — прогнозы с командной поддержкой, минимум 10 домашних "
                "игр хозяев и 10 выездных игр гостей, ранжированные по вероятности, "
                "отрыву от второго исхода, истории, xG и полноте линии. "
                "Проходные ставки — отдельный, более строгий фильтр: подтверждённый состав "
                "обеих команд и судья, согласованная котировка одного букмекера, "
                "проверка рынка, положительные Edge и EV не ниже 5% и допустимый Kelly."
            )
            m1, m2, m3 = st.columns(3)
            m1.metric("Лучшие матчи", len(best_matches))
            m2.metric("Проходные Value-ставки", len(passable_bets))
            m3.metric("Отсечены prior/league-only", best_match_diagnostics.get("unsupported_model", 0))

            st.markdown("**🏆 Лучшие матчи по качеству прогноза**")
            if best_matches:
                best_rows = []
                for number, item in enumerate(best_matches, start=1):
                    card = item.get("card") or {}
                    best_rows.append({
                        "№": number,
                        "Матч": card.get("match") or (
                            f"{card.get('home_team') or card.get('HomeTeam') or 'Хозяева'} — "
                            f"{card.get('away_team') or card.get('AwayTeam') or 'Гости'}"
                        ),
                        "Лига": card.get("league") or card.get("div") or "—",
                        "Начало": " ".join(
                            str(value) for value in (card.get("date") or card.get("date_iso"), card.get("time"))
                            if value
                        ),
                        "Наиболее вероятный исход": item.get("label") or item.get("pick") or "—",
                        "Вероятность": f"{float(item.get('probability') or 0):.1%}",
                        "Отрыв от 2-го": f"{float(item.get('probability_margin') or 0):.1%}",
                        "История дом/выезд": f"{item.get('home_sample', 0)}/{item.get('away_sample', 0)}",
                        "Калибровка": (
                            f"применена · {item.get('calibration_matches', 0)}"
                            if item.get("calibration_ready")
                            else f"не применена · {item.get('calibration_matches', 0)}"
                        ),
                        "Качество": f"{float(item.get('quality_score') or 0):.1f}/100",
                    })
                st.dataframe(
                    best_rows,
                    width="stretch",
                    hide_index=True,
                    height=min(520, 42 + 36 * len(best_rows)),
                )
            else:
                st.info(
                    "Нет матчей с подтверждённой командной историей и вероятностью фаворита от 52%. "
                    "Prior-only и league-only прогнозы в этот список не допускаются."
                )

            st.divider()
            st.markdown("**✅ Проходные ставки · только проверенный Value**")
            if passable_bets:
                bet_rows = []
                for number, card in enumerate(passable_bets, start=1):
                    best = card.get("best") or ()
                    verdict = card.get("verdict") or {}
                    if len(best) < 6:
                        continue
                    try:
                        odd = float(best[2])
                        ev = float(best[3])
                        probability = float(best[4])
                        stake = float(best[5])
                    except (TypeError, ValueError):
                        continue
                    edge = verdict.get("edge")
                    try:
                        edge_text = f"{float(edge):+.1%}"
                    except (TypeError, ValueError):
                        edge_text = "—"
                    context = card.get("context_quality") or {}
                    bet_rows.append({
                        "№": number,
                        "Матч": card.get("match") or card.get("match_ru") or "—",
                        "Лига": card.get("league") or card.get("div") or "—",
                        "Рынок": best[0],
                        "Выбор": best[1],
                        "Кэф": f"{odd:.2f}",
                        "Вероятность": f"{probability:.1%}",
                        "EV": f"{ev:+.1%}",
                        "Edge": edge_text,
                        "Составы / судья": (
                            "подтверждены"
                            if context.get("strong_value_eligible") else "не подтверждены"
                        ),
                        "Источник цены": (
                            (card.get("market_odds_source_by_pick") or {}).get(str(best[1]))
                            or card.get("best_odds_source")
                            or card.get("odds_source")
                            or "—"
                        ),
                        "Размер ставки": f"{stake:.2f}",
                    })
                if bet_rows:
                    st.dataframe(
                        bet_rows,
                        width="stretch",
                        hide_index=True,
                        height=min(520, 42 + 36 * len(bet_rows)),
                    )
                else:
                    st.info("Нет корректно сформированных Value-строк для этого скана.")
            else:
                st.info(
                    "В текущем скане нет ставки, прошедшей все фильтры Value. "
                    "Высокая вероятность без проверенного рынка и цены не считается проходной ставкой."
                )

            if passable_watchlist:
                st.divider()
                st.markdown("**🔎 На наблюдение — сильный прогноз, но калибровка ещё не допущена**")
                watch_rows = []
                for number, item in enumerate(passable_watchlist, start=1):
                    card = item.get("card") or {}
                    pick = str(item.get("pick") or "")
                    quoted = (card.get("market_odds") or {}).get(pick)
                    try:
                        odds_text = f"{float(quoted):.2f}" if quoted is not None else "—"
                    except (TypeError, ValueError):
                        odds_text = "—"
                    watch_rows.append({
                        "№": number,
                        "Матч": card.get("match") or card.get("match_ru") or "—",
                        "Лига": card.get("league") or card.get("div") or "—",
                        "Исход": item.get("label") or pick,
                        "Вероятность": f"{float(item.get('probability') or 0):.1%}",
                        "История дом/выезд": f"{item.get('home_sample', 0)}/{item.get('away_sample', 0)}",
                        "Кэф": odds_text,
                        "Калибратор": f"нет · settled n={item.get('calibration_matches', 0)}",
                    })
                st.dataframe(
                    watch_rows,
                    width="stretch",
                    hide_index=True,
                    height=min(420, 42 + 36 * len(watch_rows)),
                )
                st.caption(
                    "Это не Value и не ставка. Варианты показаны отдельно, чтобы не выдавать "
                    "сырые или shadow-only вероятности за подтверждённый сигнал."
                )

            st.caption(
                "Диагностика строгого фильтра фаворитов: "
                f"в пуле {passable_diagnostics.get('input_cards', 0)} · "
                f"отсечены prior/league-only {passable_diagnostics.get('unsupported_model', 0)} · "
                f"не хватает домашней истории {passable_diagnostics.get('insufficient_home_history', 0)} · "
                f"не хватает выездной истории {passable_diagnostics.get('insufficient_away_history', 0)} · "
                f"ниже 65% {passable_diagnostics.get('below_probability', 0)} · "
                f"без калибровки {passable_diagnostics.get('uncalibrated_candidates', 0)}."
            )

        with market_tabs[9]:
            st.caption(
                "Независимый список для глубокого анализа: минимум 5 домашних матчей хозяев "
                "и 5 выездных матчей гостей. Отбор не зависит от фаворита основной модели "
                "и наличия коэффициентов. Это исследовательские кандидаты, а не Value-ставки. "
                "ИИ не получает вероятности и котировки основной модели."
            )
            st.caption(
                "Фильтр: "
                f"матчей {top_match_diagnostics['fixtures']} → "
                f"история команд ≥5 {top_match_diagnostics['enough_venue_history']} → "
                f"xG {top_match_diagnostics['with_xg_history']} → "
                f"угловые {top_match_diagnostics['with_corners_history']} → "
                f"карточки {top_match_diagnostics['with_cards_history']} → "
                f"кандидатов {top_match_diagnostics['eligible']}."
            )
            if not top_match_candidates:
                st.info(
                    "В этом сканировании нет матчей с достаточной домашней/выездной историей для независимого анализа."
                )

            refresh_top_dossiers = st.button(
                f"🔄 Обновить SStats-досье и повторить анализ {_provider_display_name(D.get('meta', {}).get('llm_provider', 'Groq (бесплатно, быстро)'))}",
                key="top_matches_refresh_sstats_groq",
                disabled=not bool(top_match_candidates),
                help=(
                    "Повторно проверяет доступные составы, травмы и судью выбранных "
                    "матчей, затем отправляет выбранной AI-модели один пакетный запрос."
                ),
            )
            if refresh_top_dossiers:
                refresh_sstats_key = _runtime_secret(
                    "sstats_api_key_input", "SSTATS_API_KEY", "sstats_api_key"
                )
                refresh_provider = D.get("meta", {}).get(
                    "llm_provider", "Groq (бесплатно, быстро)"
                )
                refresh_provider_name = _provider_display_name(refresh_provider)
                refresh_llm_key = _provider_api_key(refresh_provider)
                if not refresh_llm_key:
                    st.error(
                        f"Для анализа {refresh_provider_name} не задан ключ "
                        f"{llm_provider_key_config(refresh_provider)['env']}."
                    )
                elif usage.llm_remaining() <= 0:
                    st.error("Дневной лимит LLM исчерпан.")
                else:
                    with st.spinner(
                        f"Обновляю SStats-контекст и отправляю {refresh_provider_name} до "
                        f"{len(top_match_candidates)} матчей; для GPT-OSS 120B также запускаю "
                        "Browser Search по точным командам и датам…"
                    ):
                        refresh_now = datetime.now(app_timezone())
                        refresh_rows = []
                        for item in top_match_candidates:
                            refresh_card = item["card"]
                            if refresh_card.get("sstats_game_id") in (None, ""):
                                continue
                            home_name = str(refresh_card.get("home_team") or "")
                            away_name = str(refresh_card.get("away_team") or "")
                            date_key = str(refresh_card.get("date_iso") or "")[:10]
                            refresh_key = fixture_row_key(
                                home_name, away_name, date_key
                            )
                            refresh_rows.append({
                                "Date": (
                                    f"{date_key} {refresh_card.get('time') or ''}"
                                ).strip(),
                                "date_iso": date_key,
                                "kickoff_utc": refresh_card.get("kickoff_utc") or "",
                                "HomeTeam": home_name,
                                "AwayTeam": away_name,
                                "sstats_game_id": refresh_card.get("sstats_game_id"),
                            })

                        refreshed_contexts = {}
                        refresh_logs = []
                        if refresh_sstats_key and refresh_rows:
                            offset = refresh_now.utcoffset()
                            offset_hours = (
                                int(round(offset.total_seconds() / 3600))
                                if offset else 0
                            )
                            try:
                                refreshed_contexts, refresh_logs = sstats_context_enrich(
                                    refresh_rows,
                                    refresh_sstats_key,
                                    date_from=refresh_now.date().isoformat(),
                                    date_to=(
                                        refresh_now.date() + timedelta(days=days - 1)
                                    ).isoformat(),
                                    now=refresh_now,
                                    max_injury_checks=MAX_TOP_MATCHES,
                                    max_lineup_checks=MAX_TOP_MATCHES,
                                    timezone_offset=offset_hours,
                                )
                            except Exception as exc:
                                refresh_logs.append(
                                    f"SStats refresh error: {type(exc).__name__}"
                                )
                        elif not refresh_sstats_key:
                            refresh_logs.append(
                                "SStats key не задан; анализ продолжен на уже имеющихся исторических данных."
                            )
                        elif not refresh_rows:
                            refresh_logs.append(
                                "SStats game IDs отсутствуют; обновить досье невозможно."
                            )

                        refresh_dossiers = []
                        refresh_cards_by_id = {}
                        for rank, item in enumerate(top_match_candidates, start=1):
                            refresh_card = item["card"]
                            home_name = str(refresh_card.get("home_team") or "")
                            away_name = str(refresh_card.get("away_team") or "")
                            date_key = str(refresh_card.get("date_iso") or "")[:10]
                            refresh_key = fixture_row_key(
                                home_name, away_name, date_key
                            )
                            refresh_card["deep_analysis_context"] = merge_context_snapshots(
                                refresh_card.get("deep_analysis_context")
                                or refresh_card.get("api_context") or {},
                                refreshed_contexts.get(refresh_key)
                                if isinstance(refreshed_contexts.get(refresh_key), dict)
                                else {},
                            )
                            dossier = build_match_dossier(refresh_card)
                            dossier["shortlist_rank"] = rank
                            refresh_dossiers.append(dossier)
                            refresh_cards_by_id[dossier["fixture_id"]] = refresh_card

                        try:
                            refreshed_analyses = analyze_top_matches({
                                "api_key": refresh_llm_key,
                                "provider": refresh_provider,
                                "model": D.get("meta", {}).get("llm_model", ""),
                                "candidates": refresh_dossiers,
                            })
                        except LLMRateLimitError as exc:
                            refreshed_analyses = {}
                            refresh_logs.append(
                                f"{refresh_provider_name} rate limit: "
                                + (
                                    f"retry after {exc.retry_after_seconds:.0f}s"
                                    if exc.retry_after_seconds is not None
                                    else "лимит провайдера"
                                )
                            )
                        except Exception as exc:
                            refreshed_analyses = {}
                            refresh_logs.append(
                                f"{refresh_provider_name} deep analysis error: {type(exc).__name__}"
                            )
                        ai_status = last_top_matches_status()
                        if ai_status.get("browser_search_status") not in (None, "", "not_enabled"):
                            refresh_logs.append(
                                "Groq Browser Search: "
                                + str(ai_status.get("browser_search_status"))
                                + " · "
                                + str(ai_status.get("message") or "")
                            )
                        for fixture_id, analysis in refreshed_analyses.items():
                            refresh_card = refresh_cards_by_id.get(fixture_id)
                            if not refresh_card:
                                continue
                            model_pick = max(
                                (("П1", float(refresh_card.get("p1") or 0.0)),
                                 ("X", float(refresh_card.get("px") or 0.0)),
                                 ("П2", float(refresh_card.get("p2") or 0.0))),
                                key=lambda pair: pair[1],
                            )[0]
                            refresh_card["deep_analysis"] = analysis
                            refresh_card["deep_analysis_provider"] = refresh_provider
                            refresh_card["deep_analysis_model"] = str(D.get("meta", {}).get("llm_model") or "")
                            refresh_card["llm_opinion"] = format_top_match_opinion(
                                analysis, model_pick_code=model_pick
                            )
                            refresh_card["deep_analysis_fixture_id"] = fixture_id
                            refresh_card["ai_crosscheck"] = build_ai_crosscheck(
                                refresh_card, analysis
                            )

                        updated_data = dict(D)
                        updated_data["cards"] = all_cards
                        updated_report = list(D.get("report") or [])
                        updated_report.extend(
                            "🏆 TOP refresh: " + str(line) for line in refresh_logs
                        )
                        updated_report.append(
                            f"🏆 TOP refresh: {refresh_provider_name} получил {len(refreshed_analyses)}/"
                            f"{len(refresh_dossiers)} досье."
                        )
                        updated_data["report"] = updated_report[-150:]
                        st.session_state.data = updated_data
                        usage.set_local_data(updated_data)
                        st.success(
                            f"Досье обновлены; {refresh_provider_name} вернул {len(refreshed_analyses)} "
                            f"из {len(refresh_dossiers)} разборов."
                        )
                        st.rerun()

            for rank, favorite_item in enumerate(top_match_candidates, start=1):
                card = favorite_item["card"]
                analysis = card.get("deep_analysis") or {}
                dossier = build_match_dossier(card)
                sstats_data = dossier.get("sstats_context") or {}
                match_title = str(card.get("match_ru") or card.get("match") or "Матч")
                st.markdown(f"### #{rank} · {html.escape(match_title)}")
                st.caption(
                    f"{html.escape(str(card.get('league') or 'Футбол'))} · "
                    f"{html.escape(str(card.get('date') or card.get('date_iso') or ''))} "
                    f"{html.escape(str(card.get('time') or ''))} · "
                    f"история команд {favorite_item['sample_games']} матчей"
                )
                top_cols = st.columns(3)
                top_cols[0].metric(
                    "История хозяева / гости",
                    f"{favorite_item['home_sample_games']} / {favorite_item['away_sample_games']}",
                    "домашние / выездные матчи",
                )
                top_cols[1].metric(
                    f"Независимый вердикт {_provider_display_name(D.get('meta', {}).get('llm_provider', 'Groq (бесплатно, быстро)'))}",
                    (
                        f"{analysis.get('independent_pick')} · "
                        f"{float((analysis.get('probabilities') or {}).get(analysis.get('independent_pick')) or 0):.1f}%"
                        if analysis else "Не получен"
                    ),
                    str(analysis.get("data_quality") or "ожидается") if analysis else "проверь LLM key/лимит",
                )
                top_cols[2].metric(
                    "Контекст SStats",
                    (
                        "Подтверждён"
                        if (
                            sstats_data.get("lineup", {}).get("home_confirmed")
                            and sstats_data.get("lineup", {}).get("away_confirmed")
                            and sstats_data.get("referee", {}).get("confirmed")
                        )
                        else "Частичный / неизвестен"
                    ),
                    "составы · травмы · судья",
                )

                if analysis:
                    st.markdown(f"**Вердикт {_provider_display_name(D.get('meta', {}).get('llm_provider', 'Groq (бесплатно, быстро)'))}**")
                    st.write(analysis.get("match_assessment") or "Общий разбор не вернулся.")
                    p = analysis.get("probabilities") or {}
                    st.caption(
                        "Независимая оценка 1X2 · "
                        f"П1 {float(p.get('П1') or 0):.1f}% · "
                        f"X {float(p.get('X') or 0):.1f}% · "
                        f"П2 {float(p.get('П2') or 0):.1f}%"
                    )
                    market_groups = (
                        ("goal_markets", "Голы и ОЗ", (
                            ("expected_home_goals", "λ хозяев"), ("expected_away_goals", "λ гостей"),
                            ("over_1_5", "ТБ 1.5"), ("over_2_5", "ТБ 2.5"),
                            ("over_3_5", "ТБ 3.5"), ("over_4_5", "ТБ 4.5"),
                            ("btts_yes", "ОЗ Да"),
                        )),
                        ("corner_markets", "Угловые", (
                            ("expected_home_corners", "хозяева"), ("expected_away_corners", "гости"),
                            ("over_8_5", "ТБ 8.5"), ("over_9_5", "ТБ 9.5"),
                            ("over_10_5", "ТБ 10.5"), ("over_11_5", "ТБ 11.5"),
                        )),
                        ("card_markets", "Карточки", (
                            ("expected_home_cards", "хозяева"), ("expected_away_cards", "гости"),
                            ("over_2_5", "ТБ 2.5"), ("over_3_5", "ТБ 3.5"),
                            ("over_4_5", "ТБ 4.5"), ("over_5_5", "ТБ 5.5"),
                        )),
                    )
                    for group_key, group_title, group_fields in market_groups:
                        group_data = analysis.get(group_key)
                        if isinstance(group_data, dict):
                            st.markdown(f"**{group_title} · независимая оценка**")
                            market_cols = st.columns(min(4, len(group_fields)))
                            for field_index, (field_key, field_label) in enumerate(group_fields):
                                value = group_data.get(field_key)
                                if value is None:
                                    continue
                                suffix = "%" if field_key.startswith("over_") or field_key == "btts_yes" else ""
                                market_cols[field_index % len(market_cols)].metric(
                                    field_label, f"{float(value):.1f}{suffix}"
                                )
                        else:
                            st.caption(f"{group_title}: недостаточно проверенной истории для независимой оценки.")
                    assessment_cols = st.columns(2)
                    with assessment_cols[0]:
                        st.markdown("**Составы и влияние**")
                        st.write(analysis.get("lineup_assessment") or "Нет отдельной оценки.")
                        st.markdown("**Травмы и дисквалификации**")
                        st.write(analysis.get("injury_assessment") or "Нет отдельной оценки.")
                        st.markdown("**Форма и история**")
                        st.write(analysis.get("form_assessment") or "Нет отдельной оценки.")
                    with assessment_cols[1]:
                        st.markdown("**Судья**")
                        st.write(analysis.get("referee_assessment") or "Нет отдельной оценки.")
                        st.markdown("**Риски**")
                        st.write(analysis.get("risks") or "Дополнительные риски не указаны.")
                    model_pick_code = max(
                        (("П1", float(card.get("p1") or 0.0)),
                         ("X", float(card.get("px") or 0.0)),
                         ("П2", float(card.get("p2") or 0.0))),
                        key=lambda pair: pair[1],
                    )[0]
                    if analysis.get("independent_pick") != model_pick_code:
                        st.warning(
                            f"Расхождение моделей: основная модель — {model_pick_code}, "
                            f"независимый ИИ — {analysis.get('independent_pick')}."
                        )
                    else:
                        st.success(
                            f"Независимый выбор ИИ совпал с основной моделью: {model_pick_code}."
                        )
                    st.caption(
                        "Качество данных ограничено полнотой и размером переданных источников: "
                        f"{analysis.get('data_quality', 'low')}."
                    )
                else:
                    st.warning(
                        f"Подбор прошёл, но подробный разбор {_provider_display_name(D.get('meta', {}).get('llm_provider', 'Groq (бесплатно, быстро)'))} не получен. "
                        "Проверь ключ выбранного провайдера, дневной лимит и журнал диагностики."
                    )

                with st.expander("Проверенные исходные данные SStats", expanded=False):
                    st.caption(
                        f"SStats game ID: {sstats_data.get('sstats_game_id') or 'нет'} · "
                        f"Flashscore event ID: {'найден' if sstats_data.get('event_id_resolved') else 'не найден'} · "
                        f"начало: {sstats_data.get('kickoff_at') or 'не указано'}"
                    )
                    xg_block = dossier.get("xg") or {}
                    provider_xg = xg_block.get("provider_history") or {}
                    proxy_xg = xg_block.get("proxy_history") or {}
                    st.caption(
                        "Исторический xG/xGA до даты матча: "
                        f"provider {provider_xg.get('home_xg_samples', 0)}/"
                        f"{provider_xg.get('away_xg_samples', 0)} наблюдений; "
                        f"proxy {proxy_xg.get('home_xg_samples', 0)}/"
                        f"{proxy_xg.get('away_xg_samples', 0)} наблюдений. "
                        f"Источник xG основной модели: {xg_block.get('primary_model_source', 'не использован')}."
                    )
                    st.caption(
                        "Наблюдения: "
                        f"досье {sstats_data.get('observed_at') or 'время неизвестно'} · "
                        f"составы {sstats_data.get('lineup_observed_at') or 'не подтверждены'} · "
                        f"травмы {sstats_data.get('injuries_observed_at') or 'нет подтверждения'} · "
                        f"судья {sstats_data.get('referee_observed_at') or 'нет подтверждения'}"
                    )
                    lineups = sstats_data.get("lineup") or {}
                    source_cols = st.columns(2)
                    with source_cols[0]:
                        st.markdown("**Стартовый XI хозяев**")
                        home_players = lineups.get("home_players") or []
                        if lineups.get("home_confirmed"):
                            st.write(", ".join(home_players) if home_players else "Подтверждённый XI без имён в ответе.")
                        else:
                            st.warning("Стартовый XI хозяев не подтверждён.")
                        st.markdown("**Стартовый XI гостей**")
                        away_players = lineups.get("away_players") or []
                        if lineups.get("away_confirmed"):
                            st.write(", ".join(away_players) if away_players else "Подтверждённый XI без имён в ответе.")
                        else:
                            st.warning("Стартовый XI гостей не подтверждён.")
                    with source_cols[1]:
                        st.markdown("**Судья**")
                        referee = sstats_data.get("referee") or {}
                        if referee.get("confirmed") and referee.get("name"):
                            st.write(referee.get("name"))
                            ref_stats = referee.get("historical_cards") or {}
                            if ref_stats.get("available"):
                                st.caption(
                                    f"История карточек: n={ref_stats.get('n', 0)} · "
                                    f"среднее {ref_stats.get('mean', 0):.2f} · "
                                    f"дисперсия {ref_stats.get('variance', 0):.2f}"
                                )
                        else:
                            st.warning("Назначение судьи не подтверждено.")
                        st.markdown("**Травмы / дисквалификации**")
                        if sstats_data.get("injury_feed_confirmed"):
                            injuries = sstats_data.get("injuries_listed") or []
                            if injuries:
                                for injury in injuries:
                                    st.caption(
                                        f"{injury.get('team', 'Команда не указана')}: "
                                        f"{injury.get('player', 'Игрок не указан')} · "
                                        f"{injury.get('reason') or injury.get('status') or 'причина не указана'}"
                                    )
                            else:
                                st.info(
                                    "SStats вернул подтверждённый пустой список. "
                                    "Это не доказывает отсутствие всех травм."
                                )
                        else:
                            st.warning("Фид травм не подтверждён.")
                st.divider()

        with market_tabs[0]:
            st.subheader("Лучшие события по всем рынкам")
            st.caption(
                "Здесь сравниваются отдельные исходы, а не только фавориты 1X2: "
                "победа/ничья, тоталы, обе забьют, угловые, карточки и азиатские форы. "
                "Один матч может дать несколько разных рыночных событий."
            )

            coverage = build_market_coverage(market_pool)
            coverage_table = [{
                "Рынок": item["market"],
                "Матчи с прогнозом": item["forecast_matches"],
                "Есть котировка": item["quoted_matches"],
                "Value подтверждён": item["value_matches"],
                "Пока только прогноз": item["forecast_only_matches"],
                "Сильный прогноз": item["strong_matches"],
            } for item in coverage]
            st.markdown("**Покрытие рынков в текущем скане**")
            st.dataframe(coverage_table, width="stretch", hide_index=True)

            events = build_best_market_events(market_pool, limit=50)
            if not events:
                st.info(
                    "Пока нет событий с корректной вероятностью. Проверь историю команд, "
                    "полноту линий SStats и наличие статистики для выбранных рынков."
                )
            else:
                verified_count = sum(bool(item["value_verified"]) for item in events)
                priced_count = sum(item.get("odds") is not None for item in events)
                strong_count = sum(bool(item["strong_forecast"]) for item in events)
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Событий в рейтинге", len(events))
                k2.metric("Есть коэффициент", priced_count)
                k3.metric("Подтверждённый Value", verified_count)
                k4.metric("Сильные прогнозы ≥65%", strong_count)

                event_filter = st.selectbox(
                    "Показывать",
                    [
                        "Сильные прогнозы ≥65% и кэф ≥1.70",
                        "Все рынки и события",
                        "Только подтверждённый Value",
                        "Сильные прогнозы ≥65%",
                        "Только события с коэффициентами",
                    ],
                    key="scanner_all_market_event_filter",
                )
                shown_events = list(events)
                if event_filter == "Сильные прогнозы ≥65% и кэф ≥1.70":
                    shown_events = [
                        item for item in shown_events
                        if item.get("strong_forecast")
                        and item.get("odds") is not None
                        and float(item.get("odds") or 0.0) >= MIN_BET_ODDS
                        and "sstats" in str(item.get("source") or "").casefold()
                        and bool(str(item.get("bookmaker") or "").strip())
                    ]
                elif event_filter == "Только подтверждённый Value":
                    shown_events = [item for item in shown_events if item["value_verified"]]
                elif event_filter == "Сильные прогнозы ≥65%":
                    shown_events = [item for item in shown_events if item["strong_forecast"]]
                elif event_filter == "Только события с коэффициентами":
                    shown_events = [item for item in shown_events if item.get("odds") is not None]

                event_table = []
                for index, item in enumerate(shown_events, start=1):
                    odds_text = f"{item['odds']:.2f}" if item.get("odds") is not None else "—"
                    fair_text = f"{item['fair_odds']:.2f}" if item.get("fair_odds") is not None else "—"
                    ev_text = (
                        f"{item['model_ev']:+.1%}"
                        if item.get("model_ev") is not None else "—"
                    )
                    edge_text = (
                        f"{item['edge']:+.1%}"
                        if item.get("edge") is not None else "не подтверждён"
                    )
                    event_table.append({
                        "№": index,
                        "Матч": item["match"],
                        "Лига": item["league"],
                        "Начало": " ".join(v for v in (item["date"], item["time"]) if v),
                        "Рынок": item["family_label"],
                        "Лучший исход": item["pick"],
                        "Вероятность": f"{item['probability']:.1%}",
                        "База вероятности": (
                            "полный + половинный выигрыш"
                            if item["family"] == "handicap" else "вероятность выбранного исхода"
                        ),
                        "Разбивка форы": item.get("outcome_detail") or "—",
                        "Коэффициент": odds_text,
                        "Букмекер": item.get("bookmaker") or "не указан",
                        "Источник": item.get("source") or "не указан",
                        "Справедливый кэф": fair_text,
                        "Модельный EV*": ev_text,
                        "Edge": edge_text,
                        "Статус": item["status_label"],
                    })
                if event_table:
                    st.dataframe(
                        event_table,
                        width="stretch",
                        hide_index=True,
                        height=min(620, 42 + 36 * len(event_table)),
                    )
                else:
                    st.info("По выбранному фильтру подходящих событий нет.")

                st.caption(
                    "*Вероятность ≥65% — оценка модели на доступной истории, а не доказанная фактическая проходимость. "
                    "Модельный EV для непроверенных рынков — предварительная оценка по вероятности модели и фактическому коэффициенту. "
                    "Это не подтверждённый Edge и не рекомендация ставить. "
                    "Edge/EV/Келли в статусе «ПРОВЕРЕННЫЙ VALUE» означают прохождение отдельной рыночной проверки."
                )
                if events:
                    with st.expander("Почему события имеют разные статусы", expanded=False):
                        st.markdown(
                            "**ПРОВЕРЕННЫЙ VALUE** — рынок прошёл собственный benchmark, "
                            "полную линию одного букмекера, Edge и EV от 5%, положительный Kelly.\n\n"
                            "**ПРОГНОЗ + КОЭФФИЦИЕНТ · НЕ VALUE** — модель и цена есть, "
                            "но независимая проверка именно этого рынка ещё не разрешает ставку.\n\n"
                            "**ФОРА · ИССЛЕДОВАТЕЛЬСКАЯ ОЦЕНКА** — расчёт по полной матрице счёта "
                            "учитывает win/half-win/push/half-loss/loss; отдельный out-of-sample benchmark "
                            "для азиатских фор пока не подключён.\n\n"
                            "**ПРОГНОЗ · НЕТ КОЭФФИЦИЕНТА** — модельная оценка есть, "
                            "но котировки именно на этот исход нет."
                        )

        st.caption(
            "✅ Общий shortlist: " + str(len(cards_view))
            + " матчей · отдельные рынки: весь пул из " + str(len(market_pool)) + " матчей"
        )
        return

    # ==================== СКАН ====================
    st.session_state["_scan_in_progress"] = True
    explicit_history_refresh = bool(
        st.session_state.pop("_scanner_force_history_refresh_next", False)
    )
    explicit_team_history_refresh = bool(
        st.session_state.pop("_scanner_team_history_refresh_next", False)
    )
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
        scan_run_id = uuid.uuid4().hex
        today_date = scan_now.date()
        scan_ledger = ScanLedger(DB_FILE)
        scan_ledger.start_run(
            scan_run_id,
            started_at=scan_now.isoformat(),
            timezone_name=getattr(scan_timezone, "key", "UTC"),
            horizon_days=int(days),
            date_from=today_date.isoformat(),
            date_to_inclusive=(today_date + timedelta(days=int(days) - 1)).isoformat(),
        )
        st.session_state["_active_scan_run_id"] = scan_run_id
        saved_fixture_snapshots = 0
        last_remote_checkpoint = 0
        logs = [
            f"📦 Сборка {APP_BUILD_ID} · " + (
                "ручной запуск полного market benchmark."
                if explicit_history_refresh else
                "обновление истории команд без market benchmark."
                if explicit_team_history_refresh else
                "обычный скан: полный market benchmark пропущен."
            )
        ]

        token = _runtime_secret("fdorg_token_input", "FDORG_TOKEN", "fdorg_token")
        sstats_key = _runtime_secret("sstats_api_key_input", "SSTATS_API_KEY", "sstats_api_key")
        footballapi_key = _runtime_secret(
            "footballapi_key_input", "FOOTBALLAPI_KEY", "footballapi_key"
        )
        fixture_sources = []
        tsdb_rows = []
        espn_rows = []
        fdorg_rows = []
        sstats_rows = []
        source_counts = {
            "TheSportsDB": 0,
            "TheSportsDB visuals": 0,
            "ESPN": 0,
            "football-data.org": 0,
            "SStats": 0,
        }

        # SStats is the primary match-and-odds source. Do not call several
        # other schedule APIs for the same horizon when SStats already returned
        # fixtures: keep those providers as failover only.
        if sstats_key:
            try:
                sstats_rows = sstats_upcoming_matches(days, sstats_key, logs)
                source_counts["SStats"] = len(sstats_rows)
            except Exception as exc:
                logs.append(f"⚠️ SStats fixtures/odds: {type(exc).__name__}")

        sstats_is_primary = bool(sstats_rows)
        if sstats_is_primary:
            # SStats remains authoritative for fixture IDs, kickoff and all
            # bookmaker prices. TheSportsDB is still queried as a metadata-only
            # source so team/league badges and venue photos do not disappear.
            # Only visual fields are overlaid onto exact SStats fixtures; extra
            # TSDB fixtures and all secondary-source odds are never admitted.
            enriched_sstats_rows = sstats_rows
            visual_metadata_rows = []
            try:
                tsdb_rows = tsdb_matches(days, logs)
                source_counts["TheSportsDB visuals"] = len(tsdb_rows)
                visual_metadata_rows.extend(tsdb_rows)
            except Exception as exc:
                logs.append(
                    f"ℹ️ TheSportsDB visual metadata unavailable: {type(exc).__name__}."
                )
            try:
                espn_rows = espn_matches(days, logs)
                source_counts["ESPN"] = len(espn_rows)
                visual_metadata_rows.extend(espn_rows)
            except Exception as exc:
                logs.append(f"ℹ️ ESPN visual metadata unavailable: {type(exc).__name__}.")
            if token:
                try:
                    fdorg_rows = fdorg_matches(days, token, logs)
                    source_counts["football-data.org"] = len(fdorg_rows)
                    visual_metadata_rows.extend(fdorg_rows)
                except Exception as exc:
                    logs.append(
                        f"ℹ️ football-data.org visual metadata unavailable: {type(exc).__name__}."
                    )
            try:
                enriched_sstats_rows, visual_stats = enrich_fixture_visual_metadata(
                    sstats_rows, visual_metadata_rows,
                )
                logs.append(
                    "🖼 Визуальные метаданные без подмены расписания/котировок: "
                    f"совпадений {visual_stats['fixtures_matched']}/"
                    f"{visual_stats['primary_fixtures']}, "
                    f"добавлено изображений и полей {visual_stats['assets_added']}, "
                    f"совпадений по сдвигу даты ±1 день {visual_stats.get('date_fallback_matches', 0)}, "
                    f"неоднозначных пар {visual_stats.get('ambiguous_team_pair_matches', 0)}, "
                    f"матчей с визуальными данными "
                    f"{visual_stats['fixtures_with_assets']}; "
                    f"источники TSDB={len(tsdb_rows)}, ESPN={len(espn_rows)}, "
                    f"football-data.org={len(fdorg_rows)}."
                )
            except Exception as exc:
                enriched_sstats_rows = sstats_rows
                logs.append(
                    f"ℹ️ Visual metadata merge unavailable: {type(exc).__name__}; "
                    "SStats fixtures continue unchanged."
                )
            # SStats remains the only source of live bookmaker prices. Other
            # schedule providers supplement fixtures that SStats did not return;
            # enforce_sstats_odds_only() strips their price fields below.
            fixture_sources.extend(enriched_sstats_rows)
            fixture_sources.extend(fdorg_rows)
            fixture_sources.extend(tsdb_rows)
            fixture_sources.extend(espn_rows)
            logs.append(
                f"✅ SStats — основной источник котировок и расписания: "
                f"{len(sstats_rows)} матчей. Дополнение расписания: "
                f"football-data.org={len(fdorg_rows)}, TheSportsDB={len(tsdb_rows)}, "
                f"ESPN={len(espn_rows)}. Резервные источники добавляют недостающие "
                "матчи; их котировки удаляются."
            )
        else:
            if sstats_key:
                logs.append(
                    "⚠️ SStats не вернул матчи; включаю резервные источники только для расписания."
                )
            else:
                logs.append(
                    "ℹ️ SSTATS_API_KEY не настроен — SStats пропущен; "
                    "включаю резервные источники только для расписания."
                )
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
            fixture_sources.extend(fdorg_rows)
            fixture_sources.extend(tsdb_rows)
            fixture_sources.extend(espn_rows)

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
        rows = enforce_sstats_odds_only(rows)

        # Team-name search is restricted on TheSportsDB's free tier.
        # Use the exact provider team IDs attached to matching event records.
        logo_by_team_id = {}
        logo_lookup_budget = 24
        logo_lookups = 0
        logo_fields_added = 0
        for fixture_row in rows:
            for badge_field, id_field in (
                ("home_badge", "home_team_tsdb_id"),
                ("away_badge", "away_team_tsdb_id"),
            ):
                existing_badge = str(fixture_row.get(badge_field) or "").strip()
                if existing_badge.startswith(("https://", "http://", "//")):
                    continue
                team_id = str(fixture_row.get(id_field) or "").strip()
                if not team_id or not team_id.isdigit():
                    continue
                if team_id not in logo_by_team_id:
                    if logo_lookups >= logo_lookup_budget:
                        continue
                    logo_by_team_id[team_id] = team_badge_url_by_id(team_id, logs=logs)
                    logo_lookups += 1
                badge_url = logo_by_team_id.get(team_id)
                if badge_url:
                    fixture_row[badge_field] = badge_url
                    logo_fields_added += 1
        if logo_lookups:
            logs.append(
                "🛡 Эмблемы команд — TheSportsDB lookupteam по ID: "
                f"проверено {logo_lookups} ID, добавлено изображений {logo_fields_added}; "
                "результаты кэшируются на 30 дней."
            )
        else:
            logs.append(
                "ℹ️ Эмблемы команд: в текущих визуальных данных нет ID TheSportsDB; "
                "добавление эмблем пропущено без поиска по названию."
            )

        # SStats rows carry their own game IDs and bookmaker prices. When SStats
        # is unavailable, public feeds are schedule-only and every price-like
        # field is stripped before the scanner can evaluate a market.

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
        if sstats_key:
            try:
                context_store = ContextStore(DB_FILE)
            except Exception as exc:
                logs.append(f"⚠️ Context DB недоступна ({type(exc).__name__}); live-скан продолжается.")
            try:
                context_by_fixture, context_logs = sstats_context_enrich(
                    rows, sstats_key,
                    date_from=today_date.isoformat(),
                    date_to=(today_date + timedelta(days=days - 1)).isoformat(),
                    now=scan_now,
                    timezone_offset=int(round(scan_now.utcoffset().total_seconds() / 3600)) if scan_now.utcoffset() else 0,
                )
                logs.extend(context_logs)
            except Exception as exc:
                logs.append(f"⚠️ SStats context: {type(exc).__name__}; Base-прогнозы сохраняются.")
        else:
            logs.append(
                "🔒 SStats не настроен: контекст недоступен; Strong/Value остаются заблокированы."
            )

        scan_divisions = list(dict.fromkeys(
            str(item.get("Div") or "").strip()
            for item in rows
            if isinstance(item, dict) and str(item.get("Div") or "").strip()
        ))
        # A fitted model is not ready for this scan merely because it exists:
        # every SStats competition in the slate must have actual historical
        # league observations. Otherwise the Poisson model can emit the same
        # prior-only 42/29/29 distribution for unrelated teams indefinitely.
        sstats_history_divisions = list(dict.fromkeys(
            str(item.get("Div") or "").strip()
            for item in rows
            if (
                isinstance(item, dict)
                and item.get("sstats_game_id") not in (None, "")
                and str(item.get("Div") or "").strip()
            )
        ))
        model_cache_ready = has_cached_engine(
            matrix_n,
            footballapi_key=footballapi_key,
            custom_historical_rows=custom_historical_rows,
            extra_divisions=scan_divisions,
            required_divisions=sstats_history_divisions if sstats_key else None,
        )
        auto_team_history_refresh = bool(
            sstats_key and sstats_history_divisions and not model_cache_ready
        )
        refresh_history = bool(
            explicit_history_refresh or explicit_team_history_refresh
            or auto_team_history_refresh or not model_cache_ready
        )
        if explicit_history_refresh or explicit_team_history_refresh or auto_team_history_refresh:
            if explicit_history_refresh:
                logs.append(
                    "🔄 Запрошено полное обновление архивов и market benchmark: "
                    "обновляю сезонную историю, SStats, xG и затем пересчитаю walk-forward."
                )
                update_loader("📈 Обновляю исторические xG-данные...", 0.16, logs)
                footballapi_xg_rows = footballapi_xg_history(
                    footballapi_key, logs, force_refresh=True,
                )
            elif explicit_team_history_refresh:
                logs.append(
                    "📚 Запрошено адресное обновление истории команд: загружаю FT-результаты "
                    "только соревнований из текущего расписания. Walk-forward market benchmark "
                    "не запускается; Value-гейты останутся fail-closed."
                )
                footballapi_xg_rows = None
            else:
                logs.append(
                    "⚙️ Автообновление истории: у обученной модели нет результатов по одному "
                    "или нескольким SStats-соревнованиям текущего расписания. Загружаю их "
                    "историю до построения прогнозов; Value-допуск останется закрытым."
                )
                footballapi_xg_rows = None

            update_loader("📚 Обновляю архивы завершённых матчей SStats текущих лиг...", 0.18, logs)
            sstats_history_rows = []
            if sstats_key and rows:
                try:
                    # sstats_historical_matches_for_fixtures resolves missing IDs
                    # from the exact Leagues catalogue before applying this limit.
                    # Do not cap it by the IDs present in the initial fixture payload.
                    history_league_limit = _runtime_int(
                        "sstats_history_max_leagues",
                        "SSTATS_HISTORY_MAX_LEAGUES",
                        default=30,
                        minimum=1,
                    )
                    sstats_history_rows = sstats_historical_matches_for_fixtures(
                        rows, sstats_key, years_back=5,
                        max_leagues=history_league_limit, logs=logs,
                        force_refresh=explicit_history_refresh,
                    )
                except Exception as exc:
                    logs.append(
                        f"⚠️ SStats historical archive: {type(exc).__name__}; "
                        "без проверенных результатов дополнительная история не добавлена."
                    )

            if sstats_history_rows:
                try:
                    merged_history, merge_stats = _merge_sstats_historical_rows(
                        custom_historical_rows, sstats_history_rows,
                    )
                    custom_historical_rows = merged_history
                    st.session_state["_custom_historical_match_rows"] = merged_history
                    if merge_stats["added"] or merge_stats["enriched"]:
                        saved_history = user_datasets.save_user_dataset(
                            "custom_historical_match_rows", merged_history,
                        )
                        st.session_state["_custom_historical_match_sha256"] = saved_history["sha256"]
                        logs.append(
                            "💾 Постоянный архив SStats FT сохранён в SQLite: "
                            f"{saved_history['row_count']} строк всего, "
                            f"новых {merge_stats['added']}, обогащено {merge_stats['enriched']}, "
                            f"дубликатов {merge_stats['duplicates']}, конфликтов счёта "
                            f"{merge_stats['conflicts']}. Данные войдут в зашифрованный SQLite backup."
                        )
                    else:
                        logs.append(
                            "ℹ️ SStats FT история уже полностью есть в постоянном архиве; "
                            f"новых строк нет, дубликатов {merge_stats['duplicates']}."
                        )
                except Exception as persist_exc:
                    logs.append(
                        "⚠️ SStats FT история доступна для текущего обучения, но не сохранена "
                        f"в SQLite ({type(persist_exc).__name__}: {str(persist_exc)[:140]}). "
                        "При замене контейнера эти новые строки придётся загрузить повторно."
                    )
            elif explicit_team_history_refresh:
                logs.append(
                    "⚠️ Адресная история SStats не вернула завершённых матчей для текущих лиг. "
                    "Проверьте строки SStats LeagueId в журнале; прогнозы останутся предварительными."
                )
        elif refresh_history:
            # Cold start still needs the core seasonal match archive and any
            # previously persisted SStats/custom rows. Never crawl the entire
            # provider history or run market validation implicitly.
            footballapi_xg_rows = None
            logs.append(
                "🧠 Первичный запуск: обучаю базовую модель на сезонных CSV и "
                f"сохранённом архиве ({len(custom_historical_rows)} дополнительных матчей). "
                "Полный массовый сбор SStats/xG и walk-forward benchmark отложены; "
                "Value-гейты останутся закрытыми."
            )
        else:
            footballapi_xg_rows = None
            logs.append(
                "⚡ Исторические источники пропущены: использую готовую обученную модель. "
                "SStats live-составы, травмы, расписание и текущие коэффициенты "
                "по-прежнему обновляются для каждого скана."
            )

        update_loader("🧠 Загрузка обученной модели...", 0.20, logs)
        engine = _load_engine(
            matrix_n, logs, update_loader,
            footballapi_key=footballapi_key,
            footballapi_xg_rows=footballapi_xg_rows,
            custom_historical_rows=custom_historical_rows,
            refresh_history=refresh_history,
            force_source_refresh=explicit_history_refresh,
            extra_divisions=scan_divisions,
            run_market_validation=explicit_history_refresh,
        )
        from model.holdout import load_fixed_holdout_gate_status
        fixed_holdout_gate = load_fixed_holdout_gate_status()
        if fixed_holdout_gate["passed"]:
            logs.append(
                "✅ Fixed Holdout Model Lab v1: PASS — вторичные рынки могут "
                "использовать только собственные отдельные benchmark/value-гейты."
            )
        else:
            logs.append(
                "🔒 Fixed Holdout Model Lab v1 ещё не PASS: O/U остаётся forecast-only; "
                "автопортфель ограничен рынком 1X2."
            )

        update_loader("🧠 Анализ матчей...", 0.80, logs)
        scan_learning_service = None
        try:
            scan_forecast_ledger = ForecastLedger(DB_FILE)
            scan_learning_service = ForecastLearningService(
                scan_forecast_ledger,
                OnlineErrorLearner(
                    scan_forecast_ledger, model_version_prefix="main-model-",
                ),
                enabled=False,
                min_history=100,
                auto_promote=True,
                promotion_min_samples=200,
                promotion_min_relative_improvement=0.01,
            )
        except Exception as learning_exc:
            logs.append(
                f"ℹ️ Онлайн-калибровка недоступна ({type(learning_exc).__name__}); "
                "сканирование продолжится с исходными вероятностями."
            )
        cards = []
        matches_with_best = 0
        xg_used_count = 0
        # Games/list compact 1X2 can be empty even when Odds/{gameId} has a
        # complete bookmaker line. Fetch details for the whole normal daily slate
        # (107-140 matches in recent scans), with an environment/Secrets override.
        sstats_detail_calls = 0
        SSTATS_DETAIL_MAX_PER_SCAN = _runtime_int(
            "sstats_detail_max_per_scan",
            "SSTATS_DETAIL_MAX_PER_SCAN",
            default=150,
            minimum=1,
        )

        funnel_counts = {"model": 0, "market_req": 0, "market_got": 0,
                          "bet": 0, "watch": 0, "skip": 0,
                          "model_only": 0, "estimated": 0, "real": 0,
                          "prior_only": 0, "league_only": 0,
                          "partial_history": 0, "team_supported": 0}
        support_by_division = {}
        support_examples = []
        logs.append(
            "🎯 Источник букмекерских коэффициентов: только SStats Games/list и Odds/{gameId}; "
            "коэффициенты других провайдеров и импортированные файлы в скане не используются."
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

            # Keep the untouched base probabilities for the audit ledger.
            # The calibrator only changes the served probability vector, and
            # only after the chronological shadow gate has passed.
            raw_model_probabilities = {
                "H": float(P["p1"]), "D": float(P["x"]), "A": float(P["p2"]),
            }
            P_decision = dict(P)
            model_support_status = str(P.get("model_support_status") or "unknown")
            if model_support_status == "prior_only":
                funnel_counts["prior_only"] = funnel_counts.get("prior_only", 0) + 1
            elif model_support_status == "league_only":
                funnel_counts["league_only"] = funnel_counts.get("league_only", 0) + 1
            elif model_support_status == "team_supported":
                funnel_counts["team_supported"] = funnel_counts.get("team_supported", 0) + 1
            else:
                funnel_counts["partial_history"] = funnel_counts.get("partial_history", 0) + 1
            # A generic 1.5/1.2 goals prior plus a fixed home Elo bonus is not a
            # match-specific forecast. Keep it visible as a coverage diagnostic,
            # but do not let online correction, market selection or the ledger
            # turn it into an apparently informed P1/X/P2 pick.
            model_bet_history_ready = bool(
                model_support_status == "team_supported"
                and int(P.get("league_history_matches") or 0) >= 20
                and int(P.get("home_venue_history_matches") or 0) >= MIN_FAVORITE_HISTORY
                and int(P.get("away_venue_history_matches") or 0) >= MIN_FAVORITE_HISTORY
            )
            division_support = support_by_division.setdefault(
                str(lg), {"league": str(r.get("League") or lg), "fixtures": 0,
                          "prior_only": 0, "league_only": 0,
                          "partial_history": 0, "team_supported": 0},
            )
            division_support["fixtures"] += 1
            division_support[model_support_status if model_support_status in {
                "prior_only", "league_only", "partial_history", "team_supported"
            } else "partial_history"] += 1
            if not model_bet_history_ready and len(support_examples) < 20:
                support_examples.append({
                    "division": str(lg),
                    "league": str(r.get("League") or lg),
                    "home": h_en,
                    "away": a_en,
                    "status": model_support_status,
                    "league_matches": int(P.get("league_history_matches") or 0),
                    "home_matches": int(P.get("home_team_history_matches") or 0),
                    "away_matches": int(P.get("away_team_history_matches") or 0),
                    "home_venue": int(P.get("home_venue_history_matches") or 0),
                    "away_venue": int(P.get("away_venue_history_matches") or 0),
                })
            learning_preview = None
            if scan_learning_service is not None and model_support_status == "team_supported":
                try:
                    learning_preview = scan_learning_service.preview(
                        raw_model_probabilities, as_of=scan_now,
                        league=str(r.get("League") or lg),
                    )
                    served = learning_preview["probabilities"]
                    P_decision["p1"] = float(served["H"])
                    P_decision["x"] = float(served["D"])
                    P_decision["p2"] = float(served["A"])
                except Exception as learning_exc:
                    logs.append(
                        f"ℹ️ Калибровка {h_ru} — {a_ru} пропущена "
                        f"({type(learning_exc).__name__}); использованы исходные вероятности."
                    )

            fh = engine.form_str(h_en, lg)
            fa = engine.form_str(a_en, lg)
            verdict, rows_, _ = build_verdict(
                P_decision, min_prob, D["bank"], kelly_frac,
                h_ru, a_ru, fh, fa, P_decision.get("h2h_n", 0))

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
                        data_source="sstats",
                    )
                except (TypeError, ValueError, OSError) as exc:
                    logs.append(f"⚠️ Context snapshot не сохранён: {type(exc).__name__}")
            context_quality = assess_context_quality(api_context)
            context_forecasts = context_probabilities(
                {"H": P_decision["p1"], "D": P_decision["x"], "A": P_decision["p2"]},
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
            one_x_two_diagnostics = dict(r.get("one_x_two_diagnostics") or {})
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
                            detailed_bookmakers = detailed.get("bookmaker_by_pick") or {}
                            detailed_diagnostic = detailed.get("one_x_two_diagnostics") or {}
                            # Never maximize each outcome independently: that would
                            # stitch prices from different books into a fake 1X2 line.
                            # Replace compact quotes only with one confirmed named line.
                            if (
                                detailed_diagnostic.get("status") == "confirmed"
                                and all(key in detailed_1x2 for key in ("П1", "X", "П2"))
                                and all(detailed_bookmakers.get(key) for key in ("П1", "X", "П2"))
                                and len({
                                    str(detailed_bookmakers.get(key)).strip().casefold()
                                    for key in ("П1", "X", "П2")
                                }) == 1
                            ):
                                market_odds = {
                                    key: float(detailed_1x2[key])
                                    for key in ("П1", "X", "П2")
                                }
                                selected_book = str(detailed_bookmakers["П1"]).strip()
                                bookmaker_by_pick = {
                                    key: selected_book for key in ("П1", "X", "П2")
                                }
                                odds_source = "sstats"
                                one_x_two_diagnostics = detailed_diagnostic
                                candidates = []
                            else:
                                status_rank = {
                                    "missing": 0, "partial": 1, "unverified_bookmaker": 2,
                                    "invalid_overround": 3, "confirmed": 4,
                                }
                                if status_rank.get(detailed_diagnostic.get("status"), 0) > status_rank.get(
                                    one_x_two_diagnostics.get("status"), 0
                                ):
                                    one_x_two_diagnostics = detailed_diagnostic
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

                    # If no provider has a verified line, leave odds empty.

                    for source_name, candidate in candidates:
                        # Defensive guard: no adapter other than SStats may supply
                        # bookmaker prices to the automatic scanner.
                        if str(source_name or "").strip().casefold() != "sstats":
                            continue
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
                pick: bool(winner_market_validation and model_bet_history_ready)
                for pick in ("П1", "X", "П2")
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
            # Preserve the exact inputs to today's Value/Kelly decision so a
            # later priority SStats XI/referee refresh can re-run the SAME gates.
            # Browser Search / LLM text is never part of this payload.
            initial_context_qualified = bool(
                context_quality["strong_value_eligible"] and model_bet_history_ready
            )
            value_recheck_inputs = None
            if not initial_context_qualified:
                base_value_verdict = copy.deepcopy(verdict)
                # The initial context block is superseded only if a fresh,
                # confirmed SStats XI + referee snapshot passes the same gates.
                # Do not carry its stale reason into a subsequently approved pick.
                base_value_verdict.pop("strong_blocked", None)
                base_value_verdict["reasons"] = [
                    reason for reason in (base_value_verdict.get("reasons") or [])
                    if not str(reason).startswith("Strong/Value заблокированы:")
                ]
                value_recheck_inputs = {
                    "base_verdict": base_value_verdict,
                    "candidate_rows": copy.deepcopy(candidate_rows),
                    "selection_odds": dict(selection_odds),
                    "bank": float(D["bank"]),
                    "kelly_frac": float(kelly_frac),
                    "min_prob": float(min_prob),
                    "min_edge": max(float(min_edge), 0.05),
                    "min_ev": 0.05,
                    "bookmaker_by_pick": dict(bookmaker_by_pick),
                    "market_validation_by_pick": dict(market_validation_by_pick),
                    "market_source_by_pick": dict(market_source_by_pick),
                    "model_bet_history_ready": bool(model_bet_history_ready),
                }
            # Evaluate each family independently before computing the one-best
            # cross-market pick. A strong 1X2 result must not suppress a goals,
            # BTTS, corners or cards signal on this fixture.
            best_by_market = best_value_markets(
                verdict, candidate_rows, selection_odds, D["bank"], kelly_frac,
                min_prob, min_edge=max(min_edge, 0.05), min_ev=0.05,
                bookmaker_by_pick=bookmaker_by_pick,
                market_validation_by_pick=market_validation_by_pick,
                context_qualified=bool(
                    context_quality["strong_value_eligible"] and model_bet_history_ready
                ),
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
                    context_qualified=bool(
                        context_quality["strong_value_eligible"] and model_bet_history_ready
                    ),
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
                if not model_bet_history_ready:
                    no_best_reason = (
                        "нет достаточной истории модели: нужны ≥20 матчей лиги и "
                        f"≥{MIN_FAVORITE_HISTORY} домашних игр хозяев / гостевых игр гостей; "
                        "общий prior не допускается к ставкам"
                    )
                elif not context_quality.get("strong_value_eligible"):
                    no_best_reason = (
                        "контекстный фильтр Strong/Value: нет подтверждённого стартового XI "
                        "обеих команд и/или назначенного судьи; ставки заблокированы"
                    )
                else:
                    no_best_reason = (
                        "ни один доступный исход не прошёл проверку для конкретной линии, "
                        "полную линию одного букмекера, минимальную вероятность, "
                        "Edge >= 5%, EV >= 5% и положительный Kelly"
                    )
                logs.append(
                    f"❌ NO BEST: {h_ru} — {a_ru} | "
                    f"is_action={verdict.get('is_action')} | reason={no_best_reason}"
                )

            cards.append({
                "div": lg,
                "league": r.get("League") or DIV_NAMES.get(lg, "Лига"),
                "home_team": h_en,
                "away_team": a_en,
                "match": f"{h_en} vs {a_en}",
                "match_ru": f"{h_ru} — {a_ru}",
                "date": d.strftime("%d.%m") +
                        (f" {kickoff_time_text}" if kickoff_time_text else ""),
                "verdict": verdict, "best": best,
                "best_by_market": best_by_market,
                "model_bet_history_ready": bool(model_bet_history_ready),
                "_value_recheck": value_recheck_inputs,
                "games": P["games"], "fh": fh, "fa": fa,
                "model_support_status": model_support_status,
                "league_history_matches": int(P.get("league_history_matches") or 0),
                "home_team_history_matches": int(P.get("home_team_history_matches") or 0),
                "away_team_history_matches": int(P.get("away_team_history_matches") or 0),
                "home_venue_history_matches": int(P.get("home_venue_history_matches") or 0),
                "away_venue_history_matches": int(P.get("away_venue_history_matches") or 0),
                # Mark a forecast calibrated only when chronological learning
                # actually promoted the correction into the served probabilities.
                # A shadow-only adjustment is not a calibrated betting signal.
                "probability_calibrated": bool(
                    isinstance(learning_preview, dict)
                    and learning_preview.get("correction_enabled")
                ),
                "calibration_matches": int(
                    (learning_preview or {}).get("history_n") or 0
                ),
                "calibration_status": str(
                    ((learning_preview or {}).get("learning_report") or {}).get("gate_reason")
                    or "not_available"
                ),
                "fixture_id": r.get("fixture_id"),
                "sstats_game_id": r.get("sstats_game_id"),
                "historical_stats": historical_stats,
                "winner_market_validation": bool(getattr(engine, "winner_market_validation", False)),
                "winner_market_validation_report": getattr(engine, "winner_market_validation_report", {}),
                "total_market_validation": bool(getattr(engine, "total_market_validation", False)),
                "fixed_holdout_1x2_passed": bool(fixed_holdout_gate["passed"]),
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
                    **historical_features(
                        engine, h_en, a_en, league=lg, match_date=d,
                    ),
                    "xg_history": historical_xg_features(
                        engine, h_en, a_en, league=lg, match_date=d,
                    ),
                    "markets": historical_stats,
                },
                "xg_used": bool(P.get("xg_used")),
                "xg_modelled_used": bool(P.get("xg_modelled_used")),
                "xg_source": P.get("xg_source"),
                "xg_weight": float(P.get("xg_weight") or 0.0),
                "xg_modelled_home_samples": int(P.get("xg_modelled_home_samples") or 0),
                "xg_modelled_away_samples": int(P.get("xg_modelled_away_samples") or 0),
                "xg_modelled_league_samples": int(P.get("xg_modelled_league_samples") or 0),
                "total_market_validation_report": getattr(engine, "total_market_validation_report", {}),
                "p1": P_decision["p1"], "px": P_decision["x"], "p2": P_decision["p2"],
                "raw_model_probabilities": raw_model_probabilities,
                "learning_decision": learning_preview,
                "over": P["over"], "btts": P["btts"],
                "odds_source": odds_source,
                "market_odds": dict(market_odds),
                "bookmaker_by_pick": dict(bookmaker_by_pick),
                "market_odds_source_by_pick": dict(market_source_by_pick),
                "one_x_two_diagnostics": dict(one_x_two_diagnostics),
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
                "api_football_fixture_id": None,
                "sstats_game_id": r.get("sstats_game_id"),
                "sstats_event_id": api_context.get("sstats_event_id"),
                "api_context": api_context,
                "context_quality": context_quality,
                "context_forecasts": context_forecasts,
                "home_badge": r.get("home_badge") or "",
                "away_badge": r.get("away_badge") or "",
                "home_team_tsdb_id": r.get("home_team_tsdb_id") or "",
                "away_team_tsdb_id": r.get("away_team_tsdb_id") or "",
                "league_badge": r.get("league_badge") or "",
                "venue": r.get("venue") or r.get("stadium") or "",
                "stadium_image": r.get("stadium_image") or r.get("venue_image") or "",
                "context": {},
            })
            # Save the fixture immediately after its first complete model snapshot.
            # The same row is enriched later with the independent AI/context result.
            try:
                _snapshot_card = cards[-1]
                _stable_fixture = forecast_fixture_identity(_snapshot_card)
                if _stable_fixture is not None:
                    _snapshot_card["forecast_fixture_id"] = _stable_fixture[0]
                    _snapshot_card["scan_run_id"] = scan_run_id
                _durable_card = dict(_snapshot_card)
                _durable_card.pop("_value_recheck", None)
                scan_ledger.save_fixture_snapshot(scan_run_id, _durable_card)
                saved_fixture_snapshots += 1
                if (
                    saved_fixture_snapshots - last_remote_checkpoint >= 25
                    and _runtime_secret("github_token_input", "GITHUB_TOKEN")
                ):
                    from storage.remote_backup import backup_database_to_gist
                    _checkpoint = backup_database_to_gist(DB_FILE)
                    last_remote_checkpoint = saved_fixture_snapshots
                    if not _checkpoint.get("ok"):
                        logs.append(
                            "⚠️ Промежуточная резервная копия скана: "
                            + str(_checkpoint.get("message") or _checkpoint.get("state"))
                        )
                    else:
                        logs.append(
                            f"☁️ Промежуточная резервная копия: сохранено матчей "
                            f"{saved_fixture_snapshots}."
                        )
            except Exception as snapshot_exc:
                logs.append(
                    "⚠️ Не удалось сохранить карточку матча сразу: "
                    f"{type(snapshot_exc).__name__}: {str(snapshot_exc)[:120]}"
                )

        if support_by_division:
            division_text = " · ".join(
                f"{item['league']} [{code}]: "
                f"team={item['team_supported']}/{item['fixtures']}, "
                f"league-only={item['league_only']}, prior={item['prior_only']}"
                for code, item in sorted(support_by_division.items())
            )
            logs.append("📚 Покрытие историей по соревнованиям: " + division_text)
        if support_examples:
            logs.append(
                "🔎 Матчи без допуска модели к Value (нужны реальные результаты команд, "
                "а не общий prior):"
            )
            for item in support_examples[:12]:
                logs.append(
                    f"  {item['league']} [{item['division']}] · "
                    f"{item['home']} ({item['home_matches']} матчей, "
                    f"{item['home_venue']} дома) — "
                    f"{item['away']} ({item['away_matches']} матчей, "
                    f"{item['away_venue']} в гостях) · "
                    f"лига={item['league_matches']} · {item['status']}"
                )

        badge_pair_count = sum(
            1 for card in cards
            if card.get("home_badge") and card.get("away_badge")
        )
        badge_any_count = sum(
            1 for card in cards
            if card.get("home_badge") or card.get("away_badge")
        )
        logs.append(
            f"🖼 Эмблемы клубов: полная пара URL у {badge_pair_count}/{len(cards)} матчей; "
            f"хотя бы одна эмблема у {badge_any_count}/{len(cards)}. "
            "Отсутствующие URL не заменяются выдуманными логотипами."
        )

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
        context_ready = sum(
            bool((card.get("context_quality") or {}).get("strong_value_eligible"))
            for card in cards
        )
        lineup_ready = sum(
            bool((card.get("context_quality") or {}).get("lineup_confirmed"))
            for card in cards
        )
        referee_ready = sum(
            bool((card.get("context_quality") or {}).get("referee_confirmed"))
            for card in cards
        )
        logs.append(
            f"🔐 Контекст Strong/Value: допущено {context_ready}/{len(cards)}; "
            f"подтверждённые XI {lineup_ready}/{len(cards)}, "
            f"судьи {referee_ready}/{len(cards)}. "
            "Без подтверждённых XI обеих команд и судьи Value-кандидаты блокируются."
        )

        # ============ TOP MATCHES · INDEPENDENT AI ANALYSIS ============
        llm_prov = D.get("meta", {}).get(
            "llm_provider", "Groq (бесплатно, быстро)"
        )
        llm_key = _provider_api_key(llm_prov)
        llm_provider_name = _provider_display_name(llm_prov)
        llm_model = D.get("meta", {}).get("llm_model", "")
        llm_done = 0

        top_diagnostics = top_match_shortlist_diagnostics(
            cards, min_team_history=MIN_FAVORITE_HISTORY,
        )
        top_match_items = select_top_matches(
            cards, limit=MAX_TOP_MATCHES, min_team_history=MIN_FAVORITE_HISTORY,
            allow_low_history=True,
        )
        strict_top_count = sum(bool(item.get("history_ready")) for item in top_match_items)
        low_history_top_count = len(top_match_items) - strict_top_count
        logs.append(
            "🏆 TOP MATCHES · независимый shortlist: "
            f"история команд ≥5: {top_diagnostics['enough_venue_history']}; "
            f"недостаточная история: {top_diagnostics['insufficient_venue_history']}; "
            f"без исторических средних: {top_diagnostics['missing_goal_features']}; "
            f"xG-история: {top_diagnostics['with_xg_history']}; "
            f"угловые: {top_diagnostics['with_corners_history']}; "
            f"карточки: {top_diagnostics['with_cards_history']}; "
            f"строгий shortlist {strict_top_count}; "
            f"ознакомительный AI-shortlist без достаточной истории {low_history_top_count}; "
            f"отобрано {len(top_match_items)}/{MAX_TOP_MATCHES}."
        )
        coverage = top_diagnostics.get("division_coverage") or []
        if coverage:
            coverage_text = " · ".join(
                f"{item['league']} [{item['code']}]: "
                f"{item['eligible']}/{item['fixtures']} shortlist, "
                f"макс. дом/выезд {item['max_home_sample']}/{item['max_away_sample']}"
                for item in coverage[:10]
            )
            logs.append("📚 TOP MATCHES · история по лигам: " + coverage_text)

        top_dossiers = []
        top_card_by_id = {}
        if top_match_items:
            # Re-check SStats context specifically for the seven selected matches.
            # This gives priority candidates their own injury/lineup/referee lookups
            # even when the general scan hit its broader provider-call caps.
            priority_rows = []
            for item in top_match_items:
                card = item["card"]
                if card.get("sstats_game_id") in (None, ""):
                    continue
                priority_rows.append({
                    "Date": f"{card.get('date_iso') or ''} {card.get('time') or ''}".strip(),
                    "date_iso": card.get("date_iso"),
                    "kickoff_utc": card.get("kickoff_utc") or "",
                    "HomeTeam": card.get("home_team") or "",
                    "AwayTeam": card.get("away_team") or "",
                    "sstats_game_id": card.get("sstats_game_id"),
                })

            priority_contexts = {}
            if sstats_key and priority_rows:
                try:
                    priority_contexts, priority_logs = sstats_context_enrich(
                        priority_rows,
                        sstats_key,
                        date_from=today_date.isoformat(),
                        date_to=(today_date + timedelta(days=days - 1)).isoformat(),
                        # Stamp the priority re-fetch at its actual request time,
                        # not at scan start; ContextStore deduplicates fixture + observed_at.
                        now=datetime.now(scan_now.tzinfo) if scan_now.tzinfo else datetime.now(),
                        max_injury_checks=MAX_TOP_MATCHES,
                        max_lineup_checks=MAX_TOP_MATCHES,
                        timezone_offset=(
                            int(round(scan_now.utcoffset().total_seconds() / 3600))
                            if scan_now.utcoffset() else 0
                        ),
                    )
                    logs.extend(
                        "🏆 TOP SStats: " + str(line) for line in priority_logs
                    )
                except Exception as exc:
                    logs.append(
                        f"⚠️ TOP SStats context: {type(exc).__name__}; "
                        f"{llm_provider_name} увидит только уже сохранённые подтверждённые данные."
                    )

            for rank, item in enumerate(top_match_items, start=1):
                card = item["card"]
                home_name = str(card.get("home_team") or "")
                away_name = str(card.get("away_team") or "")
                date_key = str(card.get("date_iso") or "")[:10]
                context_key = fixture_row_key(home_name, away_name, date_key)
                refreshed_context = priority_contexts.get(context_key)
                context_recheck = {}
                if isinstance(refreshed_context, dict) and refreshed_context:
                    previous_best = card.get("best")
                    previous_best_exists = bool(previous_best)
                    context_recheck = _recheck_value_after_priority_context(
                        card, refreshed_context, context_store=context_store,
                    )
                    if context_recheck.get("rechecked"):
                        new_best = card.get("best")
                        if not previous_best_exists and new_best:
                            matches_with_best += 1
                            funnel_counts["bet"] += 1
                            funnel_counts["real"] += 1
                            funnel_counts["model_only"] = max(
                                0, int(funnel_counts.get("model_only") or 0) - 1
                            )
                        logs.append(
                            f"🏆 TOP SStats → VALUE: {card.get('match_ru') or match_name(card)} | "
                            f"XI={'подтверждён' if context_recheck.get('new_eligible') else 'нет'} · "
                            f"Value повторно проверен · "
                            + (
                                f"допущено {new_best[1]} @ {float(new_best[2]):.2f}"
                                if new_best else "ставка не прошла остальные Value/EV/Edge/Kelly-гейты"
                            )
                        )
                    else:
                        logs.append(
                            f"ℹ️ TOP SStats → VALUE: {card.get('match_ru') or match_name(card)} | "
                            f"новый контекст сохранён, но допуск не изменён: "
                            f"{context_recheck.get('reason') or 'не хватает подтверждений/истории'}."
                        )
                if context_recheck.get("updated"):
                    card["context_forecasts"] = context_probabilities(
                        {
                            "H": float(card.get("p1") or 0.0),
                            "D": float(card.get("px") or 0.0),
                            "A": float(card.get("p2") or 0.0),
                        },
                        card.get("api_context") or {},
                        model=st.session_state.get("_context_blender"),
                    )
                    if hasattr(engine, "match_stats"):
                        try:
                            referee_name = str(
                                ((card.get("api_context") or {}).get("referee") or {}).get("name") or ""
                            ).strip()
                            refreshed_stats = engine.match_stats.predict(
                                home_name, away_name, str(card.get("div") or "G"),
                                referee_id=referee_name or None,
                            )
                            card["historical_stats"] = refreshed_stats
                            features = card.get("independent_features")
                            if isinstance(features, dict):
                                features["markets"] = refreshed_stats
                        except Exception as exc:
                            logs.append(
                                f"ℹ️ TOP SStats referee-history refresh {home_name} — {away_name}: "
                                f"{type(exc).__name__}."
                            )
                card["deep_analysis_context"] = merge_context_snapshots(
                    card.get("api_context") or {},
                    refreshed_context if isinstance(refreshed_context, dict) else {},
                )
                dossier = build_match_dossier(card)
                # Recheck inputs are one-scan working data, not forecast history.
                card.pop("_value_recheck", None)
                dossier["shortlist_rank"] = rank
                top_dossiers.append(dossier)
                top_card_by_id[dossier["fixture_id"]] = {
                    "card": card,
                    "favorite": item,
                }

            # Never persist temporary Value recheck inputs on non-shortlisted cards.
            for _card in cards:
                if isinstance(_card, dict):
                    _card.pop("_value_recheck", None)

            if not llm_key:
                reason = (
                    f"Не задан ключ {llm_provider_key_config(llm_prov)['env']}."
                )
                logs.append(
                    f"🏆 TOP MATCHES: кандидаты найдены, но {llm_provider_name} не вызван — {reason}"
                )
                logs.append(
                    f"🏆 TOP MATCHES · AI STATUS: missing_key · {llm_provider_name} · {reason}"
                )
            elif usage.llm_remaining() <= 0:
                reason = "Дневной лимит AI-запросов исчерпан."
                logs.append(
                    f"🏆 TOP MATCHES: кандидаты найдены, но {reason}"
                )
                logs.append(
                    f"🏆 TOP MATCHES · AI STATUS: daily_limit · {llm_provider_name} · {reason}"
                )
            elif top_dossiers:
                update_loader(
                    f"🏆 {llm_provider_name} Deep Analysis · {len(top_dossiers)} матчей одним пакетом",
                    0.94, logs,
                )
                try:
                    top_analyses = analyze_top_matches({
                        "api_key": llm_key,
                        "provider": llm_prov,
                        "model": llm_model,
                        "candidates": top_dossiers,
                    })
                except LLMRateLimitError as exc:
                    top_analyses = {}
                    retry_after = exc.retry_after_seconds
                    logs.append(
                        f"⛔ {llm_provider_name} ограничил пакетный TOP MATCHES анализ. "
                        + (
                            f"Повторить скан примерно через {retry_after:.0f} сек."
                            if retry_after is not None
                            else "Повторите скан после сброса лимита провайдера."
                        )
                    )
                except Exception as exc:
                    top_analyses = {}
                    logs.append(
                        f"⚠️ {llm_provider_name} TOP MATCHES: {type(exc).__name__}: {str(exc)[:140]}"
                    )

                ai_status = last_top_matches_status()
                actual_llm_provider = str(ai_status.get("provider") or llm_prov)
                actual_llm_model = str(ai_status.get("model") or (
                    LLM_PROVIDERS.get(actual_llm_provider, {}).get("model") or llm_model
                ))
                actual_llm_name = _provider_display_name(actual_llm_provider)
                logs.append(
                    "🏆 TOP MATCHES · AI STATUS: "
                    + str(ai_status.get("state") or "unknown")
                    + f" · {actual_llm_name} · {actual_llm_model or 'модель не указана'} · "
                    + str(ai_status.get("message") or "нет подробностей")
                )
                for fixture_id, analysis in top_analyses.items():
                    target = top_card_by_id.get(fixture_id)
                    if not target:
                        continue
                    card = target["card"]
                    model_pick_code = max(
                        (("П1", float(card.get("p1") or 0.0)),
                         ("X", float(card.get("px") or 0.0)),
                         ("П2", float(card.get("p2") or 0.0))),
                        key=lambda pair: pair[1],
                    )[0]
                    card["deep_analysis"] = analysis
                    card["deep_analysis_provider"] = actual_llm_provider
                    card["deep_analysis_model"] = actual_llm_model
                    if ai_status.get("used_fallback"):
                        card["deep_analysis_fallback_from"] = str(ai_status.get("fallback_from") or llm_prov)
                        card["deep_analysis_browser_search"] = False
                    card["llm_opinion"] = format_top_match_opinion(
                        analysis, model_pick_code=model_pick_code
                    )
                    card["deep_analysis_fixture_id"] = fixture_id
                    card["ai_crosscheck"] = build_ai_crosscheck(card, analysis)
                    if card["ai_crosscheck"].get("status") == "disagreement":
                        logs.append(
                            f"⚠️ AI CROSS-CHECK: {card.get('match_ru') or card.get('match')}: "
                            f"модель {card['ai_crosscheck'].get('main_pick')} · "
                            f"AI {card['ai_crosscheck'].get('ai_pick')} · "
                            "ручная проверка расхождения; Value/EV/Edge/Kelly не изменены."
                        )
                    elif card["ai_crosscheck"].get("status") == "agreement":
                        logs.append(
                            f"✅ AI CROSS-CHECK: {card.get('match_ru') or card.get('match')}: "
                            f"модель и независимый AI согласны по "
                            f"{card['ai_crosscheck'].get('main_pick')}; "
                            "это диагностическое согласие, не дополнительное доказательство Value."
                        )
                    llm_done += 1

                logs.append(
                    f"🏆 {llm_provider_name} Deep Analysis: {llm_done}/{len(top_dossiers)} "
                    "досье получено одним пакетным запросом."
                )
                for dossier in top_dossiers:
                    card = top_card_by_id[dossier["fixture_id"]]["card"]
                    if not card.get("deep_analysis"):
                        logs.append(
                            f"ℹ️ {llm_provider_name} не вернул валидный разбор: "
                            f"{dossier['match']['home']} — {dossier['match']['away']}."
                        )
        else:
            reason = (
                "Нет корректных матчей для независимого AI-разбора: нужны названия "
                "обеих команд и идентифицируемая дата/игровое событие. Недостаточная "
                "история сама по себе больше не блокирует ознакомительный разбор."
            )
            logs.append("🏆 TOP MATCHES: " + reason)
            logs.append(
                f"🏆 TOP MATCHES · AI STATUS: no_candidates · {llm_provider_name} · {reason}"
            )

        # Temporary Value recheck inputs are never persisted to cards or the ledger,
        # including scans where the AI shortlist is empty or the API key is missing.
        for _card in cards:
            if isinstance(_card, dict):
                _card.pop("_value_recheck", None)

        # Enrich the already durable per-fixture snapshots with final AI/context fields.
        # If the process stops before this point, the initial model cards remain in SQLite.
        for _snapshot_card in cards:
            try:
                scan_ledger.save_fixture_snapshot(scan_run_id, _snapshot_card)
            except Exception as snapshot_exc:
                logs.append(
                    f"⚠️ Финальное обновление карточки не записано: {type(snapshot_exc).__name__}."
                )

        # Persist both forecasters before the fixture starts. The ledger keeps the
        # first prediction immutable, making repeated scans idempotent and auditable.
        try:
            tracked = record_scan_forecasts(
                cards, provider=llm_prov, model=llm_model, db_path=DB_FILE,
                timezone_name=getattr(scan_timezone, "key", "UTC"),
                now=scan_now, scan_run_id=scan_run_id,
            )
            logs.append(
                "📚 Журнал прогнозов: "
                f"модель +{tracked['main_recorded']}, ИИ +{tracked['ai_recorded']}, "
                f"рынки +{tracked['market_recorded']}, "
                f"калибровка сохранена {tracked['learning_recorded']}, "
                f"поправка применена {tracked['learning_promoted']}, "
                f"ошибок обучения {tracked['learning_errors']}, "
                f"без времени старта пропущено {tracked['skipped_no_time']}, "
                f"после старта пропущено {tracked['skipped_started']}, "
                f"ошибок {tracked['invalid']}"
            )
            # SQLite backup is performed only after the complete scan, bets,
            # portfolio state and bank history have all been committed.
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
                # Add each independently validated family; size_bet applies the
                # combined match and daily caps across the same scan batch.
                for family_candidate in ("winner", "goals"):
                    portfolio_choice = _portfolio_auto_recommendation(
                        card, market_family=family_candidate
                    )
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
                    except (TypeError, ValueError, IndexError):
                        continue
                    if probability < max(float(min_prob), AUTO_PORTFOLIO_MIN_PROB):
                        continue
                    risk_bets = list(D.get("bets") or []) + new_bets
                    stake_value, risk_info = size_bet(
                        probability=probability,
                        odds=odds_value,
                        bank=available_bank,
                        bets=risk_bets,
                        fixture_id=card.get("fixture_id"),
                        match=card.get("match"),
                        date_iso=card.get("date_iso"),
                        market=market_code,
                        pick=pick,
                        division=card.get("div"),
                        now=datetime.now(app_timezone()),
                    )
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
                        "date_time": datetime.now(app_timezone()).strftime("%Y-%m-%d %H:%M"),
                        "fixture_id": card.get("fixture_id"),
                        "scan_run_id": card.get("scan_run_id") or scan_run_id,
                        "forecast_key": card.get("main_forecast_key"),
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
                        "kelly_fraction": KELLY_FRACTION,
                        "risk_policy": {
                            "max_market_pct": MAX_MARKET_STAKE_PCT,
                            "max_match_pct": MAX_MATCH_EXPOSURE_PCT,
                            "max_daily_pct": MAX_DAILY_EXPOSURE_PCT,
                            "stake_reason": str(risk_info.get("reason") or "approved"),
                        },
                    }
                    new_bets.append(new_bet)
                    existing_keys.add(match_key)
                    available_bank -= stake_value
                    total_stake += stake_value
                    logs.append(
                        f"🤖 AUTO-PORTFOLIO: {card.get('match_ru', card.get('match', ''))} "
                        f"| {pick} @ {odds_value:.2f} | stake {stake_value:.2f}"
                    )

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

        # Automated staking remains limited to markets that passed their own
        # out-of-sample bookmaker benchmark (currently 1X2 and validated O/U 2.5).
        # Other markets stay visible as forecasts; size_bet enforces combined
        # market, fixture and daily exposure across every added selection.

        logs.append(
            "🧠 Поддержка модели: "
            f"полная история команд={funnel_counts.get('team_supported', 0)}, "
            f"лига без данных одной из команд={funnel_counts.get('league_only', 0)}, "
            f"частичная история={funnel_counts.get('partial_history', 0)}, "
            f"prior-only без истории турнира={funnel_counts.get('prior_only', 0)}. "
            "Матчи без истории обеих команд исключаются из выбора фаворита, Value и журнала основной модели."
        )
        logs.append(
            f"🗓 Горизонт последнего скана: {days} календарн. дн. "
            f"({today_date.isoformat()} — {(today_date + timedelta(days=days - 1)).isoformat()}); "
            f"проанализировано в окне={date_filter_counts['inside_horizon']}, "
            f"вне окна={date_filter_counts['outside_horizon']}, "
            f"уже начались={date_filter_counts['past_kickoff']}, "
            f"без времени сегодня={date_filter_counts['unknown_same_day_kickoff']}."
        )
        D2 = dict(D)
        D2["cards"] = cards
        D2["report"] = logs
        D2["scan_meta"] = {
            "scan_run_id": scan_run_id,
            "scan_started_at": scan_now.isoformat(),
            "scan_completed_at": datetime.now(scan_timezone).isoformat(),
            "horizon_days": int(days),
            "date_from": today_date.isoformat(),
            "date_to_inclusive": (today_date + timedelta(days=days - 1)).isoformat(),
            "source_rows": len(fixture_sources),
            "unique_fixtures": len(rows),
            "analysed_matches": int(date_filter_counts["inside_horizon"]),
            "forecast_cards": len(cards),
            "model_support_counts": {
                "prior_only": int(funnel_counts.get("prior_only", 0)),
                "league_only": int(funnel_counts.get("league_only", 0)),
                "partial_history": int(funnel_counts.get("partial_history", 0)),
                "team_supported": int(funnel_counts.get("team_supported", 0)),
            },
            "model_support_by_division": support_by_division,
            "model_support_examples": support_examples,
            "date_filter_counts": dict(date_filter_counts),
            "source_counts": {
                str(name): int(count) for name, count in source_counts.items()
            },
            # Persist only credential presence, never the API key itself.
            "api_football_configured": bool(
                _runtime_secret("api_football_key_input", "API_FOOTBALL_KEY")
            ),
        }
        D2["bets"] = list(D.get("bets") or []) + new_bets
        D2["bank"] = max(0.0, float(D.get("bank") or 0.0) - total_stake)
        D2["funnel"] = {
            "trained": getattr(engine, "trained_n", 0),
            "src": len(rows), "found": matches_with_best,
            "added": len(new_bets), "frozen": total_stake, "llm": llm_done,
        }
        try:
            links = scan_ledger.record_run(
                scan_run_id,
                metadata={
                    **D2["scan_meta"],
                    "timezone_name": getattr(scan_timezone, "key", "UTC"),
                    "funnel": D2["funnel"],
                },
                cards=cards,
                report=logs,
            )
            logs.append(
                "🔗 Единый реестр сканирования: "
                f"{links['fixtures']} матчей, {links['forecasts']} связей с прогнозами, "
                f"{links['decisions']} решений по рынкам."
            )
            D2["report"] = logs
        except Exception as scan_ledger_exc:
            logs.append(
                f"⚠️ Реестр сканирования не записан: {type(scan_ledger_exc).__name__}."
            )
            D2["report"] = logs
        st.session_state.data = D2
        usage.set_local_data(D2)
        if db.SQLITE_BOOT_OK:
            db.insert_bets_batch(new_bets)
            db.log_bank(D2["bank"], event="scan")
            db.invalidate_caches()

        # Back up only after all scan forecasts, bets, balance changes and the
        # durable app-state row are committed to SQLite.
        try:
            from storage.remote_backup import backup_database_to_gist
            backup_status = backup_database_to_gist(DB_FILE)
            st.session_state["_sqlite_backup_status"] = {
                **backup_status,
                "checked_at": datetime.now().astimezone().isoformat(),
            }
            if not backup_status.get("ok"):
                _print_log(
                    "⚠️ SQLite remote backup: "
                    + str(backup_status.get("message") or backup_status.get("state"))
                )
        except Exception as backup_exc:
            st.session_state["_sqlite_backup_status"] = {
                "ok": False, "state": "backup_error",
                "message": f"SQLite backup failed: {type(backup_exc).__name__}",
                "checked_at": datetime.now().astimezone().isoformat(),
            }
            _print_log(f"⚠️ SQLite remote backup: {type(backup_exc).__name__}")

        # Surface source diagnostics in deployment logs, not only the in-app report.
        for source_line in logs:
            if "матчей после фильтрации" in source_line:
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
            try:
                active_run = str(st.session_state.get("_active_scan_run_id") or "")
                if active_run:
                    ScanLedger(DB_FILE).mark_run_failed(active_run, message)
                    if _runtime_secret("github_token_input", "GITHUB_TOKEN"):
                        from storage.remote_backup import backup_database_to_gist
                        backup_database_to_gist(DB_FILE)
            except Exception:
                pass
        except Exception:
            pass
        st.error(message)
        st.info("Проверь раздел «Диагностика» и последние строки журнала.")
        return
    finally:
        st.session_state["_scan_in_progress"] = False
        st.session_state.pop("_active_scan_run_id", None)
    # A completed scan should land on the dashboard with the new results visible.
    st.session_state["_pending_terminal_navigation_v16"] = "🏠 Дашборд"
    st.rerun()
