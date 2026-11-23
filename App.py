"""App.py — NEURO BET PRO v13 (исправленная версия).

Что изменено относительно оригинала:
  1. API-ключи НЕ попадают в скачиваемый бэкап; при восстановлении текущие ключи сохраняются.
  2. Calibration error считается корректно (ECE по бинам), а не как mean|p - y|.
  3. Авто-void через 48 ч убран: такие ставки помечаются stale и закрываются вручную.
  4. ROI считается от оборота (поставленных денег), статистика пересчитывается из списка ставок.
  5. Ставка закрывается только через MIN_AGE после начала матча и только по завершённому счёту.
  6. Неудачные запросы не расходуют лимит; у каждой ставки есть пауза между попытками.
  7. Часовые пояса приведены к naive-локальным (нет TypeError при сравнении дат).
  8. Восстановление из бэкапа валидируется; исправлен бесконечный rerun при загрузке файла.
  9. Константы вынесены, мусор (ERR, _print_log) заменён на logging.
 10. Удаление прокси из окружения можно отключить: NEURO_KEEP_PROXY=1.
 11. Порядок источников авто-сеттла: ESPN/TheSportsDB/OpenLigaDB → fdorg.
     Раньше fdorg шёл первым и упирался в 429; теперь он только fallback.
"""
from __future__ import annotations

import os
import json
import time
import logging
from datetime import date as dt_date, datetime, time as dt_time, timedelta, timezone

import streamlit as st

if os.environ.get("NEURO_KEEP_PROXY") != "1":
    for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
               "ALL_PROXY", "all_proxy", "FTP_PROXY", "ftp_proxy"]:
        os.environ.pop(_k, None)

st.set_page_config(page_title="NEURO BET PRO",
                   page_icon="🏟", layout="wide",
                   initial_sidebar_state="expanded")

from config import (APP_VERSION, APP_BUILD_ID, LLM_PROVIDERS, DATA_VERSION,  # noqa: E402
                    AUTO_SETTLE_THROTTLE_SEC, DB_FILE, normalize_llm_provider)
from llm.deep_analysis import check_provider_connection  # noqa: E402
from storage import sqlite_store as db  # noqa: E402
from storage import usage  # noqa: E402
from storage.secret_store import load_local_secrets, save_local_secrets  # noqa: E402
from storage.remote_backup import (
    backup_database_to_gist, create_sqlite_backup_bytes,
    remote_backup_status, restore_database_if_empty,
    restore_sqlite_backup_bytes,
)  # noqa: E402
from model.audit_restore import restore_audit_export_bytes  # noqa: E402
from data.sources import (
    parse_date, fdorg_match_result,
    sstats_account_check, sstats_match_statistics, sstats_find_match_id,
    footballapi_xg_connection_check,
)  # noqa: E402
from data.score_status import is_verified_final_score  # noqa: E402
from betting.settlement import (  # noqa: E402
    determine_outcome, determine_stat_outcome,
)


def _settlement_ready_compat(
    match_date, match_time=None, *, kickoff_at=None, now=None,
    grace=timedelta(hours=2, minutes=15),
):
    """Conservative compatibility gate for deployments with an older settlement.py."""
    def _absolute_utc(value):
        if isinstance(value, datetime):
            parsed = value
        else:
            raw = str(value or "").strip()
            if not raw:
                return None
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                return None
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)

    absolute = _absolute_utc(kickoff_at) or _absolute_utc(match_date)
    if absolute is not None:
        not_before = absolute + grace
    else:
        date_part = None
        embedded_local_time = None
        if isinstance(match_date, datetime):
            date_part = match_date.date()
            if match_date.tzinfo is None and match_date.time() != dt_time.min:
                embedded_local_time = match_date.time()
        elif isinstance(match_date, dt_date):
            date_part = match_date
        else:
            raw_date = str(match_date or "").strip()
            if raw_date:
                try:
                    parsed_date = datetime.fromisoformat(
                        raw_date.replace("Z", "+00:00")
                    )
                    date_part = parsed_date.date()
                    if parsed_date.tzinfo is None and parsed_date.time() != dt_time.min:
                        embedded_local_time = parsed_date.time()
                except ValueError:
                    for fmt in (
                        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                        "%d/%m/%Y %H:%M", "%Y-%m-%d", "%d.%m.%Y",
                        "%d/%m/%Y", "%d-%m-%Y", "%d/%m/%y",
                    ):
                        try:
                            parsed_date = datetime.strptime(raw_date[:19], fmt)
                            date_part = parsed_date.date()
                            if parsed_date.time() != dt_time.min:
                                embedded_local_time = parsed_date.time()
                            break
                        except ValueError:
                            continue
        if date_part is None:
            return None

        raw_time = str(match_time or "").strip()
        if not raw_time and embedded_local_time is not None:
            raw_time = embedded_local_time.isoformat()
        if raw_time:
            try:
                if "T" in raw_time or " " in raw_time:
                    parsed_time = datetime.fromisoformat(
                        raw_time.replace("Z", "+00:00")
                    )
                    if parsed_time.tzinfo is not None:
                        not_before = parsed_time.astimezone(timezone.utc) + grace
                    else:
                        local_kickoff = parsed_time
                        not_before = (
                            local_kickoff.replace(tzinfo=timezone.utc)
                            + timedelta(hours=12) + grace
                        )
                else:
                    parsed_clock = dt_time.fromisoformat(
                        raw_time.replace("Z", "+00:00")
                    )
                    local_kickoff = datetime.combine(date_part, parsed_clock)
                    if parsed_clock.tzinfo is not None:
                        not_before = local_kickoff.astimezone(timezone.utc) + grace
                    else:
                        not_before = (
                            local_kickoff.replace(tzinfo=timezone.utc)
                            + timedelta(hours=12) + grace
                        )
            except ValueError:
                raw_time = ""
        if not raw_time:
            latest_local_kickoff = datetime.combine(
                date_part + timedelta(days=1), dt_time(11, 59),
                tzinfo=timezone.utc,
            )
            not_before = latest_local_kickoff + grace

    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current >= not_before


try:
    from betting.settlement import settlement_ready  # noqa: E402
    _SETTLEMENT_READY_FALLBACK = False
except ImportError as exc:
    if "settlement_ready" not in str(exc):
        raise
    settlement_ready = _settlement_ready_compat
    _SETTLEMENT_READY_FALLBACK = True
from model.forecast_ledger import MarketForecastLedger  # noqa: E402
from model.independent_forecasts import settle_due_forecasts  # noqa: E402
from ui import theme  # noqa: E402
from ui.tabs import scanner, portfolio, stats, calculator, backtest, analytics, dashboard  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[APP] %(message)s")
log = logging.getLogger("neurobet")
if _SETTLEMENT_READY_FALLBACK:
    log.warning("betting.settlement is older than App.py; using conservative local settlement gate")

# ==================== КОНСТАНТЫ ====================
DEFAULT_BANK = 10000.0
AUTO_SETTLE_EVERY_SEC = AUTO_SETTLE_THROTTLE_SEC
FORECAST_AUTO_SYNC_EVERY_SEC = 600
FORECAST_AUTO_SYNC_MAX_FIXTURES = 8
FORECAST_MANUAL_SYNC_MAX_FIXTURES = 60
SETTLE_MIN_AGE = timedelta(hours=2, minutes=15)
RETRY_EVERY = timedelta(minutes=60)
STALE_AFTER = timedelta(hours=48)
MAX_PER_RUN = 20
SECRET_KEYS = (
    "fdorg_token", "highlightly_api_key", "sstats_api_key",
    "footballapi_key", "llm_api_key", "xai_api_key",
    "odds_api_key", "therundown_api_key", "rundown_key",
)


def _configured_secret(*keys):
    """Read root Streamlit secrets and common nested API sections."""
    try:
        for key in keys:
            value = st.secrets.get(key, "")
            if value:
                return str(value).strip()
        for section_name in ("api", "apis", "secrets", "highlightly", "sstats", "footballapi", "football_api", "xai"):
            section = st.secrets.get(section_name, {})
            if hasattr(section, "get"):
                for key in keys:
                    value = section.get(key, "")
                    if value:
                        return str(value).strip()
                if section_name == "sstats":
                    for alias in ("api_key", "key"):
                        value = section.get(alias, "")
                        if value:
                            return str(value).strip()
    except Exception:
        pass
    return ""
def _api_key_default(storage_key, env_name, *secret_names):
    """Configured deployment secrets take precedence over the private local UI store."""
    env_value = os.environ.get(env_name, "").strip()
    if env_value:
        return env_value
    configured = (
        _configured_secret(env_name, *secret_names, "api_key", "key")
        if storage_key == "sstats_api_key"
        else _configured_secret(env_name, *secret_names)
    )
    if configured:
        return configured
    return _saved_ui_secrets.get(storage_key, "")


def _configured_key_value(env_names, secret_names):
    for name in env_names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return _configured_secret(*secret_names)

# ==================== ХЕЛПЕРЫ ====================
def _naive(dt):
    if dt is None:
        return None
    if getattr(dt, "tzinfo", None) is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def _parse_iso(s):
    try:
        return _naive(datetime.fromisoformat(s)) if s else None
    except (TypeError, ValueError):
        return None


def _bet_date(b):
    try:
        return _naive(parse_date(b.get("date_iso") or b.get("date") or ""))
    except Exception:
        return None


def _is_finished(res) -> bool:
    """Fail closed unless a provider explicitly confirms a final regulation score."""
    return is_verified_final_score(res)


def _recompute_stats(bets, old=None):
    s = dict(old or {})
    s.update({"won": 0, "lost": 0, "push": 0, "void": 0, "profit": 0.0})
    for b in bets:
        if not isinstance(b, dict):
            continue
        st_ = b.get("status")
        try:
            stake = float(b.get("stake", 0))
            odds = float(b.get("odds", 0))
        except (TypeError, ValueError):
            continue
        if st_ == "won":
            s["won"] += 1
            s["profit"] += stake * (odds - 1)
        elif st_ == "lost":
            s["lost"] += 1
            s["profit"] -= stake
        elif st_ in ("push", "void"):
            s[st_] += 1
    return s


def _roi_from_bets(bets):
    profit = turnover = 0.0
    for b in bets:
        if not isinstance(b, dict) or b.get("status") not in ("won", "lost"):
            continue
        try:
            stake = float(b.get("stake", 0))
            odds = float(b.get("odds", 0))
        except (TypeError, ValueError):
            continue
        turnover += stake
        profit += stake * (odds - 1) if b["status"] == "won" else -stake
    return (profit / turnover * 100 if turnover else 0.0), profit, turnover


def _strip_secrets(D):
    out = dict(D)
    out["meta"] = {k: v for k, v in (D.get("meta") or {}).items()
                   if k not in SECRET_KEYS}
    return out


def _validate_data(raw):
    if isinstance(raw, dict) and isinstance(raw.get("data"), dict):
        raw = raw["data"]
    if not isinstance(raw, dict):
        raise ValueError("Файл не похож на бэкап (ожидался JSON-объект)")
    if not isinstance(raw.get("bets", []), list):
        raise ValueError("Поле 'bets' должно быть списком")
    d = dict(raw)
    d["bets"] = [b for b in d.get("bets", []) if isinstance(b, dict)]
    try:
        d["bank"] = float(d.get("bank", DEFAULT_BANK))
    except (TypeError, ValueError):
        raise ValueError("Поле 'bank' должно быть числом")
    d.setdefault("version", DATA_VERSION)
    d.setdefault("cards", [])
    d.setdefault("funnel", None)
    d.setdefault("report", [])
    d.setdefault("mode", "paper")
    if not isinstance(d.get("meta"), dict):
        d["meta"] = {}
    # Imported backups are untrusted too: never restore credentials into local state.
    for secret_key in SECRET_KEYS:
        d["meta"].pop(secret_key, None)
    d["stats"] = _recompute_stats(d["bets"], d.get("stats"))
    d["meta"].setdefault("initial_bank", d["bank"])
    return d


# ==================== BOOT ====================
# Streamlit Community Cloud local disks are ephemeral. If a private encrypted
# Gist is configured, restore only when this local SQLite file has no user rows.
# Existing local data is never overwritten by this automatic path.
if "_sqlite_restore_status" not in st.session_state:
    try:
        st.session_state["_sqlite_restore_status"] = restore_database_if_empty(DB_FILE)
    except Exception as _backup_exc:
        st.session_state["_sqlite_restore_status"] = {
            "restored": False, "state": "restore_error",
            "message": f"SQLite backup restore failed: {type(_backup_exc).__name__}",
        }
db.db_init()
theme.inject()

if "data" not in st.session_state:
    st.session_state.data = usage.get_local_data() or {
        "version": DATA_VERSION, "bank": DEFAULT_BANK, "bets": [], "cards": [],
        "funnel": None, "report": [], "meta": {},
        "stats": {"won": 0, "lost": 0, "profit": 0, "push": 0, "void": 0},
        "mode": "paper",
    }

D = st.session_state.data
D.setdefault("meta", {})
# Credentials live outside portfolio data and backup/sync files.
_saved_ui_secrets = load_local_secrets()
# API credentials are session/env/secrets-only, never persisted in local state.
_secret_keys = SECRET_KEYS
_secrets_removed = False
for _key in _secret_keys:
    if D["meta"].pop(_key, None) is not None:
        _secrets_removed = True
if _secrets_removed:
    usage.set_local_data(D)
D.setdefault("bets", [])
if "initial_bank" not in D["meta"]:
    D["meta"]["initial_bank"] = float(D.get("bank", DEFAULT_BANK))
    usage.set_local_data(D)


# ==================== AUTO-SETTLE ====================
def _auto_settle(D, force: bool = False):
    """Settle score markets from FT results and STAT markets from SStats full stats."""
    D2 = dict(D)
    bets = list(D2.get("bets", []))
    closed = tried = 0
    now = datetime.now()
    token = str(
        st.session_state.get("fdorg_token_input", "")
        or _api_key_default("fdorg_token", "FDORG_TOKEN", "fdorg_token")
    ).strip()
    sstats_key = _api_key_default(
        "sstats_api_key", "SSTATS_API_KEY", "sstats_api_key"
    ).strip()
    log.info("START: %d bets | remaining=%s", len(bets), usage.settle_remaining())

    for idx, b in enumerate(bets):
        if not isinstance(b, dict) or b.get("status") != "pending":
            continue
        is_stat_market = str(b.get("market", "")).upper() == "STAT"
        if is_stat_market and not sstats_key:
            continue
        if tried >= MAX_PER_RUN:
            log.info("MAX_PER_RUN reached")
            break
        if usage.settle_remaining() <= 0:
            log.info("LIMIT reached")
            break

        # Do not query score providers just because calendar midnight passed:
        # wait for 2h15 after the actual kickoff. Legacy bets without a kickoff
        # time are deferred to the next day to avoid false NOT FOUND lookups.
        ready = settlement_ready(
            b.get("date_iso") or b.get("date"),
            b.get("match_time") or b.get("time"),
            kickoff_at=b.get("kickoff_utc"),
            grace=SETTLE_MIN_AGE,
        )
        if ready is False:
            continue
        if not force:
            last = _parse_iso(b.get("last_settle_try"))
            if last and now - last < RETRY_EVERY:
                continue

        b2 = dict(b)
        b2["last_settle_try"] = now.isoformat()
        outcome, reason = None, ""
        source = None
        result_label = None
        actual_stat_value = None

        if is_stat_market:
            # New bets retain the SStats game ID. Legacy bets are resolved only
            # by exact local date and ordered home/away team names.
            game_id = b2.get("sstats_game_id")
            if game_id in (None, ""):
                fixture_id = str(b2.get("fixture_id") or "")
                if fixture_id.startswith("sstats:"):
                    candidate = fixture_id.split(":", 1)[1]
                    game_id = candidate if candidate.isdigit() else None
            h_team = str(b2.get("home") or "").strip()
            a_team = str(b2.get("away") or "").strip()
            if not (h_team and a_team):
                match = str(b2.get("match") or "")
                if " vs " in match:
                    h_team, a_team = [part.strip() for part in match.split(" vs ", 1)]
            d_iso = str(b2.get("date_iso") or "")[:10]
            tried += 1
            if game_id in (None, "") and h_team and a_team and d_iso:
                try:
                    game_id = sstats_find_match_id(d_iso, h_team, a_team, sstats_key)
                    if game_id not in (None, ""):
                        b2["sstats_game_id"] = int(game_id)
                except Exception as exc:
                    log.warning("SStats match ID lookup failed: %s", type(exc).__name__)
            if game_id in (None, ""):
                log.info(
                    "%s | SStats match ID not resolved; left pending",
                    b2.get("match_ru", "?"),
                )
                bets[idx] = b2
                continue
            try:
                stat_data = sstats_match_statistics(game_id, sstats_key)
            except Exception as exc:
                stat_data = None
                log.warning("SStats match statistics failed: %s", type(exc).__name__)
            if not isinstance(stat_data, dict) or not stat_data.get("finished"):
                log.info(
                    "%s | SStats match not confirmed FT; left pending",
                    b2.get("match_ru", "?"),
                )
                bets[idx] = b2
                continue

            pick = str(b2.get("pick") or "")
            stat_kind = (
                "corners" if pick.casefold().startswith("угл ")
                else "cards" if pick.casefold().startswith("карты ")
                else None
            )
            actual_stat_value = stat_data.get(stat_kind) if stat_kind else None
            if actual_stat_value is None:
                log.info(
                    "%s | SStats FT but %s statistics unavailable; left pending",
                    b2.get("match_ru", "?"), stat_kind or "selection",
                )
                bets[idx] = b2
                continue
            outcome, reason = determine_stat_outcome(
                "STAT", pick, actual_stat_value
            )
            source = "sstats"
            result_label = (
                f"Угловые: {actual_stat_value}"
                if stat_kind == "corners" else f"Карточки: {actual_stat_value}"
            )
        else:
            res, source = None, None
            tried += 1
            h_team = str(b.get("home") or "").strip()
            a_team = str(b.get("away") or "").strip()
            if not (h_team and a_team):
                match = str(b.get("match") or "")
                if " vs " in match:
                    h_team, a_team = [part.strip() for part in match.split(" vs ", 1)]
            d_iso = b.get("date_iso") or ""
            if h_team and a_team and d_iso:
                try:
                    from data.scores import find_match_score
                    # First ask ESPN only. This prevents its internal FDORG/OLB
                    # fallbacks from running before a stored FDORG fixture ID.
                    res = find_match_score(
                        h_team, a_team, d_iso, token=token, providers=("espn",)
                    )
                    if res:
                        source = res.get("source", "espn")
                except Exception as exc:
                    log.warning("find_match_score error: %s", type(exc).__name__)

            if not _is_finished(res):
                res = None
                fixture_id = str(b2.get("fixture_id") or "").strip()
                fixture_source = str(
                    b2.get("fixture_source") or b2.get("fixture_provider") or ""
                ).strip().casefold()
                # SStats fixture IDs use "sstats:<id>" and are never sent to
                # football-data.org. Direct lookups require explicit provenance.
                direct_fdorg_id = None
                if fixture_id.casefold().startswith("fdorg:"):
                    direct_fdorg_id = fixture_id.split(":", 1)[1]
                elif fixture_source in {"fdorg", "football-data.org", "football_data_org"}:
                    direct_fdorg_id = fixture_id
                if direct_fdorg_id and direct_fdorg_id.isdigit() and token:
                    try:
                        res = fdorg_match_result(direct_fdorg_id, token)
                        if _is_finished(res):
                            source = "fdorg_fixture_id"
                    except Exception as exc:
                        log.warning("fdorg fixture-ID lookup failed: %s", type(exc).__name__)

            if not _is_finished(res) and h_team and a_team and d_iso:
                try:
                    # Pass the UI token into date-based lookup explicitly;
                    # do not depend on a different session-state key.
                    res = find_match_score(
                        h_team, a_team, d_iso, token=token,
                        providers=("fdorg", "olb"),
                    )
                    if res:
                        source = res.get("source", "auto")
                except Exception as exc:
                    log.warning("score fallback error: %s", type(exc).__name__)

            if not _is_finished(res):
                bets[idx] = b2
                continue
            outcome, reason = determine_outcome(
                b.get("market"), b.get("pick"),
                int(res["home"]), int(res["away"]),
            )
            result_label = f"{int(res['home'])}:{int(res['away'])}"

        if outcome is None:
            log.info("unknown outcome: %s", reason)
            bets[idx] = b2
            continue

        usage.settle_increment(1)
        try:
            stake = float(b2["stake"])
            odds = float(b2["odds"])
        except (KeyError, TypeError, ValueError):
            bets[idx] = b2
            continue

        b2["score"] = result_label
        b2["settled_at"] = datetime.now().isoformat()
        b2["settle_source"] = source
        if actual_stat_value is not None:
            b2["actual_stat_value"] = actual_stat_value
            b2["settlement_source"] = source
        b2.pop("stale", None)

        if outcome in ("push", "void"):
            b2["status"] = outcome
            D2["bank"] = D2.get("bank", DEFAULT_BANK) + stake
        elif outcome == "won":
            b2["status"] = "won"
            D2["bank"] = D2.get("bank", DEFAULT_BANK) + stake * odds
        elif outcome == "lost":
            b2["status"] = "lost"
        else:
            bets[idx] = b2
            continue
        bets[idx] = b2

        if db.SQLITE_BOOT_OK:
            try:
                if b2.get("id") is not None:
                    db.update_bet(
                        int(b2["id"]), status=b2.get("status"),
                        score=b2.get("score"), settled_at=b2.get("settled_at"),
                    )
                else:
                    db.update_bet_by_identity(
                        b2.get("match"), b2.get("market"), b2.get("pick"),
                        b2.get("date_iso"), status=b2.get("status"),
                        score=b2.get("score"), settled_at=b2.get("settled_at"),
                    )
            except Exception as exc:
                log.warning("SQLite settle persist error: %s", type(exc).__name__)

        audit_key = b2.get("market_forecast_key")
        if audit_key:
            try:
                MarketForecastLedger(DB_FILE).settle(
                    str(audit_key), outcome, result_score=result_label,
                    actual_value=actual_stat_value,
                    settlement_source=str(source or "auto"),
                )
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("Market forecast audit settle error: %s", type(exc).__name__)
        closed += 1

    for idx, b in enumerate(bets):
        if isinstance(b, dict) and b.get("status") == "pending":
            bd = _bet_date(b)
            if bd and now - bd > STALE_AFTER and not b.get("stale"):
                b2 = dict(b)
                b2["stale"] = True
                bets[idx] = b2

    D2["bets"] = bets
    D2["stats"] = _recompute_stats(bets, D2.get("stats"))
    log.info("DONE: %d closed, %d tried", closed, tried)
    return D2, closed, tried

def _void_stale(D):
    D2 = dict(D)
    bets = list(D2.get("bets", []))
    n = 0
    for idx, b in enumerate(bets):
        if isinstance(b, dict) and b.get("status") == "pending" and b.get("stale"):
            b2 = dict(b)
            b2["status"] = "void"
            b2["score"] = "void (manual, no result)"
            b2["settled_at"] = datetime.now().isoformat()
            b2.pop("stale", None)
            try:
                D2["bank"] = D2.get("bank", DEFAULT_BANK) + float(b2["stake"])
            except (KeyError, TypeError, ValueError):
                pass
            bets[idx] = b2
            if db.SQLITE_BOOT_OK:
                try:
                    if b2.get("id") is not None:
                        db.update_bet(
                            int(b2["id"]),
                            status="void",
                            score=b2.get("score"),
                            settled_at=b2.get("settled_at"),
                        )
                    else:
                        db.update_bet_by_identity(
                            b2.get("match"),
                            b2.get("market"),
                            b2.get("pick"),
                            b2.get("date_iso"),
                            status="void",
                            score=b2.get("score"),
                            settled_at=b2.get("settled_at"),
                        )
                except Exception as e:
                    log.warning("SQLite stale-void persist error: %s", e)
            audit_key = b2.get("market_forecast_key")
            if audit_key:
                try:
                    MarketForecastLedger(DB_FILE).settle(
                        str(audit_key),
                        "void",
                        result_score=str(b2.get("score") or "void (manual, no result)"),
                        settlement_source="manual_stale_void",
                        settled_at=b2.get("settled_at"),
                    )
                except (KeyError, TypeError, ValueError) as audit_exc:
                    log.warning(
                        "Market forecast stale-void audit error: %s",
                        type(audit_exc).__name__,
                    )
            n += 1
    D2["bets"] = bets
    D2["stats"] = _recompute_stats(bets, D2.get("stats"))
    return D2, n


def _apply_settle_result(D2, closed, tried, event):
    st.session_state.data = D2
    if closed > 0 or tried > 0:
        usage.set_local_data(D2)
    if closed > 0:
        db.log_bank(D2.get("bank", DEFAULT_BANK), event=event)
        db.invalidate_caches()


def _sync_forecast_results(force: bool = False):
    """Synchronize prediction and market audit ledgers during active app use."""
    now_ts = time.time()
    last_check = float(st.session_state.get("_last_forecast_ledger_sync_ts", 0) or 0)
    if not force and now_ts - last_check < FORECAST_AUTO_SYNC_EVERY_SEC:
        return None
    # Save the attempt time even on failure so a transient outage does not hammer providers.
    st.session_state["_last_forecast_ledger_sync_ts"] = now_ts
    # The API token is kept in a password widget/local secret store, not in
    # portfolio metadata. Pass it explicitly to the forecast-ledger score lookup;
    # data.meta.fdorg_token is intentionally stripped from persisted portfolio data.
    score_token = str(
        st.session_state.get("fdorg_token_input", "")
        or _api_key_default("fdorg_token", "FDORG_TOKEN", "fdorg_token")
    ).strip()
    sstats_score_key = str(
        st.session_state.get("sstats_api_key_input", "")
        or _api_key_default("sstats_api_key", "SSTATS_API_KEY", "sstats_api_key")
    ).strip()
    try:
        with st.spinner("Автоматически сверяю прогнозы с завершёнными матчами…"):
            status = settle_due_forecasts(
                db_path=DB_FILE,
                max_fixtures=(
                    FORECAST_MANUAL_SYNC_MAX_FIXTURES
                    if force else FORECAST_AUTO_SYNC_MAX_FIXTURES
                ),
                token=score_token or None,
                sstats_api_key=sstats_score_key or None,
                force=force,
            )
        status["checked_at"] = datetime.now(timezone.utc).isoformat()
        status["mode"] = "manual" if force else "automatic"
    except Exception as exc:
        status = {
            "error": f"{type(exc).__name__}: {str(exc)[:160]}",
            "fixtures_checked": 0, "forecasts_settled": 0,
            "market_fixtures_checked": 0, "market_forecasts_settled": 0,
            "errors": 1,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "mode": "manual" if force else "automatic",
        }
    st.session_state["_forecast_ledger_sync_status"] = status
    return status


@st.fragment(run_every="5m")
def _automatic_sqlite_backup_widget():
    """Persist a changed SQLite snapshot to the configured encrypted secret Gist."""
    last_attempt = float(st.session_state.get("_last_sqlite_backup_ts", 0) or 0)
    now_ts = time.time()
    if now_ts - last_attempt < 300:
        return
    st.session_state["_last_sqlite_backup_ts"] = now_ts
    try:
        result = backup_database_to_gist(DB_FILE)
        st.session_state["_sqlite_backup_status"] = {
            **result, "checked_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        st.session_state["_sqlite_backup_status"] = {
            "ok": False, "state": "backup_error",
            "message": f"SQLite backup failed: {type(exc).__name__}",
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


@st.fragment(run_every="10m")
def _automatic_result_sync_widget():
    """Run settlement while the Streamlit session remains open, without user clicks."""
    now_ts = time.time()
    last_portfolio_check = float(st.session_state.get("_last_auto_settle_ts", 0) or 0)
    if now_ts - last_portfolio_check >= AUTO_SETTLE_EVERY_SEC:
        st.session_state["_last_auto_settle_ts"] = now_ts
        try:
            current_data = st.session_state.get("data") or {}
            updated_data, closed, tried = _auto_settle(current_data)
            _apply_settle_result(updated_data, closed, tried, "auto_settle")
            st.session_state["_portfolio_auto_settlement_status"] = {
                "closed": closed, "tried": tried,
                "checked_at": datetime.now(timezone.utc).isoformat(),
            }
            if closed:
                st.toast(f"🔄 Автоматически закрыто ставок: {closed}", icon="✅")
        except Exception as exc:
            st.session_state["_portfolio_auto_settlement_status"] = {
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"
            }

    _sync_forecast_results()
    status = st.session_state.get("_forecast_ledger_sync_status") or {}
    portfolio_status = st.session_state.get("_portfolio_auto_settlement_status") or {}
    if status.get("error") or portfolio_status.get("error"):
        st.caption(
            "Автосверка: " + str(status.get("error") or portfolio_status.get("error"))
        )
    elif status or portfolio_status:
        st.caption(
            "Автосверка (пока приложение открыто): "
            f"прогнозов закрыто {status.get('forecasts_settled', 0)} · "
            f"рыночных рекомендаций {status.get('market_forecasts_settled', 0)} · "
            f"ставок портфеля {portfolio_status.get('closed', 0)}."
        )


_now_ts = time.time()
if _now_ts - st.session_state.get("_last_auto_settle_ts", 0) > AUTO_SETTLE_EVERY_SEC:
    st.session_state["_last_auto_settle_ts"] = _now_ts
    with st.spinner("Проверяю результаты матчей…"):
        D2, n, tried = _auto_settle(D)
    _apply_settle_result(D2, n, tried, "auto_settle")
    D = D2
    if n > 0:
        st.toast(f"🔄 Авто-закрыто {n} ставок", icon="✅")

# Automatic prediction-ledger reconciliation is independent of virtual-bet
# settlement and runs on app reruns at a bounded ten-minute cadence.
_forecast_sync = _sync_forecast_results()
if _forecast_sync and (
    _forecast_sync.get("forecasts_settled", 0)
    or _forecast_sync.get("market_forecasts_settled", 0)
):
    st.toast(
        "📊 Подтверждены результаты "
        f"прогнозов: {_forecast_sync.get('forecasts_settled', 0)} · "
        f"рыночных рекомендаций: {_forecast_sync.get('market_forecasts_settled', 0)}",
        icon="✅",
    )

_bets = [b for b in D.get("bets", []) if isinstance(b, dict)]
pending_count = sum(1 for b in _bets if b.get("status") == "pending")
stale_count = sum(1 for b in _bets
                  if b.get("status") == "pending" and b.get("stale"))


# ==================== MODEL HEALTH ====================
def _ece(pairs, n_bins: int = 5) -> float:
    bins = [[] for _ in range(n_bins)]
    for p, y in pairs:
        bins[min(int(p * n_bins), n_bins - 1)].append((p, y))
    total = len(pairs)
    ece = 0.0
    for b in bins:
        if b:
            mp = sum(p for p, _ in b) / len(b)
            my = sum(y for _, y in b) / len(b)
            ece += len(b) / total * abs(mp - my)
    return ece


def _model_health(D):
    closed = [b for b in D.get("bets", [])
              if isinstance(b, dict) and b.get("status") in ("won", "lost")]
    calibration = []
    for b in closed:
        try:
            p = float(b.get("prob"))
        except (TypeError, ValueError):
            continue
        if 0 <= p <= 1:
            calibration.append((p, 1.0 if b.get("status") == "won" else 0.0))

    sample = len(calibration)
    brier = cal_error = None
    if calibration:
        brier = sum((p - y) ** 2 for p, y in calibration) / sample
        cal_error = _ece(calibration)

    clv = {}
    try:
        clv = db.clv_summary() or {}
    except Exception:
        clv = {}
    clv_n = int(clv.get("n", 0) or 0)
    clv_avg = float(clv.get("avg_clv", 0) or 0)

    checks = []
    if sample < 20:
        checks.append(("neutral", "Мало данных", f"{sample} ставок для оценки модели"))
    else:
        if brier is not None and brier > 0.25:
            checks.append(("warn", "Brier выше 0.25", f"{brier:.3f}"))
        if sample >= 50 and cal_error is not None and cal_error > 0.10:
            checks.append(("warn", "ECE > 10 п.п.", f"{cal_error * 100:.1f} п.п."))
    if clv_n >= 20 and clv_avg < 0:
        checks.append(("warn", "CLV отрицательный", f"{clv_avg * 100:+.2f}%"))

    warnings = [x for x in checks if x[0] == "warn"]
    base = {"details": warnings or checks, "sample": sample, "brier": brier,
            "cal_error": cal_error, "clv": clv_avg, "clv_n": clv_n}
    if warnings:
        return {**base, "status": "ATTENTION",
                "label": "⚠️ MODEL HEALTH · ATTENTION"}
    if sample >= 20:
        return {**base, "status": "OK", "label": "🟢 MODEL HEALTH · OK"}
    return {**base, "status": "MONITORING", "label": "🟡 MODEL HEALTH · MONITORING"}


model_health = _model_health(D)

TERMINAL_SECTIONS = {
    "🏠 Дашборд": (
        "SPORTS INTELLIGENCE", "OVERVIEW",
        "Главная панель: прогнозы, журнал результатов и качество модели.", "SYSTEM ONLINE",
    ),
    "🏟 Сканер матчей": (
        "MATCH INTELLIGENCE", "LIVE SCANNER",
        "Поиск матчей, проверка источников и предматчевая аналитика.", "LIVE WORKSPACE",
    ),
    "📊 Прогнозы": (
        "MODEL OUTPUT", "AI FORECASTS",
        "Вероятности основной модели и подтверждённые результаты.", "FORECAST DESK",
    ),
    "💎 Value / портфель": (
        "MARKET EDGE", "VALUE / PORTFOLIO",
        "Реальные котировки, Edge и управление виртуальным портфелем.", "VALUE CONTROL",
    ),
    "🧭 Качество данных": (
        "SOURCE MONITOR", "DATA QUALITY",
        "Составы, травмы, судьи, линии и свежесть данных.", "DATA AUDIT",
    ),
    "🧠 Обучение": (
        "PERFORMANCE REVIEW", "AI LEARNING CENTER",
        "Разбор ошибок на подтверждённых результатах.", "LEARNING AUDIT",
    ),
    "🛡️ Риск": (
        "EXPOSURE MONITOR", "RISK CONTROL",
        "Экспозиция, концентрация и дневные лимиты.", "RISK DESK",
    ),
    "🧪 Model Lab": (
        "VALIDATION WORKBENCH", "MODEL LAB",
        "Brier, Log Loss, калибровка и независимая проверка.", "RESEARCH ONLY",
    ),
    "⚙️ API Center": (
        "INTEGRATION HEALTH", "API CENTER",
        "Статус ключей, дневные квоты и подключения источников.", "API MONITOR",
    ),
    "🧰 Инструменты": (
        "ANALYST TOOLKIT", "TOOLS",
        "Калькулятор вероятностей, банкролла и исторический бэктест.", "TOOLBOX",
    ),
}

# Apply deferred navigation before the keyed radio widget is instantiated.
# Streamlit forbids changing a widget's session_state key later in the same run.
_pending_terminal_navigation = st.session_state.pop(
    "_pending_terminal_navigation_v16", None
)
if _pending_terminal_navigation in TERMINAL_SECTIONS:
    st.session_state["terminal_navigation_v16"] = _pending_terminal_navigation

# ==================== SIDEBAR NAVIGATION & SETTINGS ====================
with st.sidebar:
    st.markdown(
        "<div class='neo-side-brand'><div class='neo-side-mark'>⚽</div>"
        "<div><b>Sport-AI-Pro13</b><span>NEURO BET PRO · AI FOOTBALL</span></div></div>",
        unsafe_allow_html=True,
    )
    active_section = st.radio(
        "Рабочая область", list(TERMINAL_SECTIONS.keys()),
        label_visibility="collapsed", key="terminal_navigation_v16",
    )
    _backup_ok = bool(
        (st.session_state.get("_sqlite_backup_status") or {}).get("ok")
        or (st.session_state.get("_sqlite_remote_status_check") or {}).get("ok")
    )
    _model_health_label = {
        "OK": "Портфель · контроль в норме",
        "ATTENTION": "Портфель · требуется внимание",
        "MONITORING": "Портфель · мало завершённых ставок",
    }.get(str(model_health.get("status") or ""), "Модель · статус неизвестен")
    _backup_label = "Резервная копия · подтверждена" if _backup_ok else "Резервная копия · нет свежего подтверждения"
    st.markdown(
        "<div class='neo-side-status'><div class='neo-side-status-head'>"
        "<i class='neo-side-status-dot'></i> Система активна</div>"
        f"<small>{_model_health_label}<br/>Матчей в скане: {len(D.get('cards') or [])}"
        f"<br/>{_backup_label}<br/>Устаревших ставок: {stale_count}"
        f"<br/>Версия: {APP_VERSION} · сборка: {APP_BUILD_ID}</small></div>",
        unsafe_allow_html=True,
    )
    with st.expander("⚙️ Настройки и обслуживание", expanded=False):
        st.markdown(
            "<div style='font-size:1.1rem;font-weight:800;color:#e6eaf2;"
            "margin-bottom:12px;'>🔑 API ключи</div>",
            unsafe_allow_html=True)

        # Never send server-stored credentials to the browser as widget defaults.
        # An empty field means "use the server-side configured/saved value".
        fdorg_token_input = st.text_input(
            "football-data.org token",
            value="",
            type="password", key="fdorg_token_input")
        fdorg_token = str(
            fdorg_token_input or _api_key_default(
                "fdorg_token", "FDORG_TOKEN", "fdorg_token"
            )
        ).strip()
        sstats_key_input = st.text_input(
            "SStats.net API key (бесплатные футбольные коэффициенты)",
            value="",
            type="password", key="sstats_api_key_input",
            help="Оставьте пустым, чтобы использовать ключ из серверной конфигурации/хранилища. Введите значение только для замены. Ключ не подставляется в поле из хранилища.")
        sstats_key = str(
            sstats_key_input or _api_key_default(
                "sstats_api_key", "SSTATS_API_KEY", "sstats_api_key"
            )
        ).strip()

        footballapi_key_input = st.text_input(
            "Football API key (xG · бесплатно, текущий сезон АПЛ)",
            value="",
            type="password", key="footballapi_key_input",
            help="Оставьте пустым, чтобы использовать ключ из серверной конфигурации/хранилища. Введите значение только для замены. Даёт xG обеих команд для завершённых матчей текущего сезона Премьер-лиги.")
        footballapi_key = str(
            footballapi_key_input or _api_key_default(
                "footballapi_key", "FOOTBALLAPI_KEY", "footballapi_key"
            )
        ).strip()

        st.caption("SStats — единственный источник списка матчей и букмекерских коэффициентов. Контекст (составы, травмы и судьи) запрашивается у SStats там, где API подтверждает эти данные.")

        if st.button("🔎 Проверить подключение xG Football API", key="check_footballapi_xg",
                     width="stretch"):
            with st.spinner("Проверяю ключ и дневную квоту Football API…"):
                _footballapi_check = footballapi_xg_connection_check(footballapi_key)
            if _footballapi_check.get("ok"):
                st.success(_footballapi_check.get("message", "Football API подключён."))
                st.caption("xG доступен только в рамках покрытия текущего сезона АПЛ на бесплатном тарифе.")
            else:
                st.error(_footballapi_check.get("message", "Не удалось проверить Football API."))
        if st.button("🔎 Проверить подключение SStats.net", key="check_sstats_api",
                     width="stretch"):
            with st.spinner("Проверяю ключ SStats, расписание и Odds…"):
                _sstats_check = sstats_account_check(sstats_key)
            if (_sstats_check.get("ok") and _sstats_check.get("games_ok")
                    and _sstats_check.get("total_line_count", 0) > 0):
                st.success(_sstats_check.get("message", "SStats подключён; подтверждённые тоталы найдены."))
            elif _sstats_check.get("ok") and _sstats_check.get("games_ok"):
                st.warning(_sstats_check.get(
                    "message",
                    "Ключ и расписание работают, но подтверждённые тоталы в тестовой выборке не найдены."
                ))
            elif _sstats_check.get("ok"):
                st.warning(_sstats_check.get("message", "Ключ SStats подтверждён, но данные матчей недоступны."))
            else:
                st.error(_sstats_check.get("message", "Не удалось подключиться к SStats.net."))

        st.markdown(
            "<div style='font-size:.72rem;color:#8b93a7;margin-top:14px;"
            "margin-bottom:6px;'>🤖 LLM-аналитик</div>",
            unsafe_allow_html=True)
        prov_keys = list(LLM_PROVIDERS.keys())
        # Keep a stable widget key so the one-click Grok switch survives the
        # Streamlit rerun and is then persisted in the user's normal app state.
        provider_widget_key = "llm_provider_select_v2"
        if provider_widget_key not in st.session_state:
            st.session_state[provider_widget_key] = normalize_llm_provider(
                D["meta"].get("llm_provider") or "Groq GPT-OSS 120B"
            )
        else:
            normalized_provider = normalize_llm_provider(
                st.session_state.get(provider_widget_key)
            )
            if st.session_state.get(provider_widget_key) != normalized_provider:
                st.session_state[provider_widget_key] = normalized_provider
        if st.button(
            "⚡ Переключить на Groq GPT-OSS 120B",
            key="select_groq_120b_provider",
            width="stretch",
            help="Рабочая модель 120B через Groq API: openai/gpt-oss-120b. Использует ключ Groq, не ключ xAI.",
        ):
            st.session_state[provider_widget_key] = "Groq GPT-OSS 120B"
            D["meta"]["llm_provider"] = "Groq GPT-OSS 120B"
            D["meta"]["llm_model"] = str(LLM_PROVIDERS["Groq GPT-OSS 120B"]["model"])
            usage.set_local_data(D)
            st.rerun()
        if st.button(
            "Переключить на Grok (x.ai)",
            key="select_xai_grok_provider",
            width="stretch",
            help="Отдельный API xAI и модель grok-4.7; требует отдельный ключ XAI_API_KEY.",
        ):
            st.session_state[provider_widget_key] = "Grok (x.ai)"
            D["meta"]["llm_provider"] = "Grok (x.ai)"
            D["meta"]["llm_model"] = str(LLM_PROVIDERS["Grok (x.ai)"]["model"])
            usage.set_local_data(D)
            st.rerun()
        llm_prov = st.selectbox(
            "Провайдер", prov_keys,
            index=prov_keys.index(normalize_llm_provider(
                st.session_state.get(provider_widget_key)
            )),
            key=provider_widget_key,
        )
        llm_key_config = LLM_PROVIDERS.get(llm_prov, {})
        llm_key_env = str(llm_key_config.get("key_env") or "LLM_API_KEY")
        llm_key_storage = str(llm_key_config.get("key_storage") or "llm_api_key")
        llm_key_session = str(llm_key_config.get("key_session") or "llm_api_key_input")
        llm_key_label = "XAI_API_KEY · Grok" if llm_key_env == "XAI_API_KEY" else "LLM API key"
        llm_key_default = _api_key_default(
            llm_key_storage, llm_key_env, llm_key_env, llm_key_storage,
        )
        llm_key_input = st.text_input(
            llm_key_label,
            value="",
            type="password", key=llm_key_session,
            help="Оставьте пустым, чтобы использовать ключ из серверной конфигурации/хранилища. Введите значение только для замены.")
        llm_key = str(llm_key_input or llm_key_default).strip()
        if llm_key_env == "XAI_API_KEY" and not llm_key_default:
            st.caption(
                "Для Grok нужен отдельный API-ключ xAI. Создай ключ в консоли xAI, "
                "вставь его в поле XAI_API_KEY и нажми проверку подключения."
            )
            st.markdown("[Открыть консоль xAI и создать API key](https://console.x.ai)")
        if st.button(
            f"🔎 Проверить AI-подключение · {llm_prov}",
            key="check_llm_provider_connection",
            width="stretch",
            help="Отправляет короткий тестовый запрос выбранной модели. Провайдер может списать несколько токенов.",
        ):
            with st.spinner(f"Проверяю реальное подключение к {llm_prov}…"):
                _llm_check = check_provider_connection(llm_prov, llm_key)
            if _llm_check.get("ok"):
                st.success(_llm_check.get("message") or "AI API отвечает.")
            else:
                st.error(_llm_check.get("message") or "AI API не ответил.")
        # Save only UI-entered keys, separate from portfolio data and its Gist sync.
        _widget_key_map = {
            "fdorg_token": ("fdorg_token_input", ("FDORG_TOKEN",), ("FDORG_TOKEN", "fdorg_token")),
            "sstats_api_key": ("sstats_api_key_input", ("SSTATS_API_KEY",), ("SSTATS_API_KEY", "sstats_api_key")),
            "footballapi_key": ("footballapi_key_input", ("FOOTBALLAPI_KEY",), ("FOOTBALLAPI_KEY", "footballapi_key")),
            "llm_api_key": ("llm_api_key_input", ("LLM_API_KEY",), ("LLM_API_KEY", "llm_api_key")),
            "xai_api_key": ("xai_api_key_input", ("XAI_API_KEY",), ("XAI_API_KEY", "xai_api_key")),
        }
        _user_entered_keys = {}
        for _storage_key, (_widget_key, _env_names, _secret_names) in _widget_key_map.items():
            _current = str(st.session_state.get(_widget_key, "") or "").strip()
            _configured = _configured_key_value(_env_names, _secret_names)
            if _current and _current != _configured:
                _user_entered_keys[_storage_key] = _current
        # Preserve credentials for providers not selected in this rerun.
        _keys_to_save = {**_saved_ui_secrets, **_user_entered_keys}
        _keys_saved = save_local_secrets(_keys_to_save)
        if not _keys_saved:
            st.caption("⚠️ Не удалось сохранить ключи на диске. Проверьте права записи хостинга.")
        _api_key_labels = {
            "fdorg_token": "football-data.org",
            "sstats_api_key": "SStats.net",
            "footballapi_key": "Football API xG",
            "llm_api_key": "Groq / Gemini",
            "xai_api_key": "Grok xAI",
        }
        _api_key_status = {}
        for _storage_key, (_widget_key, _env_names, _secret_names) in _widget_key_map.items():
            if any(os.environ.get(name, "").strip() for name in _env_names):
                _api_key_status[_storage_key] = "серверная переменная окружения"
            elif _configured_secret(*_secret_names, "api_key", "key" if _storage_key == "sstats_api_key" else ""):
                _api_key_status[_storage_key] = "Streamlit Secrets"
            elif _storage_key in _keys_to_save and _keys_saved:
                _api_key_status[_storage_key] = "локальный файл приложения"
            elif _storage_key in _user_entered_keys:
                _api_key_status[_storage_key] = "введён, но не сохранён"
            else:
                _api_key_status[_storage_key] = "не настроен"

        st.markdown("**Состояние хранения ключей**")
        st.caption(
            "Поля ввода намеренно не заполняются при перезагрузке браузера. "
            "Пустое поле само по себе не означает, что ключ потерян: сканер читает "
            "ключ на сервере из ENV, Streamlit Secrets или локального хранилища."
        )
        for _storage_key, _label in _api_key_labels.items():
            st.caption(f"{_label}: {_api_key_status.get(_storage_key, 'не настроен')}")
        _local_only_keys = [
            _api_key_labels[key] for key, state in _api_key_status.items()
            if state == "локальный файл приложения"
        ]
        if _local_only_keys:
            st.warning(
                "Ключи " + ", ".join(_local_only_keys) +
                " есть только в локальном файле. На Streamlit Community Cloud "
                "локальные файлы могут исчезнуть после перезапуска или пересборки. "
                "Для постоянного хранения добавьте их в Manage app → Settings → Secrets."
            )
            st.code(
                'FDORG_TOKEN = "..."\n'
                'SSTATS_API_KEY = "..."\n'
                'FOOTBALLAPI_KEY = "..."\n'
                'LLM_API_KEY = "..."\n'
                'XAI_API_KEY = "..."',
                language="toml",
            )
        elif not any(
            state in {"серверная переменная окружения", "Streamlit Secrets"}
            for state in _api_key_status.values()
        ):
            st.info(
                "Ни один API-ключ не найден в постоянной серверной конфигурации. "
                "Сканирование будет работать только с доступными публичными источниками."
            )

        # Keep provider and model in sync. Older saved sessions can retain a
        # Groq model ID after changing the provider to Grok; that makes xAI return
        # an empty/HTTP error and leaves every independent analysis at zero.
        _configured_llm_model = str(llm_key_config.get("model") or "")
        if (
            llm_prov != D["meta"].get("llm_provider")
            or _configured_llm_model != str(D["meta"].get("llm_model") or "")
        ):
            D["meta"]["llm_provider"] = llm_prov
            D["meta"]["llm_model"] = _configured_llm_model
            usage.set_local_data(D)

        st.markdown("<hr style='border-color:rgba(255,255,255,.08);margin:16px 0;'>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.72rem;color:#8b93a7;margin-bottom:6px;'>"
            "🎯 Параметры</div>",
            unsafe_allow_html=True)
        min_prob = st.slider(
            "Мин. P для отбора Value %",
            30, 85, 40, 1,
            help=(
                "Минимум вероятности для Value-кандидатов. Отдельная метка «Сильный прогноз» "
                "присваивается при вероятности от 65%; этот порог не заменяет историческую "
                "проверку рынка, Edge, EV и Kelly."
            ),
        ) / 100
        min_edge = st.slider("Мин. Edge %", 0, 15, 3, 1) / 100
        kelly_frac = 0.25
        st.caption("Риск-профиль зафиксирован: ¼ Kelly · максимум 1% банка на рынок · 2% на матч · 5% суммарно за день.")
        st.caption(f"Версия приложения: {APP_VERSION} · build 2026-11-15")
        matrix_n = st.slider("Матрица", 6, 15, 12, 1)

        st.markdown("<hr style='border-color:rgba(255,255,255,.08);margin:16px 0;'>",
                    unsafe_allow_html=True)

        if st.button(
            "🔃 Проверить все результаты",
            width="stretch",
            help=(
                "Проверяет до 60 уникальных завершённых матчей за запуск. "
                "Один найденный счёт может закрыть несколько записей прогнозов. "
                "Повторно не опрашивает один и тот же матч чаще чем раз в 10 минут."
            ),
        ):
            with st.spinner("Проверяю портфель и все журналы прогнозов…"):
                D2, n, tried = _auto_settle(D, force=True)
                forecast_sync = _sync_forecast_results(force=True)
            _apply_settle_result(D2, n, tried, "manual_check")
            forecast_sync = forecast_sync or {}
            st.success(
                f"Закрыто ставок: {n} · подтверждено прогнозов модели/ИИ: "
                f"{forecast_sync.get('forecasts_settled', 0)} · "
                f"закрыто рыночных рекомендаций: {forecast_sync.get('market_forecasts_settled', 0)} · "
                f"проверено матчей: {forecast_sync.get('fixtures_checked', 0)} · "
                f"осталось прогнозов: {forecast_sync.get('pending_forecasts', 0)}"
            )
            st.rerun()

        _automatic_result_sync_widget()
        _automatic_sqlite_backup_widget()

        _sync_status = st.session_state.get("_forecast_ledger_sync_status") or {}
        if _sync_status.get("error"):
            st.caption("Сверка прогнозов: ошибка — " + str(_sync_status["error"]))
        elif _sync_status:
            st.caption(
                "Последняя сверка журналов: "
                f"матчей проверено {_sync_status.get('fixtures_checked', 0)} · "
                f"прогнозов закрыто {_sync_status.get('forecasts_settled', 0)} · "
                f"рыночных рекомендаций закрыто {_sync_status.get('market_forecasts_settled', 0)} · "
                f"ожидают FT {_sync_status.get('pending_forecasts', 0)} записей · "
                f"уникальных матчей {_sync_status.get('pending_unique_fixtures', 0)} · "
                f"ещё не прошли 2ч15 {_sync_status.get('not_due_grace_fixtures', 0)} · "
                f"в начале очереди {_sync_status.get('eligible_fixtures', 0)} · "
                f"осталось матчей {_sync_status.get('eligible_fixtures_remaining', 0)} · "
                f"старше окна {_sync_status.get('age_limited_fixtures', 0)} · "
                f"ожидают повтора {_sync_status.get('retry_deferred_fixtures', 0)} · "
                f"не найден результат {_sync_status.get('not_found_fixtures', 0)} · "
                f"SStats запросов {_sync_status.get('sstats_lookups', 0)} · "
                f"источники {', '.join(f'{name}:{count}' for name, count in sorted((_sync_status.get('result_sources') or {}).items())) or 'нет'} · "
                f"ошибок {_sync_status.get('errors', 0)}"
            )

        if stale_count and st.button(f"↩️ Вернуть стейки по устаревшим ({stale_count})",
                                     width="stretch",
                                     help="Помечает ставки как void и возвращает стейк в банк"):
            D2, n = _void_stale(D)
            _apply_settle_result(D2, n, 0, "void_stale")
            usage.set_local_data(D2)
            st.toast(f"Возвращены стейки: {n}")
            st.rerun()

        if st.button("🔄 Очистить карточки", width="stretch"):
            D["cards"] = []
            D["funnel"] = None
            D["report"] = []
            usage.set_local_data(D)
            st.toast("Карточки очищены")
            st.rerun()

        if st.button("♻️ Сбросить счётчики", width="stretch"):
            usage.settle_reset()
            usage.llm_reset()
            usage.odds_reset()
            usage.fdorg_reset()
            st.toast("Счётчики сброшены")
            st.rerun()

        st.markdown("<hr style='border-color:rgba(255,255,255,.08);margin:16px 0;'>",
                    unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:.72rem;color:#8b93a7;margin-bottom:6px;'>"
            "💾 Бэкап (API-ключи не включаются)</div>",
            unsafe_allow_html=True)

        # Local SQLite is independent from neuro_local.json and is not included in
        # the ordinary portfolio JSON backup. Offer a direct encrypted-file export
        # even before a remote Gist is configured.
        with st.expander("🛡️ Защита журнала прогнозов SQLite", expanded=False):
            _restore_state = st.session_state.get("_sqlite_restore_status") or {}
            if _restore_state.get("restored"):
                st.success(
                    "SQLite восстановлена из зашифрованной резервной копии. "
                    "Проверьте число строк и последние результаты."
                )
            elif _restore_state.get("state") == "local_data_present":
                st.caption("Локальная SQLite содержит записи; автоматическое восстановление её не заменяло.")
            elif _restore_state.get("state") == "not_configured":
                st.warning(
                    "Удалённая копия SQLite пока не настроена. Локальная база на Streamlit Cloud "
                    "может исчезнуть при пересоздании контейнера."
                )
            elif _restore_state.get("state") == "restore_error":
                st.error(str(_restore_state.get("message") or "Не удалось проверить резервную копию."))

            _remote_state = st.session_state.get("_sqlite_backup_status") or {}
            if _remote_state.get("ok"):
                st.success(
                    "Зашифрованная копия: "
                    + str(_remote_state.get("backup_at") or "сохранена")
                    + " · "
                    + str(_remote_state.get("bytes") or 0)
                    + " байт"
                )
            elif _remote_state.get("state") == "backup_error":
                st.error(str(_remote_state.get("message") or "Ошибка резервного копирования."))

            st.caption(
                "Для автоматической резервной копии достаточно одного секрета Streamlit: "
                "GITHUB_TOKEN. Приложение само создаст непубличный Secret Gist, найдёт его при "
                "следующих запусках и выведет ключ шифрования из токена. Нужен Fine-grained token "
                "только с правом Gists: Read and write. Храни токен в менеджере паролей и не меняй "
                "его без предварительного переноса/восстановления архива: он также используется "
                "для расшифровки старых копий."
            )

            _b1, _b2 = st.columns(2)
            if _b1.button("Проверить удалённую копию", key="check_neuro_remote_backup"):
                st.session_state["_sqlite_remote_status_check"] = remote_backup_status()
            _checked_remote = st.session_state.get("_sqlite_remote_status_check") or {}
            if _checked_remote:
                if _checked_remote.get("ok"):
                    st.success(str(_checked_remote.get("message") or "Удалённая копия доступна."))
                else:
                    st.warning(str(_checked_remote.get("message") or "Удалённая копия не настроена."))

            if _b2.button("Сохранить SQLite в Secret Gist", key="backup_neuro_sqlite_now"):
                with st.spinner("Создаю и проверяю зашифрованную копию SQLite…"):
                    st.session_state["_sqlite_backup_status"] = backup_database_to_gist(DB_FILE)
            if st.button("Подготовить резервную копию SQLite для скачивания", key="prepare_neuro_sqlite_download"):
                try:
                    st.session_state["_sqlite_manual_backup_bytes"] = create_sqlite_backup_bytes(DB_FILE)
                    st.session_state["_sqlite_manual_backup_name"] = (
                        f"neuro_db_{datetime.now():%Y%m%d_%H%M}.sqlite.gz"
                    )
                except Exception as _sqlite_export_exc:
                    st.error(
                        "Не удалось подготовить SQLite-бэкап: "
                        + type(_sqlite_export_exc).__name__
                    )
            if st.session_state.get("_sqlite_manual_backup_bytes"):
                st.download_button(
                    "⬇️ Скачать полную резервную копию SQLite",
                    data=st.session_state["_sqlite_manual_backup_bytes"],
                    file_name=st.session_state.get("_sqlite_manual_backup_name") or "neuro_db.sqlite.gz",
                    mime="application/gzip",
                    key="download_neuro_sqlite_backup",
                    width="stretch",
                )

            _sqlite_upload = st.file_uploader(
                "Восстановить SQLite из .sqlite.gz (только в пустую базу)",
                type=["gz"], key="restore_neuro_sqlite_upload",
            )
            if _sqlite_upload is not None and st.button(
                "Восстановить пустой журнал SQLite",
                key="restore_neuro_sqlite_button",
            ):
                try:
                    _restored_sqlite = restore_sqlite_backup_bytes(
                        _sqlite_upload.getvalue(), DB_FILE, overwrite=False,
                    )
                    db.db_init()
                    db.invalidate_caches()
                    st.session_state["_sqlite_restore_status"] = {
                        "restored": True, "state": "restored",
                        "message": "SQLite восстановлена из локального бэкапа.",
                    }
                    st.success(
                        "SQLite восстановлена. Таблиц с данными: "
                        + str(len(_restored_sqlite.get("tables") or {}))
                    )
                except Exception as _restore_sqlite_exc:
                    st.error(
                        "Восстановление SQLite отменено: "
                        + str(_restore_sqlite_exc)[:180]
                    )

            _audit_upload = st.file_uploader(
                "Восстановить журнал из архива ledger .zip",
                type=["zip"], key="restore_neuro_audit_zip",
            )
            if _audit_upload is not None and st.button(
                "Восстановить прогнозы, рынок и историю банка из audit ZIP",
                key="restore_neuro_audit_zip_button",
            ):
                try:
                    _audit_restored = restore_audit_export_bytes(
                        _audit_upload.getvalue(), DB_FILE,
                    )
                    db.db_init()
                    db.invalidate_caches()
                    st.session_state["_sqlite_restore_status"] = {
                        "restored": True, "state": "restored",
                        "message": "Журнал восстановлен из audit ZIP.",
                    }
                    st.success(
                        "Журнал восстановлен без перезаписи существующих данных: "
                        f"прогнозы {_audit_restored['forecasts']}, "
                        f"рынки {_audit_restored['market_forecasts']}, "
                        f"ставки {_audit_restored['paper_bets']}, "
                        f"банк {_audit_restored['bank_history']}, "
                        f"снимки решений {_audit_restored['decision_snapshots']}, "
                        f"исторические архивы {_audit_restored.get('user_datasets', 0)} "
                        f"({_audit_restored.get('user_dataset_rows', 0)} строк)."
                    )
                except Exception as _audit_restore_exc:
                    st.error(
                        "Восстановление audit ZIP отменено: "
                        + str(_audit_restore_exc)[:220]
                    )

            st.caption(
                "Secret Gist — это непубличная ссылка, а не приватный диск: любой с URL может его открыть. "
                "В новой одно-секретной схеме SQLite шифруется ключом, производным от GITHUB_TOKEN; "
                "старые установки с отдельным NEURO_BACKUP_KEY поддерживаются."
            )

        _backup = json.dumps(_strip_secrets(D), ensure_ascii=False, indent=2, default=str)
        st.download_button("📥 Скачать", _backup,
                           file_name=f"neuro_data_{datetime.now():%Y%m%d_%H%M}.json",
                           mime="application/json",
                           width="stretch")

        _up = st.file_uploader("📤 Загрузить", type=["json"],
                               key="restore_upload",
                               label_visibility="collapsed")
        if _up is not None:
            _up_id = f"{_up.name}:{_up.size}"
            if st.session_state.get("_restored_id") != _up_id:
                st.session_state["_restored_id"] = _up_id
                try:
                    _r = _validate_data(json.loads(_up.read().decode("utf-8")))
                    for _sk in SECRET_KEYS:
                        if D["meta"].get(_sk):
                            _r["meta"][_sk] = D["meta"][_sk]
                    st.session_state.data = _r
                    usage.set_local_data(_r)
                    db.invalidate_caches()
                    st.success("✅ Восстановлено!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Не удалось восстановить: {e}")

        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if not st.session_state.confirm_clear:
            if st.button("🗑️ Очистить портфель", width="stretch"):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("Удалить ВСЕ ставки?")
            cc1, cc2 = st.columns(2)
            if cc1.button("Да", key="confirm_yes"):
                D["bets"] = []
                D["cards"] = []
                D["bank"] = DEFAULT_BANK
                D["stats"] = {"won": 0, "lost": 0, "profit": 0, "push": 0, "void": 0}
                D["meta"]["initial_bank"] = DEFAULT_BANK
                usage.set_local_data(D)
                db.invalidate_caches()
                st.session_state.confirm_clear = False
                st.rerun()
            if cc2.button("Нет", key="confirm_no"):
                st.session_state.confirm_clear = False
                st.rerun()
    # API keys, risk settings, settlement and backup controls remain available here.

# ==================== ACTIVE WORKSPACE ====================
section_kicker, section_title, section_subtitle, section_status = TERMINAL_SECTIONS[active_section]
if active_section != "🏠 Дашборд":
    st.markdown(f"""
    <div class="terminal-section-head">
      <div>
        <div class="terminal-section-kicker">{section_kicker} / NEURO BET PRO v{APP_VERSION}</div>
        <div class="terminal-section-title">{section_title}</div>
        <div class="terminal-section-subtitle">{section_subtitle}</div>
      </div>
      <div class="terminal-section-status">● {section_status}</div>
    </div>
    """, unsafe_allow_html=True)

if active_section == "🏠 Дашборд":
    dashboard.render()
elif active_section == "🏟 Сканер матчей":
    scanner.render(min_prob, kelly_frac, matrix_n, min_edge)
elif active_section == "📊 Прогнозы":
    stats.render()
elif active_section == "💎 Value / портфель":
    portfolio.render()
elif active_section == "🧭 Качество данных":
    analytics.render_data_quality()
elif active_section == "🧠 Обучение":
    analytics.render_learning_center()
elif active_section == "🛡️ Риск":
    analytics.render_portfolio_risk()
elif active_section == "🧪 Model Lab":
    analytics.render_model_lab()
elif active_section == "⚙️ API Center":
    analytics.render_api_center()
else:
    calculator.render(matrix_n)
    st.divider()
    backtest.render(matrix_n)
