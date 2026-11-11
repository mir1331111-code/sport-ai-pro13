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

from config import (APP_VERSION, LLM_PROVIDERS, DATA_VERSION,  # noqa: E402
                    AUTO_SETTLE_THROTTLE_SEC, DB_FILE)
from storage import sqlite_store as db  # noqa: E402
from storage import usage  # noqa: E402
from storage.secret_store import load_local_secrets, save_local_secrets  # noqa: E402
from data.sources import (
    parse_date, fdorg_match_result, highlightly_connection_check,
    sstats_account_check, sstats_match_statistics, sstats_find_match_id,
    footballapi_xg_connection_check,
)  # noqa: E402
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
from ui import theme  # noqa: E402
from ui.tabs import scanner, portfolio, stats, calculator, backtest  # noqa: E402
from ui.tabs import (
    data_quality, ai_forecasts, value_engine, learning, model_lab, api_center,
)  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[APP] %(message)s")
log = logging.getLogger("neurobet")
if _SETTLEMENT_READY_FALLBACK:
    log.warning("betting.settlement is older than App.py; using conservative local settlement gate")

# ==================== КОНСТАНТЫ ====================
DEFAULT_BANK = 10000.0
AUTO_SETTLE_EVERY_SEC = AUTO_SETTLE_THROTTLE_SEC
SETTLE_MIN_AGE = timedelta(hours=2, minutes=15)
RETRY_EVERY = timedelta(minutes=60)
STALE_AFTER = timedelta(hours=48)
MAX_PER_RUN = 20
SECRET_KEYS = (
    "fdorg_token", "highlightly_api_key", "sstats_api_key",
    "footballapi_key", "api_football_key", "llm_api_key",
)


def _configured_secret(*keys):
    """Read root Streamlit secrets and common nested API sections."""
    try:
        for key in keys:
            value = st.secrets.get(key, "")
            if value:
                return str(value).strip()
        for section_name in ("api", "apis", "secrets", "highlightly", "sstats", "footballapi", "football_api"):
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

FINISHED_STATUSES = {"finished", "ft", "final", "closed", "ended",
                     "full-time", "full_time", "completed"}


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
    if not isinstance(res, dict):
        return False
    try:
        int(res["home"]), int(res["away"])
    except (KeyError, TypeError, ValueError):
        return False
    status = str(res.get("status", "")).strip().lower()
    return (not status) or status in FINISHED_STATUSES


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
_secret_keys = ("fdorg_token", "odds_api_key", "api_football_key", "therundown_api_key", "rundown_key",
                "highlightly_api_key", "llm_api_key")
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
    token = (st.session_state.get("fdorg_token_input", "")
             or os.environ.get("FDORG_TOKEN", "")
             or _configured_secret("FDORG_TOKEN", "fdorg_token")).strip()
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
                    res = find_match_score(h_team, a_team, d_iso)
                    if res:
                        source = res.get("source", "auto")
                except Exception as exc:
                    log.warning("find_match_score error: %s", type(exc).__name__)

            if not _is_finished(res):
                res = None
                fixture_id = b.get("fixture_id")
                if fixture_id and token:
                    try:
                        res = fdorg_match_result(fixture_id, token)
                        if res:
                            source = "fdorg"
                    except Exception as exc:
                        log.warning("fdorg error: %s", type(exc).__name__)

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


_now_ts = time.time()
if _now_ts - st.session_state.get("_last_auto_settle_ts", 0) > AUTO_SETTLE_EVERY_SEC:
    st.session_state["_last_auto_settle_ts"] = _now_ts
    with st.spinner("Проверяю результаты матчей…"):
        D2, n, tried = _auto_settle(D)
    _apply_settle_result(D2, n, tried, "auto_settle")
    D = D2
    if n > 0:
        st.toast(f"🔄 Авто-закрыто {n} ставок", icon="✅")

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

# ==================== HERO ====================
_today = datetime.now().strftime("%Y-%m-%d")
signals_today = sum(
    1 for c in D.get("cards", [])
    if isinstance(c, dict)
    and (c.get("date_iso") or "").startswith(_today)
    and c.get("best") is not None)
roi_pct, _profit, _turnover = _roi_from_bets(_bets)
roi_cls = "r" if roi_pct < 0 else "g"
bank_now = float(D.get("bank", DEFAULT_BANK))
mode_label = str(D.get("mode", "paper")).upper()

st.markdown(f"""
<div class="hero">
<div class="hero-head">
<div>
<div class="hero-eyebrow">TODAY · VALUE TERMINAL</div>
<h1>NEURO BET PRO</h1>
<p>Модель → рынок → value → портфель</p>
</div>
<div class="hero-status"><span class="status-dot"></span>LIVE · {mode_label}</div>
</div>
<div class="kpis">
<div class="kpi"><div class="t">Банкролл</div>
<div class="v y">{bank_now:.0f} у.е.</div></div>
<div class="kpi"><div class="t">Сигналы сегодня</div><div class="v">{signals_today}</div></div>
<div class="kpi"><div class="t">Активные ставки</div><div class="v">{pending_count}</div></div>
<div class="kpi"><div class="t">ROI (от оборота)</div>
<div class="v {roi_cls}">{roi_pct:+.1f}%</div></div>
</div></div>""", unsafe_allow_html=True)

if model_health["status"] == "ATTENTION":
    _health_items = " · ".join(f"{x[1]} ({x[2]})" for x in model_health["details"])
    st.warning(
        f"{model_health['label']} — {_health_items}. "
        "Это диагностический сигнал, а не автоматическое изменение параметров ставок.")
elif model_health["status"] == "MONITORING":
    st.info(
        f"{model_health['label']} — закрытых ставок с P: "
        f"{model_health['sample']}. Нужна более длинная выборка для оценки.")
else:
    st.success(
        f"{model_health['label']} · Brier {model_health['brier']:.3f} · "
        f"ECE {model_health['cal_error'] * 100:.1f} п.п. · "
        f"CLV {model_health['clv'] * 100:+.2f}%")

if stale_count:
    st.warning(
        f"⏳ {stale_count} ставок не удалось закрыть автоматически более 48 ч. "
        "Стейк по ним заморожен, в статистику они не входят. Проверьте результат "
        "вручную во вкладке «Портфель» или верните стейки кнопкой в боковой панели.")

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(
        "<div style='font-size:1.1rem;font-weight:800;color:#e6eaf2;"
        "margin-bottom:12px;'>🔑 API ключи</div>",
        unsafe_allow_html=True)

    fdorg_token = st.text_input(
        "football-data.org token",
        value=_api_key_default("fdorg_token", "FDORG_TOKEN", "fdorg_token"),
        type="password", key="fdorg_token_input")
    highlightly_key = st.text_input(
        "Highlightly API key (коэффициенты; нужен платный тариф)",
        value=_api_key_default("highlightly_api_key", "HIGHLIGHTLY_API_KEY", "highlightly_api_key"),
        type="password", key="highlightly_api_key_input",
        help="Реальные prematch odds из Highlightly. На Free/Basic endpoint коэффициентов недоступен.")

    sstats_key = st.text_input(
        "SStats.net API key (бесплатные футбольные коэффициенты)",
        value=_api_key_default("sstats_api_key", "SSTATS_API_KEY", "sstats_api_key"),
        type="password", key="sstats_api_key_input",
        help="Ключ SStats.net. Сканирование получает рынки; после FT API также закрывает ставки на угловые и карточки по фактической статистике.")

    footballapi_key = st.text_input(
        "Football API key (xG · бесплатно, текущий сезон АПЛ)",
        value=_api_key_default("footballapi_key", "FOOTBALLAPI_KEY", "footballapi_key"),
        type="password", key="footballapi_key_input",
        help="Даёт xG обеих команд для завершённых матчей текущего сезона Премьер-лиги. Бесплатный тариф: 50 запросов в день.")

    api_football_key = st.text_input(
        "API-Football key (составы · травмы · дисквалификации · судьи)",
        value=_api_key_default("api_football_key", "API_FOOTBALL_KEY", "api_football_key"),
        type="password", key="api_football_key_input",
        help="Вводится здесь, в панели API-ключей. Сохраняется в приватном локальном хранилище вместе с остальными ключами и автоматически используется сканером."
    )
    if st.button("🔎 Проверить подключение xG Football API", key="check_footballapi_xg",
                 use_container_width=True):
        with st.spinner("Проверяю ключ и дневную квоту Football API…"):
            _footballapi_check = footballapi_xg_connection_check(footballapi_key)
        if _footballapi_check.get("ok"):
            st.success(_footballapi_check.get("message", "Football API подключён."))
            st.caption("xG доступен только в рамках покрытия текущего сезона АПЛ на бесплатном тарифе.")
        else:
            st.error(_footballapi_check.get("message", "Не удалось проверить Football API."))
    if st.button("🔎 Проверить подключение SStats.net", key="check_sstats_api",
                 use_container_width=True):
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

    if st.button("🔎 Проверить доступ к коэффициентам", key="check_highlightly_odds",
                 use_container_width=True):
        with st.spinner("Проверяю ключ и endpoint Highlightly /odds…"):
            _highlightly_check = highlightly_connection_check(highlightly_key)
        if _highlightly_check.get("ok"):
            st.success(_highlightly_check.get("message", "Highlightly /odds доступен."))
            st.caption(f"API host: {_highlightly_check.get('base', 'автовыбор')}")
        elif _highlightly_check.get("matches_ok"):
            st.warning(_highlightly_check.get("message", "Ключ работает, но доступ к коэффициентам не подтверждён."))
        else:
            st.error(_highlightly_check.get("message", "Не удалось проверить Highlightly."))

    st.markdown(
        "<div style='font-size:.72rem;color:#8b93a7;margin-top:14px;"
        "margin-bottom:6px;'>🤖 LLM-аналитик</div>",
        unsafe_allow_html=True)
    prov_keys = list(LLM_PROVIDERS.keys())
    cur = D["meta"].get("llm_provider", prov_keys[0])
    llm_prov = st.selectbox("Провайдер", prov_keys,
                            index=prov_keys.index(cur) if cur in prov_keys else 0)
    llm_key = st.text_input(
        "LLM key",
        value=_api_key_default("llm_api_key", "LLM_API_KEY", "llm_api_key"),
        type="password", key="llm_api_key_input")
    # Save only UI-entered keys, separate from portfolio data and its Gist sync.
    _widget_key_map = {
        "fdorg_token": ("fdorg_token_input", ("FDORG_TOKEN",), ("FDORG_TOKEN", "fdorg_token")),
        "highlightly_api_key": ("highlightly_api_key_input", ("HIGHLIGHTLY_API_KEY",), ("HIGHLIGHTLY_API_KEY", "highlightly_api_key")),
        "sstats_api_key": ("sstats_api_key_input", ("SSTATS_API_KEY",), ("SSTATS_API_KEY", "sstats_api_key")),
        "footballapi_key": ("footballapi_key_input", ("FOOTBALLAPI_KEY",), ("FOOTBALLAPI_KEY", "footballapi_key")),
        "api_football_key": ("api_football_key_input", ("API_FOOTBALL_KEY",), ("API_FOOTBALL_KEY", "api_football_key")),
        "llm_api_key": ("llm_api_key_input", ("LLM_API_KEY",), ("LLM_API_KEY", "llm_api_key")),
    }
    _user_entered_keys = {}
    for _storage_key, (_widget_key, _env_names, _secret_names) in _widget_key_map.items():
        _current = str(st.session_state.get(_widget_key, "") or "").strip()
        _configured = _configured_key_value(_env_names, _secret_names)
        if _current and _current != _configured:
            _user_entered_keys[_storage_key] = _current
    if not save_local_secrets(_user_entered_keys):
        st.caption("⚠️ Не удалось сохранить ключи на диске. Для постоянного хранения настройте Secrets/Environment на хостинге.")
    if llm_prov != D["meta"].get("llm_provider"):
        D["meta"]["llm_provider"] = llm_prov
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
    kelly_frac = st.slider("Kelly", 0.05, 0.40, 0.25, 0.05)
    st.caption(f"Версия приложения: {APP_VERSION} · build 2026-10-26")
    matrix_n = st.slider("Матрица", 6, 15, 12, 1)

    st.markdown("<hr style='border-color:rgba(255,255,255,.08);margin:16px 0;'>",
                unsafe_allow_html=True)

    if st.button("🔃 Проверить результаты", use_container_width=True,
                 help="Форсировать авто-сеттл прямо сейчас (игнорирует паузу между попытками)"):
        with st.spinner("Проверяю результаты матчей…"):
            D2, n, tried = _auto_settle(D, force=True)
        _apply_settle_result(D2, n, tried, "manual_check")
        if n > 0:
            st.success(f"✅ Закрыто {n} ставок")
        else:
            st.info("Нет завершённых матчей для закрытия")
        st.rerun()

    if stale_count and st.button(f"↩️ Вернуть стейки по устаревшим ({stale_count})",
                                 use_container_width=True,
                                 help="Помечает ставки как void и возвращает стейк в банк"):
        D2, n = _void_stale(D)
        _apply_settle_result(D2, n, 0, "void_stale")
        usage.set_local_data(D2)
        st.toast(f"Возвращены стейки: {n}")
        st.rerun()

    if st.button("🔄 Очистить карточки", use_container_width=True):
        D["cards"] = []
        D["funnel"] = None
        D["report"] = []
        usage.set_local_data(D)
        st.toast("Карточки очищены")
        st.rerun()

    if st.button("♻️ Сбросить счётчики", use_container_width=True):
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

    _backup = json.dumps(_strip_secrets(D), ensure_ascii=False, indent=2, default=str)
    st.download_button("📥 Скачать", _backup,
                       file_name=f"neuro_data_{datetime.now():%Y%m%d_%H%M}.json",
                       mime="application/json",
                       use_container_width=True)

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
        if st.button("🗑️ Очистить портфель", use_container_width=True):
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

# ==================== ANALYTICS TERMINAL ====================
(
    tab_scanner,
    tab_forecasts,
    tab_value,
    tab_quality,
    tab_portfolio,
    tab_learning,
    tab_lab,
    tab_stats_tools,
    tab_api,
) = st.tabs([
    "🏟 LIVE SCANNER",
    "🧠 AI FORECASTS",
    "💎 VALUE ENGINE",
    "🧪 DATA QUALITY",
    "💼 PORTFOLIO",
    "🎓 AI LEARNING",
    "🔬 MODEL LAB",
    "📊 STATISTICS & TOOLS",
    "⚙️ API CENTER",
])

with tab_scanner:
    scanner.render(min_prob, kelly_frac, matrix_n, min_edge)
with tab_forecasts:
    ai_forecasts.render()
with tab_value:
    value_engine.render()
with tab_quality:
    data_quality.render()
with tab_portfolio:
    portfolio.render()
with tab_learning:
    learning.render()
with tab_lab:
    model_lab.render()
with tab_stats_tools:
    stats.render()
    st.divider()
    with st.expander("🧮 Калькулятор", expanded=False):
        calculator.render(matrix_n)
    with st.expander("🧪 Бэктест", expanded=False):
        backtest.render(matrix_n)
with tab_api:
    api_center.render()
