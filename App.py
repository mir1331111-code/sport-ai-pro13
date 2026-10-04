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
from datetime import datetime, timedelta

import streamlit as st

if os.environ.get("NEURO_KEEP_PROXY") != "1":
    for _k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
               "ALL_PROXY", "all_proxy", "FTP_PROXY", "ftp_proxy"]:
        os.environ.pop(_k, None)

st.set_page_config(page_title="NEURO BET PRO",
                   page_icon="🏟", layout="wide",
                   initial_sidebar_state="expanded")

from config import (APP_VERSION, LLM_PROVIDERS, DATA_VERSION,  # noqa: E402
                    AUTO_SETTLE_THROTTLE_SEC)
from storage import sqlite_store as db  # noqa: E402
from storage import usage  # noqa: E402
from data.sources import parse_date, fdorg_match_result  # noqa: E402
from betting.settlement import determine_outcome  # noqa: E402
from ui import theme  # noqa: E402
from ui.tabs import scanner, portfolio, stats, calculator, backtest  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[APP] %(message)s")
log = logging.getLogger("neurobet")

# ==================== КОНСТАНТЫ ====================
DEFAULT_BANK = 10000.0
AUTO_SETTLE_EVERY_SEC = AUTO_SETTLE_THROTTLE_SEC
SETTLE_MIN_AGE = timedelta(hours=2, minutes=15)
RETRY_EVERY = timedelta(minutes=60)
STALE_AFTER = timedelta(hours=48)
MAX_PER_RUN = 20
SECRET_KEYS = ("fdorg_token", "odds_api_key", "llm_api_key")
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
D.setdefault("bets", [])
if "initial_bank" not in D["meta"]:
    D["meta"]["initial_bank"] = float(D.get("bank", DEFAULT_BANK))
    usage.set_local_data(D)


# ==================== AUTO-SETTLE ====================
def _auto_settle(D, force: bool = False):
    """ESPN/TheSportsDB/OpenLigaDB → fdorg. Не раньше MIN_AGE. Никакого авто-void."""
    D2 = dict(D)
    bets = list(D2.get("bets", []))
    closed = tried = 0
    now = datetime.now()
    token = (D.get("meta", {}).get("fdorg_token") or "").strip()
    log.info("START: %d bets | remaining=%s", len(bets), usage.settle_remaining())

    for idx, b in enumerate(bets):
        if not isinstance(b, dict) or b.get("status") != "pending":
            continue
        if tried >= MAX_PER_RUN:
            log.info("MAX_PER_RUN reached")
            break
        if usage.settle_remaining() <= 0:
            log.info("LIMIT reached")
            break

        bd = _bet_date(b)
        if bd and now - bd < SETTLE_MIN_AGE:
            continue
        if not force:
            last = _parse_iso(b.get("last_settle_try"))
            if last and now - last < RETRY_EVERY:
                continue

        res, source = None, None
        tried += 1

        # 1) ESPN / TheSportsDB / OpenLigaDB — без лимита, быстрый путь
        h_team = (b.get("home") or "").strip()
        a_team = (b.get("away") or "").strip()
        if not (h_team and a_team):
            m = b.get("match") or ""
            if " vs " in m:
                h_team, a_team = [p.strip() for p in m.split(" vs ", 1)]
        d_iso = b.get("date_iso") or ""
        if h_team and a_team and d_iso:
            try:
                from data.scores import find_match_score
                res = find_match_score(h_team, a_team, d_iso)
                if res:
                    source = res.get("source", "auto")
            except Exception as e:
                log.warning("find_match_score error: %s", e)

        # 2) football-data.org — только если первый источник не справился
        if not _is_finished(res):
            res = None
            fid = b.get("fixture_id")
            if fid and token:
                try:
                    res = fdorg_match_result(fid, token)
                    if res:
                        source = "fdorg"
                except Exception as e:
                    log.warning("fdorg error: %s", e)

        b2 = dict(b)
        b2["last_settle_try"] = now.isoformat()
        log.info("%s | src=%s | res=%s", b.get("match_ru", "?"), source, res)

        if not _is_finished(res):
            bets[idx] = b2
            continue

        usage.settle_increment(1)
        outcome, reason = determine_outcome(
            b.get("market"), b.get("pick"), int(res["home"]), int(res["away"]))
        if outcome is None:
            log.info("unknown outcome: %s", reason)
            bets[idx] = b2
            continue

        try:
            stake = float(b2["stake"])
            odds = float(b2["odds"])
        except (KeyError, TypeError, ValueError):
            bets[idx] = b2
            continue

        b2["score"] = f"{int(res['home'])}:{int(res['away'])}"
        b2["settled_at"] = datetime.now().isoformat()
        b2["settle_source"] = source
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
        value=D["meta"].get("fdorg_token") or os.environ.get("FDORG_TOKEN", ""),
        type="password")
    if fdorg_token != D["meta"].get("fdorg_token", ""):
        D["meta"]["fdorg_token"] = fdorg_token
        usage.set_local_data(D)

    odds_key = st.text_input(
        "The Odds API key (опц.)",
        value=D["meta"].get("odds_api_key") or os.environ.get("ODDS_API_KEY", ""),
        type="password")
    if odds_key != D["meta"].get("odds_api_key", ""):
        D["meta"]["odds_api_key"] = odds_key
        usage.set_local_data(D)

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
        value=D["meta"].get("llm_api_key") or os.environ.get("LLM_API_KEY", ""),
        type="password")
    if (llm_prov != D["meta"].get("llm_provider")
            or llm_key != D["meta"].get("llm_api_key", "")):
        D["meta"]["llm_provider"] = llm_prov
        D["meta"]["llm_api_key"] = llm_key
        usage.set_local_data(D)

    st.markdown("<hr style='border-color:rgba(255,255,255,.08);margin:16px 0;'>",
                unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:.72rem;color:#8b93a7;margin-bottom:6px;'>"
        "🎯 Параметры</div>",
        unsafe_allow_html=True)
    min_prob = st.slider("Мин. P %", 30, 85, 40, 1) / 100
    kelly_frac = st.slider("Kelly", 0.05, 0.40, 0.25, 0.05)
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

# ==================== TABS ====================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["🏟 Сканер", "💼 Портфель", "📈 Статистика",
     "🧮 Калькулятор", "🧪 Бэктест"])

with tab1:
    scanner.render(min_prob, kelly_frac, matrix_n)
with tab2:
    portfolio.render()
with tab3:
    stats.render()
with tab4:
    calculator.render(matrix_n)
with tab5:
    backtest.render(matrix_n)
