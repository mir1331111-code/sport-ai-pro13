"""data/scores.py — ESPN + football-data.org + OpenLigaDB (v3).

Что изменено:
  * _same_team: мягкий матчинг (Brooklyn FC ↔ Brooklyn, Rhode Island FC ↔ Rhode Island)
  * _find_espn: логирование near-miss (для диагностики NOT FOUND)
  * Сохранены все защиты v2: токены, диакритика, алиасы, женские/юношеские
  * football-data.org: 1 запрос /v4/matches на дату
  * AET: отдельный статус (не путается со FT)
"""
from __future__ import annotations

import logging
import re
import time as _time
import unicodedata
from datetime import datetime, timedelta
from typing import Iterator, Optional

import requests

from security import cache_get, cache_put
from data.fixture_time import app_timezone

log = logging.getLogger("scores")

NO_PROXY = {"http": None, "https": None, "all": None}
ESPN_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/all/scoreboard"
FDORG_URL = "https://api.football-data.org/v4/matches"

FDORG_COMPETITIONS = ["PL", "ELC", "PD", "SA", "BL1", "FL1", "DED", "PPL", "CL"]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "application/json",
    })
    s.trust_env = False
    s.proxies = {"http": None, "https": None}
    return s


_sess = _session()


def _get_token() -> str:
    try:
        import streamlit as st
        return (st.session_state.get("data", {}).get("meta", {})
                .get("fdorg_token", "") or "").strip()
    except Exception:
        return ""


# ============ СОПОСТАВЛЕНИЕ КОМАНД ============
_TR = str.maketrans({"ß": "ss", "ø": "o", "æ": "ae", "œ": "oe", "ł": "l",
                     "đ": "d", "ð": "d", "þ": "th", "ı": "i"})
_STOP = {"fc", "cf", "afc", "sc", "ac", "as", "ssc", "cd", "ud", "rcd",
         "sv", "fk", "bv", "1"}
_FLAGS = {"women", "w", "wfc", "fem", "feminin", "femenino", "femeni", "frauen",
          "u17", "u18", "u19", "u20", "u21", "u23", "ii", "b", "reserves",
          "res", "youth", "castilla"}
_ALIASES = {
    "man utd": "manchester united", "man united": "manchester united",
    "manchester utd": "manchester united", "man city": "manchester city",
    "spurs": "tottenham", "tottenham hotspur": "tottenham",
    "wolves": "wolverhampton wanderers", "internazionale": "inter",
    "inter milan": "inter", "paris saint germain": "psg", "paris sg": "psg",
    "bayern munich": "bayern", "bayern munchen": "bayern",
    "nott m forest": "nottingham forest",
    # USL (для тех, что не матчились)
    "brooklyn": "brooklyn fc",
    "rhode island": "rhode island fc",
    "detroit city": "detroit city fc",
    "birmingham legion": "birmingham legion fc",
}


def _tokens(name: str) -> tuple[frozenset, frozenset]:
    s = unicodedata.normalize("NFKD", name or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower()
    s = s.translate(_TR).replace("&", " and ")
    words = re.findall(r"[a-zа-яё0-9]+", s)
    flags = frozenset(w for w in words if w in _FLAGS)
    core = [w for w in words if w not in _STOP and w not in _FLAGS]
    phrase = " ".join(core)
    phrase = _ALIASES.get(phrase, phrase)
    return frozenset(phrase.split()), flags


def _same_team(a: str, b: str) -> bool:
    """Мягкий матчинг: ESPN часто даёт короткие названия USL-команд."""
    ca, fa = _tokens(a)
    cb, fb = _tokens(b)
    if not ca or not cb or fa != fb:
        return False
    if ca == cb:
        return True

    # Одно подмножество другого
    small, big = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
    if small <= big:
        return True

    # 60% общих слов + хотя бы одно длинное (>=4)
    common = small & big
    if len(common) / max(1, len(small)) >= 0.6:
        if any(len(t) >= 4 for t in common):
            return True
    return False


def _parse_day(date_iso: str) -> Optional[datetime]:
    try:
        return datetime.strptime((date_iso or "")[:10], "%Y-%m-%d")
    except ValueError:
        return None


def _source_local_day(value: object) -> Optional[str]:
    """Normalize provider date/timestamp to the app's calendar day.

    UTC-offset-aware provider timestamps are converted before extracting the
    day. Date-only and timezone-naive values retain their supplied date because
    there is no reliable offset to apply.
    """
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is not None and parsed.utcoffset() is not None:
            parsed = parsed.astimezone(app_timezone())
        return parsed.date().isoformat()
    except (TypeError, ValueError, OverflowError):
        parsed_day = _parse_day(raw)
        return parsed_day.date().isoformat() if parsed_day else None


def _day_offsets(date_iso: str) -> Iterator[str]:
    d = _parse_day(date_iso)
    if d is None:
        return
    for off in (0, -1, 1):
        yield (d + timedelta(days=off)).strftime("%Y-%m-%d")


def _ttl(date_iso: str) -> int:
    d = _parse_day(date_iso)
    if d is None or (datetime.now() - d).days <= 2:
        return 600
    return 7 * 86400


# ============ ESPN ============
def _espn_day(date_iso: str) -> list:
    # v5 invalidates pre-archive cache rows that had no date/league metadata.
    ck = f"espn_day_v5_{date_iso}"
    cached = cache_get(ck, _ttl(date_iso))
    if isinstance(cached, list):
        return cached
    try:
        r = _sess.get(ESPN_URL,
                      params={"dates": date_iso.replace("-", ""), "limit": 500},
                      timeout=15, proxies=NO_PROXY)
    except requests.RequestException as e:
        log.warning("espn error %s: %s", date_iso, type(e).__name__)
        return []
    if r.status_code != 200:
        log.warning("espn HTTP %s %s", r.status_code, date_iso)
        return []
    out = []
    for e in (r.json() or {}).get("events") or []:
        try:
            comp = (e.get("competitions") or [{}])[0]
            stype = (comp.get("status") or {}).get("type") or {}
            if not stype.get("completed"):
                continue
            home = away = None
            for t in comp.get("competitors") or []:
                name = (t.get("team") or {}).get("displayName") or ""
                if t.get("homeAway") == "home":
                    home = (name, t.get("score"))
                elif t.get("homeAway") == "away":
                    away = (name, t.get("score"))
            if not home or not away or home[1] is None or away[1] is None:
                continue
            sname = str(stype.get("name") or "").upper()
            extra = any(x in sname for x in ("AET", "PEN", "EXTRA"))
            league_meta = e.get("league") or {}
            event_day = _source_local_day(e.get("date")) or date_iso
            out.append({
                "home": home[0], "away": away[0],
                "hs": int(home[1]), "as": int(away[1]),
                "status": "AET" if extra else "FT",
                "date": event_day,
                "league": str(league_meta.get("name") or "ESPN Soccer"),
                "fixture_id": str(e.get("id") or ""),
                "source": "ESPN",
            })
        except Exception:
            continue
    log.info("espn %s: %d events", date_iso, len(out))
    cache_put(ck, out)
    return out


def _find_espn(home: str, away: str, date_iso: str) -> Optional[dict]:
    """ESPN. Near-miss логирование для диагностики NOT FOUND."""
    for d_str in _day_offsets(date_iso):
        events = _espn_day(d_str)
        for e in events:
            if _same_team(home, e["home"]) and _same_team(away, e["away"]):
                return {"home": e["hs"], "away": e["as"],
                        "status": e["status"], "source": "espn"}
        # Диагностика — только когда дата = сегодня (не спамим на -1/+1)
        if d_str != date_iso:
            continue
        try:
            ht, _ = _tokens(home)
            at, _ = _tokens(away)
            near = []
            for e in events:
                eh_t, _ = _tokens(e["home"])
                ea_t, _ = _tokens(e["away"])
                if (ht & eh_t) or (at & ea_t):
                    near.append(f"{e['home']} vs {e['away']}")
            if near:
                log.info("NEAR MISS: want '%s vs %s' | ESPN has: %s",
                         home, away, "; ".join(near[:5]))
        except Exception:
            pass
    return None


# ============ FOOTBALL-DATA.ORG ============
_LAST_FDORG_CALL = [0.0]
_FDORG_COOLDOWN_UNTIL = [0.0]


def _fdorg_rate_limit() -> None:
    delta = _time.time() - _LAST_FDORG_CALL[0]
    if delta < 6.5:
        _time.sleep(6.5 - delta)
    _LAST_FDORG_CALL[0] = _time.time()


def _fdorg_range(date_iso: str, token: str | None = None) -> list:
    token = str(token or _get_token()).strip()
    d = _parse_day(date_iso)
    if not token or d is None or _time.time() < _FDORG_COOLDOWN_UNTIL[0]:
        return []
    # v6 refreshes cached UTC calendar dates into app-local fixture dates.
    ck = f"fdorg_rng_v6_{d:%Y-%m-%d}"
    cached = cache_get(ck, _ttl(date_iso))
    if isinstance(cached, list):
        return cached
    _fdorg_rate_limit()
    try:
        r = _sess.get(
            FDORG_URL, headers={"X-Auth-Token": token},
            params={"dateFrom": (d - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "dateTo": (d + timedelta(days=1)).strftime("%Y-%m-%d"),
                    "competitions": ",".join(FDORG_COMPETITIONS)},
            timeout=15, proxies=NO_PROXY)
    except requests.RequestException as e:
        log.warning("fdorg error: %s", type(e).__name__)
        return []
    if r.status_code == 429:
        _FDORG_COOLDOWN_UNTIL[0] = _time.time() + 120
        log.warning("fdorg 429 — пауза 120с")
        return []
    if r.status_code != 200:
        log.warning("fdorg HTTP %s", r.status_code)
        return []
    out = []
    for m in (r.json() or {}).get("matches") or []:
        try:
            if m.get("status") != "FINISHED":
                continue
            h = (m.get("homeTeam") or {}).get("name") or ""
            a = (m.get("awayTeam") or {}).get("name") or ""
            score = m.get("score") or {}
            duration = str(score.get("duration") or "REGULAR").upper()
            official = score.get("fullTime") or {}
            regular = official
            tag = "FT"
            if duration != "REGULAR":
                # football-data.org fullTime is the final match score; regularTime
                # is the 90-minute score. Preserve both so settlement can refuse
                # AET while the results archive displays the actual final score.
                tag = "AET"
                regular_time = score.get("regularTime") or {}
                if (
                    regular_time.get("home") is not None
                    and regular_time.get("away") is not None
                ):
                    regular = regular_time
            if (
                not h or not a
                or regular.get("home") is None or regular.get("away") is None
            ):
                continue
            official_score = (
                official if official.get("home") is not None
                and official.get("away") is not None else regular
            )
            competition = m.get("competition") or {}
            out.append({
                "home": h, "away": a, "hs": int(regular["home"]),
                "as": int(regular["away"]), "official_hs": int(official_score["home"]),
                "official_as": int(official_score["away"]), "status": tag,
                "duration": duration,
                "date": _source_local_day(m.get("utcDate")) or "",
                "league": str(competition.get("name") or "football-data.org"),
                "fixture_id": str(m.get("id") or ""),
                "source": "football-data.org",
            })
        except Exception:
            continue
    log.info("fdorg %s: %d events", date_iso, len(out))
    cache_put(ck, out)
    return out


def _closest(cands: list, date_iso: str) -> Optional[dict]:
    d = _parse_day(date_iso)
    best, best_gap = None, None
    for e in cands:
        ed = _parse_day(e.get("date", ""))
        gap = abs((ed - d).days) if (ed and d) else 99
        if gap <= 1 and (best_gap is None or gap < best_gap):
            best, best_gap = e, gap
    return best


def _find_fdorg(home: str, away: str, date_iso: str) -> Optional[dict]:
    cands = [e for e in _fdorg_range(date_iso)
             if _same_team(home, e["home"]) and _same_team(away, e["away"])]
    e = _closest(cands, date_iso)
    if not e:
        return None
    return {"home": e["hs"], "away": e["as"],
            "status": e["status"], "source": "fdorg"}


# ============ OPENLIGADB ============
def _olb_season(date_iso: str) -> Optional[int]:
    d = _parse_day(date_iso)
    if d is None:
        return None
    return d.year if d.month >= 7 else d.year - 1


def _olb_league(league: str, season: int) -> list:
    # v6 refreshes cached timestamps normalized to the app's timezone.
    ck = f"olb_{league}_{season}_v6"
    cached = cache_get(ck, 1800)
    if isinstance(cached, list):
        return cached
    try:
        r = _sess.get(f"https://api.openligadb.de/getmatchdata/{league}/{season}",
                      timeout=20, proxies=NO_PROXY)
    except requests.RequestException as e:
        log.warning("olb %s error: %s", league, type(e).__name__)
        return []
    if r.status_code != 200:
        log.warning("olb %s HTTP %s", league, r.status_code)
        return []
    out = []
    for m in r.json() or []:
        try:
            if not m.get("matchIsFinished"):
                continue
            h = (m.get("team1") or {}).get("teamName") or ""
            a = (m.get("team2") or {}).get("teamName") or ""
            results = [
                result for result in (m.get("matchResults") or [])
                if isinstance(result, dict)
                and result.get("pointsTeam1") is not None
                and result.get("pointsTeam2") is not None
            ]
            if not results:
                continue
            # Use the latest result stage, not unconditionally the 90-minute
            # score (resultTypeID=2). Cup ties can add extra-time/penalty stages.
            final_result = max(
                results,
                key=lambda result: (
                    int(result.get("resultOrderID") or result.get("resultTypeID") or 0),
                    int(result.get("resultTypeID") or 0),
                ),
            )
            result_type = int(final_result.get("resultTypeID") or 0)
            result_kind = str(final_result.get("resultTypeKind") or "").lower()
            duration = (
                "PENALTY_SHOOTOUT"
                if "penalt" in result_kind or result_type == 5 else
                "EXTRA_TIME" if "extra" in result_kind or result_type in {3, 4} else
                "REGULAR"
            )
            hs, ags = final_result.get("pointsTeam1"), final_result.get("pointsTeam2")
            if not h or not a or hs is None or ags is None:
                continue
            out.append({
                "home": h, "away": a, "hs": int(hs), "as": int(ags),
                "status": "AET" if duration != "REGULAR" else "FT",
                "duration": duration,
                "date": _source_local_day(
                    m.get("matchDateTimeUTC") or m.get("matchDateTime")
                ) or "",
                "league": {"bl1": "Bundesliga", "bl2": "2. Bundesliga",
                           "bl3": "3. Liga"}.get(league, league.upper()),
                "fixture_id": str(m.get("matchID") or ""),
                "source": "OpenLigaDB",
            })
        except Exception:
            continue
    log.info("olb %s/%s: %d finished", league, season, len(out))
    cache_put(ck, out)
    return out


def _find_olb(home: str, away: str, date_iso: str) -> Optional[dict]:
    season = _olb_season(date_iso)
    if season is None:
        return None
    for league in ("bl1", "bl2", "bl3"):
        cands = [e for e in _olb_league(league, season)
                 if _same_team(home, e["home"]) and _same_team(away, e["away"])]
        e = _closest(cands, date_iso)
        if e:
            return {"home": e["hs"], "away": e["as"],
                    "status": e["status"], "source": "openligadb"}
    return None


# ============ ПУБЛИЧНАЯ ФУНКЦИЯ ============
def find_match_score(home: str, away: str, date_iso: str) -> Optional[dict]:
    """ESPN → football-data.org → OpenLigaDB.
    status: FT | AET (доп. время — не закрывается автоматически)."""
    if not home or not away or not date_iso:
        return None
    for name, fn in (("espn", _find_espn),
                     ("fdorg", _find_fdorg),
                     ("olb", _find_olb)):
        try:
            r = fn(home, away, date_iso)
        except Exception as e:
            log.warning("%s fail: %s", name, e)
            continue
        if r:
            log.info("FOUND %s: %s vs %s = %s:%s (%s)",
                     r["source"], home, away,
                     r["home"], r["away"], r["status"])
            return r
    log.info("NOT FOUND: %s vs %s @ %s", home, away, date_iso)
    return None


def completed_matches_for_day(date_iso: str, *, token: str | None = None) -> dict:
    """Return deduplicated completed football matches for one calendar day.

    Coverage is intentionally explicit: ESPN broad soccer scoreboard,
    football-data.org's configured competition set when a token is available,
    and OpenLigaDB's German leagues (BL1/BL2/BL3). This is a match-results
    archive, not a betting ledger or a claim of exhaustive worldwide coverage.
    """
    day = str(date_iso or "").strip()[:10]
    parsed_day = _parse_day(day)
    if parsed_day is None or parsed_day.strftime("%Y-%m-%d") != day:
        raise ValueError("date_iso must be YYYY-MM-DD")

    candidates: list[dict] = []
    source_counts = {
        "ESPN": 0,
        "football-data.org": 0,
        "OpenLigaDB": 0,
    }
    source_errors: dict[str, str] = {}

    source_seen: dict[str, set[str]] = {name: set() for name in source_counts}

    def _add(rows, source: str, *, fallback_date: str | None = None):
        for original in rows or []:
            if not isinstance(original, dict):
                continue
            provider_day = _source_local_day(original.get("date")) or fallback_date
            if provider_day != day:
                continue
            try:
                home = str(original.get("home") or "").strip()
                away = str(original.get("away") or "").strip()
                home_score = int(original.get("official_hs", original.get("hs")))
                away_score = int(original.get("official_as", original.get("as")))
                status = str(original.get("status") or "").upper()
                if (
                    not home or not away or home.casefold() == away.casefold()
                    or home_score < 0 or away_score < 0
                    or status not in {"FT", "AET"}
                ):
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            home_key = " ".join(sorted(_tokens(home)[0]))
            away_key = " ".join(sorted(_tokens(away)[0]))
            unique_key = str(original.get("fixture_id") or f"{home_key}>{away_key}")
            if unique_key in source_seen[source]:
                continue
            source_seen[source].add(unique_key)
            item = {
                "date": day,
                "home": home,
                "away": away,
                "home_score": home_score,
                "away_score": away_score,
                "score": f"{home_score}:{away_score}",
                "status": status,
                "duration": str(original.get("duration") or ""),
                "league": str(original.get("league") or source),
                "fixture_id": str(original.get("fixture_id") or ""),
                "sources": [source],
                "score_conflict": False,
                "alternate_scores": [],
            }
            candidates.append(item)
            source_counts[source] += 1

    # Query adjacent provider days as timestamps close to UTC midnight can
    # belong to a different calendar day in the app's configured timezone.
    # Only the exact requested-day query may use a legacy date-less row's day
    # as a fallback; adjacent-day rows must carry a real event date.
    for request_day in _day_offsets(day):
        try:
            _add(
                _espn_day(request_day), "ESPN",
                fallback_date=day if request_day == day else None,
            )
        except Exception as exc:
            source_errors["ESPN"] = type(exc).__name__
            log.warning("daily results ESPN %s: %s", request_day, type(exc).__name__)

    if str(token or _get_token() or "").strip():
        try:
            _add(_fdorg_range(day, token=token), "football-data.org")
        except Exception as exc:
            source_errors["football-data.org"] = type(exc).__name__
            log.warning("daily results football-data.org %s: %s", day, type(exc).__name__)
    else:
        source_errors["football-data.org"] = "API token not configured"

    for league_code in ("bl1", "bl2", "bl3"):
        try:
            _add(_olb_league(league_code, _olb_season(day)), "OpenLigaDB")
        except Exception as exc:
            source_errors["OpenLigaDB"] = type(exc).__name__
            log.warning("daily results OpenLigaDB %s/%s: %s", league_code, day, type(exc).__name__)

    # Merge only exact-date, directionally matched fixtures. If two providers
    # disagree on a score, retain the first result but expose the discrepancy.
    merged: list[dict] = []
    for item in candidates:
        existing = next(
            (
                row for row in merged
                if _same_team(row["home"], item["home"])
                and _same_team(row["away"], item["away"])
            ),
            None,
        )
        if existing is None:
            merged.append(item)
            continue
        for source in item["sources"]:
            if source not in existing["sources"]:
                existing["sources"].append(source)
        if (
            existing["home_score"] != item["home_score"]
            or existing["away_score"] != item["away_score"]
        ):
            existing["score_conflict"] = True
            alternative = {
                "score": item["score"],
                "source": item["sources"][0],
            }
            if alternative not in existing["alternate_scores"]:
                existing["alternate_scores"].append(alternative)
        if existing["league"] in {"ESPN Soccer", "football-data.org"} and item["league"]:
            if item["league"] not in {"ESPN Soccer", "football-data.org"}:
                existing["league"] = item["league"]

    merged.sort(key=lambda row: (
        str(row.get("league") or "").casefold(),
        str(row.get("home") or "").casefold(),
        str(row.get("away") or "").casefold(),
    ))
    return {
        "date": day,
        "matches": merged,
        "source_counts": source_counts,
        "source_errors": source_errors,
        "total": len(merged),
        "goals": sum(row["home_score"] + row["away_score"] for row in merged),
        "home_wins": sum(row["home_score"] > row["away_score"] for row in merged),
        "draws": sum(row["home_score"] == row["away_score"] for row in merged),
        "away_wins": sum(row["home_score"] < row["away_score"] for row in merged),
        "score_conflicts": sum(bool(row["score_conflict"]) for row in merged),
    }
