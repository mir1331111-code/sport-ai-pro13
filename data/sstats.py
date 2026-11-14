"""SStats.net football API adapter.

Owns parsing, cache access and request handling for SStats. Public functions
are re-exported from :mod:`data.sources` for backwards compatibility.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
from requests.adapters import HTTPAdapter

try:
    from urllib3.util.retry import Retry
    _HAS_RETRY = True
except Exception:
    _HAS_RETRY = False

from security import cache_get, cache_put
from data.team_names import normalize_team


def _norm_name(value: str) -> str:
    try:
        return normalize_team(value)
    except Exception:
        return re.sub(r"[^a-zа-я0-9]", "", (value or "").lower())


def _parse_provider_date(value) -> Optional[datetime]:
    """Parse SStats timestamps without dropping their UTC offset."""
    if value is None:
        return None
    source = str(value).strip()
    if not source:
        return None
    try:
        # fromisoformat preserves +00:00 / Z, which is essential when converting
        # the provider kickoff to the app's configured timezone.
        return datetime.fromisoformat(source.replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        pass
    for fmt in (
        "%d/%m/%Y %H:%M", "%d/%m/%Y", "%d/%m/%y", "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    ):
        try:
            return datetime.strptime(source, fmt)
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def _session() -> requests.Session:
    session = requests.Session()
    if _HAS_RETRY:
        retry = Retry(
            total=3, connect=3, read=3, backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "HEAD"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(
            max_retries=retry, pool_connections=20, pool_maxsize=20
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0 Chrome/120.0"})
    session.trust_env = False
    session.proxies = {"http": None, "https": None}
    return session


_sess = _session()
NO_PROXY = {"http": None, "https": None, "all": None}


# ============ SSTATS.NET FREE FOOTBALL API ============
SSTATS_API_BASE = "https://api.sstats.net"


def _sstats_unwrap(payload):
    """Unwrap the documented ApiResponse envelope without assuming non-null data."""
    if isinstance(payload, dict):
        if str(payload.get("status") or "").casefold() == "error":
            raise ValueError(str(payload.get("message") or "SStats API returned an error"))
        return payload.get("data")
    return payload


def _sstats_timezone():
    """Return app-local date and integer UTC offset required by SStats."""
    zone_name = str(os.environ.get("APP_TIMEZONE", "Europe/Warsaw") or "Europe/Warsaw")
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo(zone_name))
        offset = now.utcoffset()
        offset_hours = int(round(offset.total_seconds() / 3600)) if offset else 3
        return now, max(-12, min(12, offset_hours))
    except Exception:
        return datetime.now(), 3


def _sstats_float(value):
    try:
        number = float(str(value).replace(",", "."))
        return number if 1.01 < number <= 100 else None
    except (TypeError, ValueError):
        return None


def _sstats_line(text):
    found = re.search(r"(?<!\d)(\d{1,2}(?:[.,]\d{1,2})?)(?!\d)", str(text or ""))
    if not found:
        return None
    try:
        return float(found.group(1).replace(",", "."))
    except ValueError:
        return None


def _sstats_flat_bets(odds_payload):
    """Yield (market, bookmaker) from compact Game.odds or full Odds response."""
    if isinstance(odds_payload, dict):
        odds_payload = odds_payload.get("data", odds_payload.get("odds", []))
    if not isinstance(odds_payload, list):
        return
    for item in odds_payload:
        if not isinstance(item, dict):
            continue
        # Compact /Games/list response: item itself is ApiSaBet. Some
        # detailed feeds also flatten bookmakerName onto each market row.
        if "marketName" in item or "marketId" in item:
            flat_book = item.get("bookmakerName") or item.get("bookmaker") or ""
            if isinstance(flat_book, dict):
                flat_book = (
                    flat_book.get("name") or flat_book.get("bookmakerName")
                    or flat_book.get("title")
                )
            yield item, str(flat_book or "").strip()
            continue
        # Detailed /Odds/{gameId} response: bookmaker object contains ApiSaBet[].
        nested = item.get("odds")
        bookmaker_raw = item.get("bookmakerName") or item.get("bookmaker")
        if isinstance(bookmaker_raw, dict):
            bookmaker_raw = (
                bookmaker_raw.get("name") or bookmaker_raw.get("bookmakerName")
                or bookmaker_raw.get("title")
            )
        bookmaker = str(bookmaker_raw or item.get("name") or "").strip()
        if isinstance(nested, list):
            for bet in nested:
                if isinstance(bet, dict):
                    yield bet, bookmaker


def _sstats_parse_markets(odds_payload):
    """Normalize SStats markets while keeping 1X2 odds from one coherent book.

    Only explicit full-time result markets are accepted as 1X2. A complete
    bookmaker line is selected as a unit; prices from different bookmakers
    are never stitched together for market de-vigging.
    """
    prices = {
        "total_market_odds": {},
        "btts_market_odds": {},
        "stat_market_odds": {},
        "asian_handicap_market_odds": {},
    }
    bookmaker_by_pick = {}
    one_x_two_by_book = {}

    def _record(selection, odd, bookmaker, group=None):
        target = prices[group] if group else prices
        values = target.setdefault(selection, []) if group else target.setdefault(selection, [])
        values.append((odd, bookmaker))
        
    def _market_key(value):
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()

    def _book_key(value):
        # Provider casing/spacing differences must not count as two books.
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

    def _is_full_time_1x2(value):
        market = _market_key(value)
        # Explicit period/market qualifiers must never be mistaken for full-time 1X2.
        excluded = (
            "first half", "1st half", "half time", "half-time", "second half",
            "2nd half", "extra time", "penalty", "to qualify", "corners",
            "corner", "cards", "handicap", "asian", "period",
        )
        if any(term in market for term in excluded):
            return False
        exact = {
            "1x2", "match result", "full time result", "fulltime result",
            "match winner", "90 minutes result", "90 min result",
            "result after 90 minutes", "1x2 90 minutes", "1x2 regular time",
        }
        if market in exact:
            return True
        return bool(re.search(r"\b(full time|fulltime|90 min|90 minutes)\b.*\b(result|winner)\b", market))

    for bet, bookmaker in _sstats_flat_bets(odds_payload):
        market_raw = bet.get("marketName") or bet.get("market") or bet.get("name") or ""
        if isinstance(market_raw, dict):
            market_raw = market_raw.get("name") or market_raw.get("title") or ""
        market_name = str(market_raw).strip()
        market = market_name.casefold()
        price_rows = bet.get("odds") or bet.get("values") or bet.get("outcomes") or []
        if not isinstance(price_rows, list):
            continue

        for price_row in price_rows:
            if not isinstance(price_row, dict):
                continue
            label = str(
                price_row.get("name") or price_row.get("label")
                or price_row.get("valueName") or price_row.get("outcomeName")
                or price_row.get("selectionName") or price_row.get("outcome") or ""
            ).strip()
            label_norm = label.casefold()
            raw_odd = next(
                (price_row.get(field) for field in ("value", "odd", "price", "odds")
                 if price_row.get(field) not in (None, "")),
                None,
            )
            odd = _sstats_float(raw_odd)
            if odd is None:
                continue

            # Only a named full-time 1X2 market qualifies; plain "winner" is ambiguous.
            if _is_full_time_1x2(market_name):
                selection = None
                if label_norm in ("1", "home", "home win", "п1", "1 (home)"):
                    selection = "П1"
                elif label_norm in ("x", "х", "draw", "tie", "ничья"):
                    selection = "X"
                elif label_norm in ("2", "away", "away win", "п2", "2 (away)"):
                    selection = "П2"
                if selection:
                    book_key = str(bookmaker or "").strip()
                    one_x_two_by_book.setdefault(book_key, {}).setdefault(selection, []).append(odd)
                continue

            # Both teams to score.
            if any(term in market for term in ("both teams", "btts", "обе заб", "обе команды забьют")):
                if label_norm in ("yes", "да", "both", "true"):
                    _record("BTTS Да", odd, bookmaker, "btts_market_odds")
                elif label_norm in ("no", "нет", "false"):
                    _record("BTTS Нет", odd, bookmaker, "btts_market_odds")
                continue

            # Totals, including corners/cards. Reject period-specific and handicap
            # markets before looking at the outcome label: "Over 3.5" alone does not
            # prove that the market is full-time goals.
            # SStats usually labels totals "Over 2.5"/"Under 2.5",
            # but some feeds put the line first ("2.5 Over") or use
            # the compact Russian forms "ТБ 2.5"/"ТМ 2.5".
            # A bare numeric outcome is never enough to infer a total side.
            is_over = (
                label_norm in ("over", "more", "больше", "o", "тб")
                or label_norm.startswith(("over ", "more ", "больше ", "o ", "тб "))
                or bool(re.match(
                    r"^\s*\d{1,2}(?:[.,]\d{1,2})?\s*(?:[-:]\s*)?(?:over|more|больше|тб)\b",
                    label_norm,
                ))
            )
            is_under = (
                label_norm in ("under", "less", "меньше", "u", "тм")
                or label_norm.startswith(("under ", "less ", "меньше ", "u ", "тм "))
                or bool(re.match(
                    r"^\s*\d{1,2}(?:[.,]\d{1,2})?\s*(?:[-:]\s*)?(?:under|less|меньше|тм)\b",
                    label_norm,
                ))
            )
            is_period_market = any(term in market for term in (
                "first half", "1st half", "half time", "half-time",
                "second half", "2nd half", "extra time", "period",
                "1st period", "2nd period", "3rd period",
            ))
            is_handicap_market = any(term in market for term in (
                "handicap", "asian", "фора", "spread",
            ))
            if (is_over or is_under) and not is_period_market and not is_handicap_market:
                # Some SStats responses put the line in a structured field while
                # the outcome label is only "Over"/"Under" and marketName is "Total".
                # Read these fields only after rejecting period/handicap markets above.
                line = _sstats_line(label) or _sstats_line(market_name)
                if line is None:
                    for line_value in (
                        price_row.get("point"), price_row.get("line"),
                        price_row.get("handicap"), price_row.get("total"),
                        price_row.get("points"), price_row.get("param"),
                        bet.get("point"), bet.get("line"),
                        bet.get("handicap"), bet.get("total"),
                        bet.get("points"), bet.get("param"),
                    ):
                        if isinstance(line_value, (int, float)):
                            candidate_line = float(line_value)
                            if 0 <= candidate_line <= 20:
                                line = candidate_line
                                break
                        elif isinstance(line_value, str):
                            candidate_line = _sstats_line(line_value)
                            if candidate_line is not None and 0 <= candidate_line <= 20:
                                line = candidate_line
                                break
                if line is None:
                    continue
                line_text = f"{line:g}"
                if any(term in market for term in ("corner", "corners", "углов")):
                    key = ("Угл ТБ " if is_over else "Угл ТМ ") + line_text
                    _record(key, odd, bookmaker, "stat_market_odds")
                elif any(term in market for term in ("card", "booking", "карточ")):
                    key = ("Карты ТБ " if is_over else "Карты ТМ ") + line_text
                    _record(key, odd, bookmaker, "stat_market_odds")
                elif (
                    any(term in market for term in (
                        "total goals", "goals total", "total goal", "goal total",
                        "over/under goals", "goals over/under", "goals over under",
                        "total goals over under", "total over under goals",
                        "match goals", "match total", "тотал голов", "тотал гола",
                        "голы тотал",
                    ))
                    or (
                        _market_key(market_name) in {
                            "total", "totals", "over under", "over under total",
                            "over under match", "total over under",
                        }
                        and not any(term in market for term in (
                            "team", "home", "away", "player", "corner", "corners",
                            "углов", "card", "booking", "карточ", "half", "period",
                        ))
                    )
                ):
                    key = ("ТБ " if is_over else "ТМ ") + line_text
                    _record(key, odd, bookmaker, "total_market_odds")
                continue

            # Asian/European handicap — kept separately; never treated as 1X2.
            if any(term in market for term in ("handicap", "asian", "фора")):
                line = _sstats_line(label) or _sstats_line(market_name)
                if line is None:
                    continue
                if any(term in label_norm for term in ("home", " хозя", "п1", "1")):
                    key = f"Ф1 {line:g}"
                elif any(term in label_norm for term in ("away", " гост", "п2", "2")):
                    key = f"Ф2 {line:g}"
                else:
                    continue
                _record(key, odd, bookmaker, "asian_handicap_market_odds")

    result = {
        "total_market_odds": {},
        "total_market_book_count": {},
        "total_market_confirmed": {},
        "btts_market_odds": {},
        "stat_market_odds": {},
        "asian_handicap_market_odds": {},
        "bookmaker_by_pick": bookmaker_by_pick,
        "extra_market_source": "sstats",
        "source": "sstats",
    }

    # Prefer the complete valid 1X2 line with the lowest overround. This makes
    # every no-vig probability traceable to one bookmaker's actual line.
    complete_lines = []
    for book, selections in one_x_two_by_book.items():
        if not all(selections.get(pick) for pick in ("П1", "X", "П2")):
            continue
        line = {pick: max(selections[pick]) for pick in ("П1", "X", "П2")}
        implied_sum = sum(1.0 / odd for odd in line.values())
        # Values outside this broad range are likely wrong market mapping/data.
        if not (0.85 <= implied_sum <= 1.35):
            continue
        complete_lines.append((implied_sum, book, line))
    if complete_lines:
        _, selected_book, selected_line = min(complete_lines, key=lambda item: item[0])
        result.update(selected_line)
        if selected_book:
            result["bookmaker_by_pick"] = {pick: selected_book for pick in selected_line}
        else:
            # A compact feed without book provenance cannot be safely de-vigged.
            result["bookmaker_by_pick"] = {}

    for group in (
        "total_market_odds", "btts_market_odds", "stat_market_odds",
        "asian_handicap_market_odds",
    ):
        valid_by_key = {}
        for key, values in prices[group].items():
            valid = [
                (float(odd), str(book or "").strip())
                for odd, book in values
                if 1.01 < float(odd) <= 100.0
            ]
            # Every two-way market requires bookmaker identity on both sides.
            # Anonymous compact-feed values can never form a deviggable line.
            if group == "total_market_odds":
                # Totals require identifiable bookmakers even for display;
                # BTTS/stat quotes may remain visible without a book, but they
                # will not form a coherent pair or actionable recommendation.
                valid = [(odd, book) for odd, book in valid if book]
            if not valid:
                continue

            ordered = sorted(odd for odd, _book in valid)
            middle = len(ordered) // 2
            median = (
                ordered[middle] if len(ordered) % 2
                else (ordered[middle - 1] + ordered[middle]) / 2.0
            )
            credible = [
                (odd, book) for odd, book in valid
                if odd <= median * 1.25
            ]
            if group in {"total_market_odds", "stat_market_odds"}:
                credible = [(odd, book) for odd, book in credible if odd <= 8.0]
            if credible:
                valid_by_key[key] = credible

        if group == "asian_handicap_market_odds":
            for key, credible in valid_by_key.items():
                best_odd, best_book = max(credible, key=lambda item: item[0])
                result[group][key] = best_odd
                if best_book:
                    result["bookmaker_by_pick"][key] = best_book
            continue

        pair_specs = []
        if group == "btts_market_odds":
            pair_specs.append(("BTTS Да", "BTTS Нет"))
        elif group == "total_market_odds":
            seen_lines = set()
            for key in valid_by_key:
                match = re.fullmatch(r"ТБ\s+(\d+(?:[.,]\d+)?)", key, re.I)
                if not match:
                    continue
                line = f"{float(match.group(1).replace(',', '.')):g}"
                if line not in seen_lines:
                    seen_lines.add(line)
                    pair_specs.append((f"ТБ {line}", f"ТМ {line}"))
        else:
            seen_lines = set()
            for key in valid_by_key:
                match = re.fullmatch(
                    r"(Угл|Карты)\s+ТБ\s+(\d+(?:[.,]\d+)?)", key, re.I
                )
                if not match:
                    continue
                prefix = "Угл" if match.group(1).casefold() == "угл" else "Карты"
                line = f"{float(match.group(2).replace(',', '.')):g}"
                identity = (prefix, line)
                if identity not in seen_lines:
                    seen_lines.add(identity)
                    pair_specs.append((f"{prefix} ТБ {line}", f"{prefix} ТМ {line}"))

        paired_keys = set()
        for over_key, under_key in pair_specs:
            paired_keys.update((over_key, under_key))
            by_book = {}
            for side, key in (("over", over_key), ("under", under_key)):
                for odd, book in valid_by_key.get(key, []):
                    book_key = _book_key(book)
                    if not book_key:
                        continue
                    quote = by_book.setdefault(book_key, {"name": book})
                    quote[side] = max(odd, quote.get(side, 0.0))

            complete_lines = []
            for book_key, quotes in by_book.items():
                over_odd, under_odd = quotes.get("over"), quotes.get("under")
                if over_odd is None or under_odd is None:
                    continue
                overround = (1.0 / over_odd) + (1.0 / under_odd)
                if not (0.85 <= overround <= 1.35):
                    continue
                complete_lines.append((
                    overround, -(over_odd + under_odd), book_key,
                    quotes["name"], over_odd, under_odd,
                ))

            if complete_lines:
                # Pick the most efficient complete bookmaker line. Never mix
                # the best Over from one book with the best Under from another.
                _, _, _, selected_book, over_odd, under_odd = min(
                    complete_lines, key=lambda item: (item[0], item[1], item[2])
                )
                result[group][over_key] = over_odd
                result[group][under_key] = under_odd
                result["bookmaker_by_pick"][over_key] = selected_book
                result["bookmaker_by_pick"][under_key] = selected_book
                if group == "total_market_odds":
                    common_book_count = len(complete_lines)
                    for selection in (over_key, under_key):
                        result["total_market_book_count"][selection] = common_book_count
                        result["total_market_confirmed"][selection] = common_book_count >= 2
            elif over_rows := valid_by_key.get(over_key):
                # Invalid/outlier pairs may be displayed, but deliberately lose
                # bookmaker provenance so they cannot enter the value selector.
                for selection in (over_key, under_key):
                    result["bookmaker_by_pick"].pop(selection, None)

        # Keep one-sided lines visible as raw reference prices only.
        for key, credible in valid_by_key.items():
            if key in result[group]:
                continue
            best_odd, best_book = max(credible, key=lambda item: item[0])
            result[group][key] = best_odd
            if group == "total_market_odds":
                result["total_market_book_count"][key] = 0
                result["total_market_confirmed"][key] = False
            # A truly one-sided market has no opposing selection to devig.
            # If its pair was present but invalid, never keep one side's book ID.
            invalid_pair = any(
                key in pair and pair[0] not in result[group] and pair[1] not in result[group]
                for pair in pair_specs
            )
            if best_book and not invalid_pair:
                result["bookmaker_by_pick"][key] = best_book
    return result


def _sstats_request(path: str, api_key: str, params=None, timeout=(5, 25)):
    """GET SStats API using its documented ?apikey= authentication.

    Never include request URLs or exception strings in logs: both can expose
    the query-string API key.
    """
    query = dict(params or {})
    query["apikey"] = str(api_key or "").strip()
    return _sess.get(
        f"{SSTATS_API_BASE}/{str(path).lstrip('/')}",
        params=query,
        headers={"Accept": "application/json"},
        timeout=timeout,
        proxies=NO_PROXY,
    )


def sstats_account_check(api_key: str) -> dict:
    """Verify key, upcoming fixtures endpoint and prematch odds endpoint."""
    key = str(api_key or "").strip()
    if not key:
        return {"ok": False, "message": "SStats API-ключ не найден."}
    try:
        response = _sstats_request("Account/Info", key)
    except Exception as exc:
        return {
            "ok": False,
            "message": f"Не удалось соединиться с SStats: {type(exc).__name__}.",
        }
    if response.status_code != 200:
        return {
            "ok": False, "http_status": response.status_code,
            "message": f"SStats отклонил ключ или запрос: HTTP {response.status_code}.",
        }
    try:
        account_payload = response.json() or {}
        account_data = _sstats_unwrap(account_payload) or {}
    except Exception as exc:
        return {
            "ok": False,
            "http_status": response.status_code,
            "message": f"SStats не подтвердил ключ ({type(exc).__name__}).",
        }
    if not isinstance(account_data, dict) or not account_data.get("userName"):
        message = str(account_payload.get("message") or "SStats не подтвердил аккаунт.")
        return {"ok": False, "message": message[:180]}

    now, timezone_offset = _sstats_timezone()
    params = {
        "From": now.date().isoformat(),
        "To": (now.date() + timedelta(days=14)).isoformat(),
        "Upcoming": "true",
        "Limit": 8,
        "Offset": 0,
        "TimeZone": timezone_offset,
    }
    try:
        games_response = _sstats_request("Games/list", key, params=params)
        if games_response.status_code != 200:
            return {
                "ok": True, "games_ok": False,
                "message": (
                    f"Ключ SStats подтверждён ({account_data.get('userName')}); "
                    f"но /Games/list ответил HTTP {games_response.status_code}."
                ),
            }
        games_payload = games_response.json() or {}
        games = _sstats_unwrap(games_payload) or []
        if not isinstance(games, list):
            games = []
    except Exception as exc:
        return {
            "ok": True, "games_ok": False,
            "message": (
                f"Ключ SStats подтверждён ({account_data.get('userName')}), "
                f"но проверка расписания завершилась ошибкой {type(exc).__name__}."
            ),
        }

    odds_count = 0
    odds_checked = False
    checked_matches = 0
    totals_fixtures = 0
    total_line_count = 0
    total_quote_line_count = 0
    unconfirmed_total_line_count = 0
    btts_fixtures = 0
    handicap_fixtures = 0
    markets_seen = set()
    # The first scheduled match may have no odds yet. Probe several upcoming
    # fixtures so this button measures market coverage, not one random game.
    sample_games = [
        game for game in games
        if isinstance(game, dict) and game.get("id") is not None
    ][:5]
    for game in sample_games:
        try:
            detailed = sstats_odds_fixture(game.get("id"), key)
        except Exception:
            detailed = None
        if not isinstance(detailed, dict):
            continue
        odds_checked = True
        checked_matches += 1
        totals = detailed.get("total_market_odds") or {}
        btts = detailed.get("btts_market_odds") or {}
        handicaps = detailed.get("asian_handicap_market_odds") or {}
        odds_count += sum(
            len(detailed.get(group) or {})
            for group in (
                "total_market_odds", "btts_market_odds",
                "stat_market_odds", "asian_handicap_market_odds",
            )
        )
        total_lines = set()
        for selection in totals:
            line_match = re.fullmatch(
                r"Т[БМ]\s+(\d+(?:[.,]\d+)?)", str(selection), re.I
            )
            if line_match:
                total_lines.add(
                    f"{float(line_match.group(1).replace(',', '.')):g}"
                )
        confirmed_totals = set()
        for selection, is_confirmed in (detailed.get("total_market_confirmed") or {}).items():
            if not is_confirmed:
                continue
            line_match = re.fullmatch(
                r"Т[БМ]\s+(\d+(?:[.,]\d+)?)", str(selection), re.I
            )
            if line_match:
                confirmed_totals.add(
                    f"{float(line_match.group(1).replace(',', '.')):g}"
                )
        total_quote_line_count += len(total_lines)
        unconfirmed_total_line_count += len(total_lines - confirmed_totals)
        if confirmed_totals:
            totals_fixtures += 1
            total_line_count += len(confirmed_totals)
        if btts:
            btts_fixtures += 1
        if handicaps:
            handicap_fixtures += 1
        markets_seen.update(detailed.get("market_names") or [])

    if not sample_games:
        detail_message = "в выборке нет предстоящих матчей для проверки Odds."
    elif not odds_checked:
        detail_message = "не удалось получить успешный ответ Odds ни для одного проверенного матча."
    elif total_line_count:
        detail_message = (
            f"подтверждённые тоталы: {total_line_count} линий на {totals_fixtures} "
            f"из {checked_matches} проверенных матчей."
        )
        if unconfirmed_total_line_count:
            detail_message += (
                f" Ещё {unconfirmed_total_line_count} линий пришли только от одного букмекера."
            )
    elif total_quote_line_count:
        detail_message = (
            f"получено {total_quote_line_count} реальных линий тотала, но каждая "
            "только от одного букмекера; для автоматической ставки нужна вторая независимая котировка."
        )
    else:
        detail_message = (
            f"Odds ответил для {checked_matches} матчей, но котировки на тоталы голов "
            "в этой выборке не найдены."
        )
    other_markets = []
    if btts_fixtures:
        other_markets.append(f"BTTS: {btts_fixtures} матчей")
    if handicap_fixtures:
        other_markets.append(f"форы: {handicap_fixtures} матчей")
    other_hint = (" Другие рынки: " + "; ".join(other_markets) + ".") if other_markets else ""
    market_hint = (
        " Примеры рынков API: " + ", ".join(sorted(markets_seen)[:8]) + "."
        if markets_seen and not total_line_count else ""
    )
    return {
        "ok": True, "games_ok": True, "games_count": len(games),
        "odds_checked": odds_checked, "odds_count": odds_count,
        "checked_matches": checked_matches,
        "totals_fixtures": totals_fixtures,
        "total_line_count": total_line_count,
        "total_quote_line_count": total_quote_line_count,
        "unconfirmed_total_line_count": unconfirmed_total_line_count,
        "btts_fixtures": btts_fixtures,
        "handicap_fixtures": handicap_fixtures,
        "markets_seen": sorted(markets_seen)[:12],
        "message": (
            f"SStats подключён: ключ подтверждён для {account_data.get('userName')}; "
            f"расписание отвечает ({len(games)} матчей). {detail_message}"
            f"{other_hint}{market_hint}"
        ),
    }


def sstats_upcoming_matches(days: int, api_key: str, logs=None) -> list:
    """Load upcoming fixtures and compact 1X2 odds; totals require corroborated bookmaker quotes."""
    key = str(api_key or "").strip()
    if not key:
        if logs is not None:
            logs.append("ℹ️ SStats: API-ключ SSTATS_API_KEY не задан")
        return []
    now, timezone_offset = _sstats_timezone()
    horizon = max(1, min(30, int(days or 3)))
    cache_key = (
        f"sstats_upcoming_v5_{now.date().isoformat()}_{horizon}_"
        f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"
    )
    cached = cache_get(cache_key, 900)
    if cached is not None:
        if logs is not None:
            logs.append(f"SStats: использован кэш расписания ({len(cached)} матчей)")
        return cached if isinstance(cached, list) else []

    params = {
        "From": now.date().isoformat(),
        "To": (now.date() + timedelta(days=horizon)).isoformat(),
        "Upcoming": "true",
        "Limit": 1000,
        "Offset": 0,
        "TimeZone": timezone_offset,
        "Order": 1,
    }
    try:
        response = _sstats_request("Games/list", key, params=params, timeout=(5, 30))
        if response.status_code != 200:
            if logs is not None:
                logs.append(f"⚠️ SStats /Games/list HTTP {response.status_code}")
            return []
        payload = response.json() or {}
        records = _sstats_unwrap(payload)
        if not isinstance(records, list):
            if logs is not None:
                logs.append("⚠️ SStats /Games/list: неожиданный формат data")
            return []
    except Exception as exc:
        if logs is not None:
            logs.append(f"⚠️ SStats /Games/list: {type(exc).__name__}")
        return []

    rows = []
    for game in records:
        if not isinstance(game, dict):
            continue
        home_obj = game.get("homeTeam") or {}
        away_obj = game.get("awayTeam") or {}
        home = str(home_obj.get("name") or game.get("homeTeamName") or "").strip()
        away = str(away_obj.get("name") or game.get("awayTeamName") or "").strip()
        raw_date = str(game.get("date") or "")
        if not home or not away or not raw_date:
            continue
        kickoff = _parse_provider_date(raw_date)
        if not kickoff:
            continue
        # Games/list may return either a local wall-clock time (because TimeZone
        # was requested) or an ISO timestamp with an explicit offset. Compare
        # both in the app timezone; stripping the offset used to hide evening
        # international fixtures when UTC and local calendar dates differed.
        if now.tzinfo is not None:
            kickoff_local = (
                kickoff.replace(tzinfo=now.tzinfo)
                if kickoff.tzinfo is None
                else kickoff.astimezone(now.tzinfo)
            )
        else:
            kickoff_local = kickoff.replace(tzinfo=None)
        now_local = now
        if not (
            now_local - timedelta(minutes=5)
            <= kickoff_local
            < now_local + timedelta(days=horizon)
        ):
            continue
        kickoff_utc = (
            kickoff_local.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
            if kickoff_local.tzinfo is not None
            else ""
        )

        league_obj = (game.get("season") or {}).get("league") or {}
        league_name = str(
            league_obj.get("name")
            or game.get("leagueName")
            or game.get("league")
            or "SStats"
        ).strip()
        div = _sstats_div_code(league_name)
        normalized = _sstats_parse_markets(game.get("odds") or [])
        compact_1x2 = {
            key: normalized[key]
            for key in ("П1", "X", "П2")
            if normalized.get(key) is not None
        }
        start_time = kickoff_local.strftime("%H:%M")
        row = {
            "Date": kickoff_local.strftime("%Y-%m-%d %H:%M:%S"),
            "Time": start_time,
            "kickoff_utc": kickoff_utc,
            "HomeTeam": home,
            "AwayTeam": away,
            "Div": div,
            "League": league_name,
            "fixture_id": f"sstats:{game.get('id')}",
            "sstats_game_id": game.get("id"),
            "source": "sstats",
            "odds_source": "sstats" if compact_1x2 else "no_market",
            "market_odds": compact_1x2,
            "bookmaker_by_pick": normalized.get("bookmaker_by_pick") or {},
            "total_market_odds": normalized.get("total_market_odds") or {},
            "btts_market_odds": normalized.get("btts_market_odds") or {},
            "stat_market_odds": normalized.get("stat_market_odds") or {},
            "asian_handicap_market_odds": normalized.get("asian_handicap_market_odds") or {},
            "total_odds_source": (
                "sstats" if normalized.get("total_market_odds") else "no_market"
            ),
            "extra_market_source": (
                "sstats"
                if any(normalized.get(name) for name in (
                    "total_market_odds", "btts_market_odds", "stat_market_odds",
                    "asian_handicap_market_odds",
                ))
                else ""
            ),
            "sstats_status": game.get("status"),
        }
        rows.append(row)

    # Deduplicate repeated records by kickoff date and ordered teams.
    unique = {}
    for row in rows:
        key_tuple = (
            str(row.get("Date") or "")[:10],
            _norm_name(row.get("HomeTeam") or ""),
            _norm_name(row.get("AwayTeam") or ""),
        )
        if key_tuple not in unique:
            unique[key_tuple] = row
            continue
        current = unique[key_tuple]
        current["total_market_odds"].update(row.get("total_market_odds") or {})
        current["market_odds"].update(row.get("market_odds") or {})
        current["btts_market_odds"].update(row.get("btts_market_odds") or {})
        current["stat_market_odds"].update(row.get("stat_market_odds") or {})
    rows = list(unique.values())
    cache_put(cache_key, rows)
    if logs is not None:
        quoted = sum(bool(row.get("market_odds")) for row in rows)
        totals = sum(bool(row.get("total_market_odds")) for row in rows)
        logs.append(
            f"✅ SStats: загружено {len(rows)} предстоящих матчей; "
            f"1X2 у {quoted}, тоталы у {totals}."
        )
    return rows


def _sstats_div_code(league_name: str) -> str:
    """Map SStats competitions without collapsing unknown tournaments into G."""
    name = re.sub(r"\s+", " ", str(league_name or "").casefold()).strip()
    if not name:
        return "SSTATS_UNKNOWN"

    # Gender and age are resolved before competition labels. This keeps women's
    # and youth international football out of men's senior calibration/history.
    women = any(token in name for token in ("women", "womens", "ladies", "female", "feminine"))
    age = next(
        (age for age in ("u21", "u-21", "under 21", "u19", "u-19", "under 19",
                         "u23", "u-23", "under 23", "u20", "u-20", "u18", "u-18",
                         "u17", "u-17")
         if age in name),
        "",
    )
    international = any(token in name for token in (
        "national", "international", "nations league", "world cup", "world cup qualification",
        "euro", "africa cup", "african cup", "afcon", "copa america",
        "asian cup", "asian games", "friendl", "qualifier", "qualification",
    ))
    if international or age:
        if "nations league" in name:
            competition = "UNL"
        elif "afcon" in name or "africa cup" in name or "african cup" in name:
            competition = "AFCON"
        elif "copa america" in name:
            competition = "COPA"
        elif "asian cup" in name:
            competition = "ASIAN"
        elif "world cup" in name:
            competition = "WCQ" if any(x in name for x in ("qualif", "qualification")) else "WC"
        elif "euro" in name:
            competition = "EURO"
        elif "friendl" in name:
            competition = "FR"
        elif any(x in name for x in ("qualifier", "qualification")):
            competition = "QUAL"
        else:
            competition = "INTL_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:7].upper()
        age_code = re.sub(r"[^0-9]", "", age)
        prefix = "W_NT" if women else "NT"
        return f"{prefix}_{'U' + age_code + '_' if age_code else ''}{competition}"[:24]

    if women:
        women_mappings = (
            (("women's super league", "womens super league", "fa wsl"), "W_E0"),
            (("liga f", "primera division femenina"), "W_SP1"),
            (("d1 feminine", "division 1 feminine"), "W_F1"),
            (("frauen bundesliga", "frauen-bundesliga"), "W_D1"),
            (("serie a women", "serie a femminile"), "W_I1"),
            (("women's champions league", "womens champions league"), "W_C1"),
        )
        for tokens, code in women_mappings:
            if any(token in name for token in tokens):
                return code
        return "W_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10].upper()

    mappings = (
        (("premier league", "premier league of england", "england premier", "epl"), "E0"),
        (("efl championship", "english championship", "sky bet championship"), "E1"),
        (("league one", "england league one", "efl league one"), "E2"),
        (("league two", "england league two", "efl league two"), "E3"),
        (("la liga", "primera division", "spain primera"), "SP1"),
        (("segunda division", "la liga 2", "spain segunda"), "SP2"),
        (("serie a", "italy serie a"), "I1"),
        (("serie b", "italy serie b"), "I2"),
        (("bundesliga", "germany bundesliga"), "D1"),
        (("2. bundesliga", "2 bundesliga"), "D2"),
        (("ligue 1", "france ligue 1"), "F1"),
        (("ligue 2", "france ligue 2"), "F2"),
        (("eredivisie", "netherlands eredivisie"), "N1"),
        (("primeira liga", "portugal primeira"), "P1"),
        (("jupiler pro league", "belgian pro league"), "B1"),
        (("super lig", "turkey super lig"), "T1"),
        (("uefa champions league", "champions league"), "C1"),
        (("uefa europa league", "europa league"), "EL"),
        (("uefa conference league", "conference league"), "EC"),
        (("usl", "united soccer league"), "US_USL"),
        (("mls", "major league soccer"), "US_MLS"),
    )
    for tokens, code in mappings:
        if any(token in name for token in tokens):
            return code

    # Stable per-competition namespace: no unknown league may fall back to the
    # shared "G" bucket and contaminate another competition's team samples.
    return "S_" + hashlib.sha1(name.encode("utf-8")).hexdigest()[:10].upper()


def sstats_find_match_id(
    date_iso: str, home_team: str, away_team: str, api_key: str,
    logs=None,
) -> Optional[int]:
    """Resolve legacy portfolio entries only when date and ordered teams match uniquely."""
    key = str(api_key or "").strip()
    date_text = str(date_iso or "")[:10]
    try:
        datetime.strptime(date_text, "%Y-%m-%d")
    except ValueError:
        return None
    home_key, away_key = _norm_name(home_team), _norm_name(away_team)
    if not key or not home_key or not away_key:
        return None
    cache_key = (
        f"sstats_match_id_v1_{date_text}_{home_key}_{away_key}_"
        f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"
    )
    cached = cache_get(cache_key, 6 * 60 * 60)
    if cached is not None:
        try:
            return int(cached) if cached else None
        except (TypeError, ValueError):
            return None
    _now, timezone_offset = _sstats_timezone()
    try:
        response = _sstats_request(
            "Games/list", key,
            params={
                "Date": date_text, "Ended": "true", "Limit": 1000,
                "Offset": 0, "TimeZone": timezone_offset,
            },
            timeout=(5, 25),
        )
        if response.status_code != 200:
            if logs is not None:
                logs.append(f"⚠️ SStats match lookup: HTTP {response.status_code}")
            return None
        records = _sstats_unwrap(response.json() or {})
    except Exception as exc:
        if logs is not None:
            logs.append(f"⚠️ SStats match lookup: {type(exc).__name__}")
        return None
    if not isinstance(records, list):
        return None
    matches = []
    for game in records:
        if not isinstance(game, dict):
            continue
        game_date = str(game.get("date") or "")[:10]
        h = game.get("homeTeam") or {}
        a = game.get("awayTeam") or {}
        if not isinstance(h, dict) or not isinstance(a, dict):
            continue
        if (
            game_date == date_text
            and _norm_name(str(h.get("name") or game.get("homeTeamName") or "")) == home_key
            and _norm_name(str(a.get("name") or game.get("awayTeamName") or "")) == away_key
            and game.get("id") not in (None, "")
        ):
            try:
                matches.append(int(game["id"]))
            except (TypeError, ValueError):
                continue
    # Multiple same-team fixtures on the same date are ambiguous, so do not guess.
    resolved = matches[0] if len(set(matches)) == 1 else None
    cache_put(cache_key, resolved)
    return resolved


def sstats_match_statistics(game_id, api_key: str, logs=None) -> Optional[dict]:
    """Fetch final verified corners/cards from the SStats full-match endpoint.

    SStats status 8 is a normal full-time result. Extra-time/penalty statuses are
    deliberately not accepted because these pre-match total markets are settled
    on regulation-time statistics. Missing fields stay None; they are never zero.
    """
    key = str(api_key or "").strip()
    if not key or game_id in (None, ""):
        return None
    try:
        game_id_text = str(int(game_id))
    except (TypeError, ValueError):
        game_id_text = str(game_id).strip()
    if not game_id_text or not re.fullmatch(r"\d+", game_id_text):
        return None

    cache_key = (
        f"sstats_match_stats_v1_{game_id_text}_"
        f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"
    )
    cached = cache_get(cache_key, 10 * 60)
    if isinstance(cached, dict):
        return cached

    try:
        response = _sstats_request(
            f"Games/{game_id_text}", key, timeout=(5, 25)
        )
    except Exception as exc:
        if logs is not None:
            logs.append(f"⚠️ SStats Game/{game_id_text}: {type(exc).__name__}")
        return None
    if response.status_code != 200:
        if logs is not None:
            logs.append(
                f"⚠️ SStats Game/{game_id_text}: HTTP {response.status_code}"
            )
        return None

    try:
        payload = response.json() or {}
        data = _sstats_unwrap(payload)
    except Exception as exc:
        if logs is not None:
            logs.append(f"⚠️ SStats Game/{game_id_text}: {type(exc).__name__}")
        return None
    if not isinstance(data, dict):
        return None
    game = data.get("game") or {}
    stats = data.get("statistics") or {}
    if not isinstance(game, dict) or not isinstance(stats, dict):
        return None
    try:
        status = int(game.get("status"))
    except (TypeError, ValueError):
        status = None

    result = {
        "game_id": game_id_text,
        "status": status,
        "status_name": str(game.get("statusName") or ""),
        "finished": status == 8,
        "corners": None,
        "cards": None,
        "source": "sstats",
    }
    if status != 8:
        return result

    def _count(field):
        value = stats.get(field)
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(str(value).replace(",", "."))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number) or number < 0 or not number.is_integer():
            return None
        return int(number)

    corners_home = _count("cornerKicksHome")
    corners_away = _count("cornerKicksAway")
    if corners_home is not None and corners_away is not None:
        result["corners"] = corners_home + corners_away

    yellow_home = _count("yellowCardsHome")
    yellow_away = _count("yellowCardsAway")
    red_home = _count("redCardsHome")
    red_away = _count("redCardsAway")
    if all(value is not None for value in (
        yellow_home, yellow_away, red_home, red_away
    )):
        result["cards"] = (
            yellow_home + yellow_away + red_home + red_away
        )

    # A completed-but-uncovered competition is cached briefly to avoid hammering
    # the source; the next normal retry can fill fields that arrive later.
    cache_put(cache_key, result)
    if logs is not None:
        logs.append(
            f"SStats Game/{game_id_text}: FT; "
            f"угловые={'есть' if result['corners'] is not None else 'нет'}, "
            f"карточки={'есть' if result['cards'] is not None else 'нет'}."
        )
    return result



def _sstats_context_cache_get(endpoint: str, api_key: str, params=None, *, ttl: int = 900):
    """Cached SStats GET for context endpoints; credentials never enter cache values."""
    query = dict(params or {})
    key_tag = hashlib.sha256(str(api_key or "").encode("utf-8")).hexdigest()[:12]
    params_tag = hashlib.sha256(
        repr(sorted((str(k), str(v)) for k, v in query.items())).encode("utf-8")
    ).hexdigest()[:12]
    cache_key = f"sstats_context_v2_{str(endpoint).strip('/').replace('/', '_')}_{key_tag}_{params_tag}"
    cached = cache_get(cache_key, ttl)
    if isinstance(cached, dict):
        return cached
    response = _sstats_request(endpoint, api_key, params=query, timeout=(4, 12))
    if response.status_code != 200:
        raise RuntimeError(f"SStats context HTTP {response.status_code}")
    payload = response.json() or {}
    if not isinstance(payload, dict) or str(payload.get("status") or "").casefold() == "error":
        raise ValueError("SStats context returned an API error")
    cache_put(cache_key, payload)
    return payload


def _sstats_context_data(payload):
    try:
        return _sstats_unwrap(payload)
    except Exception:
        return None


def _sstats_context_name(value):
    if isinstance(value, dict):
        for field in ("name", "teamName", "team_name", "participantName", "playerName", "player_name", "fullName", "shortName", "displayName", "title"):
            candidate = value.get(field)
            if candidate not in (None, "") and not isinstance(candidate, (dict, list)):
                return str(candidate).strip()
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _sstats_context_find_value(value, wanted_keys):
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z]", "", str(key).casefold())
            if normalized in wanted_keys:
                if isinstance(item, dict):
                    found = _sstats_context_name(item)
                    if found:
                        return found
                elif isinstance(item, (str, int, float)) and str(item).strip():
                    return str(item).strip()
        for item in value.values():
            found = _sstats_context_find_value(item, wanted_keys)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _sstats_context_find_value(item, wanted_keys)
            if found:
                return found
    return ""


def _sstats_context_player_rows(value):
    """Extract starters from common Ls/GameLineups response shapes.

    Generic squad/player collections are accepted only when each record marks
    the player as a starter. Dedicated starting-XI arrays need no extra flag.
    """
    player_fields = {
        "player", "playername", "fullname", "shortname", "displayname",
    }
    starter_arrays = {
        "lineup", "startingxi", "startxi", "startingplayers",
        "startinglineup", "starters",
    }
    generic_arrays = {"players", "squad"}

    def _walk(node, require_starter_marker=False):
        if isinstance(node, list):
            result = []
            for item in node:
                if isinstance(item, dict):
                    player_obj = item.get("player") if isinstance(item.get("player"), dict) else item
                    name = _sstats_context_name(player_obj)
                    if name:
                        is_starter = None
                        for flag in ("isStarting", "isStarter", "starting", "starter"):
                            if flag in item:
                                raw = item.get(flag)
                                is_starter = (
                                    raw is True
                                    or str(raw).strip().casefold() in {"1", "true", "yes", "starter", "starting"}
                                )
                                break
                        if is_starter is False:
                            continue
                        if require_starter_marker and is_starter is not True:
                            continue
                        result.append(name)
                    else:
                        result.extend(_walk(item, require_starter_marker))
                elif isinstance(item, str) and item.strip() and not require_starter_marker:
                    result.append(item.strip())
            return result
        if isinstance(node, dict):
            for key, item in node.items():
                normalized = re.sub(r"[^a-z]", "", str(key).casefold())
                if normalized in starter_arrays and isinstance(item, (dict, list)):
                    found = _walk(item, False)
                    if found:
                        return found
            for key, item in node.items():
                normalized = re.sub(r"[^a-z]", "", str(key).casefold())
                if normalized in generic_arrays and isinstance(item, (dict, list)):
                    found = _walk(item, True)
                    if found:
                        return found
            result = []
            for key, item in node.items():
                normalized = re.sub(r"[^a-z]", "", str(key).casefold())
                if normalized in player_fields:
                    name = _sstats_context_name(item)
                    if name:
                        result.append(name)
            if result:
                return result
            for key, item in node.items():
                normalized = re.sub(r"[^a-z]", "", str(key).casefold())
                # Dedicated and generic roster arrays have already been handled
                # above. Do not revisit a generic players/squad array without
                # its starter-marker requirement, or an unconfirmed squad of
                # eleven can be mistaken for a confirmed starting XI.
                if normalized in starter_arrays or normalized in generic_arrays:
                    continue
                found = _walk(item, require_starter_marker)
                if found:
                    return found
        return []

    names = _walk(value)
    unique = []
    seen = set()
    for name in names:
        key = re.sub(r"[^a-zа-я0-9]", "", str(name).casefold())
        if key and key not in seen:
            seen.add(key)
            unique.append(str(name))
    return unique


def _sstats_context_side_lineup(payload, side: str, team_name: str):
    data = _sstats_context_data(payload)
    if isinstance(data, dict):
        side_keys = (
            {"home", "hometeam", "homelineup", "homeplayers", "local"}
            if side == "home" else
            {"away", "awayteam", "awaylineup", "awayplayers", "visitor"}
        )
        for key, value in data.items():
            normalized = re.sub(r"[^a-z]", "", str(key).casefold())
            if normalized in side_keys:
                players = _sstats_context_player_rows(value)
                if players:
                    return players
    # Some feeds return a teams[] array; identify each team by its name.
    target = _norm_name(team_name)
    def _find_team(node):
        if isinstance(node, dict):
            team = node.get("team") if isinstance(node.get("team"), dict) else node
            label = _sstats_context_name(team)
            if label and _norm_name(label) == target:
                players = _sstats_context_player_rows(node)
                if players:
                    return players
            for child in node.values():
                found = _find_team(child)
                if found:
                    return found
        elif isinstance(node, list):
            for child in node:
                found = _find_team(child)
                if found:
                    return found
        return []
    return _find_team(data)


def _sstats_context_injuries_confirmed(data):
    """Only treat a documented list payload as a confirmed injury-feed response."""
    if isinstance(data, list):
        return True
    if isinstance(data, dict):
        return any(
            isinstance(data.get(field), list)
            for field in ("injuries", "players", "home", "away", "items", "results")
        )
    return False


def _sstats_context_injuries(payload):
    data = _sstats_context_data(payload)
    if data is None:
        return []
    if isinstance(data, dict):
        collected = []
        for field in ("injuries", "players", "home", "away", "items", "results"):
            value = data.get(field)
            if isinstance(value, list):
                collected.extend(value)
        data = collected
    if not isinstance(data, list):
        return []
    result = []
    for item in data:
        if not isinstance(item, dict):
            continue
        player = item.get("player") if isinstance(item.get("player"), dict) else item
        team = item.get("team") if isinstance(item.get("team"), dict) else {}
        name = _sstats_context_name(player)
        if not name:
            continue
        result.append({
            "player_id": player.get("id"),
            "player_name": name,
            "team_id": team.get("id"),
            "team_name": _sstats_context_name(team),
            "type": str(item.get("type") or player.get("type") or ""),
            "reason": str(item.get("reason") or player.get("reason") or ""),
        })
    return result


def _sstats_ls_event_key(event: dict):
    """Extract the ordered teams, match date and string event ID from Ls/List."""
    if not isinstance(event, dict):
        return None, ""
    info = event.get("matchInfo") if isinstance(event.get("matchInfo"), dict) else {}
    game = event.get("game") if isinstance(event.get("game"), dict) else {}
    nested_event = event.get("event") if isinstance(event.get("event"), dict) else {}
    merged = {**game, **nested_event, **event, **info}
    teams = merged.get("teams") if isinstance(merged.get("teams"), dict) else {}

    home = (
        merged.get("homeTeam") or merged.get("home") or merged.get("home_team")
        or teams.get("home") or teams.get("homeTeam") or ""
    )
    away = (
        merged.get("awayTeam") or merged.get("away") or merged.get("away_team")
        or teams.get("away") or teams.get("awayTeam") or ""
    )

    # Some Ls responses expose participants[] instead of homeTeam/awayTeam.
    participants = merged.get("participants")
    if (not home or not away) and isinstance(participants, list):
        for participant in participants:
            if not isinstance(participant, dict):
                continue
            role = " ".join(str(participant.get(field) or "") for field in (
                "side", "position", "homeAway", "venueRole", "participantType", "role",
            )).casefold()
            is_home = any(token in role for token in ("home", "host", "local"))
            is_away = any(token in role for token in ("away", "visitor"))
            if is_home and not home:
                home = participant.get("team") or participant
            elif is_away and not away:
                away = participant.get("team") or participant

    home_name = _sstats_context_name(home)
    away_name = _sstats_context_name(away)
    raw_date_value = next(
        (merged.get(field) for field in (
            "date", "startTime", "kickoff", "dateTime", "matchDate", "startDate",
        ) if merged.get(field) not in (None, "")),
        "",
    )
    if isinstance(raw_date_value, (int, float)):
        try:
            timestamp = float(raw_date_value)
            if timestamp > 10_000_000_000:
                timestamp /= 1000.0
            raw_date = datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            raw_date = ""
    else:
        raw_date = str(raw_date_value or "").strip()
    date_key = raw_date[:10]

    # Ls/GameInfo and Ls/GameLineups require the Flashscore string event ID.
    # A numeric SStats game ID must never be sent to those endpoints by mistake.
    external_id = ""
    for source in (info, nested_event, event, game):
        for field in (
            "eventId", "event_id", "flashscoreId", "flashScoreId",
            "stringId", "matchId", "id",
        ):
            candidate = source.get(field)
            if isinstance(candidate, str) and candidate.strip():
                external_id = candidate.strip()
                break
        if external_id:
            break
    referee = _sstats_context_find_value(
        merged, {"referee", "refereename", "mainreferee", "refereefullname"}
    )
    if not (date_key and home_name and away_name):
        return None, external_id
    return (
        date_key, _norm_name(home_name), _norm_name(away_name)
    ), external_id


def _sstats_ls_event_keys(event: dict, timezone_offset: int = 0):
    """Return exact home/away team keys for both source and local/UTC dates."""
    match_key, external_id = _sstats_ls_event_key(event)
    if not match_key:
        return [], external_id
    days = {match_key[0]}
    info = event.get("matchInfo") if isinstance(event.get("matchInfo"), dict) else {}
    game = event.get("game") if isinstance(event.get("game"), dict) else {}
    nested_event = event.get("event") if isinstance(event.get("event"), dict) else {}
    merged = {**game, **nested_event, **event, **info}
    raw_date = next(
        (merged.get(field) for field in (
            "date", "startTime", "kickoff", "dateTime", "matchDate", "startDate",
        ) if merged.get(field) not in (None, "")),
        "",
    )
    parsed = _parse_provider_date(raw_date)
    local_tz = timezone(timedelta(hours=int(timezone_offset)))
    if parsed is not None:
        if parsed.tzinfo is None:
            days.add(parsed.date().isoformat())
            days.add(parsed.replace(tzinfo=local_tz).astimezone(timezone.utc).date().isoformat())
        else:
            days.add(parsed.astimezone(timezone.utc).date().isoformat())
            days.add(parsed.astimezone(local_tz).date().isoformat())
    return [
        (day, match_key[1], match_key[2])
        for day in sorted(days)
    ], external_id


def sstats_context_enrich(
    rows: list[dict],
    api_key: str,
    *,
    date_from: str,
    date_to: str,
    now: datetime | None = None,
    max_injury_checks: int = 20,
    max_lineup_checks: int = 12,
    timezone_offset: int | None = None,
) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
    """Enrich SStats Games/list fixtures using SStats injuries and Flashscore Ls endpoints.

    Games/list remains the only fixture/odds source. Ls/List is queried only to
    resolve the provider's own string event IDs for GameInfo/GameLineups; it is
    never used to add or replace matches. Missing provider fields remain unknown.
    """
    key = str(api_key or "").strip()
    logs = []
    if not key:
        return {}, ["SStats context: API key отсутствует; контекст недоступен."]
    now = now or datetime.now()
    if timezone_offset is None:
        _clock, timezone_offset = _sstats_timezone()

    valid_rows = [
        row for row in (rows or [])
        if isinstance(row, dict)
        and row.get("sstats_game_id") not in (None, "")
        and row.get("HomeTeam") and row.get("AwayTeam")
    ]
    if not valid_rows:
        return {}, ["SStats context: нет матчей с SStats game ID."]

    ls_by_key = {}
    ls_event_count = 0
    ls_events_with_string_id = 0
    try:
        # The Ls endpoint is a separate Flashscore feed. Its supported filters
        # are not the Games/list date/time filters; keep this request to the
        # documented Upcoming + Limit parameters and match locally below.
        ls_payload = _sstats_context_cache_get(
            "Ls/List", key,
            {"Upcoming": "true", "Limit": 1000},
            ttl=1800,
        )
        ls_records = _sstats_context_data(ls_payload)
        if isinstance(ls_records, dict):
            ls_records = next(
                (ls_records.get(field) for field in (
                    "games", "matches", "events", "items", "results", "list", "fixtures",
                ) if isinstance(ls_records.get(field), list)),
                [],
            )
        if isinstance(ls_records, list):
            ls_event_count = len(ls_records)
            for event in ls_records:
                event_keys, external_id = _sstats_ls_event_keys(
                    event, timezone_offset=timezone_offset
                )
                if event_keys and external_id:
                    ls_events_with_string_id += 1
                    referee_name = _sstats_context_find_value(
                        event, {"referee", "refereename", "mainreferee", "refereefullname"}
                    )
                    for match_key in event_keys:
                        ls_by_key.setdefault(match_key, (external_id, referee_name))
        else:
            logs.append(
                "⚠️ SStats Ls/List: ответ не содержит ожидаемый список событий; "
                "составы и судьи оставлены неизвестными."
            )
    except Exception as exc:
        logs.append(f"⚠️ SStats Ls/List: {type(exc).__name__}; XI/судьи могут быть недоступны.")

    from datetime import timezone as dt_timezone
    local_tz = dt_timezone(timedelta(hours=int(timezone_offset)))

    def _kickoff(row):
        raw = str(row.get("kickoff_utc") or row.get("Date") or "")
        parsed = _parse_provider_date(raw)
        if parsed is None:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=local_tz).astimezone(dt_timezone.utc)
        return parsed.astimezone(dt_timezone.utc)

    current_utc = now.astimezone(dt_timezone.utc) if now.tzinfo else now.replace(tzinfo=dt_timezone.utc)
    injury_candidates = sorted(
        [
            row for row in valid_rows
            if _kickoff(row) is not None
            and timedelta(0) <= _kickoff(row) - current_utc <= timedelta(hours=24)
        ],
        key=lambda row: _kickoff(row),
    )[:max(0, int(max_injury_checks))]
    lineup_candidates = sorted(
        [
            row for row in valid_rows
            if _kickoff(row) is not None
            and timedelta(0) <= _kickoff(row) - current_utc <= timedelta(hours=6)
        ],
        key=lambda row: _kickoff(row),
    )[:max(0, int(max_lineup_checks))]

    context_by_fixture = {}
    injury_checked = injury_found = info_checked = lineup_checked = lineup_confirmed = referee_confirmed = 0
    linked_fixtures = 0

    for row in valid_rows:
        home = str(row.get("HomeTeam") or "")
        away = str(row.get("AwayTeam") or "")
        date_key = str(row.get("date_iso") or row.get("Date") or row.get("date") or "")[:10]
        fixture_key_value = (date_key, _norm_name(home), _norm_name(away))
        ls_match = ls_by_key.get(fixture_key_value)
        ls_id = ls_match[0] if ls_match else ""
        linked_fixtures += int(bool(ls_id))
        referee = ls_match[1] if ls_match else ""
        context = {
            "provider": "sstats",
            "sstats_game_id": str(row.get("sstats_game_id")),
            "sstats_event_id": ls_id or None,
            "kickoff_at": str(row.get("kickoff_utc") or row.get("Date") or "") or None,
            "timestamp": current_utc.isoformat(),
            "referee": {
                "id": None, "name": referee or None, "is_confirmed": bool(referee),
                "timestamp": current_utc.isoformat() if referee else None,
                "source_reliability": None,
            },
            "lineup": {
                "home": {"starting_xi": [], "confirmed": False},
                "away": {"starting_xi": [], "confirmed": False},
                "is_confirmed": False, "timestamp": None, "source_reliability": None,
            },
            "injuries": [], "injuries_is_confirmed": False,
            "injuries_timestamp": None, "injuries_source_reliability": None,
        }

        if row in injury_candidates:
            try:
                injury_payload = _sstats_context_cache_get(
                    "Games/injuries", key,
                    {"gameId": row.get("sstats_game_id")}, ttl=4 * 60 * 60,
                )
                injury_data = _sstats_context_data(injury_payload)
                injuries_confirmed = _sstats_context_injuries_confirmed(injury_data)
                context["injuries"] = _sstats_context_injuries(injury_payload)
                context["injuries_is_confirmed"] = injuries_confirmed
                context["injuries_timestamp"] = current_utc.isoformat() if injuries_confirmed else None
                injury_checked += int(injuries_confirmed)
                injury_found += len(context["injuries"])
                if not injuries_confirmed:
                    logs.append(
                        f"ℹ️ SStats injuries game={row.get('sstats_game_id')}: "
                        "структура ответа не подтверждает полный injury feed."
                    )
            except Exception as exc:
                logs.append(
                    f"⚠️ SStats injuries game={row.get('sstats_game_id')}: {type(exc).__name__}"
                )

        if row in lineup_candidates and ls_id:
            try:
                info_payload = _sstats_context_cache_get(
                    "Ls/GameInfo", key, {"id": ls_id}, ttl=60 * 60,
                )
                info_data = _sstats_context_data(info_payload)
                if info_data is not None:
                    info_checked += 1
                    if not context["referee"]["is_confirmed"]:
                        referee = _sstats_context_find_value(
                            info_data,
                            {"referee", "refereename", "mainreferee", "refereefullname"},
                        )
                        if referee:
                            context["referee"].update({
                                "name": referee, "is_confirmed": True,
                                "timestamp": current_utc.isoformat(),
                            })
                lineup_payload = _sstats_context_cache_get(
                    "Ls/GameLineups", key, {"id": ls_id}, ttl=10 * 60,
                )
                home_players = _sstats_context_side_lineup(
                    lineup_payload, "home", home
                )
                away_players = _sstats_context_side_lineup(
                    lineup_payload, "away", away
                )
                # Never promote a partial squad or a substitutes list to a confirmed XI.
                home_confirmed = len(home_players) == 11
                away_confirmed = len(away_players) == 11
                context["lineup"] = {
                    "home": {"starting_xi": home_players, "confirmed": home_confirmed},
                    "away": {"starting_xi": away_players, "confirmed": away_confirmed},
                    "is_confirmed": home_confirmed and away_confirmed,
                    "timestamp": current_utc.isoformat(),
                    "source_reliability": None,
                }
                lineup_checked += 1
                lineup_confirmed += int(home_confirmed and away_confirmed)
                referee_confirmed += int(bool(context["referee"]["is_confirmed"]))
            except Exception as exc:
                logs.append(
                    f"⚠️ SStats Ls details event={ls_id}: {type(exc).__name__}"
                )

        context_by_fixture[fixture_key_value] = context

    logs.append(
        "SStats context: "
        f"Ls/List events={ls_event_count}, with string ID={ls_events_with_string_id}, "
        f"linked fixtures={linked_fixtures}/{len(valid_rows)}; "
        f"injuries checked={injury_checked}, entries={injury_found}; "
        f"GameInfo={info_checked}; lineups checked={lineup_checked}, "
        f"confirmed XI={lineup_confirmed}; confirmed referees={referee_confirmed}."
    )
    if linked_fixtures == 0:
        logs.append(
            "ℹ️ SStats Ls/List не связал ни один Games/list матч с Flashscore string ID; "
            "составы и судьи оставлены неизвестными, без подстановки данных из другого матча."
        )
    return context_by_fixture, logs


def sstats_odds_fixture(game_id, api_key: str, logs=None) -> Optional[dict]:
    """Fetch full prematch bookmaker markets for one SStats fixture."""
    key = str(api_key or "").strip()
    if not key or game_id in (None, ""):
        return None
    cache_key = (
        f"sstats_full_odds_v5_{game_id}_"
        f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"
    )
    cached = cache_get(cache_key, 1800)
    if cached is not None:
        return cached if isinstance(cached, dict) else None
    try:
        response = _sstats_request(
            f"Odds/{game_id}", key, params={"opening": "false"}, timeout=(5, 25)
        )
        if response.status_code != 200:
            if logs is not None:
                logs.append(f"⚠️ SStats Odds/{game_id}: HTTP {response.status_code}")
            return None
        payload = response.json() or {}
        data = _sstats_unwrap(payload)
        if not isinstance(data, list):
            return None
        result = _sstats_parse_markets(data)
        result["market_names"] = sorted({
            str(
                (bet.get("marketName") or bet.get("market") or bet.get("name") or "")
                if not isinstance(bet.get("market"), dict)
                else (
                    bet.get("market", {}).get("name")
                    or bet.get("market", {}).get("title") or ""
                )
            ).strip()
            for bet, _book in _sstats_flat_bets(data)
            if str(bet.get("marketName") or bet.get("market") or bet.get("name") or "").strip()
        })
        result["bookmakers"] = sorted({
            book for _bet, book in _sstats_flat_bets(data) if book
        })
        market_count = sum(
            len(result.get(group) or {})
            for group in (
                "total_market_odds", "btts_market_odds", "stat_market_odds",
                "asian_handicap_market_odds",
            )
        ) + sum(key_name in result for key_name in ("П1", "X", "П2"))
        cache_put(cache_key, result)
        if logs is not None:
            group_counts = {
                "тоталы": (
                    f"{len(result.get('total_market_odds') or {})} "
                    f"(подтверждены {sum(bool(x) for x in (result.get('total_market_confirmed') or {}).values())})"
                ),
                "BTTS": len(result.get("btts_market_odds") or {}),
                "статистика": len(result.get("stat_market_odds") or {}),
                "форы": len(result.get("asian_handicap_market_odds") or {}),
                "1X2": sum(result.get(key_name) is not None for key_name in ("П1", "X", "П2")),
            }
            market_names = result.get("market_names") or []
            logs.append(
                f"SStats Odds/{game_id}: {len(data)} букмекеров, "
                f"{market_count} исходов/линий; " +
                ", ".join(f"{name}={count}" for name, count in group_counts.items()) + "."
            )
            if group_counts["тоталы"] == 0 and market_names:
                logs.append(
                    "SStats: названия рынков в ответе API: "
                    + " | ".join(market_names[:30])
                )
                # Market labels and JSON field names are sufficient to diagnose
                # parser/schema mismatches. Never log request URLs or API keys.
                outcome_samples = []
                for bet, _book in _sstats_flat_bets(data):
                    market_label = str(
                        bet.get("marketName") or bet.get("name") or "без названия"
                    ).strip()
                    rows = bet.get("odds") or bet.get("values") or bet.get("outcomes") or []
                    for row in rows[:3] if isinstance(rows, list) else []:
                        if not isinstance(row, dict):
                            continue
                        label = str(
                            row.get("name") or row.get("label")
                            or row.get("valueName") or "без подписи"
                        ).strip()
                        fields = ",".join(sorted(str(key) for key in row.keys()))
                        outcome_samples.append(f"{market_label} → {label} [{fields}]")
                        if len(outcome_samples) >= 12:
                            break
                    if len(outcome_samples) >= 12:
                        break
                if outcome_samples:
                    logs.append(
                        "SStats: примеры исходов и полей: "
                        + " | ".join(outcome_samples)
                    )
            if all(result.get(key) is not None for key in ("П1", "X", "П2")):
                selected_book = (result.get("bookmaker_by_pick") or {}).get("П1")
                logs.append(
                    "✅ SStats 1X2: проверенная полная линия"
                    + (f" от {selected_book}" if selected_book else " без подтверждённого букмекера")
                    + "."
                )
            else:
                logs.append(
                    "⚠️ SStats 1X2 отклонён/не найден: проверьте название рынка, "
                    "полноту линии и сумму обратных коэффициентов."
                )
        return result
    except Exception as exc:
        if logs is not None:
            logs.append(f"⚠️ SStats Odds/{game_id}: {type(exc).__name__}")
        return None
