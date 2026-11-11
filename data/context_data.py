"""Rate-limited API-Football context provider.

Only API-Football fixture IDs are used here. Third-party fixture IDs (SStats,
TheSportsDB, ESPN, football-data.org) are never sent to this API.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
try:
    import fcntl
except ImportError:  # pragma: no cover - Windows development fallback
    fcntl = None
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from data.team_names import normalize_team
from storage.context_store import referee_entity_id


BASE_URL = "https://v3.football.api-sports.io"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y", "да"}


def fixture_row_key(home: str, away: str, match_date: str) -> tuple[str, str, str]:
    return (
        str(match_date or "")[:10],
        normalize_team(str(home or "")),
        normalize_team(str(away or "")),
    )


class ApiFootballError(RuntimeError):
    """Safe provider error without including API credentials."""


class ApiFootballBudget:
    """Persisted UTC-day request budget with a configurable safety reserve."""

    def __init__(
        self,
        path: str | Path,
        *,
        daily_limit: int = 100,
        reserve: int = 10,
        clock=_utc_now,
    ):
        self.path = Path(path)
        self.daily_limit = max(1, int(daily_limit))
        self.reserve = max(0, min(int(reserve), self.daily_limit - 1))
        self.clock = clock
        self._lock = threading.Lock()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _read(self) -> dict:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
        except (OSError, ValueError, TypeError):
            pass
        return {}

    def used_today(self) -> int:
        day = self.clock().date().isoformat()
        value = self._read()
        return max(0, int(value.get("date") == day and value.get("used", 0) or 0))

    def can_consume(self, count: int = 1) -> bool:
        return self.used_today() + max(0, int(count)) <= self.daily_limit - self.reserve

    def consume(self, count: int = 1) -> bool:
        count = max(0, int(count))
        day = self.clock().date().isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            # Streamlit can run several sessions/processes against this shared
            # budget file. Serialize the read-modify-write across processes too.
            with open(self.lock_path, "a+", encoding="utf-8") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    value = self._read()
                    used = int(value.get("used", 0)) if value.get("date") == day else 0
                    if used + count > self.daily_limit - self.reserve:
                        return False
                    value = {"date": day, "used": used + count, "limit": self.daily_limit}
                    temporary = self.path.with_suffix(self.path.suffix + ".tmp")
                    temporary.write_text(json.dumps(value), encoding="utf-8")
                    os.replace(temporary, self.path)
                    return True
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class ApiFootballClient:
    """Small GET-only API client with disk cache and a local daily budget."""

    def __init__(
        self,
        api_key: str,
        *,
        cache_dir: str | Path | None = None,
        daily_limit: int = 100,
        reserve: int = 10,
        session=None,
        clock=_utc_now,
    ):
        self.api_key = str(api_key or "").strip()
        self.cache_dir = Path(
            cache_dir or os.environ.get("NEURO_CACHE_DIR") or "neuro_cache"
        ) / "api_football"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self.session = session or requests.Session()
        self.budget = ApiFootballBudget(
            self.cache_dir / "daily_budget.json",
            daily_limit=daily_limit,
            reserve=reserve,
            clock=clock,
        )

    def _cache_path(self, endpoint: str, params: dict) -> Path:
        canonical = json.dumps(
            [endpoint, params], ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return self.cache_dir / (key + ".json")

    def _cache_get(self, path: Path, max_age: int):
        try:
            if self.clock().timestamp() - path.stat().st_mtime > max_age:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, TypeError):
            return None

    def has_fresh_cache(self, endpoint: str, params: dict, *, ttl: int = 900) -> bool:
        """Return whether the exact GET request can be served without quota."""
        return self._cache_get(self._cache_path(endpoint, params), ttl) is not None

    def _cache_put(self, path: Path, payload: dict) -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def get(self, endpoint: str, params: dict, *, ttl: int = 900) -> dict:
        if not self.api_key:
            raise ApiFootballError("API_FOOTBALL_KEY is not configured")
        path = self._cache_path(endpoint, params)
        cached = self._cache_get(path, ttl)
        if cached is not None:
            return cached
        if not self.budget.consume():
            raise ApiFootballError("daily API-Football request budget reached")
        try:
            response = self.session.get(
                f"{BASE_URL}/{str(endpoint).lstrip('/')}",
                params=params,
                headers={"x-apisports-key": self.api_key},
                timeout=(5, 20),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            raise ApiFootballError(f"API-Football request failed: {type(exc).__name__}") from None
        if not isinstance(payload, dict):
            raise ApiFootballError("API-Football returned a non-object response")
        errors = payload.get("errors")
        if errors:
            # Provider messages may echo request data. Avoid forwarding them to UI/logs.
            raise ApiFootballError("API-Football reported a provider error")
        self._cache_put(path, payload)
        return payload


class ApiFootballContextProvider:
    """Enrich upcoming fixtures with lineups, injuries and referee metadata.

    Fixture list is fetched in one date-range request. Injuries are batched by
    up to 20 API fixture IDs. Lineups are queried only close to kickoff and
    under a per-scan cap, avoiding wasteful calls days before an XI is released.
    """

    def __init__(
        self,
        client: ApiFootballClient,
        *,
        timezone_name: str = "Europe/London",
        max_lineup_calls: int = 12,
        lineup_window_hours: int = 6,
    ):
        self.client = client
        self.timezone_name = timezone_name
        self.max_lineup_calls = max(0, int(max_lineup_calls))
        self.lineup_window = timedelta(hours=max(0, int(lineup_window_hours)))

    def _api_date(self, fixture: dict) -> str:
        value = _parse_datetime((fixture.get("fixture") or {}).get("date"))
        if value is None:
            return ""
        try:
            return value.astimezone(ZoneInfo(self.timezone_name)).date().isoformat()
        except Exception:
            return value.date().isoformat()

    def _row_date(self, row: dict) -> str:
        # An absolute kickoff is authoritative. Convert it into the exact same
        # app timezone used by the API /fixtures request before comparing dates.
        raw_utc = row.get("kickoff_utc")
        if raw_utc:
            kickoff = _parse_datetime(raw_utc)
            if kickoff is not None:
                try:
                    return kickoff.astimezone(ZoneInfo(self.timezone_name)).date().isoformat()
                except Exception:
                    return kickoff.date().isoformat()
        raw = str(row.get("date_iso") or row.get("Date") or row.get("date") or "")[:10]
        if len(raw) == 10 and raw[4:5] == "-":
            return raw
        # Avoid guessing ambiguous non-ISO date formats here. The scanner owns
        # the canonical parsed date, and CSV loaders normalize to ISO dates.
        return ""

    def _kickoff(self, fixture: dict) -> datetime | None:
        return _parse_datetime((fixture.get("fixture") or {}).get("date"))

    @staticmethod
    def _api_fixture_id(fixture: dict) -> int | None:
        try:
            value = int((fixture.get("fixture") or {}).get("id"))
            return value if value > 0 else None
        except (TypeError, ValueError):
            return None

    def _lineups_map(self, payload: dict) -> dict:
        mapped = {}
        for item in payload.get("response") or []:
            if not isinstance(item, dict):
                continue
            team = item.get("team") or {}
            team_id = str(team.get("id") or "")
            start_xi = item.get("startXI") or item.get("start_xi") or []
            players = []
            for entry in start_xi:
                player = (entry or {}).get("player") if isinstance(entry, dict) else None
                if isinstance(player, dict):
                    players.append({
                        "id": player.get("id"),
                        "name": str(player.get("name") or "").strip(),
                        "number": player.get("number"),
                        "pos": player.get("pos"),
                    })
            if team_id:
                mapped[team_id] = {
                    "team_name": str(team.get("name") or ""),
                    "starting_xi": players,
                    "confirmed": len(players) >= 11,
                }
        return mapped

    def enrich(
        self,
        rows: list[dict],
        *,
        date_from: str,
        date_to: str,
        now: datetime | None = None,
    ) -> tuple[dict[tuple[str, str, str], dict], list[str]]:
        logs: list[str] = []
        if not self.client.api_key:
            return {}, ["API-Football: API key отсутствует; контекст недоступен."]
        now = (now or self.client.clock()).astimezone(timezone.utc)
        try:
            fixtures_payload = self.client.get(
                "fixtures",
                {"from": date_from, "to": date_to, "timezone": self.timezone_name},
                ttl=1800,
            )
        except ApiFootballError as exc:
            return {}, [f"API-Football: список fixtures недоступен ({exc})."]

        api_fixtures = [
            item for item in (fixtures_payload.get("response") or [])
            if isinstance(item, dict)
        ]
        by_key: dict[tuple[str, str, str], dict] = {}
        for fixture in api_fixtures:
            teams = fixture.get("teams") or {}
            home = str((teams.get("home") or {}).get("name") or "")
            away = str((teams.get("away") or {}).get("name") or "")
            match_date = self._api_date(fixture)
            key = fixture_row_key(home, away, match_date)
            if all(key):
                by_key[key] = fixture

        matches = []
        result = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            home = row.get("HomeTeam") or row.get("home_team") or ""
            away = row.get("AwayTeam") or row.get("away_team") or ""
            key = fixture_row_key(home, away, self._row_date(row))
            fixture = by_key.get(key)
            if fixture is None:
                continue
            fixture_id = self._api_fixture_id(fixture)
            if fixture_id is None:
                continue
            row_key = fixture_row_key(home, away, self._row_date(row))
            kickoff = self._kickoff(fixture)
            referee = str((fixture.get("fixture") or {}).get("referee") or "").strip()
            context = {
                "provider": "api-football",
                "api_fixture_id": fixture_id,
                "kickoff_at": kickoff.isoformat() if kickoff is not None else None,
                "referee": {
                    "id": referee_entity_id(referee) if referee else None,
                    "name": referee or None,
                    "is_confirmed": bool(referee),
                    "timestamp": now.isoformat(),
                    "source_reliability": None,
                },
                "lineup": {
                    "home": {"starting_xi": [], "confirmed": False},
                    "away": {"starting_xi": [], "confirmed": False},
                    "is_confirmed": False,
                    "timestamp": None,
                    "source_reliability": None,
                },
                "injuries": [],
                "injuries_is_confirmed": False,
                "injuries_timestamp": None,
                "injuries_source_reliability": None,
                "timestamp": now.isoformat(),
            }
            row["api_football_fixture_id"] = fixture_id
            row["api_football_referee"] = referee or None
            result[row_key] = context
            matches.append((fixture_id, fixture, context, kickoff))

        # Injuries endpoint accepts up to 20 fixture IDs in one call.
        injury_matches = sorted(
            matches,
            key=lambda item: item[3] or datetime.max.replace(tzinfo=timezone.utc),
        )
        injury_batches = 0
        for batch_start in range(0, len(injury_matches), 20):
            batch = injury_matches[batch_start:batch_start + 20]
            ids = "-".join(str(item[0]) for item in batch)
            try:
                payload = self.client.get("injuries", {"ids": ids}, ttl=4 * 60 * 60)
                injuries_by_fixture: dict[int, list] = {
                    item[0]: [] for item in batch
                }
                for item in payload.get("response") or []:
                    if not isinstance(item, dict):
                        continue
                    injury_fixture = item.get("fixture") or {}
                    try:
                        fixture_id = int(injury_fixture.get("id"))
                    except (TypeError, ValueError):
                        continue
                    if fixture_id in injuries_by_fixture:
                        player = item.get("player") or {}
                        team = item.get("team") or {}
                        injuries_by_fixture[fixture_id].append({
                            "player_id": player.get("id"),
                            "player_name": str(player.get("name") or ""),
                            "team_id": team.get("id"),
                            "team_name": str(team.get("name") or ""),
                            "type": str(player.get("type") or ""),
                            "reason": str(player.get("reason") or ""),
                        })
                for fixture_id, _, context, _ in batch:
                    context["injuries"] = injuries_by_fixture.get(fixture_id, [])
                    # Successful response means the provider was queried, not
                    # that injury coverage is guaranteed for that competition.
                    context["injuries_is_confirmed"] = True
                    context["injuries_timestamp"] = now.isoformat()
                injury_batches += 1
            except ApiFootballError as exc:
                logs.append(f"API-Football: травмы/дисквалификации недоступны ({exc}).")
                break

        # The lineup endpoint is only worth calling near kickoff; query the
        # closest matches first and never burn the full quota on a wide horizon.
        lineup_candidates = [
            item for item in matches
            if item[3] is not None
            and timedelta(0) <= item[3] - now <= self.lineup_window
        ]
        lineup_candidates.sort(key=lambda item: item[3])
        lineup_calls = 0
        for fixture_id, _, context, _ in lineup_candidates:
            lineup_params = {"fixture": fixture_id}
            cache_hit = self.client.has_fresh_cache(
                "fixtures/lineups", lineup_params, ttl=10 * 60
            )
            if not cache_hit and lineup_calls >= self.max_lineup_calls:
                continue
            try:
                if not cache_hit:
                    # Count attempts that consume quota, including provider errors.
                    lineup_calls += 1
                payload = self.client.get(
                    "fixtures/lineups", lineup_params, ttl=10 * 60
                )
                lineup_map = self._lineups_map(payload)
                teams = (payload.get("response") or [])
                expected_team_ids = {
                    str((item.get("team") or {}).get("id") or "")
                    for item in teams if isinstance(item, dict)
                }
                # The fixture's team IDs are authoritative for home/away identity.
                fixture_obj = next(
                    (match[1] for match in matches if match[0] == fixture_id), {}
                )
                fixture_teams = fixture_obj.get("teams") or {}
                home_id = str((fixture_teams.get("home") or {}).get("id") or "")
                away_id = str((fixture_teams.get("away") or {}).get("id") or "")
                home_lineup = lineup_map.get(home_id, {"starting_xi": [], "confirmed": False})
                away_lineup = lineup_map.get(away_id, {"starting_xi": [], "confirmed": False})
                context["lineup"] = {
                    "home": home_lineup,
                    "away": away_lineup,
                    "is_confirmed": bool(home_lineup["confirmed"] and away_lineup["confirmed"]),
                    "timestamp": now.isoformat(),
                    "source_reliability": None,
                }
                context["lineup"]["expected_team_ids"] = [home_id, away_id]
                context["lineup"]["returned_team_ids"] = sorted(expected_team_ids)
            except ApiFootballError as exc:
                logs.append(f"API-Football: стартовый состав fixture={fixture_id} недоступен ({exc}).")

        logs.append(
            f"API-Football: fixtures={len(api_fixtures)}, сопоставлено={len(matches)}, "
            f"injuries batches={injury_batches}, "
            f"lineups calls={lineup_calls}, бюджет использован={self.client.budget.used_today()}."
        )
        return result, logs
