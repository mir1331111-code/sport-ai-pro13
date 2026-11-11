"""Validated custom historical match CSV import for women's leagues and national teams."""
from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from typing import Any


ALIASES = {
    "date": ("date", "match_date", "date_iso"),
    "home": ("hometeam", "home_team", "home"),
    "away": ("awayteam", "away_team", "away"),
    "home_goals": ("fthg", "home_goals", "score_home"),
    "away_goals": ("ftag", "away_goals", "score_away"),
    "division": ("div", "division", "league_code", "competition_code"),
}
OPTIONAL_COLUMNS = (
    "league", "time", "kickoff_utc", "HTHG", "HTAG", "HTHC", "HTAC",
    "HC", "AC", "HY", "AY", "HR", "AR", "Referee",
    "HxG", "AxG", "xg_source",
)


def _date(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    for fmt in (
        "%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y",
        "%d-%m-%Y", "%d-%m-%y", "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(raw[:10], fmt).date().isoformat()
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed.date().isoformat()
    except ValueError:
        return None


def _score(value: Any, column: str) -> int:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{column} must be an integer score") from None
    if number < 0 or number > 20 or not number.is_integer():
        raise ValueError(f"{column} must be an integer from 0 to 20")
    return int(number)


def _first(row: dict, names: tuple[str, ...]) -> str:
    for key in names:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def parse_historical_matches_csv(source: str | bytes) -> dict:
    """Parse dated final scores without synthesizing missing goals or statistics.

    The Div/competition_code column is mandatory so each competition trains in
    its own model namespace (e.g. W_E0, W_SP1, NT_EURO, NT_WC).
    """
    if isinstance(source, bytes):
        source = source.decode("utf-8-sig", errors="strict")
    reader = csv.DictReader(io.StringIO(str(source or "").lstrip("\ufeff")))
    headers = {
        str(name or "").strip().casefold(): str(name or "").strip()
        for name in (reader.fieldnames or [])
    }
    missing = [
        key for key in ("date", "home", "away", "home_goals", "away_goals", "division")
        if not any(alias in headers for alias in ALIASES[key])
    ]
    if missing:
        raise ValueError("missing required columns: " + ", ".join(missing))

    rows, rejected = [], []
    seen = set()
    for line, original in enumerate(reader, start=2):
        source_row = {
            str(key or "").strip().casefold(): value
            for key, value in (original or {}).items()
        }
        if not any(str(value or "").strip() for value in source_row.values()):
            continue
        try:
            date_value = _date(_first(source_row, ALIASES["date"]))
            home = _first(source_row, ALIASES["home"])
            away = _first(source_row, ALIASES["away"])
            division = _first(source_row, ALIASES["division"]).strip()
            if not date_value or not home or not away:
                raise ValueError("date and both teams are required")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,24}", division):
                raise ValueError("Div code must contain 1-24 letters, digits, _ or -")
            home_goals = _score(_first(source_row, ALIASES["home_goals"]), "FTHG")
            away_goals = _score(_first(source_row, ALIASES["away_goals"]), "FTAG")
            row = {
                "Date": date_value,
                "HomeTeam": home,
                "AwayTeam": away,
                "FTHG": home_goals,
                "FTAG": away_goals,
                "Div": division,
                "League": _first(source_row, ("league",)) or division,
                "history_source": "custom_csv",
            }
            for column in OPTIONAL_COLUMNS:
                actual = headers.get(column.casefold())
                if actual is not None and str(source_row.get(actual.casefold()) or "").strip():
                    row[column] = str(source_row[actual.casefold()]).strip()
            key = (
                division.casefold(), date_value,
                home.casefold(), away.casefold(),
            )
            if key in seen:
                rejected.append({"line": line, "reason": "duplicate fixture"})
                continue
            seen.add(key)
            rows.append(row)
        except ValueError as exc:
            rejected.append({"line": line, "reason": str(exc)})

    rows.sort(key=lambda item: (
        item["Date"], item["Div"], item["HomeTeam"].casefold(), item["AwayTeam"].casefold()
    ))
    return {
        "rows": rows,
        "rejected": rejected,
        "accepted": len(rows),
        "divisions": sorted({row["Div"] for row in rows}),
        "schema_version": 1,
    }
