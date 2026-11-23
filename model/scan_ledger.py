"""Persistent cross-module scan ledger.

A scan run ties fixtures, immutable forecast keys and per-market gate decisions
together. The forecast ledgers remain the source of probability/result metrics;
this ledger stores only run provenance and references to those immutable rows.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from config import DB_FILE

SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_runs (
    scan_run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    timezone_name TEXT,
    horizon_days INTEGER NOT NULL CHECK(horizon_days BETWEEN 1 AND 14),
    date_from TEXT,
    date_to_inclusive TEXT,
    source_rows INTEGER NOT NULL DEFAULT 0,
    unique_fixtures INTEGER NOT NULL DEFAULT 0,
    analysed_matches INTEGER NOT NULL DEFAULT 0,
    forecast_cards INTEGER NOT NULL DEFAULT 0,
    source_counts_json TEXT NOT NULL DEFAULT '{}',
    funnel_json TEXT NOT NULL DEFAULT '{}',
    report_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_scan_runs_completed ON scan_runs(completed_at);
CREATE TABLE IF NOT EXISTS scan_run_fixtures (
    scan_run_id TEXT NOT NULL REFERENCES scan_runs(scan_run_id) ON DELETE CASCADE,
    fixture_id TEXT NOT NULL,
    provider_fixture_id TEXT,
    kickoff_at TEXT,
    home_team TEXT,
    away_team TEXT,
    league TEXT,
    model_support_status TEXT,
    has_bookmaker_odds INTEGER NOT NULL DEFAULT 0,
    quality_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (scan_run_id, fixture_id)
);
CREATE INDEX IF NOT EXISTS idx_scan_fixtures_fixture ON scan_run_fixtures(fixture_id);
CREATE TABLE IF NOT EXISTS scan_run_forecasts (
    scan_run_id TEXT NOT NULL REFERENCES scan_runs(scan_run_id) ON DELETE CASCADE,
    fixture_id TEXT NOT NULL,
    forecast_key TEXT NOT NULL,
    forecast_kind TEXT NOT NULL,
    PRIMARY KEY (scan_run_id, forecast_key, forecast_kind)
);
CREATE INDEX IF NOT EXISTS idx_scan_forecasts_key ON scan_run_forecasts(forecast_key);
CREATE TABLE IF NOT EXISTS scan_run_decisions (
    scan_run_id TEXT NOT NULL REFERENCES scan_runs(scan_run_id) ON DELETE CASCADE,
    fixture_id TEXT NOT NULL,
    market_family TEXT NOT NULL,
    market TEXT,
    selection TEXT,
    status TEXT NOT NULL,
    probability REAL,
    odds REAL,
    expected_value REAL,
    edge REAL,
    gates_json TEXT NOT NULL,
    PRIMARY KEY (scan_run_id, fixture_id, market_family)
);
CREATE INDEX IF NOT EXISTS idx_scan_decisions_status ON scan_run_decisions(scan_run_id, status);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


class ScanLedger:
    """SQLite run index connecting scanner cards to forecasts and approval gates."""

    FAMILIES = ("winner", "goals", "btts", "corners", "cards")
    MARKET_CODES = {
        "winner": "1X2", "goals": "OU", "btts": "BTTS",
        "corners": "STAT", "cards": "STAT",
    }

    def __init__(self, db_path: str | Path = DB_FILE):
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            columns = {
                str(row[1]) for row in conn.execute(
                    "PRAGMA table_info(scan_run_fixtures)"
                ).fetchall()
            }
            if "quality_json" not in columns:
                conn.execute(
                    "ALTER TABLE scan_run_fixtures ADD COLUMN quality_json TEXT NOT NULL DEFAULT '{}'"
                )

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def record_run(
        self,
        scan_run_id: str,
        *,
        metadata: Mapping[str, Any],
        cards: list[dict],
        report: list[str] | None = None,
        status: str = "completed",
    ) -> dict[str, int]:
        run_id = str(scan_run_id or "").strip()
        if not run_id:
            raise ValueError("scan_run_id is required")
        if not isinstance(metadata, Mapping):
            raise ValueError("scan metadata must be a mapping")
        horizon = int(metadata.get("horizon_days") or 0)
        if not 1 <= horizon <= 14:
            raise ValueError("horizon_days must be between 1 and 14")

        started = str(metadata.get("scan_started_at") or _now())
        completed = str(metadata.get("scan_completed_at") or _now())
        source_counts = metadata.get("source_counts")
        source_counts = source_counts if isinstance(source_counts, Mapping) else {}
        funnel = metadata.get("funnel")
        funnel = funnel if isinstance(funnel, Mapping) else {}
        logs = report if isinstance(report, list) else []
        fixture_rows = []
        forecast_rows = []
        decision_rows = []

        # Import locally to keep this storage primitive independent of scanner UI.
        from betting.gate_report import classify_card
        from betting.market_ranking import market_forecast_summary
        from model.analytics_quality import card_quality

        api_football_configured = bool(metadata.get("api_football_configured"))

        for card in cards or []:
            if not isinstance(card, dict):
                continue
            fixture_id = str(
                card.get("forecast_fixture_id") or card.get("fixture_id") or ""
            ).strip()
            if not fixture_id:
                continue
            match = str(card.get("match") or "")
            if " vs " in match:
                home, away = match.split(" vs ", 1)
            else:
                parts = str(card.get("match_ru") or "").split(" — ", 1)
                home, away = (parts[0], parts[1]) if len(parts) == 2 else ("", "")
            fixture_rows.append((
                run_id, fixture_id,
                str(card.get("provider_fixture_id") or "") or None,
                str(card.get("kickoff_utc") or "") or None,
                home.strip(), away.strip(),
                str(card.get("league") or card.get("div") or "") or None,
                str(card.get("model_support_status") or "") or None,
                int(bool(card.get("market_odds")) or bool(card.get("total_market_odds"))),
                _json(card_quality(
                    card, api_football_configured=api_football_configured,
                )),
            ))

            keys = [
                (card.get("main_forecast_key"), "main_1x2"),
                (card.get("ai_forecast_key"), "independent_ai_1x2"),
            ]
            market_keys = dict(card.get("market_forecast_keys") or {})
            keys.extend(
                (key, "market_" + str(family))
                for family, key in market_keys.items()
            )
            legacy_market_key = card.get("market_forecast_key")
            if legacy_market_key and legacy_market_key not in market_keys.values():
                keys.append((legacy_market_key, "market_legacy"))
            for key, kind in keys:
                if key:
                    forecast_rows.append((run_id, fixture_id, str(key), kind))

            recommendations = card.get("best_by_market")
            recommendations = recommendations if isinstance(recommendations, Mapping) else {}
            for family in self.FAMILIES:
                try:
                    status_value, gates = classify_card(card, family)
                    summary = market_forecast_summary(card, family)
                except Exception as exc:
                    status_value, gates, summary = "provisional", [], None
                    gate_payload = [{
                        "name": "Диагностика", "passed": False,
                        "detail": f"ошибка расчёта гейтов: {type(exc).__name__}",
                    }]
                else:
                    gate_payload = [
                        {"name": gate.name, "passed": bool(gate.passed), "detail": gate.detail}
                        for gate in gates
                    ]
                recommendation = recommendations.get(family)
                recommendation = recommendation if isinstance(recommendation, Mapping) else {}
                best = recommendation.get("best")
                verdict = recommendation.get("verdict")
                if not isinstance(verdict, Mapping):
                    verdict = {}
                if isinstance(best, (tuple, list)) and len(best) >= 5:
                    market = str(best[0] or self.MARKET_CODES[family])
                    selection = str(best[1] or "")
                    try:
                        odds = float(best[2])
                    except (TypeError, ValueError):
                        odds = None
                    try:
                        probability = float(best[4])
                    except (TypeError, ValueError):
                        probability = None
                else:
                    market = self.MARKET_CODES[family]
                    selection = str((summary or {}).get("pick") or "")
                    odds = None
                    probability = (
                        float(summary["probability"])
                        if summary and summary.get("probability") is not None else None
                    )
                try:
                    expected_value = float(verdict.get("ev")) if verdict.get("ev") is not None else None
                except (TypeError, ValueError):
                    expected_value = None
                try:
                    edge = float(verdict.get("edge")) if verdict.get("edge") is not None else None
                except (TypeError, ValueError):
                    edge = None
                decision_rows.append((
                    run_id, fixture_id, family, market, selection or None,
                    status_value, probability, odds, expected_value, edge,
                    _json(gate_payload),
                ))

        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                """INSERT INTO scan_runs (
                    scan_run_id,status,started_at,completed_at,timezone_name,
                    horizon_days,date_from,date_to_inclusive,source_rows,
                    unique_fixtures,analysed_matches,forecast_cards,
                    source_counts_json,funnel_json,report_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(scan_run_id) DO UPDATE SET
                    status=excluded.status, completed_at=excluded.completed_at,
                    source_rows=excluded.source_rows, unique_fixtures=excluded.unique_fixtures,
                    analysed_matches=excluded.analysed_matches,
                    forecast_cards=excluded.forecast_cards,
                    source_counts_json=excluded.source_counts_json,
                    funnel_json=excluded.funnel_json, report_json=excluded.report_json""",
                (
                    run_id, str(status), started, completed,
                    str(metadata.get("timezone_name") or "") or None,
                    horizon, metadata.get("date_from"), metadata.get("date_to_inclusive"),
                    int(metadata.get("source_rows") or 0),
                    int(metadata.get("unique_fixtures") or 0),
                    int(metadata.get("analysed_matches") or 0),
                    int(metadata.get("forecast_cards") or 0),
                    _json(dict(source_counts)), _json(dict(funnel)), _json(logs[-200:]),
                ),
            )
            for table in ("scan_run_fixtures", "scan_run_forecasts", "scan_run_decisions"):
                conn.execute(f"DELETE FROM {table} WHERE scan_run_id=?", (run_id,))
            conn.executemany(
                """INSERT INTO scan_run_fixtures (
                    scan_run_id,fixture_id,provider_fixture_id,kickoff_at,
                    home_team,away_team,league,model_support_status,has_bookmaker_odds,
                    quality_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                fixture_rows,
            )
            conn.executemany(
                """INSERT INTO scan_run_forecasts
                   (scan_run_id,fixture_id,forecast_key,forecast_kind)
                   VALUES (?,?,?,?)""",
                forecast_rows,
            )
            conn.executemany(
                """INSERT INTO scan_run_decisions (
                    scan_run_id,fixture_id,market_family,market,selection,status,
                    probability,odds,expected_value,edge,gates_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                decision_rows,
            )
            conn.commit()

        return {
            "fixtures": len(fixture_rows),
            "forecasts": len(forecast_rows),
            "decisions": len(decision_rows),
        }

    @staticmethod
    def _decode_run(row: Mapping[str, Any]) -> dict:
        item = dict(row)
        for source, target in (
            ("source_counts_json", "source_counts"),
            ("funnel_json", "funnel"),
            ("report_json", "report"),
        ):
            try:
                item[target] = json.loads(item.pop(source) or "{}")
            except (TypeError, ValueError):
                item[target] = {} if target != "report" else []
        return item

    def runs(self, limit: int = 20) -> list[dict]:
        safe_limit = max(1, min(200, int(limit)))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM scan_runs ORDER BY completed_at DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        return [self._decode_run(row) for row in rows]

    def latest_run(self) -> dict | None:
        rows = self.runs(1)
        return rows[0] if rows else None

    def decisions(self, scan_run_id: str | None = None, limit: int = 500) -> list[dict]:
        query = "SELECT * FROM scan_run_decisions"
        params: list[Any] = []
        if scan_run_id:
            query += " WHERE scan_run_id=?"
            params.append(str(scan_run_id))
        query += " ORDER BY scan_run_id DESC, fixture_id, market_family LIMIT ?"
        params.append(max(1, min(5000, int(limit))))
        with self._connect() as conn:
            rows = [dict(row) for row in conn.execute(query, params).fetchall()]
        for row in rows:
            try:
                row["gates"] = json.loads(row.pop("gates_json") or "[]")
            except (TypeError, ValueError):
                row["gates"] = []
        return rows

    def quality_rows(self, scan_run_id: str | None = None, limit: int = 500) -> list[dict]:
        """Load persisted data-quality snapshots for one scan, newest run by default."""
        run_id = str(scan_run_id or "").strip()
        if not run_id:
            latest = self.latest_run()
            if not latest:
                return []
            run_id = str(latest.get("scan_run_id") or "")
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT fixture_id,provider_fixture_id,kickoff_at,home_team,away_team,
                          league,model_support_status,has_bookmaker_odds,quality_json
                   FROM scan_run_fixtures WHERE scan_run_id=?
                   ORDER BY kickoff_at,fixture_id LIMIT ?""",
                (run_id, max(1, min(5000, int(limit)))),
            ).fetchall()
        result = []
        for raw in rows:
            item = dict(raw)
            try:
                item["quality"] = json.loads(item.pop("quality_json") or "{}")
            except (TypeError, ValueError):
                item["quality"] = {}
            result.append(item)
        return result

    def forecast_links(self, scan_run_id: str) -> list[dict]:
        with self._connect() as conn:
            return [
                dict(row) for row in conn.execute(
                    """SELECT fixture_id,forecast_key,forecast_kind
                       FROM scan_run_forecasts WHERE scan_run_id=?
                       ORDER BY fixture_id,forecast_kind""",
                    (str(scan_run_id),),
                )
            ]
