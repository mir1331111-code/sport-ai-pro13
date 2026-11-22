"""Safe append-only continuation of chronological model training.

A stable key represents model schema/configuration. The row manifest verifies
the full previously trained prefix; backfills, edits and deletions force rebuild.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from config import APP_VERSION, DATA_VERSION
from model.training import prepare_completed_training_rows, train_engine_chronologically

_STABLE_SCHEMA = "neuro-engine-stable-v2"


class IncrementalHistoryChanged(RuntimeError):
    """Current source history cannot safely extend the cached model."""


def _manifest_entry(item: dict) -> tuple[str, str]:
    stamp = item["match_date"].date().isoformat()
    payload = {
        "date": stamp,
        "league": str(item.get("league") or ""),
        "home": str(item.get("home") or ""),
        "away": str(item.get("away") or ""),
        "home_goals": int(item.get("home_goals")),
        "away_goals": int(item.get("away_goals")),
        "row": item.get("row") if isinstance(item.get("row"), dict) else {},
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str,
    ).encode("utf-8")
    return stamp, hashlib.sha256(encoded).hexdigest()


def training_manifest(rows: list[dict]) -> list[tuple[str, str]]:
    return sorted(_manifest_entry(item) for item in rows)


def record_training_manifest(engine: Any, rows: list[dict]) -> None:
    manifest = training_manifest(rows)
    engine._incremental_manifest = manifest
    engine._incremental_cutoff = max((entry[0] for entry in manifest), default=None)


def stable_engine_fingerprint(
    *, matrix_n: int, xg_enabled: bool, divisions, history_years=None,
) -> str:
    payload = {
        "schema": _STABLE_SCHEMA,
        "app_version": str(APP_VERSION),
        "data_version": int(DATA_VERSION),
        "matrix_n": int(matrix_n),
        "xg_enabled": bool(xg_enabled),
        "divisions": sorted(str(value) for value in (divisions or ())),
        "history_years": sorted(int(value) for value in (history_years or ())),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "neuro-engine-stable-" + hashlib.sha256(encoded).hexdigest()[:24]


def bring_engine_up_to_date(engine: Any, history: dict, train_divs, logs: list) -> int:
    prepared, skipped = prepare_completed_training_rows(history, train_divs)
    previous = list(getattr(engine, "_incremental_manifest", None) or [])
    if not previous:
        raise IncrementalHistoryChanged("у модели нет манифеста обученной истории")

    current = training_manifest(prepared)
    previous_counts, current_counts = Counter(previous), Counter(current)
    missing = previous_counts - current_counts
    if missing:
        raise IncrementalHistoryChanged(
            f"изменены или удалены ранее обученные строки: {sum(missing.values())}"
        )

    cutoff = str(
        getattr(engine, "_incremental_cutoff", None)
        or max(entry[0] for entry in previous)
    )
    remaining = previous_counts.copy()
    delta = []
    for item in prepared:
        signature = _manifest_entry(item)
        if remaining[signature] > 0:
            remaining[signature] -= 1
        elif signature[0] <= cutoff:
            raise IncrementalHistoryChanged(
                f"новый результат на уже обученной дате {signature[0]}"
            )
        else:
            delta.append(item)

    if any(remaining.values()):
        raise IncrementalHistoryChanged("обученную историю не удалось сверить полностью")
    if not delta:
        engine._incremental_manifest = current
        engine._incremental_cutoff = cutoff
        return 0

    before = int(getattr(engine, "trained_n", 0) or 0)
    train_engine_chronologically(
        engine, engine.match_stats, delta,
        processed_offset=before, total_count=before + len(delta),
    )
    record_training_manifest(engine, prepared)
    logs.append(
        f"🔧 Инкрементальное обучение: +{len(delta)} матчей после {cutoff}; "
        f"всего обучено {engine.trained_n}; пропущено строк источника="
        f"{sum(int(value or 0) for value in skipped.values())}."
    )
    return len(delta)
