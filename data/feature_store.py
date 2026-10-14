"""Timestamp-safe external features (xG, lineups, injuries) for football matches.

Callers must supply when the information became available, not just match date.
The store refuses future information for a prediction cutoff.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable


@dataclass(frozen=True)
class FeatureSnapshot:
    fixture_id: str
    source: str
    feature_type: str
    available_at: datetime
    payload: dict[str, Any]


class FeatureStore:
    """In-memory feature snapshot registry; persistence is delegated to the caller."""

    def __init__(self) -> None:
        self._items: dict[str, list[FeatureSnapshot]] = {}

    def add(self, snapshot: FeatureSnapshot) -> None:
        if not snapshot.fixture_id.strip() or not snapshot.source.strip():
            raise ValueError("fixture_id and source are required")
        if snapshot.available_at.tzinfo is None:
            raise ValueError("available_at must be timezone-aware")
        if snapshot.feature_type not in {"xg", "lineup", "injury", "rest", "weather"}:
            raise ValueError("unsupported feature_type")
        self._items.setdefault(snapshot.fixture_id, []).append(snapshot)
        self._items[snapshot.fixture_id].sort(key=lambda x: x.available_at)

    def latest(self, fixture_id: str, feature_type: str, as_of: datetime) -> FeatureSnapshot | None:
        if as_of.tzinfo is None:
            raise ValueError("as_of must be timezone-aware")
        eligible = [
            item for item in self._items.get(fixture_id, [])
            if item.feature_type == feature_type and item.available_at <= as_of
        ]
        return eligible[-1] if eligible else None

    def features_as_of(self, fixture_id: str, as_of: datetime) -> dict[str, dict[str, Any]]:
        out = {}
        for kind in ("xg", "lineup", "injury", "rest", "weather"):
            item = self.latest(fixture_id, kind, as_of)
            if item is not None:
                out[kind] = dict(item.payload)
        return out

    def history(self, fixture_id: str) -> tuple[FeatureSnapshot, ...]:
        return tuple(self._items.get(fixture_id, ()))
