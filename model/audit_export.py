"""Build a portable, credential-free forecast audit archive."""
from __future__ import annotations

import io
import json
import zipfile
from typing import Any, Mapping


def build_audit_export(payload: Mapping[str, Any]) -> bytes:
    """Serialize explicitly supplied audit data into a ZIP containing audit.json.

    The caller controls which fields enter the archive. This helper does not
    inspect environment variables, Streamlit secrets, or application settings.
    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    serialized = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        default=str,
        allow_nan=False,
    ).encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("audit.json", serialized)
    return buffer.getvalue()
