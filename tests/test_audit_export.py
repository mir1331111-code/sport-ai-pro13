import io
import json
import unittest
import zipfile

from model.audit_export import build_audit_export


class AuditExportTests(unittest.TestCase):
    def test_builds_zip_with_round_trippable_audit_json(self):
        payload = {
            "schema_version": 1,
            "summary": {"main_forecasts": 2, "main_settled": 1},
            "forecasts": [
                {"fixture_id": "2026-11-06:home-away", "xg_used": True},
                {"fixture_id": "2026-11-07:other-fixture", "xg_used": False},
            ],
            "main_reliability_bins": {"rows": []},
        }
        data = build_audit_export(payload)
        with zipfile.ZipFile(io.BytesIO(data), "r") as archive:
            self.assertEqual(archive.namelist(), ["audit.json"])
            decoded = json.loads(archive.read("audit.json"))
        self.assertEqual(decoded, payload)

    def test_export_rejects_non_finite_json_numbers(self):
        with self.assertRaises(ValueError):
            build_audit_export({"metric": float("nan")})

    def test_export_does_not_read_environment_or_add_secrets(self):
        payload = {"forecasts": [], "summary": {"main_forecasts": 0}}
        with zipfile.ZipFile(io.BytesIO(build_audit_export(payload))) as archive:
            decoded = json.loads(archive.read("audit.json"))
        self.assertEqual(set(decoded), {"forecasts", "summary"})
        self.assertNotIn("api_key", decoded)


if __name__ == "__main__":
    unittest.main()
