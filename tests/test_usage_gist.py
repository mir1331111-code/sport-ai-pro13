import json
import unittest
from unittest.mock import patch

from storage import usage


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class UsageGistVisibilityTests(unittest.TestCase):
    def test_portfolio_write_refuses_public_gist_before_patch(self):
        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(
                usage.urllib.request, "urlopen",
                return_value=_Response({"public": True, "files": {}}),
            ) as open_url,
        ):
            result = usage._gist_write({"bank": 1000, "bets": []})
        self.assertFalse(result)
        self.assertEqual(open_url.call_count, 1)
        self.assertEqual(open_url.call_args.args[0].get_method(), "GET")

    def test_portfolio_write_only_patches_confirmed_non_public_gist(self):
        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(
                usage.urllib.request, "urlopen",
                side_effect=[
                    _Response({"public": False, "files": {}}),
                    _Response({"public": False, "files": {"neuro_local.json": {}}}),
                ],
            ) as open_url,
        ):
            result = usage._gist_write({"bank": 1000, "bets": []})
        self.assertTrue(result)
        self.assertEqual(open_url.call_count, 2)
        self.assertEqual(open_url.call_args.args[0].get_method(), "PATCH")

    def test_read_keeps_public_legacy_backup_available_but_logs_warning(self):
        legacy = {"data": {"bank": 1000, "bets": []}}
        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(
                usage.urllib.request, "urlopen",
                return_value=_Response({
                    "public": True,
                    "files": {"neuro_local.json": {"content": json.dumps(legacy)}},
                }),
            ),
            self.assertLogs("usage", level="ERROR") as captured,
        ):
            result = usage._gist_read()
        self.assertEqual(result, legacy)
        self.assertTrue(any("PUBLIC" in line for line in captured.output))


if __name__ == "__main__":
    unittest.main()
