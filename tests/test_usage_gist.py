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


class _RawResponse(_Response):
    def __init__(self, content: str):
        self.content = content

    def read(self):
        return self.content.encode("utf-8")


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

    def test_read_fetches_full_raw_file_when_gist_content_is_truncated(self):
        expected = {"data": {"bank": 2500, "bets": [], "cards": []}}
        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(
                usage.urllib.request, "urlopen",
                side_effect=[
                    _Response({
                        "public": False,
                        "files": {
                            "neuro_local.json": {
                                "content": "{\"data\":",
                                "truncated": True,
                                "raw_url": "https://gist.githubusercontent.com/user/id/raw/neuro_local.json",
                            }
                        },
                    }),
                    _RawResponse(json.dumps(expected)),
                ],
            ) as open_url,
        ):
            result = usage._gist_read()
        self.assertEqual(result, expected)
        self.assertEqual(open_url.call_count, 2)
        self.assertIn("gist.githubusercontent.com", open_url.call_args.args[0].full_url)

    def test_gist_backup_never_serializes_provider_credentials(self):
        secrets = {
            "fdorg_token": "FDORG_SENTINEL",
            "odds_api_key": "ODDS_SENTINEL",
            "therundown_api_key": "RUNDOWN_SENTINEL",
            "rundown_key": "RUNDOWN_ALIAS_SENTINEL",
            "highlightly_api_key": "HIGHLIGHTLY_SENTINEL",
            "sstats_api_key": "SSTATS_SENTINEL",
            "footballapi_key": "FOOTBALL_SENTINEL",
            "llm_api_key": "LLM_SENTINEL",
            "xai_api_key": "XAI_SENTINEL",
            "grok_api_key": "GROK_SENTINEL",
            "groq_api_key": "GROQ_SENTINEL",
        }
        captured = {}

        class _PatchResponse(_Response):
            pass

        def fake_urlopen(request, timeout):
            if request.get_method() == "GET":
                return _Response({"public": False, "files": {}})
            captured["payload"] = request.data.decode("utf-8")
            return _PatchResponse({"public": False, "files": {"neuro_local.json": {}}})

        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(usage.urllib.request, "urlopen", side_effect=fake_urlopen),
        ):
            result = usage._gist_write({
                "bank": 1000, "bets": [],
                "meta": {**secrets, "initial_bank": 1000},
            })

        self.assertTrue(result)
        request_payload = json.loads(captured["payload"])
        saved_json = request_payload["files"]["neuro_local.json"]["content"]
        saved_data = json.loads(saved_json)
        self.assertEqual(saved_data["meta"]["initial_bank"], 1000)
        for secret_name in secrets:
            self.assertNotIn(secret_name, saved_data["meta"])
        for sentinel in secrets.values():
            self.assertNotIn(sentinel, captured["payload"])

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
