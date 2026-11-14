import json
import unittest
from cryptography.fernet import Fernet
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
        cipher = Fernet(Fernet.generate_key())
        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(usage, "_portfolio_fernet", return_value=cipher),
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
        cipher = Fernet(Fernet.generate_key())
        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(usage, "_portfolio_fernet", return_value=cipher),
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

        cipher = Fernet(Fernet.generate_key())
        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(usage, "_portfolio_fernet", return_value=cipher),
            patch.object(usage.urllib.request, "urlopen", side_effect=fake_urlopen),
        ):
            result = usage._gist_write({
                "bank": 1000, "bets": [],
                "meta": {**secrets, "initial_bank": 1000},
            })

        self.assertTrue(result)
        request_payload = json.loads(captured["payload"])
        envelope = json.loads(request_payload["files"]["neuro_local.json"]["content"])
        self.assertEqual(envelope["format"], "neuro-local-fernet-v1")
        plaintext = cipher.decrypt(envelope["ciphertext"].encode("ascii")).decode("utf-8")
        saved_data = json.loads(plaintext)
        self.assertEqual(saved_data["meta"]["initial_bank"], 1000)
        for secret_name in secrets:
            self.assertNotIn(secret_name, saved_data["meta"])
        for sentinel in secrets.values():
            self.assertNotIn(sentinel, captured["payload"])

    def test_encrypted_portfolio_gist_round_trips_with_shared_backup_key(self):
        cipher = Fernet(Fernet.generate_key())
        expected = {"data": {"bank": 3600, "bets": [{"pick": "П1"}]}}
        envelope = {
            "format": "neuro-local-fernet-v1",
            "ciphertext": cipher.encrypt(
                json.dumps(expected, ensure_ascii=False).encode("utf-8")
            ).decode("ascii"),
        }
        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(usage, "_portfolio_fernet", return_value=cipher),
            patch.object(
                usage.urllib.request, "urlopen",
                return_value=_Response({
                    "public": False,
                    "files": {"neuro_local.json": {"content": json.dumps(envelope)}},
                }),
            ),
        ):
            result = usage._gist_read()
        self.assertEqual(result, expected)

    def test_portfolio_gist_write_refuses_to_store_plaintext_without_backup_key(self):
        with (
            patch.object(usage, "_gist_config", return_value=("gist-id", "token")),
            patch.object(usage, "_portfolio_fernet", return_value=None),
            patch.object(
                usage.urllib.request, "urlopen",
                return_value=_Response({"public": False, "files": {}}),
            ) as open_url,
            self.assertLogs("usage", level="ERROR"),
        ):
            result = usage._gist_write({"bank": 1000, "bets": []})
        self.assertFalse(result)
        # Without a cipher key, refuse locally before making any remote request.
        self.assertEqual(open_url.call_count, 0)

    def test_legacy_portfolio_is_migrated_to_encrypted_gist_after_recovery(self):
        legacy = {"bank": 2200, "bets": [], "meta": {"initial_bank": 1000}}
        persisted = {}
        with (
            patch.object(usage, "_safe_read", return_value={}),
            patch.object(usage, "_gist_read", return_value=legacy),
            patch.object(usage, "_update", side_effect=lambda mutate: mutate(persisted)),
            patch.object(usage, "_gist_write", return_value=True) as write_remote,
        ):
            result = usage.get_local_data()
        self.assertEqual(result, legacy)
        self.assertEqual(persisted["data"], legacy)
        write_remote.assert_called_once_with(legacy)

    def test_legacy_portfolio_recovery_keeps_local_copy_when_encryption_key_is_missing(self):
        legacy = {"bank": 2200, "bets": []}
        persisted = {}
        with (
            patch.object(usage, "_safe_read", return_value={}),
            patch.object(usage, "_gist_read", return_value=legacy),
            patch.object(usage, "_update", side_effect=lambda mutate: mutate(persisted)),
            patch.object(usage, "_gist_write", return_value=False) as write_remote,
        ):
            result = usage.get_local_data()
        self.assertEqual(result, legacy)
        self.assertEqual(persisted["data"], legacy)
        write_remote.assert_called_once_with(legacy)

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
