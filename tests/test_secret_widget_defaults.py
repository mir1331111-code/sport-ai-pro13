"""Regression tests for server-side-only API-key defaults in Streamlit widgets."""

import ast
from pathlib import Path
import unittest


class SecretWidgetDefaultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app_path = Path(__file__).resolve().parents[1] / "App.py"
        cls.source = cls.app_path.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source, filename=str(cls.app_path))

    def test_server_stored_api_keys_are_not_widget_default_values(self):
        protected_keys = {
            "fdorg_token_input",
            "sstats_api_key_input",
            "footballapi_key_input",
            "llm_key_session",
        }
        checked = set()
        for node in ast.walk(self.tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "text_input":
                continue
            key_kw = next((kw.value for kw in node.keywords if kw.arg == "key"), None)
            key_name = None
            if isinstance(key_kw, ast.Constant) and isinstance(key_kw.value, str):
                key_name = key_kw.value
            elif isinstance(key_kw, ast.Name):
                key_name = key_kw.id
            if key_name not in protected_keys:
                continue
            value_kw = next((kw.value for kw in node.keywords if kw.arg == "value"), None)
            self.assertIsInstance(
                value_kw, ast.Constant,
                f"{key_name} must use a literal empty widget default",
            )
            self.assertEqual(
                value_kw.value, "",
                f"{key_name} must not receive a server-side secret as its default",
            )
            checked.add(key_name)

        self.assertEqual(checked, protected_keys)


    def test_provider_api_key_aliases_are_supported_without_revealing_values(self):
        from config import LLM_PROVIDERS, llm_provider_key_config

        groq_names = {
            str(name).casefold()
            for name in llm_provider_key_config("Groq GPT-OSS 120B")["aliases"]
        }
        self.assertIn("groq_api_key", groq_names)
        self.assertIn("llm_api_key", self.source.casefold())
        self.assertIn('"groq"', self.source.casefold())
        for provider, config in LLM_PROVIDERS.items():
            if provider.startswith("Groq"):
                self.assertIn("GROQ_API_KEY", config["key_aliases"])

    def test_background_score_lookups_resolve_keys_server_side(self):
        self.assertIn(
            '_api_key_default("fdorg_token", "FDORG_TOKEN", "fdorg_token")',
            self.source,
        )
        self.assertIn(
            '_api_key_default("sstats_api_key", "SSTATS_API_KEY", "sstats_api_key")',
            self.source,
        )
        self.assertIn('st.session_state.get("fdorg_token_input", "")', self.source)
        self.assertIn('st.session_state.get("sstats_api_key_input", "")', self.source)


if __name__ == "__main__":
    unittest.main()
