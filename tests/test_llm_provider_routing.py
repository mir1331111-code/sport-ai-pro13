import unittest

from config import LLM_PROVIDERS, normalize_llm_provider


class LLMProviderRoutingTests(unittest.TestCase):
    def test_default_groq_provider_uses_official_endpoint(self):
        provider = LLM_PROVIDERS["Groq (бесплатно, быстро)"]
        self.assertEqual(
            provider["base"], "https://api.groq.com/openai/v1"
        )
        self.assertEqual(provider["model"], "openai/gpt-oss-20b")

    def test_retired_groq_70b_selection_migrates_to_supported_gpt_oss_120b(self):
        provider_name = normalize_llm_provider("Groq 70B (бесплатно, мощнее)")
        self.assertEqual(
            provider_name, "Groq GPT-OSS 120B (оплата по токенам)"
        )
        self.assertEqual(
            LLM_PROVIDERS[provider_name]["model"], "openai/gpt-oss-120b"
        )
        self.assertNotIn(
            "llama-3.3-70b-versatile",
            {provider["model"] for provider in LLM_PROVIDERS.values()},
        )

    def test_grok_uses_current_xai_model_and_dedicated_credential(self):
        from config import llm_provider_key_config

        provider = LLM_PROVIDERS["Grok (x.ai)"]
        self.assertEqual(provider["base"], "https://api.x.ai/v1")
        self.assertEqual(provider["model"], "grok-4.7")
        self.assertEqual(llm_provider_key_config("Grok (x.ai)"), {
            "env": "XAI_API_KEY",
            "storage": "xai_api_key",
            "session": "xai_api_key_input",
        })

    def test_provider_endpoints_are_https_and_not_author_proxy_hosts(self):
        for name, provider in LLM_PROVIDERS.items():
            with self.subTest(provider=name):
                base = str(provider.get("base") or "")
                self.assertTrue(base.startswith("https://"))
                self.assertNotIn("workers.dev", base.casefold())
                self.assertNotIn("mir1331111.workers.dev", base.casefold())


if __name__ == "__main__":
    unittest.main()
