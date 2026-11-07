import unittest

from config import LLM_PROVIDERS


class LLMProviderRoutingTests(unittest.TestCase):
    def test_default_groq_provider_uses_official_endpoint(self):
        provider = LLM_PROVIDERS["Groq (бесплатно, быстро)"]
        self.assertEqual(
            provider["base"], "https://api.groq.com/openai/v1"
        )
        self.assertEqual(provider["model"], "openai/gpt-oss-20b")

    def test_provider_endpoints_are_https_and_not_author_proxy_hosts(self):
        for name, provider in LLM_PROVIDERS.items():
            with self.subTest(provider=name):
                base = str(provider.get("base") or "")
                self.assertTrue(base.startswith("https://"))
                self.assertNotIn("workers.dev", base.casefold())
                self.assertNotIn("mir1331111.workers.dev", base.casefold())


if __name__ == "__main__":
    unittest.main()
