import unittest

from config import (
    DEFAULT_LLM_PROVIDER,
    llm_provider_key_config,
    normalize_llm_provider,
)


class GrokProviderConfigurationTests(unittest.TestCase):
    def test_fresh_or_invalid_provider_defaults_to_xai_grok(self):
        self.assertEqual(DEFAULT_LLM_PROVIDER, "Grok (x.ai)")
        self.assertEqual(normalize_llm_provider(None), "Grok (x.ai)")
        self.assertEqual(normalize_llm_provider("unknown-provider"), "Grok (x.ai)")

    def test_explicit_existing_provider_selection_is_preserved(self):
        self.assertEqual(
            normalize_llm_provider("Groq GPT-OSS 120B"),
            "Groq GPT-OSS 120B",
        )

    def test_grok_uses_its_own_xai_credential_names(self):
        config = llm_provider_key_config("Grok (x.ai)")
        self.assertEqual(config["env"], "XAI_API_KEY")
        self.assertEqual(config["storage"], "xai_api_key")
        self.assertEqual(config["session"], "xai_api_key_input")


if __name__ == "__main__":
    unittest.main()
