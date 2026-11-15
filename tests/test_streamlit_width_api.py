import unittest
from pathlib import Path


class StreamlitWidthApiCompatibilityTests(unittest.TestCase):
    def test_application_modules_do_not_use_deprecated_container_width_kwarg(self):
        root = Path(__file__).resolve().parents[1]
        sources = (
            "App.py",
            "ui/tabs/scanner.py",
            "ui/tabs/analytics.py",
            "ui/tabs/backtest.py",
            "ui/tabs/calculator.py",
            "ui/tabs/portfolio.py",
            "ui/tabs/stats.py",
        )
        deprecated_kwarg = "use_" + "container_width"
        for relative_path in sources:
            with self.subTest(path=relative_path):
                source = (root / relative_path).read_text(encoding="utf-8")
                self.assertNotIn(deprecated_kwarg, source)

    def test_minimum_streamlit_version_supports_width_parameter(self):
        root = Path(__file__).resolve().parents[1]
        requirements = (root / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("streamlit>=1.50,<2", requirements)


if __name__ == "__main__":
    unittest.main()
