import unittest

from llm.analyst import _parse_llm_response


class IndependentAnalystTests(unittest.TestCase):
    def test_formats_independent_pick_and_disagreement(self):
        raw = '''{
          "independent_pick": "X",
          "probabilities": {"П1": 30, "X": 45, "П2": 25},
          "opinion": "Ничья выглядит правдоподобно, но преимущество небольшое.",
          "risks": "Недостаточно данных о составах.",
          "data_quality": "low"
        }'''
        result = _parse_llm_response(raw, "П1")
        self.assertIn("САМОСТОЯТЕЛЬНЫЙ ПРОГНОЗ ИИ: ничья (X)", result)
        self.assertIn("П1 30% · X 45% · П2 25%", result)
        self.assertIn("РАСХОЖДЕНИЕ", result)
        self.assertIn("основная модель — П1", result)

    def test_normalizes_probabilities_and_uses_the_largest_class(self):
        raw = '''{
          "independent_pick": "П2",
          "probabilities": {"П1": 40, "X": 30, "П2": 20},
          "opinion": "Преимущество хозяев по ожидаемым голам.",
          "risks": "",
          "data_quality": "medium"
        }'''
        result = _parse_llm_response(raw, "П1")
        self.assertIn("победа хозяев (П1)", result)
        self.assertIn("П1 45% · X 33% · П2 22%", result)
        self.assertIn("исход совпадает (П1)", result)

    def test_rejects_opinion_without_probabilities(self):
        result = _parse_llm_response('{"opinion":"Да, согласен","agree":true}', "П1")
        self.assertIsNone(result)

    def test_rejects_invalid_or_empty_distribution(self):
        raw = '{"probabilities":{"П1":0,"X":0,"П2":0},"opinion":"Нет данных"}'
        self.assertIsNone(_parse_llm_response(raw))



    def test_data_quality_is_capped_by_sample_size(self):
        raw = '''{
          "independent_pick": "П1",
          "probabilities": {"П1": 60, "X": 25, "П2": 15},
          "opinion": "Небольшое преимущество хозяев.",
          "risks": "Малая выборка и нет данных о составах.",
          "data_quality": "high"
        }'''
        low_sample = _parse_llm_response(raw, quality_cap="low")
        enough_sample = _parse_llm_response(raw, quality_cap="medium")
        self.assertIn("Качество исходных данных: низкое", low_sample)
        self.assertIn("Качество исходных данных: среднее", enough_sample)

    def test_comparison_is_added_after_independent_answer(self):
        from llm.analyst import _append_model_comparison
        raw = '''{
          "independent_pick": "X",
          "probabilities": {"П1": 20, "X": 60, "П2": 20},
          "opinion": "Ничья возможна, но неопределённость высокая.",
          "risks": "Небольшая выборка.",
          "data_quality": "low"
        }'''
        base = _parse_llm_response(raw)
        self.assertNotIn("основной моделью", base)
        self.assertIn("РАСХОЖДЕНИЕ", _append_model_comparison(base, "П1"))
        self.assertIn("исход совпадает (X)", _append_model_comparison(base, "X"))

    def test_llm_prompt_does_not_receive_model_pick_or_lambdas(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import llm.analyst as analyst

        content = '''{
          "independent_pick": "X",
          "probabilities": {"П1": 30, "X": 45, "П2": 25},
          "opinion": "Ничья может быть вероятнее с учётом небольшой разницы результатов.",
          "risks": "Выборки небольшие.",
          "data_quality": "high"
        }'''
        response = SimpleNamespace(
            status_code=200,
            text=content,
            json=lambda: {"choices": [{"message": {"content": content}}]},
        )
        ctx = {
            "api_key": "unit-test-key", "provider": "Groq (бесплатно, быстро)",
            "model": "unit-test-model",
            "home": "Home FC", "away": "Away FC", "league": "Test League",
            "date": "2026-10-25", "model_pick_code": "П1",
            "p1": .92, "px": .05, "p2": .03, "lam_h": 5.67, "lam_a": .12,
            "pick": "П1", "prob": .92, "ev": .4, "market_odds": {"П1": 1.05},
            "fh": "ВВПВВ", "fa": "ППВНП",
            "independent_features": {
                "home_scored_at_home_avg": 1.2, "home_conceded_at_home_avg": 0.8,
                "away_scored_away_avg": 1.0, "away_conceded_away_avg": 1.3,
                "home_home_sample": 2, "away_away_sample": 1,
            },
        }
        with (
            patch.object(analyst, "cache_get", return_value=None),
            patch.object(analyst, "cache_put"),
            patch.object(analyst.usage, "llm_remaining", return_value=5),
            patch.object(analyst.usage, "llm_increment"),
            patch.object(analyst._sess, "post", return_value=response) as post,
        ):
            result = analyst.analyze_match(ctx)
        request = post.call_args.kwargs["json"]
        user_prompt = request["messages"][1]["content"]
        self.assertIn("Голы хозяев дома за матч", user_prompt)
        self.assertNotIn("5.67", user_prompt)
        self.assertNotIn("0.92", user_prompt)
        self.assertNotIn("П1", user_prompt)
        self.assertIn("РАСХОЖДЕНИЕ", result)
        self.assertIn("Качество исходных данных: низкое", result)


if __name__ == "__main__":
    unittest.main()
