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


if __name__ == "__main__":
    unittest.main()
