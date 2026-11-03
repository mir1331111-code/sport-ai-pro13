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
                "home_home_sample": 5, "away_away_sample": 5,
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
        self.assertIn("Качество исходных данных: среднее", result)

    def test_prompt_includes_corners_cards_and_goal_markets(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import llm.analyst as analyst

        content = '''{
          "independent_pick": "П1",
          "probabilities": {"П1": 45, "X": 30, "П2": 25},
          "goal_markets": {
            "expected_home_goals": 1.5, "expected_away_goals": 1.0,
            "over_1_5": 80, "over_2_5": 58, "over_3_5": 34,
            "over_4_5": 16, "btts_yes": 52
          },
          "corner_markets": {
            "expected_home_corners": 5.1, "expected_away_corners": 4.2,
            "over_8_5": 63, "over_9_5": 54, "over_10_5": 42, "over_11_5": 28
          },
          "card_markets": {
            "expected_home_cards": 2.0, "expected_away_cards": 1.8,
            "over_2_5": 72, "over_3_5": 55, "over_4_5": 36, "over_5_5": 19
          },
          "opinion": "У хозяев небольшое преимущество.",
          "risks": "Нет подтверждённых составов.",
          "data_quality": "medium"
        }'''
        response = SimpleNamespace(
            status_code=200, text=content,
            json=lambda: {"choices": [{"message": {"content": content}}]},
        )
        features = {
            "home_scored_at_home_avg": 1.4, "home_conceded_at_home_avg": 0.8,
            "away_scored_away_avg": 1.0, "away_conceded_away_avg": 1.2,
            "home_home_sample": 7, "away_away_sample": 6,
            "markets": {
                "corners": {
                    "home_for_avg": 5.1, "home_against_avg": 4.0,
                    "away_for_avg": 4.2, "away_against_avg": 4.8,
                    "history_matches": 320, "home_team_games": 8, "away_team_games": 7,
                },
                "cards": {
                    "home_for_avg": 2.0, "home_against_avg": 1.8,
                    "away_for_avg": 1.8, "away_against_avg": 2.0,
                    "history_matches": 320, "home_team_games": 8, "away_team_games": 7,
                },
            },
        }
        ctx = {
            "api_key": "unit-test-key", "provider": "Groq (бесплатно, быстро)",
            "model": "unit-test-model", "home": "Home FC", "away": "Away FC",
            "league": "Test League", "date": "2026-11-03",
            "fh": "ВВПВВ", "fa": "ППВНП",
            "independent_features": features, "model_pick_code": "П1",
            "p1": .7, "px": .2, "p2": .1, "lam_h": 1.5, "lam_a": 1.0,
        }
        with (
            patch.object(analyst, "cache_get", return_value=None),
            patch.object(analyst, "cache_put"),
            patch.object(analyst.usage, "llm_remaining", return_value=5),
            patch.object(analyst.usage, "llm_increment"),
            patch.object(analyst._sess, "post", return_value=response) as post,
        ):
            result = analyst.analyze_match(ctx)
        prompt = post.call_args.kwargs["json"]["messages"][1]["content"]
        self.assertIn("Угловые:", prompt)
        self.assertIn("Карточки:", prompt)
        self.assertIn("Оцени 1X2 обязательно", prompt)
        self.assertIn("ГОЛЫ (оценка Groq)", result)
        self.assertIn("УГЛОВЫЕ (оценка Groq)", result)
        self.assertIn("КАРТОЧКИ (оценка Groq)", result)
        self.assertNotIn("Edge", prompt)
        self.assertNotIn("market_odds", prompt)
        self.assertNotIn("1.81", prompt)
        self.assertNotIn("2.20", prompt)

    def test_llm_does_not_publish_corner_probabilities_without_real_history(self):
        from llm.analyst import _format_multi_market_sections
        raw = {
            "corner_markets": {
                "expected_home_corners": 6, "expected_away_corners": 5,
                "over_8_5": 70, "over_9_5": 60, "over_10_5": 50, "over_11_5": 40,
            }
        }
        summary = _format_multi_market_sections(raw, {
            "markets": {"corners": {
                "history_matches": 20, "home_team_games": 2, "away_team_games": 2,
            }}
        })
        self.assertIn("недостаточно подтверждённой истории", summary)
        self.assertNotIn("70%", summary)

    def test_llm_rejects_non_monotone_corner_line_probabilities(self):
        from llm.analyst import _format_multi_market_sections
        raw = {
            "corner_markets": {
                "expected_home_corners": 6, "expected_away_corners": 5,
                "over_8_5": 55, "over_9_5": 65, "over_10_5": 35, "over_11_5": 20,
            }
        }
        summary = _format_multi_market_sections(raw, {
            "markets": {"corners": {
                "history_matches": 300, "home_team_games": 8, "away_team_games": 7,
            }}
        })
        self.assertIn("Groq не вернул корректную оценку", summary)
        self.assertNotIn("55%", summary)

    def test_groq_gpt_oss_request_uses_low_reasoning_and_strict_json(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import llm.analyst as analyst

        content = '''{
          "independent_pick": "П1",
          "probabilities": {"П1": 48, "X": 28, "П2": 24},
          "goal_markets": null,
          "corner_markets": null,
          "card_markets": null,
          "opinion": "Небольшое преимущество хозяев.",
          "risks": "Нет подтверждённых составов.",
          "data_quality": "medium"
        }'''
        response = SimpleNamespace(
            status_code=200,
            text=content,
            json=lambda: {"choices": [{"message": {"content": content}}]},
        )
        ctx = {
            "api_key": "unit-test-key",
            "provider": "Groq (бесплатно, быстро)",
            "model": "openai/gpt-oss-20b",
            "home": "Home FC", "away": "Away FC", "league": "Test League",
            "date": "2026-11-03", "model_pick_code": "П1",
            "fh": "ВВПВВ", "fa": "ППВНП",
            "independent_features": {
                "home_scored_at_home_avg": 1.2, "home_conceded_at_home_avg": 0.8,
                "away_scored_away_avg": 1.0, "away_conceded_away_avg": 1.3,
                "home_home_sample": 5, "away_away_sample": 5,
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
        self.assertEqual(request["max_completion_tokens"], 4096)
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertFalse(request["include_reasoning"])
        self.assertEqual(request["response_format"]["type"], "json_schema")
        self.assertTrue(request["response_format"]["json_schema"]["strict"])
        self.assertEqual(len(request["messages"]), 1)
        self.assertIn("САМОСТОЯТЕЛЬНЫЙ ПРОГНОЗ ИИ", result)

    def test_gpt_oss_retries_empty_length_completion_with_larger_budget(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import llm.analyst as analyst

        raw = '''{
          "independent_pick": "П1",
          "probabilities": {"П1": 48, "X": 28, "П2": 24},
          "goal_markets": null,
          "corner_markets": null,
          "card_markets": null,
          "opinion": "Небольшое преимущество хозяев.",
          "risks": "Нет подтверждённых составов.",
          "data_quality": "medium"
        }'''
        empty = SimpleNamespace(
            status_code=200,
            text='{"choices":[{"message":{"content":null},"finish_reason":"length"}]}',
            json=lambda: {
                "choices": [{
                    "message": {"role": "assistant", "content": None, "reasoning": "private reasoning"},
                    "finish_reason": "length",
                }],
                "usage": {
                    "completion_tokens": 4096,
                    "completion_tokens_details": {"reasoning_tokens": 3900},
                },
            },
        )
        recovered = SimpleNamespace(
            status_code=200,
            text=raw,
            json=lambda: {"choices": [{
                "message": {"role": "assistant", "content": raw},
                "finish_reason": "stop",
            }]},
        )
        ctx = {
            "api_key": "unit-test-key",
            "provider": "Groq (бесплатно, быстро)",
            "model": "openai/gpt-oss-20b",
            "home": "Home FC", "away": "Away FC", "league": "Test League",
            "date": "2026-11-03", "model_pick_code": "П1",
            "fh": "ВВПВВ", "fa": "ППВНП",
            "independent_features": {
                "home_scored_at_home_avg": 1.2, "home_conceded_at_home_avg": 0.8,
                "away_scored_away_avg": 1.0, "away_conceded_away_avg": 1.3,
                "home_home_sample": 5, "away_away_sample": 5,
            },
        }
        with (
            patch.object(analyst, "cache_get", return_value=None),
            patch.object(analyst, "cache_put"),
            patch.object(analyst.usage, "llm_remaining", return_value=5),
            patch.object(analyst.usage, "llm_increment") as increment,
            patch.object(analyst._sess, "post", side_effect=[empty, recovered]) as post,
        ):
            result = analyst.analyze_match(ctx)

        self.assertIsNotNone(result)
        self.assertIn("П1 48% · X 28% · П2 24%", result)
        self.assertEqual(post.call_count, 2)
        self.assertEqual(post.call_args_list[0].kwargs["json"]["max_completion_tokens"], 4096)
        self.assertEqual(post.call_args_list[1].kwargs["json"]["max_completion_tokens"], 8192)
        self.assertEqual(increment.call_count, 2)
        self.assertNotIn("private reasoning", result)

    def test_chat_completion_extractor_reads_text_blocks_but_never_reasoning(self):
        from llm.analyst import _extract_chat_completion

        content, diagnostics = _extract_chat_completion({
            "choices": [{
                "message": {
                    "content": [
                        {"type": "text", "text": r'{"probabilities": {'},
                        {"type": "text", "text": "}"},
                    ],
                    "reasoning": "private chain of thought",
                },
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 20},
        })
        self.assertEqual(content, r'{"probabilities": {}}')
        self.assertEqual(diagnostics["finish_reason"], "stop")
        self.assertNotIn("private chain", content)

        empty, meta = _extract_chat_completion({
            "choices": [{
                "message": {"content": None, "reasoning": "must not be shown"},
                "finish_reason": "length",
            }],
            "usage": {
                "completion_tokens": 50,
                "completion_tokens_details": {"reasoning_tokens": 50},
            },
        })
        self.assertEqual(empty, "")
        self.assertEqual(meta["reasoning_tokens"], 50)
        self.assertNotIn("must not be shown", empty)

    def test_analyze_match_skips_without_minimum_venue_history(self):
        from unittest.mock import patch
        import llm.analyst as analyst

        ctx = {
            "api_key": "unit-test-key",
            "independent_features": {
                "home_scored_at_home_avg": 1.2,
                "home_conceded_at_home_avg": 0.8,
                "away_scored_away_avg": 1.0,
                "away_conceded_away_avg": 1.3,
                "home_home_sample": 4,
                "away_away_sample": 8,
            },
        }
        with patch.object(analyst._sess, "post") as post:
            self.assertIsNone(analyst.analyze_match(ctx))
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
