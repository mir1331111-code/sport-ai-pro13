import unittest

from data.context_history import latest_pre_kickoff_rows, parse_context_csv


class ContextHistoryTests(unittest.TestCase):
    def test_rejects_post_kickoff_snapshots_and_keeps_latest_pre_kickoff(self):
        raw = (
            "fixture_id,home_team,away_team,kickoff_at,observed_at,is_confirmed,"
            "lineup_is_confirmed,referee_is_confirmed,source_reliability,"
            "lineup_strength_home,lineup_strength_away,key_players_missing_home,"
            "key_players_missing_away,base_p_home,base_p_draw,base_p_away,result\n"
            "f1,Home,Away,2026-11-12T20:00:00Z,2026-11-12T18:00:00Z,true,"
            "true,true,0.9,0.7,0.4,0,1,0.55,0.25,0.20,H\n"
            "f1,Home,Away,2026-11-12T20:00:00Z,2026-11-12T19:00:00Z,true,"
            "true,true,0.95,0.8,0.3,0,0,0.56,0.24,0.20,H\n"
            "f1,Home,Away,2026-11-12T20:00:00Z,2026-11-12T20:01:00Z,true,"
            "true,true,0.99,0.9,0.1,0,0,0.60,0.20,0.20,H\n"
        )
        parsed = parse_context_csv(raw)
        self.assertEqual(parsed["accepted"], 2)
        self.assertEqual(len(parsed["rejected"]), 1)
        latest = latest_pre_kickoff_rows(parsed["rows"])
        self.assertEqual(len(latest), 1)
        self.assertEqual(latest[0]["observed_at"], "2026-11-12T19:00:00+00:00")
        self.assertEqual(latest[0]["source_reliability"], 0.95)
        self.assertTrue(latest[0]["lineup_is_confirmed"])
        self.assertTrue(latest[0]["referee_is_confirmed"])

    def test_aggregates_per_feed_reliability_when_overall_value_is_blank(self):
        parsed = parse_context_csv(
            "fixture_id,home_team,away_team,kickoff_at,observed_at,is_confirmed,"
            "lineup_is_confirmed,referee_is_confirmed,lineup_source_reliability,"
            "referee_source_reliability,lineup_strength_home,lineup_strength_away,"
            "key_players_missing_home,key_players_missing_away,base_p_home,base_p_draw,base_p_away,result\n"
            "f2,H,A,2026-11-12T20:00:00Z,2026-11-12T19:00:00Z,true,true,true,"
            "0.91,0.84,0.8,0.4,0,0,0.5,0.3,0.2,H\n"
        )
        row = parsed["rows"][0]
        self.assertEqual(row["source_reliability"], 0.84)

    def test_requires_timezone_aware_timestamps(self):
        with self.assertRaises(ValueError):
            parse_context_csv(
                "fixture_id,home_team,away_team,kickoff_at,observed_at\n"
                "f3,H,A,2026-11-12T20:00:00,2026-11-12T19:00:00\n"
            )


if __name__ == "__main__":
    unittest.main()
