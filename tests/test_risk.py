import unittest
from datetime import datetime

from betting.kelly import kelly
from betting.risk import (
    KELLY_FRACTION,
    MIN_BET_ODDS,
    MAX_MARKET_STAKE_PCT,
    MAX_MATCH_EXPOSURE_PCT,
    MAX_DAILY_EXPOSURE_PCT,
    risk_snapshot,
    size_bet,
)


NOW = datetime(2026, 11, 11, 12, 0)


def _bet(*, stake, market="1X2", pick="П1", fixture_id="fixture-1", date_time="2026-11-11 09:00"):
    return {
        "status": "pending",
        "stake": stake,
        "market": market,
        "pick": pick,
        "fixture_id": fixture_id,
        "match": "Home FC vs Away FC",
        "date_iso": "2026-11-11",
        "date_time": date_time,
    }


class RiskSizingTests(unittest.TestCase):
    def test_policy_constants_match_user_approved_limits(self):
        self.assertEqual(KELLY_FRACTION, 0.25)
        self.assertEqual(MIN_BET_ODDS, 1.70)
        self.assertEqual(MAX_MARKET_STAKE_PCT, 0.01)
        self.assertEqual(MAX_MATCH_EXPOSURE_PCT, 0.02)
        self.assertEqual(MAX_DAILY_EXPOSURE_PCT, 0.05)

    def test_kelly_callers_cannot_raise_fraction_or_market_cap(self):
        stake = kelly(0.90, 2.0, 1000, frac=0.90, cap=0.20)
        self.assertLessEqual(stake, 10.0)

    def test_new_market_stake_is_capped_at_one_percent(self):
        stake, info = size_bet(
            probability=0.90,
            odds=2.0,
            bank=1000,
            bets=[],
            fixture_id="fixture-1",
            match="Home FC vs Away FC",
            date_iso="2026-11-11",
            market="1X2",
            pick="П1",
            now=NOW,
        )
        self.assertEqual(stake, 10.0)
        self.assertEqual(info["reason"], "approved")

    def test_odds_below_1_70_are_never_staked(self):
        stake, info = size_bet(
            probability=0.95,
            odds=1.69,
            bank=1000,
            bets=[],
            fixture_id="fixture-1",
            match="Home FC vs Away FC",
            date_iso="2026-11-11",
            market="1X2",
            pick="П1",
            now=NOW,
        )
        self.assertEqual(stake, 0.0)
        self.assertEqual(info["reason"], "odds_below_minimum_1_70")

    def test_same_market_exposure_is_counted_towards_one_percent(self):
        stake, info = size_bet(
            probability=0.90,
            odds=2.0,
            bank=992,
            bets=[_bet(stake=8.0)],
            fixture_id="fixture-1",
            match="Home FC vs Away FC",
            date_iso="2026-11-11",
            market="1X2",
            pick="П2",
            now=NOW,
        )
        self.assertEqual(stake, 2.0)
        self.assertEqual(info["market_used"], 8.0)

    def test_different_markets_share_two_percent_match_limit(self):
        stake, info = size_bet(
            probability=0.90,
            odds=2.0,
            bank=992,
            bets=[_bet(stake=8.0, market="1X2", pick="П1")],
            fixture_id="fixture-1",
            match="Home FC vs Away FC",
            date_iso="2026-11-11",
            market="OU",
            pick="ТБ 2.5",
            now=NOW,
        )
        self.assertEqual(stake, 10.0)
        self.assertEqual(info["match_used"], 8.0)

    def test_two_percent_match_cap_is_a_hard_combined_ceiling(self):
        stake, info = size_bet(
            probability=0.90,
            odds=2.0,
            bank=985,
            bets=[_bet(stake=15.0, market="1X2", pick="П1")],
            fixture_id="fixture-1",
            match="Home FC vs Away FC",
            date_iso="2026-11-11",
            market="OU",
            pick="ТБ 2.5",
            now=NOW,
        )
        self.assertEqual(stake, 5.0)
        self.assertEqual(info["match_used"], 15.0)

    def test_daily_exposure_blocks_stake_above_five_percent(self):
        stake, info = size_bet(
            probability=0.90,
            odds=2.0,
            bank=952,
            bets=[_bet(stake=48.0, fixture_id="another-fixture")],
            fixture_id="fixture-2",
            match="Other FC vs Away FC",
            date_iso="2026-11-11",
            market="1X2",
            pick="П1",
            now=NOW,
        )
        self.assertEqual(stake, 2.0)
        self.assertEqual(info["daily_used"], 48.0)

    def test_bets_on_other_days_do_not_use_today_daily_budget(self):
        stake, info = size_bet(
            probability=0.90,
            odds=2.0,
            bank=950,
            bets=[_bet(stake=50.0, fixture_id="old-fixture", date_time="2026-11-10 20:00")],
            fixture_id="fixture-2",
            match="Other FC vs Away FC",
            date_iso="2026-11-11",
            market="1X2",
            pick="П1",
            now=NOW,
        )
        self.assertEqual(stake, 10.0)
        self.assertEqual(info["daily_used"], 0.0)

    def test_daily_snapshot_uses_all_open_bets_and_available_bank(self):
        snapshot = risk_snapshot(
            950,
            [_bet(stake=30.0), _bet(stake=20.0, fixture_id="fixture-2")],
            now=NOW,
        )
        self.assertEqual(snapshot["bankroll_basis"], 1000.0)
        self.assertEqual(snapshot["pending_exposure"], 50.0)
        self.assertEqual(snapshot["daily_exposure"], 50.0)
        self.assertEqual(snapshot["daily_remaining"], 0.0)
        self.assertAlmostEqual(snapshot["daily_exposure_pct"], 0.05)

    def test_settled_bets_still_count_towards_cumulative_daily_risk(self):
        settled = _bet(stake=48.0, fixture_id="settled-fixture")
        settled["status"] = "lost"
        stake, info = size_bet(
            probability=0.90,
            odds=2.0,
            bank=952,
            bets=[settled],
            fixture_id="new-fixture",
            match="New FC vs Other FC",
            date_iso="2026-11-11",
            market="1X2",
            pick="П1",
            now=NOW,
        )
        self.assertEqual(stake, 2.0)
        self.assertEqual(info["daily_used"], 48.0)

    def test_non_finite_prices_and_probabilities_are_rejected(self):
        self.assertEqual(kelly(float("nan"), 2.0, 1000), 0.0)
        self.assertEqual(kelly(0.75, float("inf"), 1000), 0.0)


if __name__ == "__main__":
    unittest.main()
