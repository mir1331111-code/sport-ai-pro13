if real_odds:
    verdict, best = refine_with_real_odds(
        verdict, rows_, real_odds, D["bank"], kelly_frac
    )
    odds_source = "market"
if best is None:
    est_odd = verdict.get("fair_odd")
    if est_odd and est_odd > 1.01:
        prob_ = verdict["prob"]
        ev_ = prob_ * est_odd - 1
        min_stake = round(D["bank"] * 0.005, 2)
        stake_ = max(
            kelly(prob_, est_odd, D["bank"], kelly_frac),
            min_stake
        )
        verdict["real_odds"] = False
        verdict["odd"] = est_odd
        verdict["ev"] = ev_
        best = (
            market_type(verdict["pick"]),
            verdict["pick"],
            est_odd,
            ev_,
            prob_,
            stake_
        )
        odds_source = "estimated"
