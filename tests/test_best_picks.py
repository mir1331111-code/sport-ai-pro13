from betting.best_picks import best_value_market


def _verdict():
    return {
        "pick": "П1", "label": "Победа хозяев", "prob": 0.70,
        "fair_odd": 1.43, "is_action": True, "confidence": "хорошая",
        "confidence_score": 70, "confidence_label": "B", "reasons": [],
    }


def _rows():
    return [
        {"pick": "П1", "label": "Победа хозяев", "prob": 0.70, "fair_odd": 1 / 0.70},
        {"pick": "X", "label": "Ничья", "prob": 0.20, "fair_odd": 5.0},
        {"pick": "П2", "label": "Победа гостей", "prob": 0.10, "fair_odd": 10.0},
        {"pick": "ТБ 2.5", "label": "Тотал Больше 2.5", "prob": 0.60, "fair_odd": 1 / 0.60},
    ]


def test_selects_best_qualified_market_not_just_highest_probability():
    # Home win has no value at 1.30; over 2.5 is a real positive-value price.
    chosen, best = best_value_market(
        _verdict(), _rows(), {"П1": 1.30, "ТБ 2.5": 1.90},
        bank=1000, kelly_frac=0.25, min_prob=0.55,
    )
    assert chosen is not None
    assert chosen["pick"] == "ТБ 2.5"
    assert chosen["real_odds"] is True
    assert chosen["value_passed"] is True
    assert best[1] == "ТБ 2.5"
    assert best[2] == 1.90


def test_never_creates_bet_from_fair_odds_or_missing_market():
    chosen, best = best_value_market(
        _verdict(), _rows(), None, bank=1000, kelly_frac=0.25, min_prob=0.55,
    )
    assert chosen is None
    assert best is None


def test_rejects_price_without_edge_and_ev_thresholds():
    chosen, best = best_value_market(
        _verdict(), _rows(), {"П1": 1.35, "ТБ 2.5": 1.55},
        bank=1000, kelly_frac=0.25, min_prob=0.55,
    )
    assert chosen is None
    assert best is None


def test_probability_threshold_is_respected():
    chosen, best = best_value_market(
        _verdict(), _rows(), {"П1": 1.30, "ТБ 2.5": 1.90},
        bank=1000, kelly_frac=0.25, min_prob=0.65,
    )
    assert chosen is None
    assert best is None
