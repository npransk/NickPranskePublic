from yahtzee_expectimax.scoring import score_counts


def test_scoring_rules() -> None:
    assert score_counts("sixes", (0, 0, 0, 0, 0, 3)) == 18
    assert score_counts("three_of_kind", (0, 0, 0, 2, 0, 3)) == 26
    assert score_counts("four_of_kind", (0, 0, 0, 2, 0, 3)) == 0
    assert score_counts("full_house", (0, 2, 0, 0, 3, 0)) == 25
    assert score_counts("small_straight", (1, 1, 1, 1, 1, 0)) == 30
    assert score_counts("large_straight", (0, 1, 1, 1, 1, 1)) == 40
    assert score_counts("yahtzee", (0, 0, 5, 0, 0, 0)) == 50
