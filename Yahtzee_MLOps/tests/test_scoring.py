from yahtzee_mlops.scoring import score_category


def test_upper_section_scores_matching_faces() -> None:
    assert score_category("sixes", (6, 6, 6, 2, 1)) == 18
    assert score_category("ones", (1, 2, 3, 4, 5)) == 1


def test_kind_scores_use_dice_total() -> None:
    assert score_category("three_of_kind", (4, 4, 4, 2, 6)) == 20
    assert score_category("four_of_kind", (4, 4, 4, 2, 6)) == 0


def test_full_house_and_straights() -> None:
    assert score_category("full_house", (2, 2, 5, 5, 5)) == 25
    assert score_category("small_straight", (1, 2, 3, 4, 6)) == 30
    assert score_category("large_straight", (2, 3, 4, 5, 6)) == 40


def test_yahtzee_and_chance() -> None:
    assert score_category("yahtzee", (3, 3, 3, 3, 3)) == 50
    assert score_category("chance", (1, 2, 3, 4, 6)) == 16
