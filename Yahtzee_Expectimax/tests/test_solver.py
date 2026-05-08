from yahtzee_expectimax.tournament import TournamentPolicy


def test_tournament_policy_scores_when_no_rolls_left() -> None:
    policy = TournamentPolicy()
    kind, payload = policy.choose(0, 0, False, (0, 0, 0, 0, 0, 5), 0)
    assert kind == "score"
    assert payload in {"yahtzee", "sixes", "chance", "three_of_kind", "four_of_kind"}
