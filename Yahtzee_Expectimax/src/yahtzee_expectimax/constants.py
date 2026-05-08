CATEGORIES = (
    "ones",
    "twos",
    "threes",
    "fours",
    "fives",
    "sixes",
    "three_of_kind",
    "four_of_kind",
    "full_house",
    "small_straight",
    "large_straight",
    "yahtzee",
    "chance",
)

UPPER = CATEGORIES[:6]
UPPER_BONUS_TARGET = 63
UPPER_BONUS = 35
YAHTZEE_BONUS = 100
N_DICE = 5
N_FACES = 6
ALL_USED_MASK = (1 << len(CATEGORIES)) - 1
