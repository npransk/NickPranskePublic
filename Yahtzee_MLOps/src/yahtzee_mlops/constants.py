"""Shared constants for Yahtzee."""

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

UPPER_CATEGORIES = CATEGORIES[:6]
LOWER_CATEGORIES = CATEGORIES[6:]
UPPER_BONUS_TARGET = 63
UPPER_BONUS_POINTS = 35
YAHTZEE_BONUS_POINTS = 100
NUM_DICE = 5
NUM_FACES = 6
MAX_ROLLS = 3
