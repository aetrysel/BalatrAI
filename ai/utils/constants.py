# Balatro game state ID constants
# These match the G.STATES values in the Lua mod

SELECTING_HAND_STATE_ID = 1
SHOP_STATE_ID = 5
RESTART_RUN_STATE_ID = 6
BLIND_SELECT_STATE_ID = 7
ROUND_EVAL_STATE_ID = 8
GAME_OVER_STATE_ID = 11

# Pack state IDs: all states where a booster pack is open
# 999 = G.STATES.SMODS_BOOSTER_OPENED (Steamodded replaces all vanilla pack states)
PACK_STATE_IDS = frozenset({9, 10, 15, 17, 18, 999})

# Max blinds-cleared-while-held counted per joker (obs normalization)
TENURE_CAP = 6

# Jokers whose value doesn't pass through scoring/money/card-creation hooks
# (rule changes, hand size/discards, deck edits, sell-for-effect); floor their contribution in joker potential
PASSIVE_CONTRIBUTION_FLOOR = 0.3
PASSIVE_JOKERS = frozenset({
    "Four Fingers", "Shortcut", "Pareidolia", "Smeared Joker", "Splash", "Oops! All 6s", "Showman",
    "Juggler", "Drunkard", "Troubadour", "Merry Andy", "Turtle Bean", "Burglar",
    "Credit Card", "Chaos the Clown", "Astronomer", "To the Moon", "Gift Card", "Egg",
    "Mr. Bones", "Chicot", "Luchador", "Diet Cola", "Invisible Joker", "Perkeo",
    "Space Joker", "Burnt Joker", "DNA", "Certificate", "Marble Joker", "Midas Mask", "Hiker",
})
