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
