# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning project for the card game Balatro. The goal is to create a mod that can:
1. Extract game state data from Balatro
2. Allow an RL AI to observe the data and perform actions
3. Enable the AI to learn over time through gameplay
4. Facilitate speed as much as possible which means skipping animations, and timers whenever possible
5. We don't want to modify the game in any way to give an unfair advantage, we just want to play the game how it was designed

## Architecture

```
balatro-rl/
├── RLBridge/           # Lua mod injected into Balatro via Lovely
│   ├── lovely/         # Lovely patch .toml files
│   ├── ai.lua          # Main update loop, orchestration, change detection
│   ├── actions.lua     # Action registry (IDs, availability, execution routing)
│   ├── input.lua       # Translates actions into game inputs
│   ├── output.lua      # Serializes game state to JSON
│   ├── communication.lua # Pipe/socket IPC with Python
│   ├── init.lua        # Run startup, speed/animation settings
│   ├── utils.lua       # Logging, helpers
│   └── dkjson.lua      # JSON encode/decode library
├── ai/                 # Python RL training code
│   ├── train_balatro.py         # Entry point; training orchestration
│   ├── config/
│   │   └── seeds.json           # Named seeds (persisted between sessions)
│   ├── environment/
│   │   ├── balatro_env.py       # Gymnasium environment wrapper
│   │   └── reward.py            # Reward calculator
│   └── utils/
│       ├── communication.py     # Pipe/socket layer (Python side)
│       ├── constants.py         # Game state ID constants (SELECTING_HAND_STATE_ID etc.)
│       ├── mappers.py           # Game state → observation vector (BalatroStateMapper)
│       ├── callbacks.py         # SB3 training callbacks, TensorBoard metrics
│       ├── validation.py        # JSON schema validators
│       ├── replay.py            # Saves top-10 winning episodes
│       ├── history_logger.py    # Human-readable per-session action/event log
│       └── debug.py             # Timestamped print helpers
├── balatro-source-code/ # Decompiled Balatro source — reference only, do not edit
├── models/             # Saved model checkpoints (run_N_YYYYMMDD_HHMMSS/)
├── logs/               # Log groups: NNN_name/{tensorboard,history}/ (one group per feature change)
├── game_history_logs/  # Legacy per-session history logs (pre log groups)
└── tensorboard_logs/   # Legacy TensorBoard event files (pre log groups)
```

### Lovely Patching System
The mod uses `RLBridge/lovely/` to define patches:
- `init.toml` — injects all 9 Lua modules (incl. `stats.lua`), hooks `G:start_up()`, patches animation skipping
- `ai.toml` — hooks `G:update(dt)` to call `AI.update()` every frame
- Any additional patch can be created by adding a `.toml` file in the lovely dir

### Balatro Source Code
The `balatro-source-code/` directory contains the decompiled Balatro source code for reference and understanding game internals.
Editing this file won't do nothing so don't edit it, it's just reference

## Development Setup

### Linux Native (on this machine)


### Linux with Proton (Steam)
Create a symlink to the Steam mods directory:
```bash
ln -s ~/dev/balatro-rl/RLBridge /mnt/gamerlinuxssd/SteamLibrary/steamapps/compatdata/2379780/pfx/drive_c/users/steamuser/AppData/Roaming/Balatro/Mods/RLBridge
```

### Windows
Symlink or copy `RLBridge/` to:
```
%APPDATA%\Balatro\Mods\RLBridge
```

### Python Dependencies
```bash
pip install -r ai/requirements.txt
```
Key packages: `gymnasium`, `stable-baselines3`, `sb3-contrib` (MaskablePPO), `torch`, `tensorboard`

### Mod Dependencies
- Uses Lovely for code injection

## How to Run Training

```bash
python -m ai.train_balatro
```

All prompts come before "Press Enter to start training". The first launch asks for every setting; later launches load `ai/config/last_session.json` and show a settings menu — press Enter to reuse everything, or type option numbers (e.g. `1 6`) to change only those:
1. **Seed** (named seeds in `ai/config/seeds.json`, or random)
2. **Deck**, 3. **Training steps** (additional steps when resuming), 4. **Endless**, 5. **Debug mode** (`n_steps=128`)
6. **Model**: resume the latest checkpoint of a `models/run_N_*` dir, or start a new run
7. **Log group**: an existing `logs/NNN_name/` (default: latest) or a new one (number auto-increments)
8. **TB session** (resumed runs only): append to an existing session of that run, or start a new one

Then start Balatro and press `R` in-game to activate the RL loop.

Checkpoints go to `models/run_N_YYYYMMDD_HHMMSS/`. Each session writes `logs/NNN_name/tensorboard/<session>/` and `logs/NNN_name/history/history_<session>.log`, where `<session>` = `YYYYMMDD_HHMMSS_run_N` (timestamp first so sessions sort chronologically). Appending to a session keeps one continuous TensorBoard curve (`reset_num_timesteps=False`) and appends to its history log. Point TensorBoard at a group: `tensorboard --logdir logs/NNN_name/tensorboard`.

Endless/seed/deck are passed to the mod via the run config file (`/tmp/balatro_run_config.json`, or `%APPDATA%/Balatro/Mods/RLBridge/run_config.json` on Windows), read by `UTIL.get_run_config()`. On the ante 8 win the mod dismisses the win overlay and keeps playing (endless) or returns to the menu to start a new run.

## Key Implementation Details

### Game State Management
Balatro uses numbered states accessed via `G.STATES.*`:
- `G.STATES.SELECTING_HAND = 1` - Player selects cards to play/discard
- `G.STATES.SHOP = 5` - Shop between rounds
- `G.STATES.BLIND_SELECT = 7` - Player chooses which blind to fight
- `G.STATES.ROUND_EVAL` - After blind cleared, before cash out
- `G.STATES.GAME_OVER` - Lost a blind
- `G.STATES.WIN` - Beat Ante 8 boss (game victory)

### Critical Patterns
- All mod functions are called from `ai.update()` which runs every frame
- Use flags to prevent repeated actions in frame-based updates
- Game state transitions happen through calling existing game functions rather than direct state manipulation
- UI elements may not exist immediately when states change
- `UTIL.is_game_ready_for_action()` must return true before executing any action

### Auto-skipped vs AI-controlled States
Some states are handled automatically without consulting the AI:
- **Auto-skipped**: START_RUN (menu), SELECT_BLIND, CASH_OUT (after blind cleared), run restart on win
- **AI-controlled**: SELECTING_HAND, SHOP, GAME_OVER (restart decision)

### Key Balatro Globals
- `G.STATE` - Current game state number
- `G.STATES` - Table of named state constants
- `G.GAME` - Main game data (ante, round, dollars, etc.)
- `G.FUNCS` - Game function callbacks
- `G.hand` - Player's current hand cards
- `G.jokers` - Active jokers
- `G.shop_jokers` / `G.shop_vouchers` / `G.shop_booster` - Shop item areas
- `G.STATE_COMPLETE` - True during state transitions; block actions when set

## Communication Architecture

### Transport
Two modes, configured via environment variable `BALATRO_COMM_MODE` (default: `"pipe"`):

**Pipe mode (default):**
- Linux: Unix domain sockets (FIFOs) at `/tmp/balatro_request` (Balatro writes) and `/tmp/balatro_response` (Balatro reads)
- Windows: Named pipes `\\.\pipe\balatro_request` and `\\.\pipe\balatro_response`
- Persistent handles — opened once, kept open for the session

**Socket mode:**
- TCP; Python acts as server, Balatro connects as client
- `BALATRO_SOCKET_HOST` (default: `127.0.0.1`), `BALATRO_SOCKET_PORT` (default: `9000`)
- Enable with `BALATRO_COMM_MODE=socket`

### Protocol
Line-delimited JSON (one JSON object per line, terminated with `\n`).

**Request (Balatro → Python):**
```json
{
  "game_state": {
    "state": 1,
    "ante": 1, "round_count": 1,
    "blind_name": "Small Blind", "blind_chips": 300,
    "blind_cleared": 0, "game_win": 0, "game_over": 0,
    "round": {"hands_left": 4, "discards_left": 3},
    "hand": {
      "size": 5, "highlighted_count": 0,
      "cards": [{"base": {"nominal": 11, "value": "A"}, "highlighted": false, "suit": "H"}]
    },
    "chips": 0, "gold": 4,
    "current_hand": {"chips": 0, "mult": 0, "score": 0, "handname": ""},
    "jokers": [{"name": "Joker", "sell_cost": 5, "slot": 1}],
    "shop": {"items": [{"name": "Joker", "cost": 8, "type": "joker", "slot": 1}]},
    "seed": 12345, "retry_count": 0
  },
  "available_actions": [1, 2, 3]
}
```

**Response (Python → Balatro):**
```json
{"action": 2, "params": [], "seed": "JFKGEEMG"}
```

## Action Space

| ID | Name | Active State | Parameters |
|----|------|-------------|------------|
| 1 | SELECT_HAND | SELECTING_HAND | `card_indices`: array of 1-based indices (1–5 cards) |
| 2 | PLAY_HAND | SELECTING_HAND (after select) | none |
| 3 | DISCARD_HAND | SELECTING_HAND (after select) | none |
| 4 | START_RUN | Menu | `seed`: optional string |
| 5 | SELECT_BLIND | BLIND_SELECT | none |
| 6 | RESTART_RUN | GAME_OVER | `seed`: optional string |
| 7 | BUY_JOKER | SHOP | `slot`: 1-based shop slot index |
| 8 | SELL_JOKER | SHOP | `slot`: 1-based joker slot index |
| 9 | REROLL_SHOP | SHOP | none |
| 10 | SKIP_SHOP | SHOP | none |
| 11 | CASH_OUT | ROUND_EVAL | none |

The Lua `ACTIONS` module validates availability before execution and returns `{success, error}`. The Python side uses `ActionMasker` (sb3-contrib) to mask unavailable actions in the policy.

## Python AI Architecture

### Algorithm
MaskablePPO (from sb3-contrib) with MlpPolicy. Key hyperparameters:
- `n_steps`: 4096, `batch_size`: 64, `n_epochs`: 10
- `gamma`: 0.99, `ent_coef`: 0.01, `learning_rate`: 3e-4

### Observation Space
Shape `(417,)` float32 vector built by `ai/utils/mappers.py` (`BalatroStateMapper`):

| Slice | Features | Count |
|-------|----------|-------|
| Round | hands_left, discards_left | 2 |
| Blind | blind_chips, current chips | 2 |
| Progress | ante (norm /8), round_count (norm /24, = `G.GAME.round`, increments on blind selection), gold (norm /100), reroll_cost (norm /10) | 4 |
| Game state | one-hot (20) + game_over + game_win + retry_count | 23 |
| Blind identity | blind_name_id (norm /30), blind_is_boss | 2 |
| Deck | deck name one-hot | 16 |
| Jokers | 5 slots × 5 (occupied, name_id, sell_cost, edition, rounds_held norm /6) | 25 |
| Consumables | 5 slots × 2 (occupied, name_id) | 10 |
| Shop | 4 joker + 1 voucher + 2 booster slots × 3 each | 21 |
| Pack cards | 5 slots × 2 (occupied, name_id) | 10 |
| Deck stats | size + 4 seals + 8 enhancements + 4 editions | 17 |
| Hand | size, highlighted_count + 12 cards × 21 features | 254 |
| Scoring | chips, mult, score + hand type one-hot × 13 | 16 |
| Action mask | 17 actions | 17 |

**Note:** `blind_name` is encoded as a vocabulary index (`BLIND_NAMES` list in `mappers.py`). `blind_is_boss` is a binary flag from `G.GAME.blind.boss`. `round_count` maps `G.GAME.round` (not a career stat — it's a per-run counter reset on new run).

### Reward System (`ai/environment/reward.py`)
Four components (summed every step):

**Inventory reward** (any state, potential-based so buy→sell nets 0):
- Φ = Σ_jokers (4 + 1.5 × min(rounds_held, 6)) + 1.5 × consumables owned; reward = Φ(s') − Φ(s)
- `rounds_held` = blinds cleared while holding that joker, tracked per `sort_id` in `BalatroEnv._annotate_joker_tenure` and injected into each joker dict; selling refunds all tenure reward
- Selling a joker with `rounds_held == 0` (any state): −1 per joker
- Selling during SELECTING_HAND: −2 (except a consumable sell when consumable slots are full)
- Savings on leaving the shop: +1 per $5 interest tier held (max +5 at $25)

**Play reward** (any state, clipped to [−25, +120]):
- Chip gain: up to +10
- Blind clear: +(15 + 5 × ante); one-hand bonuses (+10/+6/+3); deck thinning +1 per card removed
- Ante progression: +5 × ante; boss blind: +5 × ante
- Money earned (non-sell sources): +0.5 × gain; consumable use: +1.5 (= consumable potential, so acquire→use nets 0)
- Game over: −15 + up to +8 progress credit

**Shop reward** (tanh-scaled to ±5):
- Money spent: −0.3 × log1p(spent); buy-and-use consumable: +2; voucher: +8
- Leave shop: +1 × joker slot utilization; −3 if broke with no jokers

**Pack reward**: planet pick +4, tarot/spectral kept +3, standard pick +1 (buffoon picks rewarded via joker potential only)

Step total clipped to [−30, +120]. No `VecNormalize`; SB3 normalizes advantages per batch.

### TensorBoard Metrics (`ai/utils/callbacks.py`)
Everything is recorded once per rollout (policy update). `<metric>` = all-time in this training process; `<metric>_rollout` = episodes that ended during the last rollout. Rates/ratios are ratio-of-sums (`Σnum / max(1, Σden)`), `mean_*_rollout` are per-episode averages. No per-episode metrics. Categories are kept ≤12 cards; per-object categories are exempt.

| Category | Contents |
|---|---|
| `progress/` | episodes, ante_8_boss_wins, win_rate, best/mean ante + round |
| `ante_reached/` | episodes reaching ante 2–8 |
| `blinds/` | clears, boss clears, skips, clear/boss/skip rates, mean_hands_per_clear, deadliest_death_rate |
| `first_blind/` | ante 1 round 1 clears and one-shots (+ rates) |
| `scoring/`, `hand_play/`, `money/`, `shop/` | scores; plays/discards/deck size; gold earned/spent/held; visits, rerolls, voucher purchase rate |
| `boosters/`, `tags/` | packs bought/opened/free/picked/skipped + buy:skip, pick:skip; tags acquired by source (skip/double/anaglyph/diet_cola) |
| `jokers/`, `joker_tenure/`, `consumables/` | gained/purchased/sold/end count + buy:sell, gain:sell; rounds held; used/failed/sold + rates |
| Per-object | `blind_by_name/`, `voucher_by_name/`, `booster_by_type/`, `tag_by_name/`, `joker_by_name/`, `hand_<type>/`, `consumable_<type>/` |

Tag and booster counts come from `RLBridge/stats.lua` (read-only wrappers on `add_tag`, `skip_blind`, `Card:open`, `use_card`, `skip_booster`), exported in `run_stats.tags` / `run_stats.packs`. The deadliest blind's name is printed in the policy-update CLI line.

## Communication with Claude
- Please keep chats as efficient as possible and brief. No fluff talk, just get to the point
- Please do NOT remove existing comments from the code
- Try to break things down in first principles effectively
- Do not add explanatory comments about why values changed (e.g., "# Changed from X to Y because...") - just use the correct values
- Comments should explain what something does, not why it was modified
- When in plan mode, do not edit files. just write a plan