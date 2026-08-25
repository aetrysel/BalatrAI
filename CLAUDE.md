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
│   ├── environment/
│   │   ├── balatro_env.py       # Gymnasium environment wrapper
│   │   └── reward.py            # Reward calculator
│   └── utils/
│       ├── communication.py     # Pipe/socket layer (Python side)
│       ├── mappers.py           # Game state → observation vector
│       ├── callbacks.py         # SB3 training callbacks, TensorBoard metrics
│       ├── validation.py        # JSON schema validators
│       ├── replay.py            # Saves top-10 winning episodes
│       └── debug.py             # Timestamped print helpers
├── balatro-source-code/ # Decompiled Balatro source — reference only, do not edit
├── models/             # Saved model checkpoints (run_N_YYYYMMDD_HHMMSS/)
└── tensorboard_logs/   # TensorBoard event files
```

### Lovely Patching System
The mod uses `RLBridge/lovely/` to define patches:
- `init.toml` — injects all 8 Lua modules, hooks `G:start_up()`, patches animation skipping
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

Interactive prompts will ask:
1. **Seed choice**: `0` = training seed (JFKGEEMG), `1` = test seed (FK76PMFU), `2` = random
2. **Resume or fresh**: pick an existing `models/run_N_*` directory or start a new run
3. Start Balatro, then press `R` in-game to activate the RL loop

Training saves checkpoints to `models/run_N_YYYYMMDD_HHMMSS/` and TensorBoard logs to `tensorboard_logs/`.

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
Shape `(313,)` float32 vector built by `ai/utils/mappers.py`:
- Round info: hands_left, discards_left (2)
- Blind: blind_chips, current_chips (2)
- Game state: one-hot (20) + game_over + game_win + retry_count (23)
- Joker slots: occupancy flags × 5 (5)
- Hand: size, highlighted_count + per-card features × 12 cards (254)
- Scoring: chips, mult, score + hand type one-hot × 13 (16)
- Available actions mask × 11 (11)

### Reward System (`ai/environment/reward.py`)
Two components:

**Play reward** (SELECTING_HAND):
- Chip gain: quadratic scaling up to +20
- Blind clear: +30 × 1.5^(ante-1); one-hand bonuses (+25/+15/+8)
- Ante progression: +10 × ante; boss blind: +15 × ante
- Game over: −20

**Shop reward** (SHOP):
- Joker purchase: +10; joker sell: −1
- Money earned: +0.5 × gain; joker slot utilization: +0.5/step
- Leave shop broke with no jokers: −3

Reward clipped to [−25, +500].

## Communication with Claude
- Please keep chats as efficient as possible and brief. No fluff talk, just get to the point
- Try to break things down in first principles effectively
- Do not add explanatory comments about why values changed (e.g., "# Changed from X to Y because...") - just use the correct values
- Comments should explain what something does, not why it was modified
- Please do NOT remove existing comments from the code