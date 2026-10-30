#!/usr/bin/env python3
"""
Balatro RL Training Script

Main training script for teaching AI to play Balatro using Stable-Baselines3.
This script creates the Balatro environment, sets up the RL model, and runs training.

Usage:
    python train_balatro.py

Requirements:
    - Balatro game running with RLBridge mod
    - file_watcher.py should NOT be running (this replaces it)
"""
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# SB3 imports
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure as configure_sb3_logger
from stable_baselines3.common.monitor import Monitor

# SB3 Contrib for action masking
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Our custom environment
from .environment.balatro_env import BalatroEnv
from .utils.callbacks import BalatroMetricsCallback
from .utils.mappers import DECK_NAMES

DEFAULT_TRAINING_STEPS = 10000
SEEDS_CONFIG_PATH = Path(__file__).parent / "config" / "seeds.json"
LAST_SESSION_PATH = Path(__file__).parent / "config" / "last_session.json"
MODELS_DIR = Path("./models")
LOGS_ROOT = Path("./logs")


# --- Seed persistence ---

def load_seeds() -> list:
    """Load named seeds from config JSON. Returns list of {name, value} dicts."""
    if SEEDS_CONFIG_PATH.exists():
        with open(SEEDS_CONFIG_PATH, "r") as f:
            data = json.load(f)
        return data.get("seeds", [])
    return []


def save_seeds(seeds: list) -> None:
    """Persist named seeds list to config JSON."""
    SEEDS_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SEEDS_CONFIG_PATH, "w") as f:
        json.dump({"seeds": seeds}, f, indent=2)


# --- Interactive menus ---

def select_seed(seeds: list) -> tuple[str | None, list]:
    """
    Prompt user to pick a seed. Returns (seed_value_or_None, updated_seeds).
    Supports add / delete sub-menus.
    """
    while True:
        print("\nSeed selection:")
        for i, s in enumerate(seeds):
            print(f"  [{i + 1}] {s['name']}  ({s['value']})")
        print(f"  [A] Add new seed")
        if seeds:
            print(f"  [D] Delete a seed")
        print(f"  [R] Random (no seed)")

        choice = input("Select seed: ").strip().upper()

        if choice == "R":
            return None, seeds

        if choice == "A":
            name = input("  Seed name: ").strip()
            value = input("  Seed value: ").strip().upper()
            if name and value:
                seeds.append({"name": name, "value": value})
                save_seeds(seeds)
                print(f"  Saved '{name}' = {value}")
            else:
                print("  Name and value required — skipped.")
            continue

        if choice == "D" and seeds:
            print("\n  Delete which seed?")
            for i, s in enumerate(seeds):
                print(f"    [{i + 1}] {s['name']}  ({s['value']})")
            print(f"    [0] Cancel")
            sub = input("  Select: ").strip()
            if sub == "0":
                continue
            try:
                idx = int(sub) - 1
                if 0 <= idx < len(seeds):
                    removed = seeds.pop(idx)
                    save_seeds(seeds)
                    print(f"  Deleted '{removed['name']}'.")
            except ValueError:
                pass
            continue

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(seeds):
                return seeds[idx]["value"], seeds
        except ValueError:
            pass

        print(f"  Invalid choice.")


def select_deck() -> str:
    """Prompt user to pick a deck. Returns deck name string."""
    print("\nDeck selection:")
    for i, name in enumerate(DECK_NAMES):
        print(f"  [{i + 1}] {name}")
    while True:
        choice = input("Select deck (Default [1] Red Deck): ").strip()
        if choice == "":
            return DECK_NAMES[0]
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(DECK_NAMES):
                return DECK_NAMES[idx]
        except ValueError:
            pass
        print(f"  Enter a number between 1 and {len(DECK_NAMES)}.")


def select_training_steps() -> int:
    """Prompt user to enter training steps. Returns int."""
    while True:
        raw = input(f"Training steps [{DEFAULT_TRAINING_STEPS}]: ").strip()
        if raw == "":
            return DEFAULT_TRAINING_STEPS
        try:
            val = int(raw)
            if val > 0:
                return val
        except ValueError:
            pass
        print("  Enter a positive integer.")




def select_debug_mode() -> bool:
    """Prompt for debug mode (faster policy updates). Returns True if debug."""
    choice = input("Debug mode — faster policy updates for testing? (y/N): ").strip().lower()
    return choice == "y"


def select_endless_mode() -> bool:
    """Prompt for endless mode (keep playing past the ante 8 boss). Returns True if endless."""
    choice = input("Endless run? (y/N): ").strip().lower()
    return choice == "y"


def write_run_config(seed: str | None, deck: str, endless: bool) -> None:
    """Write seed, deck and endless flag to a config file that the Lua mod reads for auto-skip START_RUN."""
    config = {"seed": seed, "deck": deck, "endless": endless}
    if os.name == "nt":
        appdata = os.getenv("APPDATA", "")
        path = Path(appdata) / "Balatro" / "Mods" / "RLBridge" / "run_config.json"
    else:
        path = Path("/tmp/balatro_run_config.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f)


# --- Run directory helpers (unchanged) ---

def get_next_run_number(models_dir: Path) -> int:
    """Scan existing run_N_* directories and return the next run number."""
    existing = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    nums = []
    for d in existing:
        parts = d.name.split("_")
        if len(parts) >= 2:
            try:
                nums.append(int(parts[1]))
            except ValueError:
                pass
    return max(nums, default=0) + 1


def run_number(run_dir_name: str) -> str:
    """'run_29_20261029_125706' → '29'."""
    parts = run_dir_name.split("_")
    return parts[1] if len(parts) >= 2 else "0"


def latest_checkpoint(run_dir: Path) -> Path | None:
    """Most recently written checkpoint in a run directory, or None."""
    checkpoints = sorted(
        run_dir.glob("balatro_model_*_steps.zip"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return checkpoints[0] if checkpoints else None


def pick_run_to_resume(models_dir: Path):
    """
    List all run directories that contain at least one checkpoint,
    let the user pick one, and return (checkpoint_path, run_dir).
    Returns (None, None) if the user cancels.
    """
    run_dirs = sorted(
        [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("run_")],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )

    runs_with_checkpoints = []
    for run_dir in run_dirs:
        ckpt = latest_checkpoint(run_dir)
        if ckpt:
            runs_with_checkpoints.append((run_dir, ckpt))

    if not runs_with_checkpoints:
        return None, None

    print("\n📂 Available runs:")
    for i, (run_dir, ckpt) in enumerate(runs_with_checkpoints):
        print(f"  [{i + 1}] {run_dir.name}  (latest: {ckpt.name})")
    print(f"  [0] Cancel — start fresh instead")

    while True:
        choice = input("Select run to resume: ").strip()
        if choice == "0":
            return None, None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(runs_with_checkpoints):
                run_dir, ckpt = runs_with_checkpoints[idx]
                return ckpt, run_dir
        except ValueError:
            pass
        print(f"  Enter a number between 0 and {len(runs_with_checkpoints)}")


def select_resume_run(models_dir: Path) -> str | None:
    """Ask whether to resume; returns the chosen models/run_* dir name, or None for a fresh run."""
    has_runs = any(latest_checkpoint(d) for d in models_dir.iterdir()
                   if d.is_dir() and d.name.startswith("run_"))
    if not has_runs:
        print("No checkpoints found — starting a fresh run.")
        return None
    if input("Resume from previous training? (y/N): ").strip().lower() != "y":
        return None
    _, run_dir = pick_run_to_resume(models_dir)
    if run_dir is None:
        print("No run selected, starting fresh.")
        return None
    return run_dir.name


# --- Log groups: logs/NNN_name/{tensorboard,history}/ ---

def list_log_groups(logs_root: Path) -> list[Path]:
    """Existing NNN_name log groups, oldest first."""
    if not logs_root.exists():
        return []
    return sorted(d for d in logs_root.iterdir() if d.is_dir() and d.name[:3].isdigit())


def create_log_group(logs_root: Path, name: str) -> Path:
    """Create logs/NNN_name with tensorboard/ and history/ subdirs, numbered after the latest group."""
    nums = [int(d.name[:3]) for d in list_log_groups(logs_root)]
    safe = "_".join("".join(c if c.isalnum() or c in "-_" else " " for c in name).split()) or "logs"
    group = logs_root / f"{max(nums, default=0) + 1:03d}_{safe}"
    (group / "tensorboard").mkdir(parents=True, exist_ok=True)
    (group / "history").mkdir(parents=True, exist_ok=True)
    return group


def select_log_group(logs_root: Path) -> str:
    """Pick an existing log group (default: latest) or create a new one. Returns the group dir name."""
    groups = list_log_groups(logs_root)
    print("\nLog group (holds TensorBoard + history logs for one feature change):")
    if not groups:
        name = input("  No log groups yet — new log group name: ").strip()
        group = create_log_group(logs_root, name)
        print(f"  Created {group}")
        return group.name
    for i, g in enumerate(groups):
        print(f"  [{i + 1}] {g.name}")
    print(f"  [N] New log group")
    default = groups[-1].name if groups else None
    while True:
        prompt = f"Select log group (default: {default}): " if default else "Select log group: "
        choice = input(prompt).strip().upper()
        if choice == "" and default:
            return default
        if choice == "N" or choice == "":
            name = input("  New log group name: ").strip()
            group = create_log_group(logs_root, name)
            print(f"  Created {group}")
            return group.name
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(groups):
                return groups[idx].name
        except ValueError:
            pass
        print("  Invalid choice.")


def list_sessions(group_dir: Path, run_num: str) -> list[str]:
    """TensorBoard sessions in a log group that belong to models/run_<run_num>_*, oldest first."""
    tb_dir = group_dir / "tensorboard"
    if not tb_dir.exists():
        return []
    return sorted(d.name for d in tb_dir.iterdir() if d.is_dir() and d.name.endswith(f"_run_{run_num}"))


def select_session(group_dir: Path, run_num: str) -> str | None:
    """Pick an existing session of this run to append to (default: latest), or None for a new session."""
    sessions = list_sessions(group_dir, run_num)
    if not sessions:
        return None
    print(f"\nTensorBoard session for run_{run_num} in {group_dir.name}:")
    for i, s in enumerate(sessions):
        print(f"  [{i + 1}] Append to {s}")
    print(f"  [N] New session")
    while True:
        choice = input(f"Select session (default: append to {sessions[-1]}): ").strip().upper()
        if choice == "":
            return sessions[-1]
        if choice == "N":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return sessions[idx]
        except ValueError:
            pass
        print("  Invalid choice.")


# --- Settings menu (persisted between launches) ---

SETTING_LABELS = [
    ("seed",           "Seed"),
    ("deck",           "Deck"),
    ("training_steps", "Training steps"),
    ("endless",        "Endless"),
    ("debug",          "Debug mode"),
    ("resume_run",     "Model"),
    ("log_group",      "Log group"),
    ("tb_session",     "TB session"),
]


def load_last_settings() -> dict | None:
    if LAST_SESSION_PATH.exists():
        try:
            with open(LAST_SESSION_PATH, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None
    return None


def save_last_settings(settings: dict) -> None:
    LAST_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_SESSION_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def validate_settings(settings: dict) -> dict:
    """Drop saved values that no longer point at something on disk."""
    run = settings.get("resume_run")
    if run and not latest_checkpoint(MODELS_DIR / run):
        print(f"⚠️  Saved run '{run}' has no checkpoints — switching to a fresh run.")
        settings["resume_run"] = None
    group = settings.get("log_group")
    if group and not (LOGS_ROOT / group).is_dir():
        print(f"⚠️  Saved log group '{group}' not found.")
        settings["log_group"] = None
    session = settings.get("tb_session")
    if session and (not settings.get("resume_run") or not settings.get("log_group")
                    or session not in list_sessions(LOGS_ROOT / settings["log_group"],
                                                    run_number(settings["resume_run"]))):
        settings["tb_session"] = None
    return settings


def format_setting(key: str, settings: dict, seeds: list) -> str:
    value = settings.get(key)
    if key == "seed":
        if not value:
            return "random"
        name = next((s["name"] for s in seeds if s["value"] == value), None)
        return f"{name} ({value})" if name else value
    if key in ("endless", "debug"):
        return "ON" if value else "OFF"
    if key == "resume_run":
        if not value:
            return "new run"
        ckpt = latest_checkpoint(MODELS_DIR / value)
        return f"resume {value} ({ckpt.name if ckpt else 'no checkpoint'})"
    if key == "tb_session":
        if not settings.get("resume_run"):
            return "new session"
        return f"append to {value}" if value else "new session"
    return str(value)


def edit_setting(key: str, settings: dict, seeds: list) -> list:
    """Prompt for one setting. Returns the (possibly updated) seeds list."""
    if key == "seed":
        settings["seed"], seeds = select_seed(seeds)
    elif key == "deck":
        settings["deck"] = select_deck()
    elif key == "training_steps":
        settings["training_steps"] = select_training_steps()
    elif key == "endless":
        settings["endless"] = select_endless_mode()
    elif key == "debug":
        settings["debug"] = select_debug_mode()
    elif key == "resume_run":
        settings["resume_run"] = select_resume_run(MODELS_DIR)
        settings["tb_session"] = None
        if settings["resume_run"] and settings.get("log_group"):
            edit_setting("tb_session", settings, seeds)
    elif key == "log_group":
        settings["log_group"] = select_log_group(LOGS_ROOT)
        settings["tb_session"] = None
        if settings.get("resume_run"):
            edit_setting("tb_session", settings, seeds)
    elif key == "tb_session":
        if not settings.get("resume_run"):
            print("  A new run always starts a new TensorBoard session.")
            settings["tb_session"] = None
        else:
            settings["tb_session"] = select_session(LOGS_ROOT / settings["log_group"],
                                                    run_number(settings["resume_run"]))
    return seeds


def configure_session(seeds: list) -> tuple[dict, list]:
    """
    Load last-used settings and show an edit menu; on first launch, prompt for everything.
    Returns (settings, seeds).
    """
    defaults = {"seed": None, "deck": DECK_NAMES[0], "training_steps": DEFAULT_TRAINING_STEPS,
                "endless": False, "debug": False, "resume_run": None, "log_group": None, "tb_session": None}
    saved = load_last_settings()
    if saved is None:
        settings = dict(defaults)
        for key, _ in SETTING_LABELS:
            seeds = edit_setting(key, settings, seeds)
    else:
        settings = validate_settings({**defaults, **saved})
        if not settings["log_group"]:
            seeds = edit_setting("log_group", settings, seeds)

    while True:
        print("\n⚙️  Training settings:")
        for i, (key, label) in enumerate(SETTING_LABELS):
            print(f"  [{i + 1}] {label:<15} {format_setting(key, settings, seeds)}")
        choice = input("Press Enter to accept, or enter option numbers to change (e.g. '1 3'): ").strip()
        if choice == "":
            return settings, seeds
        for token in choice.replace(",", " ").split():
            try:
                idx = int(token) - 1
            except ValueError:
                idx = -1
            if 0 <= idx < len(SETTING_LABELS):
                seeds = edit_setting(SETTING_LABELS[idx][0], settings, seeds)
            else:
                print(f"  Ignoring '{token}' — enter numbers 1-{len(SETTING_LABELS)}.")


# --- Logging ---

def setup_logging():
    """Setup logging for training"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
        ]
    )


# --- Environment and model setup ---


# --- Environment and model setup ---

def mask_fn(env):
    """Extract action mask from the environment's action_masks() method"""
    return env.action_masks()


def create_environment(seed=None, deck="Red Deck", is_endless=False, run_label: str = "", run_dir: str = "",
                       history_dir: str = "game_history_logs", history_append: bool = False):
    """Create and wrap the Balatro environment"""
    env = BalatroEnv(seed=seed, deck=deck, is_endless=is_endless, run_label=run_label, run_dir=run_dir,
                     history_dir=history_dir, history_append=history_append)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env, filename="training_monitor.csv")
    return env


def create_model(env, model_path=None, n_steps=4096):
    """
    Create MaskablePPO model for training

    Args:
        env: Balatro environment
        model_path: Path to load existing model (optional)
        n_steps: Steps per rollout before policy update (128 for debug, 4096 for training)

    Returns:
        MaskablePPO model ready for training
    """
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
    )

    if model_path and Path(model_path).exists():
        model = load_model(model_path, env, n_steps)
        print(f"Loaded existing model from {model_path}")

    return model


def load_model(model_path, env, n_steps):
    """Load a checkpoint; n_steps goes through custom_objects so the rollout buffer is rebuilt at that size."""
    return MaskablePPO.load(model_path, env=env, custom_objects={"n_steps": n_steps})


def create_callbacks(save_freq=1000, checkpoint_dir="./models/"):
    """Create training callbacks for saving and evaluation"""
    callbacks = []

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="balatro_model"
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(BalatroMetricsCallback(log_freq=10, verbose=1))

    return callbacks


def train_agent(total_timesteps=100000, save_path="./models/balatro_final", resume_from=None, checkpoint_dir="./models/", seed=None, deck="Red Deck", is_endless=False, run_label: str = "", n_steps: int = 4096, tb_session_dir: str = "./tensorboard_logs/MaskablePPO", run_dir: str = "", history_dir: str = "game_history_logs", history_append: bool = False):
    """
    Main training function

    Args:
        total_timesteps: Number of training steps
        save_path: Where to save final model
        resume_from: Path to checkpoint to resume from
        checkpoint_dir: Directory to save checkpoints
        seed: Seed string or None for random
        deck: Deck name string
        tb_session_dir: Exact TensorBoard run directory (appended to if it already exists)
        history_dir: Directory for the history log of this session
        history_append: Append to an existing history log instead of overwriting
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Balatro RL training with MaskablePPO")
    logger.info(f"Training for {total_timesteps} timesteps")

    try:
        env = create_environment(seed=seed, deck=deck, is_endless=is_endless, run_label=run_label, run_dir=run_dir,
                                 history_dir=history_dir, history_append=history_append)

        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming training from: {resume_from}")
            model = load_model(resume_from, env, n_steps)
        else:
            logger.info("Starting training from scratch")
            model = create_model(env, n_steps=n_steps)

        # Exact TB directory (no SB3 _N suffix) so it pairs with the history log of the same name
        model.set_logger(configure_sb3_logger(tb_session_dir, ["tensorboard"]))

        callbacks = create_callbacks(save_freq=max(1000, total_timesteps // 20), checkpoint_dir=checkpoint_dir)

        logger.info("Starting training...")
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not resume_from,
        )

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")

        model.save(save_path)
        logger.info(f"Model saved to {save_path}")

        if hasattr(env, 'cleanup'):
            env.cleanup()
        elif hasattr(env.env, 'cleanup'):  # Monitor wrapper
            env.env.cleanup()

        return model

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if hasattr(env, 'cleanup'):
            env.cleanup()
        elif hasattr(env.env, 'cleanup'):
            env.env.cleanup()
        return None
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if hasattr(env, 'cleanup'):
            env.cleanup()
        elif hasattr(env.env, 'cleanup'):
            env.env.cleanup()
        raise


def test_trained_model(model_path, num_episodes=5):
    """
    Test a trained model

    Args:
        model_path: Path to trained model
        num_episodes: Number of episodes to test
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Testing model from {model_path}")

    env = create_environment()
    model = MaskablePPO.load(model_path)

    episode_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            if done:
                break

        episode_rewards.append(total_reward)
        logger.info(f"Episode {episode + 1}: {steps} steps, reward: {total_reward:.2f}")

    avg_reward = sum(episode_rewards) / len(episode_rewards)
    logger.info(f"Average reward over {num_episodes} episodes: {avg_reward:.2f}")

    env.cleanup()
    return episode_rewards



if __name__ == "__main__":
    setup_logging()

    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_ROOT.mkdir(exist_ok=True)

    # --- All prompts: settings menu (seed, deck, steps, endless, debug, model, log group, TB session) ---
    seeds = load_seeds()
    settings, seeds = configure_session(seeds)
    save_last_settings(settings)

    seed_choice = settings["seed"]
    deck_choice = settings["deck"]
    training_steps = settings["training_steps"]
    is_endless = settings["endless"]
    n_steps = 128 if settings["debug"] else 4096

    # Write config for the Lua mod auto-skip
    write_run_config(seed_choice, deck_choice, is_endless)

    # --- Resolve model + log paths ---
    resume_from = None
    if settings["resume_run"]:
        run_dir = MODELS_DIR / settings["resume_run"]
        resume_from = str(latest_checkpoint(run_dir))
        checkpoint_dir = str(run_dir) + "/"
        run_num_str = run_number(run_dir.name)
    else:
        run_num_str = str(get_next_run_number(MODELS_DIR))
        checkpoint_dir = f"./models/run_{run_num_str}_{datetime.now().strftime('%Y%m%d_%H%M%S')}/"

    group_dir = LOGS_ROOT / settings["log_group"]
    history_append = settings["tb_session"] is not None
    # Session name: timestamp first so sessions sort chronologically; shared by TB dir and history log
    session_name = settings["tb_session"] or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_run_{run_num_str}"
    tb_session_dir = group_dir / "tensorboard" / session_name
    history_dir = group_dir / "history"

    print(f"\nUsing seed: {seed_choice if seed_choice else '(none — game picks randomly)'}")
    print(f"Using deck: {deck_choice}")
    print(f"Training steps: {training_steps}")
    print(f"Endless mode: {'ON' if is_endless else 'OFF (stops at ante 8 boss)'}")
    print(f"Policy update every: {n_steps} steps{'  (debug)' if settings['debug'] else ''}")
    if resume_from:
        print(f"Resuming {Path(checkpoint_dir).name} from {Path(resume_from).name}")
    else:
        print(f"Starting fresh. Checkpoints → {checkpoint_dir}")
    print(f"TensorBoard → {tb_session_dir}{'  (append)' if history_append else ''}")
    print(f"History log → {history_dir / f'history_{session_name}.log'}")

    # Train the agent
    print("\n🎮 Starting Balatro RL Training!")
    print("Setup steps:")
    print("1. ✅ Balatro is running with RLBridge mod")
    print("2. ✅ Balatro is in menu state")

    input("Press Enter to start training then press 'R' in Balatro)...")

    try:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        model = train_agent(
            total_timesteps=training_steps,
            save_path=f"{checkpoint_dir}balatro_trained",
            resume_from=resume_from,
            checkpoint_dir=checkpoint_dir,
            seed=seed_choice,
            deck=deck_choice,
            is_endless=is_endless,
            run_label=session_name,
            n_steps=n_steps,
            tb_session_dir=str(tb_session_dir),
            run_dir=checkpoint_dir,
            history_dir=str(history_dir),
            history_append=history_append,
        )

        if model:
            print("\n🎉Training completed successfully! Ready for next training session.")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Check the logs for more details.")
    finally:
        # Pipes will be cleaned up by the environment
        print("🧹 Training session ended")