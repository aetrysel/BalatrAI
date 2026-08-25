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
        choice = input("Select deck [1]: ").strip()
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


def write_run_config(seed: str | None, deck: str) -> None:
    """Write seed and deck to a config file that the Lua mod reads for auto-skip START_RUN."""
    config = {"seed": seed, "deck": deck}
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
        checkpoints = sorted(
            run_dir.glob("balatro_model_*_steps.zip"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if checkpoints:
            runs_with_checkpoints.append((run_dir, checkpoints[0]))

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

def mask_fn(env):
    """Extract action mask from the environment's action_masks() method"""
    return env.action_masks()


def create_environment(seed=None, deck="Red Deck", is_endless=False):
    """Create and wrap the Balatro environment"""
    env = BalatroEnv(seed=seed, deck=deck, is_endless=is_endless)
    env = ActionMasker(env, mask_fn)
    env = Monitor(env, filename="training_monitor.csv")
    return env


def create_model(env, model_path=None):
    """
    Create MaskablePPO model for training

    Args:
        env: Balatro environment
        model_path: Path to load existing model (optional)

    Returns:
        MaskablePPO model ready for training
    """
    model = MaskablePPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        #CHANGE BASED ON HOW MANY TRAINING STEPS
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.01,
        tensorboard_log="./tensorboard_logs/"
    )

    if model_path and Path(model_path).exists():
        model = MaskablePPO.load(model_path, env=env, tensorboard_log="./tensorboard_logs/")
        print(f"Loaded existing model from {model_path}")

    return model


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


def train_agent(total_timesteps=100000, save_path="./models/balatro_final", resume_from=None, checkpoint_dir="./models/", seed=None, deck="Red Deck", is_endless=False):
    """
    Main training function

    Args:
        total_timesteps: Number of training steps
        save_path: Where to save final model
        resume_from: Path to checkpoint to resume from
        checkpoint_dir: Directory to save checkpoints
        seed: Seed string or None for random
        deck: Deck name string
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Balatro RL training with MaskablePPO")
    logger.info(f"Training for {total_timesteps} timesteps")

    try:
        env = create_environment(seed=seed, deck=deck, is_endless=is_endless)

        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming training from: {resume_from}")
            model = MaskablePPO.load(resume_from, env=env, tensorboard_log="./tensorboard_logs/")
        else:
            logger.info("Starting training from scratch")
            model = create_model(env)

        callbacks = create_callbacks(save_freq=max(1000, total_timesteps // 20), checkpoint_dir=checkpoint_dir)

        logger.info("Starting training...")
        start_time = time.time()

        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            tb_log_name="Claude_skip_blinds",
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

    Path("./models").mkdir(exist_ok=True)
    Path("./tensorboard_logs").mkdir(exist_ok=True)

    # --- Seed selection ---
    seeds = load_seeds()
    seed_choice, seeds = select_seed(seeds)
    print(f"Using seed: {seed_choice if seed_choice else '(none — game picks randomly)'}")

    # --- Deck selection ---
    deck_choice = select_deck()
    print(f"Using deck: {deck_choice}")

    # --- Training steps ---
    training_steps = select_training_steps()
    print(f"Training steps: {training_steps}")

    # --- Endless mode ---
    endless_raw = input("Endless run? (y/N): ").strip().lower()
    is_endless = (endless_raw == 'y')
    print(f"Endless mode: {'ON' if is_endless else 'OFF (stops at ante 8 boss)'}")

    # Write config for the Lua mod auto-skip
    write_run_config(seed_choice, deck_choice)

    # Train the agent
    print("\n🎮 Starting Balatro RL Training!")
    print("Setup steps:")
    print("1. ✅ Balatro is running with RLBridge mod")
    print("2. ✅ Balatro is in menu state")

    input("Press Enter to start training then press 'R' in Balatro)...")

    try:
        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)

        # Check if any runs with checkpoints exist
        has_runs = any(
            list(d.glob("balatro_model_*_steps.zip"))
            for d in models_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ) if models_dir.exists() else False

        resume_from = None
        checkpoint_dir = None

        if has_runs:
            resume_choice = input("Resume from a previous run? (y/N): ").strip().lower()
            if resume_choice == "y":
                ckpt, run_dir = pick_run_to_resume(models_dir)
                if ckpt:
                    resume_from = str(ckpt)
                    checkpoint_dir = str(run_dir) + "/"
                    print(f"Resuming {run_dir.name} from {ckpt.name}")
                else:
                    print("No run selected, starting fresh.")

        if checkpoint_dir is None:
            run_num = get_next_run_number(models_dir)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = f"./models/run_{run_num}_{run_id}/"
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            print(f"Starting fresh. Checkpoints → {checkpoint_dir}")

        model = train_agent(
            total_timesteps=training_steps,
            save_path=f"{checkpoint_dir}balatro_trained",
            resume_from=resume_from,
            checkpoint_dir=checkpoint_dir,
            seed=seed_choice,
            deck=deck_choice,
            is_endless=is_endless,
        )

        if model:
            print("\n🎉Training completed successfully! Ready for next training session.")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Check the logs for more details.")
    finally:
        # Pipes will be cleaned up by the environment
        print("🧹 Training session ended")