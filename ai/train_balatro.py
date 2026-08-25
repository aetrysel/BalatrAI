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
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from ai import global_var
import re

# SB3 imports
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

# SB3 Contrib for action masking
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# Our custom environment
from .environment.balatro_env import BalatroEnv
from .utils.callbacks import BalatroMetricsCallback

TRAINING_STEPS = 10000  # Total training steps // old default was 250000

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

    # Filter to only runs that actually have checkpoints
    runs_with_checkpoints = []
    for run_dir in run_dirs:
        checkpoints = sorted(
            run_dir.glob("balatro_model_*_steps.zip"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if checkpoints:
            runs_with_checkpoints.append((run_dir, checkpoints[0]))  # latest checkpoint per run

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

def update_seed_in_lua(filepath, new_seed):
    with open(filepath, "r") as f:
        content = f.read()
    
    updated = re.sub(
        r'(exec_params\.seed\s*=\s*")[^"]*(")',
        rf'\g<1>{new_seed}\g<2>',
        content
    )
    
    with open(filepath, "w") as f:
        f.write(updated)

def setup_logging():
    """Setup logging for training"""
    # env_logger = logging.getLogger('ai.environment.balatro_env')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),

        ]
    )
    # console = logging.StreamHandler()
    # console.setLevel(logging.WARNING)
    # logging.getLogger('').addHandler(console)


def mask_fn(env):
    """Extract action mask from the environment's action_masks() method"""
    return env.action_masks()

def create_environment():
    """Create and wrap the Balatro environment"""
    # Create base environment
    env = BalatroEnv()
    
    # Use ActionMasker wrapper
    env = ActionMasker(env, mask_fn)

    # Wrap with Monitor for logging episode stats
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
    
    # Load existing model if path provided
    if model_path and Path(model_path).exists():
        model = MaskablePPO.load(model_path, env=env, tensorboard_log="./tensorboard_logs/")
        print(f"Loaded existing model from {model_path}")
    
    return model


def create_callbacks(save_freq=1000, checkpoint_dir="./models/"):
    """Create training callbacks for saving and evaluation"""
    callbacks = []
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=checkpoint_dir,
        name_prefix="balatro_model"
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(BalatroMetricsCallback(log_freq=10,verbose=1))
    
    return callbacks


def train_agent(total_timesteps=100000, save_path="./models/balatro_final", resume_from=None, checkpoint_dir="./models/"):
    """
    Main training function
    
    Args:
        total_timesteps: Number of training steps
        save_path: Where to save final model
        resume_from: Path to checkpoint to resume from
        checkpoint_dir: Directory to save checkpoints
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting Balatro RL training with MaskablePPO")
    logger.info(f"Training for {total_timesteps} timesteps")
    
    try:
        # Create environment and model
        env = create_environment()
        
        if resume_from and Path(resume_from).exists():
            logger.info(f"Resuming training from: {resume_from}")
            model = MaskablePPO.load(resume_from, env=env, tensorboard_log="./tensorboard_logs/")
        else:
            logger.info("Starting training from scratch")
            model = create_model(env)
        
        # Create callbacks
        callbacks = create_callbacks(save_freq=max(1000, total_timesteps // 20), checkpoint_dir=checkpoint_dir)
        
        # Train the model
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
        
        # Save final model
        model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
        # Clean up environment  
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
    
    # Create environment and load model
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
    # Setup
    setup_logging()
    
    # Create necessary directories
    Path("./models").mkdir(exist_ok=True)
    Path("./tensorboard_logs").mkdir(exist_ok=True)

    # Query user for seed to use
    seed_choice_option = input("Input 0 to use the training seed, 1 for the testing seed, or 2 for random seed: ")
 
    # Determine actual seed string and export to environment for the game mod
    if seed_choice_option == "0":
        seed_choice = "JFKGEEMG"  # training seed
    elif seed_choice_option == "1":
        seed_choice = "FK76PMFU"  # testing seed (example)
    else:
        seed_choice = None
    global_var.choosen_seed = seed_choice
    print(f"Using seed: {seed_choice if seed_choice else '(none — game picks randomly)'}")

    if seed_choice:
        if os.name == 'nt':
            appdata_path = os.getenv('APPDATA')
            balatro_mod_path = Path(appdata_path) / "Balatro" / "Mods" / "RLBridge" / "ai.lua"
            update_seed_in_lua(balatro_mod_path, seed_choice)
        else:
            home = Path.home()
            balatro_mod_path = home / ".local" / "share" / "love" / "Mods" / "RLBridge" / "ai.lua"
            update_seed_in_lua(balatro_mod_path, seed_choice)

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
            total_timesteps=TRAINING_STEPS,
            save_path=f"{checkpoint_dir}balatro_trained",
            resume_from=resume_from,
            checkpoint_dir=checkpoint_dir,
        )
        
        if model:
            print("\n🎉Training completed successfully! Ready for next training session.")
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Check the logs for more details.")
    finally:
        # Pipes will be cleaned up by the environment
        print("🧹 Training session ended")