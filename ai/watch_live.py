#!/usr/bin/env python3
"""
Live viewer: one windowed Balatro instance playing the newest checkpoint at normal speed

Runs alongside train_parallel.py without feeding it data. Between episodes it
reloads the newest checkpoint in the run dir, so you watch the current agent.

Usage:
    python -m ai.watch_live --run models/run_N_YYYYMMDD_HHMMSS
    python -m ai.watch_live --run models/run_N_... --gamespeed 2 --deterministic
"""
import argparse
import logging
import os
from pathlib import Path

from sb3_contrib import MaskablePPO

from .environment.balatro_env import BalatroEnv
from .train_balatro import write_run_config
from .train_parallel import resolve_seed
from .utils.instances import BalatroInstance


def latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = list(run_dir.glob("*.zip"))
    return max(ckpts, key=lambda f: f.stat().st_mtime) if ckpts else None


def main():
    p = argparse.ArgumentParser(description="Watch the latest checkpoint play at normal speed")
    p.add_argument("--run", required=True, help="run dir to follow (models/run_N_...)")
    p.add_argument("--seed", default=None, help="seed value or name from seeds.json (default: random)")
    p.add_argument("--deck", default="Red Deck")
    p.add_argument("--gamespeed", type=float, default=1.0, help="G.SETTINGS.GAMESPEED (1 = normal)")
    p.add_argument("--deterministic", action="store_true", help="always take the most likely action")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - live - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler("watch_live.log")])
    run_dir = Path(args.run)
    seed = resolve_seed(args.seed)

    os.environ["BALATRO_RUN_CONFIG"] = str(Path("instances/live/run_config.json").resolve())
    Path("instances/live").mkdir(parents=True, exist_ok=True)
    write_run_config(seed, args.deck)

    instance = BalatroInstance(name="live", worker_id="live", headless=False,
                               gamespeed=args.gamespeed, animations=True)
    env = BalatroEnv(seed=seed, deck=args.deck, run_label="live", worker_id="live",
                     instance=instance, read_timeout=900.0)

    ckpt, model = None, None
    episode = 0
    try:
        while True:
            newest = latest_checkpoint(run_dir)
            if newest is None:
                raise SystemExit(f"No checkpoints in {run_dir} yet")
            if newest != ckpt:
                try:
                    model = MaskablePPO.load(newest, device="cpu")
                    ckpt = newest
                    print(f"🎬 Loaded {ckpt.name}")
                except Exception as e:  # checkpoint still being written
                    if model is None:
                        raise
                    logging.warning(f"Could not load {newest.name}, keeping {ckpt.name}: {e}")

            obs, _ = env.reset()
            episode += 1
            total, done, info = 0.0, False, {}
            while not done:
                action, _ = model.predict(obs, action_masks=env.action_masks(), deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                done = terminated or truncated
            result = "WON" if info.get("won") else f"died ante {info.get('ante', '?')} round {info.get('round', '?')}"
            print(f"Episode {episode} ({ckpt.name}): {result}, reward {total:.1f}")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
