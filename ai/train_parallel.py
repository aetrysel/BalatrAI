#!/usr/bin/env python3
"""
Parallel Balatro RL training on headless instances

Launches N headless Balatro instances (one per worker subprocess) and trains a
single AsyncMaskablePPO policy on all of them. Each worker is stepped as soon as
its game is ready, so no game waits on the others except at the end of a rollout.

Usage:
    python -m ai.train_parallel --workers 4 --timesteps 1000000
    python -m ai.train_parallel --workers 4 --resume models/run_7_.../balatro_model_50000_steps.zip

Watch it:
    python -m ai.monitor                       # grid of every instance's screen + status
    python -m ai.watch_live --run models/run_N # one GUI instance playing the latest checkpoint
    tensorboard --logdir tensorboard_logs      # parallel/* scalars: throughput, idle, restarts
"""
import argparse
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from .async_ppo import AsyncMaskablePPO
from .environment.balatro_env import BalatroEnv
from .train_balatro import (
    create_callbacks, create_model, get_next_run_number, load_seeds, mask_fn, setup_logging, write_run_config,
)
from .utils.async_vec_env import AsyncSubprocVecEnv
from .utils.instances import BalatroInstance


def make_env(worker_id: int, args, run_label: str, run_dir: str):
    """Env factory run inside the worker subprocess: launches that worker's Balatro instance."""
    def _init():
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - w{worker_id} - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.FileHandler(Path(run_dir) / f"worker_{worker_id}.log")],
        )
        instance = BalatroInstance(
            name=f"w{worker_id}",
            worker_id=worker_id,
            headless=not args.gui,
            gamespeed=args.gamespeed,
            fps_cap=args.fps_cap,
            snapshot_secs=args.snapshot_secs,
        )
        env = BalatroEnv(
            seed=args.seed, deck=args.deck, is_endless=args.endless,
            run_label=run_label, run_dir=run_dir,
            worker_id=worker_id, instance=instance, read_timeout=args.read_timeout,
        )
        env = ActionMasker(env, mask_fn)
        env = Monitor(env, filename=str(Path(run_dir) / f"monitor_w{worker_id}"))
        return env
    return _init


def resolve_seed(seed_arg: str | None) -> str | None:
    """Accept a seed value or a name from ai/config/seeds.json; None = random."""
    if not seed_arg:
        return None
    for s in load_seeds():
        if s["name"].lower() == seed_arg.lower():
            return s["value"]
    return seed_arg.upper()


def parse_args():
    p = argparse.ArgumentParser(description="Parallel headless Balatro training")
    p.add_argument("--workers", type=int, default=4, help="number of Balatro instances")
    p.add_argument("--timesteps", type=int, default=1_000_000, help="total env steps across all workers")
    p.add_argument("--rollout", type=int, default=4096, help="total steps per policy update (split across workers)")
    p.add_argument("--seed", default=None, help="seed value or name from seeds.json (default: random)")
    p.add_argument("--deck", default="Red Deck")
    p.add_argument("--endless", action="store_true")
    p.add_argument("--resume", default=None, help="checkpoint .zip to resume from (its run dir is reused)")
    p.add_argument("--tb-name", default=None, help="TensorBoard run name (default: parallel_<N>w)")
    p.add_argument("--gui", action="store_true", help="show windows instead of running headless (debugging)")
    p.add_argument("--gamespeed", type=float, default=None, help="override G.SETTINGS.GAMESPEED (default: RLBridge's 100)")
    p.add_argument("--fps-cap", type=int, default=None, help="override G.FPS_CAP (default 500)")
    p.add_argument("--snapshot-secs", type=float, default=2.0, help="seconds between monitor snapshots")
    p.add_argument("--read-timeout", type=float, default=300.0, help="seconds without a request before an instance is treated as hung")
    return p.parse_args()


def main():
    args = parse_args()
    args.seed = resolve_seed(args.seed)
    setup_logging()
    logger = logging.getLogger(__name__)

    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    if args.resume:
        run_dir = Path(args.resume).parent
    else:
        run_dir = models_dir / f"run_{get_next_run_number(models_dir)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir.mkdir(parents=True)
    checkpoint_dir = str(run_dir) + "/"
    run_label = f"{run_dir.name.split('_')[0]}_{run_dir.name.split('_')[1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tb_name = args.tb_name or f"parallel_{args.workers}w"

    # Run config for the Lua START_RUN auto-skip, kept separate from a single-instance GUI session's /tmp file
    os.environ["BALATRO_RUN_CONFIG"] = str((run_dir / "run_config.json").resolve())
    write_run_config(args.seed, args.deck)

    n_steps = max(64, args.rollout // args.workers)
    print(f"🎮 Parallel training: {args.workers} {'GUI' if args.gui else 'headless'} instances")
    print(f"   seed={args.seed or 'random'} deck={args.deck} endless={args.endless}")
    print(f"   {n_steps} steps/worker per update ({n_steps * args.workers} total), {args.timesteps} timesteps")
    print(f"   run dir: {run_dir}   TensorBoard: {tb_name}")
    print(f"   monitor: python -m ai.monitor")

    venv = AsyncSubprocVecEnv([make_env(i, args, run_label, str(run_dir)) for i in range(args.workers)])
    model = None
    try:
        if args.resume:
            model = AsyncMaskablePPO.load(args.resume, env=venv, tensorboard_log="./tensorboard_logs/")
            model.resize_rollout_buffer(n_steps)
            print(f"Resumed from {args.resume}")
        else:
            model = create_model(venv, n_steps=n_steps, algo=AsyncMaskablePPO)

        callbacks = create_callbacks(save_freq=max(1000, args.timesteps // 20), checkpoint_dir=checkpoint_dir)
        start = time.time()
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=False,
            tb_log_name=tb_name,
            reset_num_timesteps=not args.resume,
        )
        logger.info(f"Training completed in {time.time() - start:.0f}s")
        model.save(f"{checkpoint_dir}balatro_trained")
        print(f"\n🎉 Training complete → {checkpoint_dir}balatro_trained.zip")
    except KeyboardInterrupt:
        if model is not None:
            print("\nInterrupted — saving checkpoint")
            model.save(f"{checkpoint_dir}balatro_interrupted")
    finally:
        try:
            venv.close()
        except Exception:
            pass
        print("🧹 Instances stopped")


if __name__ == "__main__":
    main()
