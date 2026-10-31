"""
MaskablePPO with asynchronous per-env rollout collection

Standard collection steps all envs together, so every step waits for the slowest
game. Here each env is acted on as soon as its previous step returns; the only
synchronization point is the end of a rollout, where an env that has filled its
n_steps quota parks until the others finish theirs, then the policy updates.

The rollout buffer keeps SB3's [n_steps, n_envs] layout (each column is one env's
contiguous trajectory), so GAE and the PPO update are unchanged.
"""

import time

import numpy as np
import torch as th

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

from .utils.async_vec_env import AsyncSubprocVecEnv


def _select_env(callback: BaseCallback, env_idx: int) -> None:
    """Tell every callback that supports it which env the next transition belongs to."""
    for cb in getattr(callback, "callbacks", [callback]):
        if hasattr(cb, "callbacks"):
            _select_env(cb, env_idx)
        elif hasattr(cb, "select_env"):
            cb.select_env(env_idx)


class AsyncMaskablePPO(MaskablePPO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._async_masks = None
        self.instance_restarts = 0

    def resize_rollout_buffer(self, n_steps: int) -> None:
        """Rebuild the rollout buffer for a new per-env n_steps (e.g. after loading a checkpoint)."""
        self.n_steps = n_steps
        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps, use_masking=True) -> bool:  # noqa: C901
        assert isinstance(env, AsyncSubprocVecEnv), "AsyncMaskablePPO needs an AsyncSubprocVecEnv"
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()

        n_envs = env.num_envs
        if self._async_masks is None:
            self._async_masks = np.stack(env.env_method("action_masks")).astype(bool)

        counts = np.zeros(n_envs, dtype=int)
        ready = list(range(n_envs))  # envs holding an observation that needs an action
        in_flight = set()
        step_secs = [[] for _ in range(n_envs)]
        finished_at = np.zeros(n_envs)
        rollout_start = time.monotonic()
        aborted = False

        while True:
            act_ids = [i for i in ready if counts[i] < n_rollout_steps]
            ready = [i for i in ready if counts[i] >= n_rollout_steps]
            if act_ids and not aborted:
                obs = self._last_obs[act_ids]
                masks = self._async_masks[act_ids]
                with th.no_grad():
                    actions, values, log_probs = self.policy(
                        obs_as_tensor(obs, self.device), action_masks=masks if use_masking else None
                    )
                actions = actions.cpu().numpy()
                values = values.cpu().numpy().flatten()
                log_probs = log_probs.cpu().numpy()
                for k, i in enumerate(act_ids):
                    t = counts[i]
                    rollout_buffer.observations[t, i] = self._last_obs[i]
                    rollout_buffer.actions[t, i] = actions[k]
                    rollout_buffer.episode_starts[t, i] = self._last_episode_starts[i]
                    rollout_buffer.values[t, i] = values[k]
                    rollout_buffer.log_probs[t, i] = log_probs[k]
                    rollout_buffer.action_masks[t, i] = masks[k]
                    env.send_step(i, actions[k])
                    in_flight.add(i)

            if not in_flight:
                break

            for i in env.wait_ready(in_flight):
                in_flight.discard(i)
                new_obs, reward, done, info, mask, secs = env.recv_step(i)
                step_secs[i].append(secs)
                if info.get("instance_lost"):
                    self.instance_restarts += 1

                # Bootstrap truncated episodes (timeouts / lost instances) with the value of the final obs
                if done and info.get("terminal_observation") is not None and info.get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(info["terminal_observation"])[0]
                    with th.no_grad():
                        reward += self.gamma * self.policy.predict_values(terminal_obs)[0].item()

                rollout_buffer.rewards[counts[i], i] = reward
                counts[i] += 1
                if counts[i] == n_rollout_steps:
                    finished_at[i] = time.monotonic()
                self._last_obs[i] = new_obs
                self._last_episode_starts[i] = done
                self._async_masks[i] = mask
                self.num_timesteps += 1

                dones = np.array([done])
                infos = [info]
                _select_env(callback, i)
                callback.update_locals(locals())
                if not callback.on_step():
                    aborted = True
                self._update_info_buffer(infos, dones)
                ready.append(i)

        if aborted:
            return False

        rollout_buffer.pos = rollout_buffer.buffer_size
        rollout_buffer.full = True
        with th.no_grad():
            last_values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, dones=self._last_episode_starts)

        self._record_parallel_stats(step_secs, finished_at, rollout_start)
        callback.on_rollout_end()
        return True

    def _record_parallel_stats(self, step_secs, finished_at, rollout_start) -> None:
        """TensorBoard: throughput, per-worker step time, and how long fast workers idled at the rollout barrier."""
        rollout_end = finished_at.max()
        wall = max(rollout_end - rollout_start, 1e-6)
        n_envs = len(step_secs)
        total_steps = sum(len(s) for s in step_secs)
        idle = [(rollout_end - f) for f in finished_at]
        self.logger.record("parallel/steps_per_sec", total_steps / wall)
        self.logger.record("parallel/barrier_idle_frac", float(np.mean(idle)) / wall)
        self.logger.record("parallel/instance_restarts", self.instance_restarts)
        for i in range(n_envs):
            if step_secs[i]:
                self.logger.record(f"parallel/w{i}_step_ms", 1000 * float(np.mean(step_secs[i])))
