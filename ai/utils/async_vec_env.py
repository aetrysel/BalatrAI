"""
Async subprocess VecEnv

SubprocVecEnv steps every env in lockstep: step_wait() blocks until the slowest
env returns, so fast envs (and their games) sit idle every step. This variant
adds a per-env API (send_step / wait_ready / recv_step) so the trainer can
act on each env the moment its result arrives. The standard VecEnv API still works.
"""

import multiprocessing as mp
import time
from multiprocessing.connection import wait
from typing import Any, Callable, Iterable

import gymnasium as gym
import numpy as np

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def _async_worker(remote, parent_remote, env_fn_wrapper: CloudpickleWrapper) -> None:  # noqa: C901
    """SB3's subprocess worker plus a 'step_masked' command that also returns the next action mask."""
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = _patch_env(env_fn_wrapper.var())
    reset_info: dict[str, Any] | None = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd in ("step", "step_masked"):
                t0 = time.monotonic()
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                if cmd == "step":
                    remote.send((observation, reward, done, info, reset_info))
                else:
                    mask = env.get_wrapper_attr("action_masks")()
                    remote.send((observation, reward, done, info, mask, time.monotonic() - t0))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped":
                remote.send(is_wrapped(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except EOFError:
            env.close()
            break
        except KeyboardInterrupt:
            env.close()
            break


class AsyncSubprocVecEnv(SubprocVecEnv):
    """SubprocVecEnv whose envs can also be stepped independently."""

    def __init__(self, env_fns: list[Callable[[], gym.Env]], start_method: str | None = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        if start_method is None:
            start_method = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)], strict=True)
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns, strict=True):
            args = (work_remote, remote, CloudpickleWrapper(env_fn))
            process = ctx.Process(target=_async_worker, args=args, daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()

        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, n_envs, observation_space, action_space)

    # --- Per-env async API ---

    def send_step(self, env_idx: int, action: np.ndarray) -> None:
        self.remotes[env_idx].send(("step_masked", action))

    def wait_ready(self, env_indices: Iterable[int], timeout: float | None = None) -> list[int]:
        """Block until at least one of env_indices has a step result; return those that do."""
        by_remote = {self.remotes[i]: i for i in env_indices}
        return [by_remote[r] for r in wait(list(by_remote), timeout=timeout)]

    def recv_step(self, env_idx: int):
        """Returns (obs, reward, done, info, next_action_mask, step_seconds)."""
        return self.remotes[env_idx].recv()
