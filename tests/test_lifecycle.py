"""Tests for the in-game lifecycle: session flag, SHUTDOWN, keybind, settings file, stop-key handling."""
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import gymnasium as gym
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.utils import communication, lifecycle  # noqa: E402
from ai.utils.communication import BalatroPipeIO  # noqa: E402


@pytest.fixture
def flag_path(tmp_path, monkeypatch):
    path = tmp_path / "session_ready"
    monkeypatch.setenv("BALATRO_SESSION_FLAG", str(path))
    return path


# --- Session-ready flag ---

def test_flag_written_with_pid_and_removed(flag_path):
    lifecycle.write_session_flag()
    assert flag_path.read_text() == str(os.getpid())
    lifecycle.remove_session_flag()
    assert not flag_path.exists()
    lifecycle.remove_session_flag()  # already gone: no error


def test_pipe_io_init_clears_stale_flag_then_writes_and_cleanup_removes(flag_path, monkeypatch):
    flag_path.write_text("999999")  # stale flag from a crashed session
    seen = {}
    monkeypatch.setattr(BalatroPipeIO, "create_pipes", lambda self: seen.setdefault("stale_gone", not flag_path.exists()))
    monkeypatch.setattr(BalatroPipeIO, "open_persistent_handles", lambda self: seen.setdefault("flag_before_open", flag_path.exists()))
    io = BalatroPipeIO()
    assert seen == {"stale_gone": True, "flag_before_open": True}
    assert flag_path.read_text() == str(os.getpid())
    monkeypatch.setattr(os.path, "exists", lambda p: False)  # keep cleanup away from the real /tmp FIFOs
    io.cleanup()
    assert not flag_path.exists()


def test_pipe_io_ctrl_c_while_waiting_removes_flag(flag_path, monkeypatch):
    def interrupted(self):
        raise KeyboardInterrupt
    monkeypatch.setattr(BalatroPipeIO, "create_pipes", lambda self: None)
    monkeypatch.setattr(BalatroPipeIO, "open_persistent_handles", interrupted)
    monkeypatch.setattr(os.path, "exists", lambda p: False)
    with pytest.raises(KeyboardInterrupt):
        BalatroPipeIO()
    assert not flag_path.exists()


# --- send_shutdown is best-effort ---

def _bare_pipe_io(monkeypatch):
    monkeypatch.setattr(communication, "SHUTDOWN_DRAIN_SECS", 0.01)
    io = BalatroPipeIO.__new__(BalatroPipeIO)
    io.logger = MagicMock()
    io.request_handle = None
    io.response_handle = None
    return io


def test_send_shutdown_delivers_line(monkeypatch):
    io = _bare_pipe_io(monkeypatch)
    r, w = os.pipe()
    io.response_handle = os.fdopen(w, "w")
    assert io.send_shutdown() is True
    assert json.loads(os.read(r, 100)) == {"action": "SHUTDOWN"}
    os.close(r)
    io.response_handle.close()


def test_send_shutdown_no_handle(monkeypatch):
    assert _bare_pipe_io(monkeypatch).send_shutdown() is False


def test_send_shutdown_closed_handle(monkeypatch):
    io = _bare_pipe_io(monkeypatch)
    r, w = os.pipe()
    io.response_handle = os.fdopen(w, "w")
    io.response_handle.close()
    os.close(r)
    assert io.send_shutdown() is False


def test_send_shutdown_no_reader(monkeypatch):
    io = _bare_pipe_io(monkeypatch)
    r, w = os.pipe()
    io.response_handle = os.fdopen(w, "w")
    os.close(r)  # reader gone -> EPIPE
    assert io.send_shutdown() is False
    io.response_handle.close()


def test_send_shutdown_full_pipe_does_not_hang(monkeypatch):
    io = _bare_pipe_io(monkeypatch)
    r, w = os.pipe()
    os.set_blocking(w, False)
    try:
        while True:
            os.write(w, b"x" * 4096)
    except BlockingIOError:
        pass
    os.set_blocking(w, True)
    io.response_handle = os.fdopen(w, "w")
    assert io.send_shutdown() is False  # returns instead of blocking
    os.close(r)
    io.response_handle.close()


def test_env_cleanup_removes_flag_before_shutdown_before_unlink(flag_path):
    from ai.environment.balatro_env import BalatroEnv
    order = []
    env = BalatroEnv.__new__(BalatroEnv)
    env.history_logger = MagicMock()
    env.pipe_io = MagicMock()
    lifecycle.write_session_flag()
    env.pipe_io.send_shutdown.side_effect = lambda: order.append(("shutdown", flag_path.exists()))
    env.pipe_io.cleanup.side_effect = lambda: order.append(("cleanup", None))
    env.cleanup()
    assert order == [("shutdown", False), ("cleanup", None)]


# --- Keybind validation ---

@pytest.mark.parametrize("text,expected", [
    ("ctrl+r", "ctrl+r"), ("Ctrl+R", "ctrl+r"), ("ctrl+shift+t", "ctrl+shift+t"),
    ("shift+ctrl+t", "ctrl+shift+t"), ("f9", "f9"), ("F18", "f18"), ("enter", "return"), ("ctrl+enter", "ctrl+return"), ("alt+f4", "alt+f4"),
    ("control+space", "ctrl+space"), ("r", "r"), (" ctrl + r ", "ctrl+r"),
])
def test_keybind_valid(text, expected):
    assert lifecycle.validate_keybind(text) == expected


@pytest.mark.parametrize("text", [
    "", "   ", None, 5, "ctrl+", "+r", "ctrl", "ctrl+shift", "ctrl+ctrl+r", "meta+r",
    "ctrl+rr", "f19", "f24", "f25", "f0", "ctrl+r+t", "ctrl++r",
])
def test_keybind_invalid(text):
    assert lifecycle.validate_keybind(text) is None


# --- Settings file ---

def test_settings_defaults_when_missing(tmp_path):
    assert lifecycle.load_bridge_settings(tmp_path / "nope.json") == lifecycle.DEFAULT_SETTINGS


def test_settings_defaults_when_corrupt(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json")
    assert lifecycle.load_bridge_settings(p) == lifecycle.DEFAULT_SETTINGS
    p.write_text("[1, 2]")
    assert lifecycle.load_bridge_settings(p) == lifecycle.DEFAULT_SETTINGS


def test_settings_partial_merge_and_invalid_dropped(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"start_key": "f9", "start_key_mode": "bogus", "win_hold_secs": -3,
                             "auto_wake": "yes", "win_hold_secs_endless": 5}))
    s = lifecycle.load_bridge_settings(p)
    assert s["start_key"] == "f9"
    assert s["start_key_mode"] == "arm"
    assert s["win_hold_secs"] == 60
    assert s["auto_wake"] is True
    assert s["win_hold_secs_endless"] == 5


def test_save_merges_and_keeps_unknown_keys(tmp_path):
    p = tmp_path / "sub" / "s.json"
    lifecycle.save_bridge_settings({"start_key": "Ctrl+Shift+T", "start_key_mode": "toggle"}, p)
    p.write_text(json.dumps({**json.loads(p.read_text()), "future_ui_key": 1}))
    lifecycle.save_bridge_settings({"start_key": "not a key", "win_hold_secs": 10}, p)  # invalid key ignored
    data = json.loads(p.read_text())
    assert data["start_key"] == "ctrl+shift+t"
    assert data["start_key_mode"] == "toggle"
    assert data["win_hold_secs"] == 10
    assert data["future_ui_key"] == 1


def test_settings_path_override_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv("BALATRO_SETTINGS_PATH", str(tmp_path / "x.json"))
    assert lifecycle.settings_path() == tmp_path / "x.json"
    monkeypatch.delenv("BALATRO_SETTINGS_PATH")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    monkeypatch.setattr(lifecycle.platform, "system", lambda: "Linux")
    if os.name != "nt":
        assert lifecycle.settings_path() == tmp_path / "love" / "Balatro" / "rlbridge_settings.json"
    monkeypatch.setattr(lifecycle.platform, "system", lambda: "Darwin")
    assert lifecycle.save_dir() == Path.home() / "Library" / "Application Support" / "Balatro"


# --- Stop request handling ---

def test_env_raises_stop_training_requested():
    from ai.environment.balatro_env import BalatroEnv
    env = BalatroEnv.__new__(BalatroEnv)
    env.pipe_io = MagicMock()
    env.pipe_io.wait_for_request.return_value = {"game_state": {}, "stop_requested": True}
    with pytest.raises(lifecycle.StopTrainingRequested):
        env._wait_for_request()
    env.pipe_io.wait_for_request.return_value = {"game_state": {}}
    assert env._wait_for_request() == {"game_state": {}}
    env.pipe_io.wait_for_request.return_value = None
    assert env._wait_for_request() is None


class _StubBalatroEnv(gym.Env):
    """Minimal env with the custom cleanup() BalatroEnv has; gymnasium wrappers do not forward it."""
    observation_space = gym.spaces.Box(-1, 1, (1,))
    action_space = gym.spaces.MultiDiscrete([2])

    def __init__(self):
        self.cleaned = 0

    def cleanup(self):
        self.cleaned += 1

    def action_masks(self):
        return [True, True]


def _wrapped_stub():
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.monitor import Monitor
    stub = _StubBalatroEnv()
    return stub, Monitor(ActionMasker(stub, lambda e: e.action_masks()))


def _run_train_agent(monkeypatch, tmp_path, exc):
    from ai import train_balatro
    stub, wrapped = _wrapped_stub()
    assert not hasattr(wrapped, "cleanup")  # the wrapper stack hides it
    model = MagicMock()
    model.learn.side_effect = exc
    monkeypatch.setattr(train_balatro, "create_environment", lambda **kw: wrapped)
    monkeypatch.setattr(train_balatro, "create_model", lambda e, n_steps: model)
    monkeypatch.setattr(train_balatro, "create_callbacks", lambda **kw: [train_balatro.BalatroMetricsCallback()])
    monkeypatch.setattr(train_balatro, "configure_sb3_logger", lambda *a, **k: None)
    result = train_balatro.train_agent(total_timesteps=10, save_path=str(tmp_path / "m"), tb_session_dir=str(tmp_path / "tb"))
    return result, stub, model


def test_train_agent_handles_stop_like_ctrl_c(monkeypatch, tmp_path):
    result, stub, model = _run_train_agent(monkeypatch, tmp_path, lifecycle.StopTrainingRequested())
    assert result is None
    assert stub.cleaned == 1  # reached through Monitor(ActionMasker(...))
    model.save.assert_not_called()


def test_train_agent_keyboard_interrupt_same_path(monkeypatch, tmp_path):
    result, stub, _ = _run_train_agent(monkeypatch, tmp_path, KeyboardInterrupt())
    assert result is None
    assert stub.cleaned == 1


def test_train_agent_error_path_cleans_up(monkeypatch, tmp_path):
    with pytest.raises(RuntimeError):
        _run_train_agent(monkeypatch, tmp_path, RuntimeError("boom"))


def test_cleanup_environment_handles_none():
    from ai import train_balatro
    train_balatro.cleanup_environment(None)
