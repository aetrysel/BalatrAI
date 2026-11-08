"""
Session lifecycle helpers shared with the RLBridge Lua mod.

Two small files connect Python and Lua without ever blocking either side:

- Session-ready flag (plain file): written by Python once the FIFOs exist, removed on cleanup.
  The idle game polls it once a second and auto-wakes when it appears. Contains Python's PID
  so Lua can ignore a stale flag left by a killed process.
- rlbridge_settings.json: persistent user settings (start keybind and mode, auto-start, win-hold
  durations). Lives in the LÖVE save directory of Balatro (Linux: ~/.local/share/love/Balatro,
  Windows: %APPDATA%/Balatro), the same directory as lovely.log, so it is outside the repo and
  never tracked by git. Lua resolves it with love.filesystem.getSaveDirectory().
  Override with BALATRO_SETTINGS_PATH (honoured by both sides).
"""
import json
import os
import platform
import re
import tempfile
from pathlib import Path

DEFAULT_SETTINGS = {
    "start_key": "ctrl+r",
    "start_key_mode": "arm",
    "auto_start": False,
    "win_hold_secs": 60,
    "win_hold_secs_endless": 15,
}
START_KEY_MODES = ("arm", "toggle")

_MODIFIERS = ("ctrl", "shift", "alt")
_MODIFIER_ALIASES = {"control": "ctrl", "lctrl": "ctrl", "rctrl": "ctrl", "lshift": "shift",
                     "rshift": "shift", "lalt": "alt", "ralt": "alt"}
_NAMED_KEYS = {
    "space", "return", "escape", "tab", "backspace", "delete", "insert", "home", "end",
    "pageup", "pagedown", "up", "down", "left", "right",
}
_KEY_RE = re.compile(r"^(?:[a-z0-9]|f(?:[1-9]|1[0-8])|[-=\[\];',./\\`])$")


class StopTrainingRequested(Exception):
    """The in-game start key (toggle mode) asked training to stop; handled like Ctrl+C."""


def save_dir() -> Path:
    """Balatro's LÖVE save directory (where lovely.log and the settings file live)."""
    if os.name == "nt":
        return Path(os.getenv("APPDATA", "")) / "Balatro"
    if platform.system() == "Darwin":
        return Path.home() / "Library" / "Application Support" / "Balatro"
    data_home = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    return Path(data_home) / "love" / "Balatro"


def settings_path() -> Path:
    override = os.environ.get("BALATRO_SETTINGS_PATH")
    return Path(override) if override else save_dir() / "rlbridge_settings.json"


def session_flag_path() -> Path:
    override = os.environ.get("BALATRO_SESSION_FLAG")
    if override:
        return Path(override)
    if os.name == "nt":
        return Path(os.getenv("APPDATA", "")) / "Balatro" / "Mods" / "RLBridge" / "session_ready"
    return Path("/tmp/balatro_session_ready")


def write_session_flag(path: Path | None = None) -> None:
    """Announce a ready session (FIFOs created). Never raises."""
    path = Path(path) if path else session_flag_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(os.getpid()))
    except OSError:
        pass


def remove_session_flag(path: Path | None = None) -> None:
    """Remove the session-ready flag (also used at startup to drop a stale one). Never raises."""
    path = Path(path) if path else session_flag_path()
    try:
        path.unlink()
    except OSError:
        pass


# --- Start keybind ---

def validate_keybind(text) -> str | None:
    """Normalise 'Ctrl+Shift+T' -> 'ctrl+shift+t'. Returns None if invalid.

    Format: zero or more modifiers (ctrl, shift, alt) then exactly one key (a letter, digit,
    F1-F18, punctuation or a named key such as space/return/escape), joined with '+'.
    """
    if not isinstance(text, str):
        return None
    raw = text.strip().lower()
    if not raw:
        return None
    # A trailing '+' key ("ctrl++") is not supported; the key must be spelled out.
    tokens = [t.strip() for t in raw.split("+")]
    if any(not t for t in tokens):
        return None
    *mods, key = tokens
    mods = [_MODIFIER_ALIASES.get(m, m) for m in mods]
    if any(m not in _MODIFIERS for m in mods) or len(set(mods)) != len(mods):
        return None
    if key == "enter":  # LÖVE calls it "return"
        key = "return"
    if key in _MODIFIER_ALIASES or key in _MODIFIERS:
        return None
    if not (key in _NAMED_KEYS or _KEY_RE.match(key)):
        return None
    ordered = [m for m in _MODIFIERS if m in mods]
    return "+".join(ordered + [key])


# --- Settings file ---

def _clean(raw: dict) -> dict:
    """Keep only valid values from a raw settings dict (invalid entries are dropped)."""
    out = {}
    key = validate_keybind(raw.get("start_key"))
    if key:
        out["start_key"] = key
    if raw.get("start_key_mode") in START_KEY_MODES:
        out["start_key_mode"] = raw["start_key_mode"]
    # "auto_wake" (old name, default True) was only ever written by a default, so it is ignored
    if isinstance(raw.get("auto_start"), bool):
        out["auto_start"] = raw["auto_start"]
    for name in ("win_hold_secs", "win_hold_secs_endless"):
        val = raw.get(name)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and val >= 0:
            out[name] = val
    return out


def load_bridge_settings(path: Path | None = None) -> dict:
    """Settings with defaults applied for a missing/corrupt file or missing/invalid keys."""
    path = Path(path) if path else settings_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    return {**DEFAULT_SETTINGS, **_clean(raw if isinstance(raw, dict) else {})}


def save_bridge_settings(updates: dict, path: Path | None = None) -> dict:
    """Merge valid `updates` into the settings file, keeping unknown keys. Returns the merged file content."""
    path = Path(path) if path else settings_path()
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            existing = {}
    except (OSError, ValueError):
        existing = {}
    existing.pop("auto_wake", None)  # old name of auto_start
    merged = {**DEFAULT_SETTINGS, **existing, **_clean(updates)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, path)
    return merged
