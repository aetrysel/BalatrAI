"""
Balatro instance manager

Launches and supervises Balatro processes for parallel / headless training.
Each instance gets its own directory under ./instances/<name>/:
    xdg/love/Balatro/   save dir (settings, profiles, Handy config) copied from the real one
    mods/               symlinks to the allowed mods only (+ lovely's own log/dump dirs)
    screen.png          latest snapshot (written by RLBridge/headless.lua)
    status.json         latest run status (written by RLBridge/headless.lua)
    game.log            stdout/stderr of the game process
"""

import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Optional

GAME_DIR = Path(os.environ.get("BALATRO_GAME_DIR", Path.home() / "dev/Balatro"))
SOURCE_SAVE_DIR = Path(os.environ.get("BALATRO_SAVE_DIR", Path.home() / ".local/share/love/Balatro"))
SOURCE_MODS_DIR = Path(os.environ.get("BALATRO_MODS_DIR", Path.home() / ".local/share/love/Mods"))
INSTANCES_ROOT = Path(os.environ.get("BALATRO_INSTANCES_DIR", "instances"))
# Instances load RLBridge from this checkout, so the Lua side always matches the Python side
REPO_RLBRIDGE = Path(__file__).resolve().parents[2] / "RLBridge"

# Mods loaded into managed instances, matched by directory-name prefix
DEFAULT_MOD_ALLOWLIST = ("RLBridge", "smods", "HandyBalatro")

# Save-dir entries copied into a fresh instance (save.jkr inside profiles is skipped — runs restart anyway)
SAVE_FILES = ("settings.jkr", "meta.jkr")
SAVE_DIRS = ("config", "1", "2", "3")


class BalatroInstance:
    """One Balatro process with its own save dir, mod set, and snapshot output."""

    def __init__(
        self,
        name: str,
        worker_id: Optional[str] = None,
        headless: bool = True,
        gamespeed: Optional[float] = None,
        animations: bool = False,
        fps_cap: Optional[int] = None,
        snapshot_secs: float = 2.0,
        mod_allowlist=DEFAULT_MOD_ALLOWLIST,
    ):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.worker_id = worker_id if worker_id is not None else name
        self.headless = headless
        self.gamespeed = gamespeed
        self.animations = animations
        self.fps_cap = fps_cap
        self.snapshot_secs = snapshot_secs
        self.mod_allowlist = mod_allowlist

        self.root = (INSTANCES_ROOT / name).resolve()
        self.xdg_dir = self.root / "xdg"
        self.save_dir = self.xdg_dir / "love" / "Balatro"
        self.mods_dir = self.root / "mods"
        self.process: Optional[subprocess.Popen] = None
        self.launch_count = 0

    # --- Directory setup ---

    def prepare(self) -> None:
        """Create the instance's save + mods dirs (save dir is copied once, then kept so in-game tweaks persist)."""
        self.root.mkdir(parents=True, exist_ok=True)

        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
            for fname in SAVE_FILES:
                src = SOURCE_SAVE_DIR / fname
                if src.exists():
                    shutil.copy2(src, self.save_dir / fname)
            for dname in SAVE_DIRS:
                src = SOURCE_SAVE_DIR / dname
                if src.is_dir():
                    shutil.copytree(src, self.save_dir / dname, ignore=shutil.ignore_patterns("save.jkr"))

        self.mods_dir.mkdir(exist_ok=True)
        targets = {e.name: e.resolve() for e in SOURCE_MODS_DIR.iterdir()
                   if e.name.startswith(tuple(self.mod_allowlist)) and e.name != "RLBridge"}
        if "RLBridge" in self.mod_allowlist:
            targets["RLBridge"] = REPO_RLBRIDGE
        for name, target in targets.items():
            link = self.mods_dir / name
            if link.is_symlink() and link.resolve() != target:
                link.unlink()
            if not link.exists():
                link.symlink_to(target, target_is_directory=True)

    # --- Process lifecycle ---

    def env_vars(self) -> dict:
        env = os.environ.copy()
        env.update({
            "LD_PRELOAD": str(GAME_DIR / "liblovely.so"),
            "XDG_DATA_HOME": str(self.xdg_dir),
            "BALATRO_OS": "Linux",
            "BALATRO_COMM_MODE": "pipe",
            "BALATRO_WORKER_ID": str(self.worker_id),
            "BALATRO_AUTOSTART": "1",
            "BALATRO_SNAPSHOT_DIR": str(self.root),
            "BALATRO_SNAPSHOT_SECS": str(self.snapshot_secs),
            "BALATRO_ANIMATIONS": "1" if self.animations else "0",
        })
        if self.gamespeed is not None:
            env["BALATRO_GAMESPEED"] = str(self.gamespeed)
        if self.fps_cap is not None:
            env["BALATRO_FPS_CAP"] = str(self.fps_cap)
        if self.headless:
            env["BALATRO_HEADLESS"] = "1"
            env["SDL_VIDEODRIVER"] = "offscreen"  # GPU-backed EGL context, no window
            env["ALSOFT_DRIVERS"] = "null"         # no audio device
        return env

    def _kill_stale(self) -> None:
        """Kill a game left running by a previous trainer that exited without cleanup."""
        pidfile = self.root / "pid"
        if not pidfile.exists():
            return
        try:
            pid = int(pidfile.read_text().strip())
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"love" in cmdline and str(self.mods_dir).encode() in cmdline:
                self.logger.warning(f"[{self.name}] killing stale instance pid={pid}")
                os.killpg(pid, signal.SIGKILL)
        except (ValueError, FileNotFoundError, ProcessLookupError, PermissionError):
            pass
        pidfile.unlink(missing_ok=True)

    def start(self) -> None:
        self.prepare()
        self._kill_stale()
        log = open(self.root / "game.log", "ab")
        self.process = subprocess.Popen(
            ["love", "Balatro.exe", "--mod-dir", str(self.mods_dir)],
            cwd=GAME_DIR,
            env=self.env_vars(),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group so stop() reaches child processes
        )
        (self.root / "pid").write_text(str(self.process.pid))
        self.launch_count += 1
        self.logger.info(f"[{self.name}] launched pid={self.process.pid} (launch #{self.launch_count})")

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        if not self.process:
            return
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=timeout)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
        self.process = None
        (self.root / "pid").unlink(missing_ok=True)

    def restart(self) -> None:
        self.logger.warning(f"[{self.name}] restarting instance")
        self.stop()
        time.sleep(1.0)
        self.start()
