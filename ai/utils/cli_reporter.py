"""
Training CLI output: episode tracking, formatting, verbosity levels and live level switching.

Nothing here touches reward code. BalatroMetricsCallback feeds per-step info dicts to an EpisodeTracker
(pure state), and a CLIReporter turns the tracker's events / results into printed lines.

    OutputLevel.MINIMAL  policy-update lines only
    OutputLevel.SUMMARY  + one summary per finished episode (header, one line per blind, jokers)
    OutputLevel.ROUND    + live episode header, live blind-cleared lines, hands under each summary blind (default)
    OutputLevel.JOKERS   + joker roster at the end of each round
    OutputLevel.VERBOSE  + every hand live, joker buy/sell events

Blind identity is (ante, slot Small/Big/Boss), never the round counter (skipped blinds don't advance it).
"""
import atexit
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Callable, List, Optional, Tuple


class OutputLevel(IntEnum):
    MINIMAL = 1
    SUMMARY = 2
    ROUND = 3
    JOKERS = 4
    VERBOSE = 5


DEFAULT_LEVEL = OutputLevel.ROUND


def coerce_level(value) -> OutputLevel:
    """OutputLevel from a saved / typed value; anything invalid falls back to ROUND."""
    try:
        return OutputLevel(int(value))
    except (TypeError, ValueError):
        return DEFAULT_LEVEL


_warned_missing_hand = False  # one-time warning when infos lack last_played_hand

LEVEL_HELP = {
    OutputLevel.MINIMAL: "policy-update lines only",
    OutputLevel.SUMMARY: "+ episode summaries",
    OutputLevel.ROUND:   "+ live blind clears, hands in summaries",
    OutputLevel.JOKERS:  "+ joker roster each round",
    OutputLevel.VERBOSE: "+ every hand live, joker buy/sell",
}


def blind_slot(blind_name: str) -> str:
    """Blind slot within an ante: 'Small', 'Big', or 'Boss' (any other blind name)."""
    if blind_name == "Small Blind":
        return "Small"
    if blind_name == "Big Blind":
        return "Big"
    return "Boss"


# --- Tracker (pure state, no printing) ---

@dataclass
class HandRecord:
    type: str
    chips: int    # score gained by this hand
    total: int    # blind score after this hand


@dataclass
class BlindRecord:
    ante: int
    name: str
    round: int
    target: int
    scored: int = 0
    hands: List[HandRecord] = field(default_factory=list)
    cleared: bool = False
    defeated: bool = False

    @property
    def slot(self) -> str:
        return blind_slot(self.name)

    @property
    def is_boss(self) -> bool:
        return self.slot == "Boss"

    @property
    def key(self) -> Tuple[int, str]:
        return (self.ante, self.slot)


@dataclass
class EpisodeResult:
    number: int
    outcome: str  # "defeat" | "won" | "cutoff" | "timeout"
    reward: float
    blinds: List[BlindRecord]
    jokers: List[str]
    # Blind the episode ended on (for header lines): identity + score at that point
    ante: int = 0
    name: str = ""
    round: int = 0
    scored: int = 0
    target: int = 0


@dataclass
class HandEvent:
    blind: BlindRecord
    number: int          # 1-based hand number within the blind
    hand: HandRecord


@dataclass
class ClearEvent:
    blind: BlindRecord
    oneshot: bool
    jokers: List[str]


class EpisodeTracker:
    """Builds the blind / hand history of the episode in progress from per-step info dicts."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.number = 0
        self.blinds: List[BlindRecord] = []
        self.jokers: List[str] = []
        self.steps = 0
        self._last_seq = 0
        self._prev_cleared = False
        # Latched from the last non-cleared step (a cleared step can already report the next ante / blind)
        self.latched_ante = 0
        self.latched_name = ""
        self.latched_round = 0
        self.latched_target = 0

    def _blind(self, ante: int) -> BlindRecord:
        """Current blind record for the latched identity, creating it on first use (skip-safe)."""
        key = (ante, blind_slot(self.latched_name))
        if self.blinds and self.blinds[-1].key == key:
            return self.blinds[-1]
        blind = BlindRecord(ante=ante, name=self.latched_name, round=self.latched_round, target=self.latched_target)
        self.blinds.append(blind)
        return blind

    def on_step(self, info: dict, step_ante: int, episode_number: int = 0) -> list:
        """Fold one step in. step_ante is the ante of the blind being played (latched by the caller on cleared steps).
        Returns the events to show live: HandEvent / ClearEvent."""
        self.number = episode_number or self.number
        self.steps += 1
        events = []
        cleared = bool(info.get("blind_cleared", False))

        if info.get("jokers") is not None:
            self.jokers = [j.get("name", "Unknown") for j in info["jokers"]]

        if not cleared or not self.latched_name:
            # (a cleared step only latches when nothing is latched yet, e.g. an episode joined mid-blind)
            # Ante / target / round latch together with the name: a step with no blind (shop after a boss)
            # already reports the next ante.
            name = info.get("blind_name", "")
            if name and name not in ("None", ""):
                self.latched_name = name
                self.latched_ante = step_ante or self.latched_ante
                if info.get("blind_chips", 0) > 0:
                    self.latched_target = info["blind_chips"]
                self.latched_round = info.get("round", self.latched_round)

        ante = step_ante or self.latched_ante

        # New hand: the Lua-side record's seq changed (covers the clearing hand and the killing hand)
        lph = info.get("last_played_hand") or {}
        seq = lph.get("seq", 0)
        global _warned_missing_hand
        if "last_played_hand" not in info and not _warned_missing_hand and not info.get("timeout", False):
            _warned_missing_hand = True
            print("⚠️  Step info has no last_played_hand: the RLBridge Lua mod probably needs redeploying "
                  "(hands will be missing from episode output).")
        if self.steps == 1 and seq > 0:
            # Joined mid-run (resume): the Lua record is a stale hand from before we started watching
            self._last_seq = seq
        elif seq and seq != self._last_seq and self.latched_name and ante:
            self._last_seq = seq
            blind = self._blind(ante)
            total = int(lph.get("chips_before", 0)) + int(lph.get("score", 0))
            hand = HandRecord(type=lph.get("handname", "Unknown") or "Unknown", chips=int(lph.get("score", 0)), total=total)
            blind.hands.append(hand)
            blind.scored = total
            events.append(HandEvent(blind, len(blind.hands), hand))

        if cleared and not self._prev_cleared and self.latched_name and ante:
            blind = self._blind(ante)
            if not blind.hands:
                blind.scored = max(blind.scored, info.get("chips", 0))
            blind.cleared = True
            events.append(ClearEvent(blind, oneshot=len(blind.hands) == 1, jokers=list(self.jokers)))
        self._prev_cleared = cleared
        return events

    def finish(self, info: dict, reward: float, outcome: Optional[str] = None) -> EpisodeResult:
        """Close the episode from its done-step info. outcome defaults to won / timeout / defeat from info."""
        if outcome is None:
            if info.get("won", False):
                outcome = "won"
            elif info.get("timeout", False):
                outcome = "timeout"
            else:
                outcome = "defeat"
        if outcome == "defeat" and self.latched_name and self.latched_ante:
            # The blind being played (latched identity), not necessarily blinds[-1]
            blind = self._blind(self.latched_ante)
            blind.defeated = True
            blind.cleared = False
        # --- Previous: marked blinds[-1] even when it was an earlier cleared blind ---
        # if outcome == "defeat" and self.blinds:
        #     self.blinds[-1].defeated = True
        #     self.blinds[-1].cleared = False
        return self._result(outcome, reward)

    def win_snapshot(self, reward: float) -> EpisodeResult:
        """A WON result for an episode that keeps going (endless mode)."""
        return self._result("won", reward)

    def snapshot(self, reward: float) -> Optional[EpisodeResult]:
        """The episode in progress as a CUT OFF result, or None if nothing has happened yet."""
        if self.steps == 0 or not self.latched_ante:
            return None
        return self._result("cutoff", reward)

    def _result(self, outcome: str, reward: float) -> EpisodeResult:
        last = self.blinds[-1] if self.blinds else None
        if outcome == "cutoff" or last is None:
            # Where the run actually was, even if no hand was played in that blind yet
            ante, name, rnd, target = self.latched_ante, self.latched_name, self.latched_round, self.latched_target
            scored = last.scored if last and last.key == (ante, blind_slot(name)) else 0
        else:
            ante, name, rnd, target, scored = last.ante, last.name, last.round, last.target, last.scored
        return EpisodeResult(number=self.number, outcome=outcome, reward=reward, blinds=list(self.blinds),
                             jokers=list(self.jokers), ante=ante, name=name, round=rnd, scored=scored, target=target)


# --- Reporter (all formatting and level gating) ---

@dataclass
class PolicySnapshot:
    steps: int
    episodes: int                       # all-time episodes
    rollout_episodes: int
    reward_mean: Optional[float]        # mean episode reward over the rollout (None if no episodes)
    clear_rate: Optional[float]         # blind clear rate over the rollout, 0..1
    mean_ante: Optional[float]
    mean_round: Optional[float]
    best_ante: int
    best_round: int
    wins_total: int
    wins_rollout: int
    # (ante, slot, round or None, rate) of the deadliest position, all-time
    deadliest_position: Optional[Tuple[int, str, Optional[int], float]] = None
    deadliest_blind: Optional[Tuple[str, float]] = None   # (name, rate) over all blinds
    deadliest_boss: Optional[Tuple[str, float]] = None    # (name, rate) over boss blinds only
    deadliest_boss_position: Optional[Tuple[int, str, Optional[int], float]] = None  # deadliest Boss (ante, slot) position
    position_names: dict = field(default_factory=dict)    # (ante, slot) -> blind name last seen there


class CLIReporter:
    def __init__(self, level: OutputLevel = DEFAULT_LEVEL, seeded: bool = False,
                 out: Callable[[str], None] = print, clock: Callable[[], float] = time.time,
                 wrap_width: int = 140):
        self.level = OutputLevel(level)
        self.seeded = seeded
        self.out = out
        self.clock = clock
        self.wrap_width = wrap_width
        self._header_printed_for = None
        self._blind_header_for = None  # (episode, ante, blind name) whose live blind header was printed
        # Previous policy update (for deltas and elapsed time)
        self._prev_time: Optional[float] = None
        self._prev_values: dict = {}

    # -- level control --

    def set_level(self, level) -> None:
        self.level = OutputLevel(level)
        self.out(f"Output level: {self.level.name} ({int(self.level)}/5) - {LEVEL_HELP[self.level]}")

    def help_text(self) -> str:
        lines = ["Output levels (press the key to switch, '?' for this help):"]
        for lvl in OutputLevel:
            marker = "*" if lvl == self.level else " "
            lines.append(f" {marker} [{int(lvl)}] {lvl.name:<8} {LEVEL_HELP[lvl]}")
        from .debug import debug_output_enabled
        lines.append(f"   [d] debug output (currently {'ON' if debug_output_enabled() else 'OFF'})")
        return "\n".join(lines)

    def toggle_debug_output(self) -> None:
        from .debug import debug_output_enabled, set_debug_output
        set_debug_output(not debug_output_enabled())
        self.out(f"Debug output: {'ON' if debug_output_enabled() else 'OFF'} (press 'd' to toggle)")

    def show_help(self) -> None:
        self.out(self.help_text())

    # -- formatting --

    @staticmethod
    def format_clear(blind: BlindRecord, oneshot: bool) -> str:
        icon, label = ("🎯", "ONESHOT!") if oneshot else ("✅", "Blind cleared!")
        return f"{icon} {label} Ante {blind.ante} - {blind.name} | {blind.scored}/{blind.target} chips [round {blind.round}]"

    @staticmethod
    def format_hand(number: int, hand: HandRecord, target: int, indent: int = 3) -> str:
        return f"{' ' * indent}Hand {number}: {hand.type} +{hand.chips} ({hand.total}/{target})"

    @staticmethod
    def format_blind_line(blind: BlindRecord) -> str:
        n = len(blind.hands)
        line = (f"   Ante {blind.ante} - {blind.name} | {blind.scored}/{blind.target} chips"
                f" · {n} hand{'s' if n != 1 else ''} [round {blind.round}]")
        if blind.cleared:
            line += " [cleared!]"
        elif blind.defeated:
            line += " [DEFEATED!] ❌"
        return line

    @staticmethod
    def format_jokers(jokers: List[str]) -> str:
        return f"   Jokers: [{', '.join(jokers) if jokers else 'None'}]"

    @staticmethod
    def format_episode_header(result: EpisodeResult) -> str:
        where = f"Ante {result.ante} - {result.name}"
        if result.outcome == "won":
            return (f"🎯 EPISODE {result.number} WON! {where} | {result.scored}/{result.target} chips "
                    f"[round {result.round}] | reward: {result.reward:.1f}")
        if result.outcome == "cutoff":
            return f"⏹️ EPISODE {result.number} CUT OFF after {where} [round {result.round}] | reward: {result.reward:.1f}"
        if result.outcome == "timeout":
            return f"🎯 EPISODE {result.number}: Timed out at {where} [round {result.round}] | reward: {result.reward:.1f}"
        return (f"🎯 EPISODE {result.number}: Defeated by Ante {result.ante} {result.name} "
                f"({result.scored}/{result.target} chips) [round {result.round}] | reward: {result.reward:.1f}")

    def format_episode_end(self, result: EpisodeResult) -> List[str]:
        """Summary lines for a finished (or cut off) episode, gated on the current level."""
        if self.level < OutputLevel.SUMMARY:
            return []
        lines = [self.format_episode_header(result)]
        for blind in result.blinds:
            lines.append(self.format_blind_line(blind))
            if self.level >= OutputLevel.ROUND:
                for i, hand in enumerate(blind.hands, 1):
                    lines.append(self.format_hand(i, hand, blind.target, indent=6))
        lines.append(self.format_jokers(result.jokers))
        return lines

    # -- live output --

    def _live_header(self, episode_number: int) -> None:
        if self._header_printed_for != episode_number:
            self._header_printed_for = episode_number
            self.out(f"── EPISODE {episode_number} ──")

    def live_events(self, events: list, episode_number: int) -> None:
        for ev in events:
            if isinstance(ev, HandEvent):
                if self.level >= OutputLevel.VERBOSE:
                    self._live_header(episode_number)
                    self._live_blind_header(ev.blind, episode_number)
                    self.out(self.format_hand(ev.number, ev.hand, ev.blind.target))
            elif isinstance(ev, ClearEvent):
                if self.level >= OutputLevel.ROUND:
                    self._live_header(episode_number)
                    self.out(self.format_clear(ev.blind, ev.oneshot))
                if self.level >= OutputLevel.JOKERS:
                    self._live_header(episode_number)
                    self.out(self.format_jokers(ev.jokers))

    @staticmethod
    def format_blind_header(blind) -> str:
        return f"   ── Ante {blind.ante} - {blind.name} [round {blind.round}] ──"

    def _live_blind_header(self, blind, episode_number: int) -> None:
        """Print an ante/blind header once, before the first live hand of each blind."""
        key = (episode_number, blind.ante, blind.name)
        if self._blind_header_for != key:
            self._blind_header_for = key
            self.out(self.format_blind_header(blind))

    def joker_event(self, kind: str, name: str, episode_number: int, source: str = "") -> None:
        """kind: bought / sold / gained. source: how a gained joker arrived (e.g. 'pack', 'consumable')."""
        if self.level < OutputLevel.VERBOSE:
            return
        self._live_header(episode_number)
        icon, verb = {"bought": ("🃏", "Bought"), "sold": ("💸", "Sold"), "gained": ("🃏", "Gained")}[kind]
        suffix = f" (from {source})" if source and kind == "gained" else ""
        self.out(f"   {icon} {verb} {name}{suffix}")

    def episode_end(self, result: EpisodeResult) -> None:
        for line in self.format_episode_end(result):
            self.out(line)
        if self.level >= OutputLevel.SUMMARY:
            self.out("")

    # -- policy update line --

    @staticmethod
    def _delta(value: Optional[float], prev: Optional[float], fmt: str, suffix: str = "") -> str:
        if value is None or prev is None:
            return ""
        d = value - prev
        sign = "+" if d >= 0 else "-"
        return f" ({sign}{format(abs(d), fmt)}{suffix})"

    @staticmethod
    def _position_label(snap: PolicySnapshot, ante: int, slot: str, rnd: Optional[int]) -> str:
        name = snap.position_names.get((ante, slot)) or slot
        return f"A{ante} {name}" + (f" [round {rnd}]" if rnd else "")

    def _deadliest_text(self, snap: PolicySnapshot) -> str:
        boss = f"boss: {snap.deadliest_boss[0]} {snap.deadliest_boss[1]:.0%}" if snap.deadliest_boss else None
        if self.seeded:
            # Previous seeded format: deadliest blind by name only (no ante/round)
            # if not snap.deadliest_blind:
            #     return "n/a"
            # name, rate = snap.deadliest_blind
            # text = f"{name} {rate:.0%}"
            # if boss and blind_slot(name) != "Boss":
            #     text += f" · {boss}"
            # return text
            # Seeded: each (ante, slot) always holds the same blind, so name it alongside its ante & round
            if not snap.deadliest_position:
                return "n/a"
            ante, slot, rnd, rate = snap.deadliest_position
            text = self._position_label(snap, ante, slot, rnd) + f" {rate:.0%}"
            if slot != "Boss" and snap.deadliest_boss_position:
                b_ante, b_slot, b_rnd, b_rate = snap.deadliest_boss_position
                text += f" · boss: {self._position_label(snap, b_ante, b_slot, b_rnd)} {b_rate:.0%}"
            return text
        if not snap.deadliest_position:
            return "n/a"
        ante, slot, rnd, rate = snap.deadliest_position
        text = f"A{ante} {slot}" + (f" [round {rnd}]" if rnd else "") + f" {rate:.0%}"
        if boss:
            text += f" · {boss}"
        return text

    def format_policy(self, snap: PolicySnapshot, title: str = "Policy update", update_state: bool = True) -> str:
        now = self.clock()
        prev = self._prev_values
        stamp = datetime.fromtimestamp(now).strftime("%H:%M:%S")
        if self._prev_time is not None:
            dt = int(round(now - self._prev_time))
            stamp += f", +{dt // 60}m{dt % 60:02d}s" if dt >= 60 else f", +{dt}s"
        clear_pct = None if snap.clear_rate is None else snap.clear_rate * 100

        head = f"🔄 [{stamp}] {title} at {snap.steps} steps"
        episodes = f"Episodes: {snap.episodes}" + (f" (+{snap.rollout_episodes})" if title == "Policy update" else "")
        if snap.reward_mean is None:
            reward = "Reward: n/a"
            clear = "Clear rate: n/a"
            progress = "Ante: n/a Round: n/a"
        else:
            reward = f"Reward: {snap.reward_mean:.1f}" + self._delta(snap.reward_mean, prev.get("reward"), ".1f")
            clear = f"Clear rate: {clear_pct:.0f}%" + self._delta(clear_pct, prev.get("clear"), ".1f", "pp")
            progress = (f"Ante: {snap.mean_ante:.1f}" + self._delta(snap.mean_ante, prev.get("ante"), ".1f")
                        + f" Round: {snap.mean_round:.1f}" + self._delta(snap.mean_round, prev.get("round"), ".1f"))
        best = f"Best ante: {snap.best_ante} round: {snap.best_round}"
        deadliest = f"Deadliest: {self._deadliest_text(snap)}"
        line1 = " | ".join([head, episodes, reward, clear])
        rest = [progress, best, deadliest]
        if snap.wins_total > 0:
            wins = f"Wins: total {snap.wins_total}"
            if snap.wins_rollout > 0 and snap.rollout_episodes > 0:
                wins += f" (rollout {snap.wins_rollout}, {100 * snap.wins_rollout / snap.rollout_episodes:.0f}%)"
            rest.append(wins)
        line2 = " | ".join(rest)
        full = f"{line1} | {line2}"
        text = full if len(full) <= self.wrap_width else f"{line1}\n   {line2}"

        if update_state:
            self._prev_time = now
            if snap.reward_mean is not None:
                self._prev_values = {"reward": snap.reward_mean, "clear": clear_pct,
                                     "ante": snap.mean_ante, "round": snap.mean_round}
        return text

    def policy_update(self, snap: PolicySnapshot) -> None:
        self.out("\n" + self.format_policy(snap))

    def cutoff(self, result: Optional[EpisodeResult], final_snap: PolicySnapshot) -> None:
        """Easter egg printed when training stops: the in-progress episode, then final stats."""
        if result is not None:
            self.episode_end(result)
        self.out(self.format_policy(final_snap, title="Final stats", update_state=False))


# --- Live level switching ---

class KeyListener:
    """Daemon thread that reads single keypresses ('1'-'5' switch level, '?' help). Does nothing unless stdin is a tty.
    The terminal is put in cbreak mode and restored when the thread stops and at interpreter exit."""

    def __init__(self, reporter: CLIReporter, stream=None):
        self.reporter = reporter
        self.stream = stream if stream is not None else sys.stdin
        self._stop = False
        self._saved = None
        self._thread = None
        self.active = False

    def start(self) -> bool:
        """Start listening. Returns False (and does nothing) if stdin is not an interactive terminal."""
        try:
            if not self.stream.isatty():
                return False
            import termios
            import threading
            import tty
            fd = self.stream.fileno()
            self._saved = termios.tcgetattr(fd)
            tty.setcbreak(fd)
        except Exception:
            return False
        atexit.register(self.stop)
        self.active = True
        self._install_signals()
        self._thread = threading.Thread(target=self._run, name="cli-key-listener", daemon=True)
        self._thread.start()
        return True

    def _install_signals(self):
        """Restore the terminal on SIGTERM / SIGHUP, then chain to the previous handler / default action."""
        import signal
        self._prev_handlers = {}
        for name in ("SIGTERM", "SIGHUP"):
            sig = getattr(signal, name, None)
            if sig is None:
                continue
            try:
                self._prev_handlers[sig] = signal.signal(sig, self._on_signal)
            except (ValueError, OSError):  # not the main thread
                pass

    def _on_signal(self, signum, frame):
        import signal
        self._restore()
        prev = self._prev_handlers.get(signum, signal.SIG_DFL)
        if callable(prev):
            prev(signum, frame)
        elif prev == signal.SIG_DFL:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    def _uninstall_signals(self):
        import signal
        for sig, prev in getattr(self, "_prev_handlers", {}).items():
            try:
                signal.signal(sig, prev if prev is not None else signal.SIG_DFL)
            except (ValueError, OSError):
                pass
        self._prev_handlers = {}

    def handle_key(self, key: str) -> None:
        if key in "12345" and key:
            self.reporter.set_level(OutputLevel(int(key)))
        elif key == "?":
            self.reporter.show_help()
        elif key in ("d", "D"):
            self.reporter.toggle_debug_output()

    def _run(self):
        import select
        try:
            fd = self.stream.fileno()
            while not self._stop:
                ready, _, _ = select.select([fd], [], [], 0.2)
                if not ready:
                    continue
                data = os.read(fd, 1)
                if not data:
                    break
                self.handle_key(data.decode(errors="ignore"))
        except Exception:
            pass
        finally:
            self._restore()

    def _restore(self):
        if self._saved is not None:
            try:
                import termios
                termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._saved)
            except Exception:
                pass
            self._saved = None

    def stop(self):
        self._stop = True
        self._restore()
        self._uninstall_signals()
        self.active = False
