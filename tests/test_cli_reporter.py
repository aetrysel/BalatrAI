"""Tests for ai/utils/cli_reporter.py and its wiring in BalatroMetricsCallback (synthetic step sequences)."""
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.utils.callbacks import BalatroMetricsCallback
from ai.utils.cli_reporter import (CLIReporter, EpisodeTracker, KeyListener, OutputLevel, PolicySnapshot)


# --- helpers ---

def info(ante=1, name="Small Blind", target=300, rnd=1, cleared=False, seq=0, hand="Pair", score=0, before=0,
         jokers=(), **extra):
    d = {"ante": ante, "round": rnd, "blind_name": name, "blind_chips": target, "blind_cleared": cleared,
         "chips": before + score, "jokers": [{"id": i, "name": n} for i, n in enumerate(jokers)],
         "won": False, "action_taken": 2, "gold": 4, "consumable_count": 0, "state": 1}
    if seq:
        d["last_played_hand"] = {"handname": hand, "score": score, "chips_before": before, "seq": seq}
    d.update(extra)
    return d


class Harness:
    """Drives a BalatroMetricsCallback with synthetic infos and records printed lines."""

    def __init__(self, level=OutputLevel.VERBOSE, seeded=False):
        self.lines = []
        self.reporter = CLIReporter(level=level, seeded=seeded, out=self.lines.append, clock=lambda: 1_700_000_000)
        self.cb = BalatroMetricsCallback(reporter=self.reporter, seeded=seeded)
        self.cb.model = SimpleNamespace(num_timesteps=4096,
                                        logger=SimpleNamespace(record=lambda *a, **k: None, dump=lambda *a, **k: None))
        self.cb.num_timesteps = 4096

    def step(self, i, done=False, reward=0.0, **ep):
        if done:
            i = dict(i, episode={"r": ep.get("r", reward)})
        self.cb.locals = {"dones": np.array([done]), "infos": [i], "rewards": np.array([reward])}
        self.cb._on_step()

    def text(self):
        return "\n".join(self.lines)


def play_small_blind_two_hands(h, ante=1):
    h.step(info(ante=ante, action_taken=5))  # select blind
    h.step(info(ante=ante, seq=1, hand="Pair", score=100, before=0))
    h.step(info(ante=ante, seq=2, hand="Flush", score=224, before=100, cleared=True, jokers=("Joker",)))


# --- live lines ---

def test_multi_hand_clear_line():
    h = Harness(OutputLevel.ROUND)
    play_small_blind_two_hands(h)
    assert "── EPISODE 1 ──" in h.lines
    assert "✅ Blind cleared! Ante 1 - Small Blind | 324/300 chips [round 1]" in h.lines
    assert not any("ONESHOT" in l for l in h.lines)


def test_oneshot_line_has_no_hand_count():
    h = Harness(OutputLevel.ROUND)
    h.step(info(action_taken=5))
    h.step(info(seq=1, hand="Flush", score=324, before=0, cleared=True))
    assert "🎯 ONESHOT! Ante 1 - Small Blind | 324/300 chips [round 1]" in h.lines
    assert not any("hand" in l.lower() for l in h.lines if "ONESHOT" in l)


def test_clearing_hand_type_comes_from_last_played_hand():
    h = Harness(OutputLevel.VERBOSE)
    h.step(info(action_taken=5))
    h.step(info(seq=1, hand="Pair", score=100))
    # Stale poll-dependent fields on the clear step must be ignored
    h.step(info(seq=2, hand="Full House", score=224, before=100, cleared=True, hand_type="Pair", hand_score=100))
    assert "   Hand 2: Full House +224 (324/300)" in h.lines


def test_boss_clear_uses_latched_ante_not_next_ante():
    h = Harness(OutputLevel.ROUND)
    h.step(info(ante=1, name="The Hook", target=600, rnd=3))
    h.step(info(ante=2, name="The Hook", target=600, rnd=3, seq=1, hand="Flush", score=700, cleared=True))
    assert "🎯 ONESHOT! Ante 1 - The Hook | 700/600 chips [round 3]" in h.lines


def test_skipped_blinds_do_not_create_records():
    t = EpisodeTracker()
    t.on_step(info(name="Big Blind", target=450, rnd=2), 1, 1)  # Small Blind was skipped
    t.on_step(info(name="Big Blind", target=450, rnd=2, seq=1, hand="Pair", score=500, cleared=True), 1, 1)
    assert [b.name for b in t.blinds] == ["Big Blind"]  # Small was skipped: never appears


# --- episode end ---

def test_defeat_summary_includes_killing_hand_and_markers():
    h = Harness(OutputLevel.ROUND)
    play_small_blind_two_hands(h)  # cleared Small
    h.step(info(name="Big Blind", target=800, rnd=2, seq=3, hand="Pair", score=300, before=0))
    # Game-over step: info is from the state before the killing hand, but last_played_hand is the killing hand
    h.step(info(name="Big Blind", target=800, rnd=2, seq=4, hand="High Card", score=50, before=300, jokers=("Joker",)),
           done=True, reward=-12.5)
    out = h.text()
    assert "🎯 EPISODE 1: Defeated by Ante 1 Big Blind (350/800 chips) [round 2] | reward: -12.5" in out
    assert "   Ante 1 - Small Blind | 324/300 chips · 2 hands [round 1] [cleared!]" in out
    assert "   Ante 1 - Big Blind | 350/800 chips · 2 hands [round 2] [DEFEATED!] ❌" in out
    lines = h.lines
    assert lines.index("      Hand 1: Pair +300 (300/800)") < lines.index("      Hand 2: High Card +50 (350/800)")
    assert "   Jokers: [Joker]" in out


def test_summary_level_has_no_hand_lines_or_live_lines():
    h = Harness(OutputLevel.SUMMARY)
    play_small_blind_two_hands(h)
    h.step(info(name="Big Blind", target=800, rnd=2, seq=3, score=10), done=True, reward=1.0)
    out = h.text()
    assert "Defeated by" in out and "[cleared!]" in out
    assert "Hand 1" not in out and "Blind cleared" not in out and "── EPISODE" not in out


def test_real_win_only():
    h = Harness(OutputLevel.SUMMARY)
    h.step(info(ante=8, name="Cerulean Bell", target=50000, rnd=24))
    h.step(info(ante=8, name="Cerulean Bell", target=50000, rnd=24, seq=1, hand="Flush", score=60000, cleared=True))
    h.step(info(ante=9, name="Cerulean Bell", target=50000, rnd=24, cleared=True, won=True), done=True, reward=900.0)
    assert any(l.startswith("🎯 EPISODE 1 WON! Ante 8 - Cerulean Bell | 60000/50000 chips [round 24] | reward: 900.0") for l in h.lines)
    # A cleared blind at the end of an episode that did not win is a defeat, never WON
    h2 = Harness(OutputLevel.SUMMARY)
    h2.step(info())
    h2.step(info(seq=1, score=400, cleared=True))
    h2.step(info(name="Big Blind", target=450, rnd=2), done=True)
    assert "WON" not in h2.text() and "Defeated by" in h2.text()


def test_cut_off_output_then_final_stats():
    h = Harness(OutputLevel.SUMMARY)
    h.step(info(ante=3, name="Big Blind", target=1000, rnd=8, seq=1, score=200), reward=3.0)
    h.cb.print_cutoff()
    out = h.text()
    assert "⏹️ EPISODE 1 CUT OFF after Ante 3 - Big Blind [round 8] | reward: 3.0" in out
    assert "Final stats at 4096 steps" in out
    h.cb.print_cutoff()  # prints once
    assert out.count("CUT OFF") == 1


def test_cut_off_with_no_episode_in_progress_prints_only_final_stats():
    h = Harness(OutputLevel.SUMMARY)
    h.cb.print_cutoff()
    assert "CUT OFF" not in h.text() and "Final stats" in h.text()


# --- level gating ---

@pytest.mark.parametrize("level,live_header,live_clear,roster,live_hand,summary,hands_in_summary", [
    (OutputLevel.MINIMAL, False, False, False, False, False, False),
    (OutputLevel.SUMMARY, False, False, False, False, True, False),
    (OutputLevel.ROUND,   True,  True,  False, False, True, True),
    (OutputLevel.JOKERS,  True,  True,  True,  False, True, True),
    (OutputLevel.VERBOSE, True,  True,  True,  True,  True, True),
])
def test_level_gating(level, live_header, live_clear, roster, live_hand, summary, hands_in_summary):
    h = Harness(level)
    play_small_blind_two_hands(h)
    live = h.text()
    assert ("── EPISODE 1 ──" in live) == live_header
    assert ("Blind cleared!" in live) == live_clear
    assert ("   Jokers: [Joker]" in live) == roster
    assert ("   Hand 2: Flush +224 (324/300)" in live) == live_hand
    h.lines.clear()
    h.step(info(name="Big Blind", target=800, rnd=2), done=True, reward=1.0)
    end = h.text()
    assert ("Defeated by" in end) == summary
    assert ("      Hand 1: Pair +100 (100/300)" in end) == hands_in_summary


def test_joker_events_only_at_verbose():
    for level, expect in ((OutputLevel.JOKERS, False), (OutputLevel.VERBOSE, True)):
        h = Harness(level)
        h.step(info(jokers=(), gold=10))
        h.step(info(jokers=("Jolly Joker",), gold=5, action_taken=7))
        h.step(info(jokers=(), gold=8, action_taken=8))
        assert ("   🃏 Bought Jolly Joker" in h.lines) == expect
        assert ("   💸 Sold Jolly Joker" in h.lines) == expect


def test_set_level_prints_confirmation():
    r_lines = []
    r = CLIReporter(out=r_lines.append)
    r.set_level(OutputLevel.MINIMAL)
    assert r.level == OutputLevel.MINIMAL and len(r_lines) == 1 and "MINIMAL" in r_lines[0]


# --- policy line ---

def snap(**kw):
    base = dict(steps=4096, episodes=20, rollout_episodes=8, reward_mean=41.2, clear_rate=0.34, mean_ante=2.1,
                mean_round=5.4, best_ante=5, best_round=18, wins_total=0, wins_rollout=0,
                deadliest_position=(2, "Boss", 6, 0.56), deadliest_blind=("The Mouth", 0.61),
                deadliest_boss=("The Mouth", 0.61))
    base.update(kw)
    return PolicySnapshot(**base)


def test_policy_line_first_then_deltas_and_no_wins_segment():
    t = [1_700_000_000]
    r = CLIReporter(out=lambda s: None, clock=lambda: t[0])
    first = r.format_policy(snap())
    assert "Policy update at 4096 steps | Episodes: 20 (+8) | Reward: 41.2 | Clear rate: 34%" in first
    assert "Wins" not in first and "Win rate" not in first and "(+" not in first.split("Episodes")[1].split("|")[1]
    t[0] += 42
    second = r.format_policy(snap(reward_mean=43.5, clear_rate=0.352, mean_ante=2.2, mean_round=5.2))
    assert ", +42s]" in second
    assert "Reward: 43.5 (+2.3)" in second
    assert "Clear rate: 35% (+1.2pp)" in second
    assert "Ante: 2.2 (+0.1) Round: 5.2 (-0.2)" in second
    assert "Best ante: 5 round: 18" in second


def test_policy_line_unseeded_and_seeded_deadliest():
    assert "Deadliest: A2 Boss [round 6] 56% · boss: The Mouth 61%" in CLIReporter(seeded=False).format_policy(snap())
    seeded = CLIReporter(seeded=True)
    assert "Deadliest: The Mouth 61%" in seeded.format_policy(snap())
    # A deadliest blind that is not a boss gets the deadliest boss appended
    text = CLIReporter(seeded=True).format_policy(snap(deadliest_blind=("Big Blind", 0.4)))
    assert "Deadliest: Big Blind 40% · boss: The Mouth 61%" in text


def test_policy_wins_segment():
    r = CLIReporter()
    assert "Wins: total 3 |" not in r.format_policy(snap(wins_total=3)) and "Wins: total 3" in r.format_policy(snap(wins_total=3))
    assert "(rollout" not in r.format_policy(snap(wins_total=3))
    assert "Wins: total 3 (rollout 1, 12%)" in CLIReporter().format_policy(snap(wins_total=3, wins_rollout=1))


def test_policy_line_wraps_when_long_and_handles_empty_rollout():
    long = CLIReporter(wrap_width=60).format_policy(snap())
    assert "\n   Ante:" in long
    empty = CLIReporter().format_policy(snap(rollout_episodes=0, reward_mean=None, clear_rate=None,
                                              mean_ante=None, mean_round=None, deadliest_position=None))
    assert "Reward: n/a" in empty and "Deadliest: n/a" in empty


def test_policy_update_from_callback_uses_rollout_counters():
    h = Harness(OutputLevel.MINIMAL)
    h.step(info(action_taken=5))
    h.step(info(seq=1, score=400, cleared=True))
    h.step(info(name="Big Blind", target=450, rnd=2, action_taken=5))
    h.step(info(name="Big Blind", target=450, rnd=2), done=True, reward=10.0)
    h.cb._on_rollout_end()
    assert any("Policy update at 4096 steps | Episodes: 1 (+1) | Reward: 10.0 | Clear rate: 50%" in l for l in h.lines)
    assert not any("Win rate" in l for l in h.lines)


# --- boss ante latching ---

def test_boss_clear_counts_against_latched_ante_in_position_stats():
    h = Harness(OutputLevel.MINIMAL)
    h.step(info(ante=1, name="The Hook", target=600, rnd=3))
    h.step(info(ante=2, name="The Hook", target=600, rnd=3, cleared=True))  # end_round already eased the ante
    assert h.cb.blind_position_clears[(1, "Boss")] == 1
    assert h.cb.blind_position_clears[(2, "Boss")] == 0


def test_deadliest_boss_only_considers_boss_slot():
    cb = BalatroMetricsCallback(reporter=CLIReporter(out=lambda s: None))
    for _ in range(6):
        cb.blind_deaths_by_name["Small Blind"] += 1
    for _ in range(3):
        cb.blind_deaths_by_name["The Mouth"] += 1
        cb.blind_clears_by_name["The Mouth"] += 1
    assert cb._deadliest_blind()[0] == "Small Blind"
    assert cb._deadliest_blind(boss_only=True)[0] == "The Mouth"


# --- KeyListener ---

def test_key_listener_noop_when_not_tty():
    r = CLIReporter(out=lambda s: None)
    listener = KeyListener(r, stream=io.StringIO())
    assert listener.start() is False
    assert listener.active is False
    listener.stop()  # safe


def test_key_listener_handles_keys_without_tty():
    lines = []
    r = CLIReporter(out=lines.append)
    listener = KeyListener(r, stream=io.StringIO())
    listener.handle_key("1")
    assert r.level == OutputLevel.MINIMAL
    listener.handle_key("5")
    assert r.level == OutputLevel.VERBOSE
    listener.handle_key("?")
    listener.handle_key("x")  # ignored
    assert any("Output levels" in l for l in lines)


# --- review fixes ---

def test_stale_hand_on_first_step_is_not_recorded_but_fresh_run_hand_is():
    t = EpisodeTracker()
    ev = t.on_step(info(seq=7, hand="Flush", score=500), 1, 1)
    assert ev == [] and t.blinds == []
    ev = t.on_step(info(seq=8, hand="Pair", score=50), 1, 1)
    assert len(ev) == 1
    fresh = EpisodeTracker()
    fresh.on_step(info(action_taken=5), 1, 1)
    assert len(fresh.on_step(info(seq=1, score=100), 1, 1)) == 1


def test_cutoff_in_shop_after_boss_keeps_boss_ante():
    h = Harness(OutputLevel.SUMMARY)
    h.step(info(ante=1, name="The Hook", target=600, rnd=3))
    h.step(info(ante=2, name="The Hook", target=600, rnd=3, seq=1, score=700, cleared=True))
    h.step(info(ante=2, name="", target=0, rnd=3))  # shop: no blind, next ante
    h.cb.print_cutoff()
    assert "CUT OFF after Ante 1 - The Hook [round 3]" in h.text()


def test_defeat_marks_current_blind_not_earlier_cleared_one():
    h = Harness(OutputLevel.ROUND)
    play_small_blind_two_hands(h)
    h.step(info(name="Big Blind", target=800, rnd=2), done=True, reward=0.0)  # died before any hand
    out = h.text()
    assert "Ante 1 - Small Blind | 324/300 chips · 2 hands [round 1] [cleared!]" in out
    assert "Ante 1 - Big Blind | 0/800 chips · 0 hands [round 2] [DEFEATED!] ❌" in out


def test_missing_last_played_hand_warns_once(capsys):
    import ai.utils.cli_reporter as m
    m._warned_missing_hand = False
    t = EpisodeTracker()
    t.on_step(info(), 1, 1)
    t.on_step(info(), 1, 1)
    assert capsys.readouterr().out.count("needs redeploying") == 1


def test_endless_win_prints_won_and_episode_continues():
    h = Harness(OutputLevel.SUMMARY)
    h.step(info(ante=8, name="Cerulean Bell", target=50000, rnd=24))
    h.step(info(ante=8, name="Cerulean Bell", target=50000, rnd=24, seq=1, score=60000, cleared=True))
    h.step(info(ante=9, name="Cerulean Bell", target=50000, rnd=24, cleared=True, won=True), reward=5.0)
    assert any("WON! Ante 8 - Cerulean Bell" in l for l in h.lines)
    assert h.cb.tracker.steps == 3  # episode still in progress


def test_key_listener_restores_on_sigterm_and_chains():
    import os, pty, signal, termios
    m, sl = pty.openpty()
    stream = os.fdopen(sl, "r")
    before = termios.tcgetattr(sl)
    called = []
    old = signal.signal(signal.SIGTERM, lambda s, f: called.append(s))
    try:
        k = KeyListener(CLIReporter(out=lambda s: None), stream=stream)
        assert k.start()
        assert termios.tcgetattr(sl) != before
        os.kill(os.getpid(), signal.SIGTERM)
        assert called == [signal.SIGTERM]
        assert termios.tcgetattr(sl) == before
        k.stop()
        assert signal.getsignal(signal.SIGTERM).__name__ == "<lambda>"
    finally:
        signal.signal(signal.SIGTERM, old)


def test_invalid_output_level_falls_back_to_round():
    from ai.utils.cli_reporter import coerce_level
    for bad in (0, 9, -1, "x", None):
        assert coerce_level(bad) == OutputLevel.ROUND
    assert coerce_level(5) == OutputLevel.VERBOSE
