"""Tests for ai/utils/history_logger.py (synthetic prev/next game states)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.utils.history_logger import HistoryLogger


# --- helpers ---

def card(value, suit, nominal=None, hl=False):
    nominal = nominal if nominal is not None else {"A": 11, "K": 10, "Q": 10, "J": 10}.get(value, int(value) if value.isdigit() else 10)
    return {"base": {"value": value, "nominal": nominal}, "suit": suit, "highlighted": hl}


def gs(state=1, **kw):
    d = {"state": state, "ante": 1, "round_count": 1, "blind_name": "Small Blind", "blind_chips": 300,
         "blind_cleared": 0, "game_over": 0, "gold": 6, "chips": 0, "deck_name": "Red Deck", "seed": "ABCD1234",
         "round": {"hands_left": 4, "discards_left": 3}, "hand": {"cards": []}, "jokers": [], "consumables": [],
         "shop": {"items": []}, "pack_cards": [], "pack_choices": 0, "pack_name": "",
         "last_played_hand": {"handname": "None", "score": 0, "chips_before": 0, "seq": 0},
         "current_hand": {"handname": ""}}
    d.update(kw)
    return d


def st(**kw):
    return {"game_state": gs(**kw), "available_actions": []}


@pytest.fixture
def log(tmp_path):
    lg = HistoryLogger(log_dir=str(tmp_path), run_label="run")
    lg.path = tmp_path / "run.log"
    return lg


def text(lg):
    lg._file.flush()
    return lg.path.read_text(encoding="utf-8")


SPADE, DIAMOND, HEART = "Spades", "Diamonds", "Hearts"


def play_pair(lg, seq=1, before=0, score=64, blind_after=0, next_extra=None):
    hand = [card("A", SPADE, hl=True), card("A", DIAMOND, hl=True), card("J", DIAMOND, hl=True), card("6", HEART, hl=True)]
    prev = st(hand={"cards": hand}, blind_chips=300, chips=before,
              last_played_hand={"handname": "None", "score": 0, "chips_before": 0, "seq": seq - 1})
    nxt = st(chips=before + score, blind_chips=blind_after, round={"hands_left": 3, "discards_left": 3},
             last_played_hand={"handname": "Pair", "score": score, "chips_before": before, "seq": seq},
             **(next_extra or {}))
    lg.log_step(prev, {"action": 2, "params": []}, nxt)


# --- PLAY / SELECT / DISCARD ---

def test_play_shows_hand_type_feature_and_target(log):
    log.new_episode(1, deck="Red Deck", seed="S")
    play_pair(log)  # next_gs.blind_chips is 0, current_hand.handname is empty
    out = text(log)
    assert "▶ PLAY Pair (of Aces) [A♠, A♦, J♦, 6♥]  →  +64 chips (64/300) | hands left: 3" in out
    assert "[A1 Small Blind R1 $6] ▶ PLAY" in out


def test_select_and_discard_formatting(log):
    log.new_episode(1)
    hand = [card("10", HEART), card("6", DIAMOND)]
    prev = st(hand={"cards": hand})
    log.log_step(prev, {"action": 1, "params": [1, 2]}, prev)
    hand_hl = [dict(c, highlighted=True) for c in hand]
    prev2 = st(hand={"cards": hand_hl})
    log.log_step(prev2, {"action": 3, "params": []}, st(round={"hands_left": 4, "discards_left": 2}))
    out = text(log)
    assert "[A1 Small Blind R1 $6]     select [10♥, 6♦]" in out
    assert "[A1 Small Blind R1 $6] ✗ DISCARD [10♥, 6♦]  →  discards left: 2" in out


def test_play_followed_by_blank_line(log):
    log.new_episode(1)
    play_pair(log)
    lines = text(log).split("\n")
    i = next(k for k, l in enumerate(lines) if "▶ PLAY" in l)
    assert lines[i + 1] == ""


def test_play_without_record_falls_back_to_chip_diff(log):
    log.new_episode(1)
    hand = [card("K", SPADE, hl=True)]
    prev = st(hand={"cards": hand}, chips=10)
    nxt = st(chips=60, blind_chips=0)
    log.log_step(prev, {"action": 2, "params": []}, nxt)
    assert "+50 chips (60/300)" in text(log)


# --- context labels / header ---

def test_state_labels_replace_unknown(log):
    log.new_episode(1)
    log.log_step(st(state=7, blind_name=""), {"action": 5, "params": []}, st(state=1))
    log.log_step(st(state=5, blind_name="None"), {"action": 10, "params": []}, st(state=7))
    out = text(log)
    assert "Unknown" not in out
    assert "[A1 Blind Select R1 $6]" in out
    assert "[A1 Shop R1 $6]" in out


def test_seed_filled_from_game_state(log):
    log.new_episode(1, seed="")
    log.log_step(st(), {"action": 3, "params": []}, st())
    assert "Seed: ABCD1234" in text(log)


def test_blind_select_and_skip_blind(log):
    log.new_episode(1)
    sel = st(state=7, blind_on_deck="Small", blind_choices={"Small": "Small Blind", "Big": "Big Blind", "Boss": "The Goad"},
             blind_choice_chips={"Small": 300, "Big": 450, "Boss": 600}, blind_name="")
    log.log_step(sel, {"action": 5, "params": []}, st(state=1))
    sel2 = st(state=7, blind_on_deck="Big", blind_choices=sel["game_state"]["blind_choices"], blind_name="",
              run_stats={"tags": {"by_key": {}}})
    after = st(state=7, blind_on_deck="Boss", blind_name="", run_stats={"tags": {"by_key": {"tag_double": 1}}})
    log.log_step(sel2, {"action": 17, "params": []}, after)
    out = text(log)
    assert 'SELECT_BLIND "Small Blind" (300)' in out
    assert 'SKIP_BLIND "Big Blind"  →  tag: Double' in out


def test_reroll_boss(log):
    log.new_episode(1)
    prev = st(state=7, gold=20, blind_choices={"Small": "", "Big": "", "Boss": "The Goad"}, blind_name="")
    nxt = st(state=7, gold=10, blind_choices={"Small": "", "Big": "", "Boss": "The Wall"}, blind_name="")
    log.log_step(prev, {"action": 20, "params": []}, nxt)
    assert 'REROLL_BOSS "The Goad" → "The Wall" ($10)' in text(log)


# --- shop ---

def shop_items():
    return [
        {"name": "Joker", "cost": 2, "type": "joker", "item_set": "Joker", "slot": 1},
        {"name": "The Devil", "cost": 3, "type": "consumable", "item_set": "Tarot", "slot": 2},
        {"name": "Overstock", "cost": 10, "type": "voucher", "item_set": "Voucher", "slot": 1, "can_buy": True},
        {"name": "Buffoon Pack", "cost": 4, "type": "booster", "item_set": "Booster", "slot": 1},
    ]


def test_buy_tarot_labelled_tarot_and_voucher_no_slot(log):
    log.new_episode(1)
    prev = st(state=5, shop={"items": shop_items()}, gold=10)
    log.log_step(prev, {"action": 7, "params": [2]},
                 st(state=5, gold=7, consumables=[{"name": "The Devil"}], shop={"items": shop_items()}))
    log.log_step(prev, {"action": 12, "params": []}, st(state=5, gold=0))
    out = text(log)
    assert 'BUY "The Devil" (Tarot, slot 2, $3)' in out
    assert 'consumables: 0 → 1 (+"The Devil")' in out
    assert 'BUY_VOUCHER "Overstock" ($10)' in out
    assert "slot:None" not in out
    log.log_episode_end(1.0, [])
    out = text(log)
    assert 'BUY "The Devil": -3' in out
    assert 'BUY_VOUCHER "Overstock": -10' in out


def test_sell_consumable_shows_set_and_value(log):
    log.new_episode(1)
    prev = st(state=5, gold=4, consumables=[{"name": "Mercury", "item_set": "Planet", "sell_cost": 2}])
    log.log_step(prev, {"action": 19, "params": [1]}, st(state=5, gold=6))
    assert 'SELL_CONSUMABLE "Mercury" (Planet, +$2)' in text(log)


# --- packs ---

def test_pack_nested_under_shop_without_new_shop(log):
    log.new_episode(1)
    items = shop_items()
    cashout = st(state=8, blind_cleared=1)
    log.log_step(cashout, {"action": 11, "params": []}, st(state=5, gold=10, shop={"items": items}))
    shop = st(state=5, gold=10, shop={"items": items}, blind_name="")
    pack_cards = [{"name": "Joker", "item_set": "Joker", "slot": 1}, {"name": "Baron", "item_set": "Joker", "slot": 2}]
    pack = st(state=18, gold=6, pack_name="Buffoon Pack", pack_cards=pack_cards, pack_choices=1, blind_name="")
    log.log_step(shop, {"action": 13, "params": [1]}, pack)
    log.log_step(pack, {"action": 15, "params": [2]},
                 st(state=5, gold=6, jokers=[{"name": "Baron"}], pack_cards=[], pack_choices=0, blind_name=""))
    log.log_step(st(state=5, gold=6, blind_name=""), {"action": 10, "params": []}, st(state=7))
    log.log_episode_end(0.0, [])
    out = text(log)
    assert out.count("=== SHOP") == 1
    assert "=== SHOP 1 ===" in out
    assert "Shops: 1" in out
    assert "    --- Buffoon Pack (choose 1 of 2) ---" in out
    assert "    1: Joker (Joker)  2: Baron (Joker)" in out
    assert '    [A1 Pack R1 $6]   TAKE slot 2: "Baron" (Joker)' in out
    assert 'jokers: 0 → 1 (+"Baron")' in out
    assert "Unknown" not in out


def test_standard_pack_playing_cards(log):
    log.new_episode(1)
    cards = [{"name": "Default Base", "item_set": "Default", "slot": 1, "value": "K", "suit": "Spades",
              "enhancement": "None", "seal": "None", "edition": "none"},
             {"name": "Bonus", "item_set": "Enhanced", "slot": 2, "value": "5", "suit": "Hearts",
              "enhancement": "Bonus", "seal": "Red", "edition": "foil"}]
    pack = st(state=17, pack_name="Standard Pack", pack_cards=cards, pack_choices=1)
    log.log_step(pack, {"action": 15, "params": [2]}, st(state=5, pack_cards=[]))
    out = text(log)
    assert "    1: K♠  2: 5♥ Bonus +Red Seal +Foil" in out
    assert "TAKE slot 2: 5♥ Bonus +Red Seal +Foil" in out


def test_pack_from_skip_tag_nests_under_blind_select(log):
    log.new_episode(1)
    bs = st(state=7, blind_name="")
    log.log_step(bs, {"action": 17, "params": []}, st(state=9))
    pack = st(state=9, pack_name="Arcana Pack", pack_cards=[{"name": "Death", "item_set": "Tarot", "slot": 1}], pack_choices=1)
    log.log_step(pack, {"action": 16, "params": []}, st(state=7))
    log.log_step(st(state=7), {"action": 5, "params": []}, st(state=1))
    out = text(log)
    assert out.count("=== BLIND SELECT ===") == 1
    assert "--- Arcana Pack (choose 1 of 1) ---" in out


# --- episode header / end ---

def test_reward_patch_keeps_file_length_and_other_bytes(log):
    log.new_episode(3, deck="Red Deck", seed="S")
    play_pair(log)
    log._file.flush()
    before = log.path.read_bytes()
    assert b"Reward:    pending" in before
    log.log_episode_end(12.345, ["a: 1"])
    # compare the file as it would be without the patch by re-reading and diffing the header only
    after = log.path.read_bytes()
    assert b"Reward:     +12.35" in after
    assert b"pending" not in after
    hdr_start = before.index(b"EPISODE 3")
    hdr_end = before.index(b"\n", hdr_start)
    assert after[:hdr_start] == before[:hdr_start]
    assert len(after[hdr_start:hdr_end]) == len(before[hdr_start:hdr_end])
    assert after[hdr_end:hdr_end + (len(before) - hdr_end)] == before[hdr_end:]
    assert text(log).rstrip().endswith("─" * 70)


def test_crash_leaves_pending(log):
    log.new_episode(1)
    play_pair(log)
    assert "Reward:    pending" in text(log)


def test_cleared_blinds_use_latched_target_and_hand_count(log):
    log.new_episode(1)
    log.log_step(st(state=7, blind_name=""), {"action": 5, "params": []}, st(state=1))
    play_pair(log, seq=1, before=0, score=100)
    # clearing hand: next_gs has blind_chips 0 and state ROUND_EVAL
    play_pair(log, seq=2, before=100, score=264,
              next_extra={"state": 8, "blind_cleared": 1})
    log.log_episode_end(5.0, [])
    out = text(log)
    assert "Ante 1 Small Blind  364/300 chips  2 hands  Round 1" in out
    assert "/0 " not in out


def test_defeated_blind_appended(log):
    log.new_episode(1)
    play_pair(log, seq=1, before=0, score=100)
    play_pair(log, seq=2, before=100, score=50, next_extra={"state": 6, "game_over": 1})
    log.log_episode_end(-5.0, [])
    out = text(log)
    assert "Ante 1 Small Blind  150/300 chips  2 hands  Round 1  [DEFEATED ❌]" in out


# --- rotation ---

def test_rotation_at_episode_boundary(tmp_path):
    lg = HistoryLogger(log_dir=str(tmp_path), run_label="run")
    lg.ROTATE_BYTES = 200
    lg.new_episode(1)
    play_pair(lg)
    play_pair(lg, seq=2)
    assert not (tmp_path / "run_part_02.log").exists()  # no rotation mid-episode
    lg.log_episode_end(1.0, [])
    part2 = tmp_path / "run_part_02.log"
    assert part2.exists()
    lg.new_episode(2)
    play_pair(lg)
    lg.log_episode_end(2.0, [])
    assert (tmp_path / "run_part_03.log").exists()
    lg.close()
    first = (tmp_path / "run.log").read_text(encoding="utf-8")
    assert "EPISODE 1" in first and "EPISODE 2" not in first
    assert "continued (part 02)" in part2.read_text(encoding="utf-8")
    assert "EPISODE 2" in part2.read_text(encoding="utf-8")


def test_next_history_session_ignores_parts(tmp_path, monkeypatch):
    import ai.train_balatro as tb
    hist = tmp_path / "001_g" / "history"
    hist.mkdir(parents=True)
    for n in ("run_session_01.log", "run_session_02.log", "run_session_02_part_02.log", "run_session_02_part_09.log"):
        (hist / n).write_text("x")
    monkeypatch.setattr(tb, "LOGS_ROOT", tmp_path)
    assert tb.next_session_number("run") == 3


def test_malformed_step_is_logged_not_raised(log):
    log.new_episode(1, deck="Red Deck", seed="X")
    bad_prev = {'game_state': {'state': 5, 'gold': 'ten', 'shop': {'items': [{'type': 'joker', 'slot': 1}]}}}
    bad_next = {'game_state': {'state': 5, 'gold': None}}
    log.log_step(bad_prev, {'action': 7, 'params': [1]}, bad_next)
    log.log_episode_end(None, None)
    assert "history logger error" in text(log)
