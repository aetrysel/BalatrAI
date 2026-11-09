"""
History logger — writes a human-readable record of the bot's actions and their effects.
One log file per training session, stored in the log group's history/ dir as <run name>_session_NN.log.
Steps are written in real-time; cleared-blinds TOC and money events appear at episode end.

Layout per episode:
  - EPISODE header with a fixed-width `Reward:` field ("pending") that is patched in place at episode end,
    then a Deck / Seed / Stake line written on the first step (the seed comes from the game state when
    the configured seed is empty).
  - One `=== SECTION ===` header per state change (blind, blind select, cashout, shop n).
  - Booster packs are nested under the section that opened them (shop or blind select) as an indented
    `--- Pack name (choose k of n) ---` block listing the pack contents; returning to the shop neither
    re-prints the shop header nor bumps the shop counter.
  - Hands come from the evaluation-time `last_played_hand` record (hand type, score, chips toward the
    blind target); PLAY lines are marked with a play marker, DISCARD with a cross, SELECT lines are indented.
  - Episode end appends CLEARED BLINDS (with per-blind hand counts and the losing blind as DEFEATED),
    MONEY EVENTS and the reward breakdown.

A session file that exceeds ROTATE_BYTES at an episode boundary is continued in <name>_part_NN.log.
"""

import functools
import os
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from .constants import SHOP_STATE_ID, PACK_STATE_IDS

STATE_NAMES = {
    1: "SELECTING_HAND", 4: "MENU", 5: "SHOP",
    6: "GAME_OVER", 7: "BLIND_SELECT", 8: "ROUND_EVAL",
    9: "TAROT_PACK", 10: "PLANET_PACK", 15: "SPECTRAL_PACK",
    17: "STANDARD_PACK", 18: "BUFFOON_PACK", 999: "SMODS_BOOSTER_OPENED",
}

ACTION_NAMES = {
    1: "SELECT_HAND", 2: "PLAY_HAND", 3: "DISCARD",
    4: "START_RUN", 5: "SELECT_BLIND", 6: "RESTART_RUN",
    7: "BUY_JOKER", 8: "SELL_JOKER", 9: "REROLL_SHOP",
    10: "SKIP_SHOP", 11: "CASH_OUT", 12: "BUY_VOUCHER",
    13: "BUY_BOOSTER", 14: "USE_CONSUMABLE",
    15: "SELECT_PACK_CARD", 16: "SKIP_BOOSTER", 17: "SKIP_BLIND",
    18: "BUY_AND_USE_CONSUMABLE", 19: "SELL_CONSUMABLE", 20: "REROLL_BOSS",
}

# Context-bracket labels for states that have no blind name
STATE_LABELS = {
    1: "Hand", 4: "Menu", 5: "Shop", 6: "Game Over", 7: "Blind Select", 8: "Cashout",
    9: "Pack", 10: "Pack", 15: "Pack", 17: "Pack", 18: "Pack", 999: "Pack",
}

SUIT_SYMBOLS = {"Spades": "♠", "Hearts": "♥", "Clubs": "♣", "Diamonds": "♦"}

BLIND_SLOT_NAMES = {"Small": "Small Blind", "Big": "Big Blind", "Boss": "Boss Blind"}

# Width of the Reward field in the EPISODE header (patched in place at episode end)
REWARD_FIELD_WIDTH = 10


def _playing_card_str(card: dict) -> str:
    """Pack card as text: 'K♠ Bonus +Red Seal +Foil' for playing cards, 'Name (Set)' otherwise."""
    if card.get('value') is not None and card.get('suit') is not None:
        suit = SUIT_SYMBOLS.get(card['suit'], str(card['suit'])[:1])
        text = f"{card['value']}{suit}"
        enhancement = card.get('enhancement', 'None')
        if enhancement and enhancement not in ('None', 'Default Base'):
            text += f" {enhancement}"
        seal = card.get('seal', 'None')
        if seal and seal != 'None':
            text += f" +{seal} Seal"
        edition = card.get('edition', 'none')
        if edition and edition != 'none':
            text += f" +{str(edition).title()}"
        return text
    return f"{card.get('name', 'Unknown')} ({card.get('item_set', 'Unknown')})"


def _card_str(card):
    suit = SUIT_SYMBOLS.get(card.get('suit', ''), card.get('suit', '')[:1])
    value = card.get('base', {}).get('value', '?')
    return f"{value}{suit}"


def _shop_jokers_item(shop_items: list, slot):
    """Item at a raw G.shop_jokers slot (jokers and consumables share that area)."""
    return next((it for it in shop_items
                 if it.get('type') in ('joker', 'consumable') and it.get('slot') == slot), None)


def _hand_primary_feature(hand_name: str, played_cards: list) -> str:
    """Returns a short description of the primary identifying feature of a played hand."""
    if not played_cards or not hand_name:
        return ""

    values = [c.get('base', {}).get('value', '?') for c in played_cards]
    suits = [c.get('suit', '') for c in played_cards]
    rank_counts = Counter(values)
    suit_counts = Counter(suits)
    rank_nominal = {
        c.get('base', {}).get('value', '?'): c.get('base', {}).get('nominal', 0)
        for c in played_cards
    }

    def rank_plural(r: str) -> str:
        return {'A': 'Aces', 'K': 'Kings', 'Q': 'Queens', 'J': 'Jacks'}.get(r, f"{r}s")

    dominant_suit = suit_counts.most_common(1)[0][0] if suit_counts else ''

    if hand_name == 'High Card':
        high = max(played_cards, key=lambda c: c.get('base', {}).get('nominal', 0))
        return high.get('base', {}).get('value', '?')

    elif hand_name == 'Pair':
        pairs = [r for r, cnt in rank_counts.items() if cnt >= 2]
        if pairs:
            return f"of {rank_plural(max(pairs, key=lambda r: rank_nominal.get(r, 0)))}"
        return ""

    elif hand_name == 'Two Pair':
        pairs = sorted(
            [r for r, cnt in rank_counts.items() if cnt >= 2],
            key=lambda r: rank_nominal.get(r, 0), reverse=True
        )
        if len(pairs) >= 2:
            return f"{rank_plural(pairs[0])} and {rank_plural(pairs[1])}"
        return ""

    elif hand_name == 'Three of a Kind':
        trips = [r for r, cnt in rank_counts.items() if cnt >= 3]
        if trips:
            return rank_plural(max(trips, key=lambda r: rank_nominal.get(r, 0)))
        return ""

    elif hand_name == 'Straight':
        high = max(played_cards, key=lambda c: c.get('base', {}).get('nominal', 0))
        return f"{high.get('base', {}).get('value', '?')}-high"

    elif hand_name == 'Flush':
        return dominant_suit

    elif hand_name == 'Full House':
        trips = [r for r, cnt in rank_counts.items() if cnt >= 3]
        pairs = [r for r, cnt in rank_counts.items() if cnt == 2]
        if trips and pairs:
            t = max(trips, key=lambda r: rank_nominal.get(r, 0))
            p = max(pairs, key=lambda r: rank_nominal.get(r, 0))
            return f"{rank_plural(t)} over {rank_plural(p)}"
        return ""

    elif hand_name == 'Four of a Kind':
        quads = [r for r, cnt in rank_counts.items() if cnt >= 4]
        if quads:
            return rank_plural(max(quads, key=lambda r: rank_nominal.get(r, 0)))
        return ""

    elif hand_name == 'Straight Flush':
        high = max(played_cards, key=lambda c: c.get('base', {}).get('nominal', 0))
        return f"{dominant_suit} {high.get('base', {}).get('value', '?')}-high"

    elif hand_name == 'Five of a Kind':
        if rank_counts:
            return rank_plural(rank_counts.most_common(1)[0][0])
        return ""

    elif hand_name == 'Flush House':
        trips = [r for r, cnt in rank_counts.items() if cnt >= 3]
        pairs = [r for r, cnt in rank_counts.items() if cnt == 2]
        if trips and pairs:
            t = max(trips, key=lambda r: rank_nominal.get(r, 0))
            p = max(pairs, key=lambda r: rank_nominal.get(r, 0))
            return f"{rank_plural(t)} over {rank_plural(p)} in {dominant_suit}"
        return ""

    elif hand_name == 'Flush Five':
        if rank_counts:
            r = rank_counts.most_common(1)[0][0]
            suit_sym = SUIT_SYMBOLS.get(dominant_suit, dominant_suit[:1])
            return f"{r}{suit_sym}"
        return ""

    return ""



def _never_raise(method):
    """Runs a logger method, writing any exception to the log instead of raising it into the training loop."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception:
            try:
                self._write(f"  [history logger error in {method.__name__}] "
                            + traceback.format_exc().strip().splitlines()[-1])
            except Exception:
                pass
    return wrapper


class HistoryLogger:
    # Session file size (bytes) above which the log continues in a _part_NN file at the next episode boundary
    ROTATE_BYTES = 15 * 1024 * 1024

    def __init__(self, log_dir="game_history_logs", run_label: str = "", joker_sell_mode: str = "shop"):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        if run_label:
            self._base_path = os.path.join(log_dir, f"{run_label}")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._base_path = os.path.join(log_dir, f"history_{ts}")
        self._part = 1
        self._run_label = run_label
        self._file = open(f"{self._base_path}.log", 'w', buffering=1, encoding='utf-8', newline='\n')
        self.episode = 0
        self._file_line_count = 0
        self._last_line_blank = False

        self._cleared_blinds: list[dict] = []
        self._defeat: dict | None = None
        self._shop_count: int = 0
        self._money_log: list[str] = []
        self._ep_deck: str = "Unknown"
        self._ep_seed: str = ""
        self._ep_stake: str = "White"
        self._ep_ts: str = ""
        self._last_logged_state: int = -1
        self._in_pack: bool = False
        self._pending_pack_name: str = ""
        self._info_pending: bool = False
        self._header_pos: int | None = None
        self._blind_target: int = 0
        self._blind_hands: int = 0
        self._blind_total: int = 0

        session_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"{'='*70}")
        self._write(f"Training session: {session_ts}" + (f"  |  Run: {run_label}" if run_label else "")
                    + f"  |  Joker selling: {joker_sell_mode}")
        self._write(f"{'='*70}")

    def _write(self, line: str) -> None:
        self._file.write(line + '\n')
        self._file_line_count += 1
        self._last_line_blank = (line == "")

    def _blank(self) -> None:
        """Blank separator line, never doubled."""
        if not self._last_line_blank:
            self._write("")

    def _episode_header_line(self, reward_text: str) -> str:
        return f"EPISODE {self.episode} | {self._ep_ts} | Reward: {reward_text:>{REWARD_FIELD_WIDTH}}"

    def new_episode(self, episode_num: int, deck: str = "Unknown", seed: str = "", stake: str = "White") -> None:
        self.episode = episode_num
        self._cleared_blinds = []
        self._defeat = None
        self._shop_count = 0
        self._money_log = []
        self._ep_deck = deck
        self._ep_seed = seed
        self._ep_stake = stake
        self._ep_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._last_logged_state = -1
        self._in_pack = False
        self._pending_pack_name = ""
        self._blind_target = 0
        self._blind_hands = 0
        self._blind_total = 0

        self._write("")
        self._write("─" * 70)
        # Byte offset of the header line so log_episode_end can patch the reward in place
        self._file.flush()
        self._header_pos = self._file.tell()
        self._write(self._episode_header_line("pending"))
        self._write("─" * 70)
        # Deck / Seed / Stake line is written with the first step, when the game state is known
        self._info_pending = True

    def _flush_episode_info(self, gs: dict) -> None:
        if not self._info_pending:
            return
        self._info_pending = False
        deck = self._ep_deck
        gs_deck = gs.get('deck_name')
        if gs_deck and gs.get('state') != 4:
            deck = gs_deck
        seed = self._ep_seed or (str(gs['seed']) if gs.get('seed') else "")
        self._write(f"  Deck: {deck}  |  Seed: {seed or 'random'}  |  Stake: {self._ep_stake}")

    @staticmethod
    def _context_label(gs: dict) -> str:
        blind_name = gs.get('blind_name', '')
        if not blind_name or blind_name in ('None', 'Unknown'):
            return STATE_LABELS.get(gs.get('state', 0), 'Round')
        return blind_name

    def _get_section_title(self, state_id: int, prev_gs: dict) -> str:
        if state_id == 7:
            return "=== BLIND SELECT ==="
        elif state_id == 1:
            blind_name = prev_gs.get('blind_name', '')
            if not blind_name or blind_name in ('None', 'Unknown'):
                blind_name = 'Blind'
            target = prev_gs.get('blind_chips', 0) or self._blind_target
            return f"=== {blind_name.upper()}{f' ({target})' if target else ''} ==="
        elif state_id == 8:
            return "=== CASHOUT ==="
        elif state_id == 5:
            return f"=== SHOP {self._shop_count} ==="
        return f"=== {STATE_NAMES.get(state_id, f'STATE {state_id}')} ==="

    @staticmethod
    def _tag_counts(gs: dict) -> dict:
        tags = (gs.get('run_stats') or {}).get('tags', {})
        by_key = tags.get('by_key', {}) if isinstance(tags, dict) else {}
        return by_key if isinstance(by_key, dict) else {}

    @staticmethod
    def _blind_on_deck_name(gs: dict, fallback_gs: dict = None) -> str:
        """Name of the blind currently offered at blind select."""
        slot = gs.get('blind_on_deck', '')
        choices = gs.get('blind_choices') or {}
        name = choices.get(slot, '') if isinstance(choices, dict) else ''
        if name:
            return name
        if slot in BLIND_SLOT_NAMES:
            if slot == 'Boss' and fallback_gs:
                boss = fallback_gs.get('blind_name', '')
                if boss and boss not in ('None', 'Unknown'):
                    return boss
            return BLIND_SLOT_NAMES[slot]
        return 'Blind'

    @staticmethod
    def _play_record(prev_gs: dict, next_gs: dict):
        """New evaluation-time hand record produced by this step, or None when seq did not advance."""
        rec = next_gs.get('last_played_hand') or {}
        prev_seq = (prev_gs.get('last_played_hand') or {}).get('seq', 0)
        if rec.get('seq', 0) and rec.get('seq') != prev_seq:
            return rec
        return None

    def _open_pack_block(self, prev_gs: dict) -> None:
        pack_name = prev_gs.get('pack_name') or self._pending_pack_name or \
            STATE_NAMES.get(prev_gs.get('state', 0), 'Pack').replace('_', ' ').title()
        cards = prev_gs.get('pack_cards', [])
        choose = prev_gs.get('pack_choices', 1)
        self._write(f"    --- {pack_name} (choose {choose} of {len(cards)}) ---")
        if cards:
            self._write("    " + "  ".join(f"{i}: {_playing_card_str(c)}" for i, c in enumerate(cards, 1)))
        self._pending_pack_name = ""

    @_never_raise
    def log_step(self, prev_state, action_data, next_state):
        if not prev_state or not next_state or not action_data:
            return

        prev_gs = prev_state.get('game_state', {})
        next_gs = next_state.get('game_state', {})

        self._flush_episode_info(prev_gs)

        curr_state = prev_gs.get('state', 0)
        next_state_id = next_gs.get('state', 0)

        # Shop visit tracking: count transitions into SHOP state from outside the shop and its packs
        if (curr_state != SHOP_STATE_ID and curr_state not in PACK_STATE_IDS
                and next_state_id == SHOP_STATE_ID):
            self._shop_count += 1

        # Section header on state change; pack states nest under the section that opened them
        in_pack = curr_state in PACK_STATE_IDS
        if in_pack:
            if not self._in_pack:
                self._open_pack_block(prev_gs)
            self._in_pack = True
        else:
            self._in_pack = False
            if curr_state != self._last_logged_state:
                if self._last_logged_state != -1:
                    self._blank()
                self._write(self._get_section_title(curr_state, prev_gs))
                self._last_logged_state = curr_state

        action_id = action_data.get('action', 0)
        params = action_data.get('params', [])

        ante = prev_gs.get('ante', 1)
        gold = prev_gs.get('gold', 0)
        round_count = prev_gs.get('round_count', 0)

        # Per-blind latches: target, hands played, last cumulative chips
        if curr_state == 1 and prev_gs.get('blind_chips', 0) > 0:
            self._blind_target = prev_gs['blind_chips']
        if action_id == 5:
            self._blind_hands = 0
            self._blind_total = 0
            chosen = (prev_gs.get('blind_choice_chips') or {}).get(prev_gs.get('blind_on_deck', ''), 0)
            self._blind_target = chosen or next_gs.get('blind_chips', 0) or 0
        if action_id == 2 and curr_state == 1:
            self._blind_hands += 1
            record = self._play_record(prev_gs, next_gs)
            if record:
                self._blind_total = record.get('chips_before', 0) + record.get('score', 0)
            else:
                self._blind_total = max(self._blind_total, next_gs.get('chips', 0))

        action_str = self._format_action(action_id, params, prev_gs, next_gs=next_gs)
        effect_str = self._format_effect(action_id, params, prev_gs, next_gs)

        marker = {2: "▶", 3: "✗"}.get(action_id, " ")
        indent = "    " if in_pack else "  "
        context = f"[A{ante} {self._context_label(prev_gs)} R{round_count} ${gold}]"
        line = f"{indent}{context} {marker} {action_str}"
        if effect_str:
            if effect_str.startswith("("):
                line += f" {effect_str}"
            else:
                line += f"  →  {effect_str}"
        self._write(line)
        if action_id == 2:
            self._write("")

        self._track_money(action_id, params, prev_gs, next_gs)

        # Blind clear detection
        if next_gs.get('blind_cleared', 0) == 1 and prev_gs.get('blind_cleared', 0) != 1:
            blind_nm = next_gs.get('blind_name', '') or prev_gs.get('blind_name', '')
            self._cleared_blinds.append({
                'name': blind_nm,
                'ante': next_gs.get('ante', ante),
                'chips': self._blind_total or next_gs.get('chips', 0),
                'blind_chips': self._blind_target or prev_gs.get('blind_chips', 0),
                'hands': self._blind_hands,
                'round_num': next_gs.get('round_count', round_count),
                'is_boss': blind_nm not in ('Small Blind', 'Big Blind', 'None', ''),
            })
            self._blind_hands = 0
            self._blind_total = 0
            self._blind_target = 0

        # Losing blind (episode ends in game over)
        if (next_gs.get('game_over', 0) == 1 or next_state_id == 6) and curr_state != 6 and self._defeat is None:
            blind_nm = prev_gs.get('blind_name', '') or next_gs.get('blind_name', '')
            self._defeat = {
                'name': blind_nm,
                'ante': prev_gs.get('ante', ante),
                'chips': self._blind_total or next_gs.get('chips', 0),
                'blind_chips': self._blind_target or prev_gs.get('blind_chips', 0),
                'hands': self._blind_hands,
                'round_num': prev_gs.get('round_count', round_count),
                'is_boss': blind_nm not in ('Small Blind', 'Big Blind', 'None', ''),
            }

    def _track_money(self, action_id, params, prev_gs, next_gs) -> None:
        """Money event tracking for the MONEY EVENTS summary."""
        prev_gold = prev_gs.get('gold', 0)
        next_gold = next_gs.get('gold', 0)
        delta = next_gold - prev_gold
        if delta == 0:
            return
        slot = params[0] if params else None
        shop_items = prev_gs.get('shop', {}).get('items', [])
        span = f"{delta:+d} (${prev_gold} → ${next_gold})"
        if action_id == 7:  # BUY_JOKER
            item = _shop_jokers_item(shop_items, slot) or {}
            self._money_log.append(f"  BUY \"{item.get('name', 'item')}\": {span}")
        elif action_id == 8:  # SELL_JOKER
            jokers_list = prev_gs.get('jokers', [])
            joker = jokers_list[slot - 1] if slot and 0 < slot <= len(jokers_list) else {}
            self._money_log.append(f"  SELL_JOKER \"{joker.get('name', 'joker')}\": {span}")
        elif action_id == 9:  # REROLL_SHOP
            self._money_log.append(f"  REROLL: {span}")
        elif action_id == 11:  # CASH_OUT (end-of-round payout)
            self._money_log.append(f"  CASH_OUT (round payout): {span}")
        elif action_id == 12:  # BUY_VOUCHER
            item = self._voucher_item(shop_items) or {}
            self._money_log.append(f"  BUY_VOUCHER \"{item.get('name', 'voucher')}\": {span}")
        elif action_id == 13:  # BUY_BOOSTER
            item = self._booster_item(shop_items, slot) or {}
            self._money_log.append(f"  BUY_BOOSTER \"{item.get('name', 'pack')}\": {span}")
        elif action_id == 18:  # BUY_AND_USE_CONSUMABLE
            item = _shop_jokers_item(shop_items, slot) or {}
            self._money_log.append(f"  BUY_AND_USE \"{item.get('name', 'item')}\": {span}")
        elif action_id == 19:  # SELL_CONSUMABLE
            consumables_list = prev_gs.get('consumables', [])
            item = consumables_list[slot - 1] if slot and 0 < slot <= len(consumables_list) else {}
            self._money_log.append(f"  SELL_CONSUMABLE \"{item.get('name', 'consumable')}\": {span}")

    @staticmethod
    def _voucher_item(shop_items: list):
        """Voucher targeted by BUY_VOUCHER (which has no params): the first buyable one, else the first."""
        vouchers = [it for it in shop_items if it.get('type') == 'voucher']
        return next((v for v in vouchers if v.get('can_buy')), vouchers[0] if vouchers else None)

    @staticmethod
    def _booster_item(shop_items: list, slot):
        boosters = [it for it in shop_items if it.get('type') == 'booster']
        item = next((b for b in boosters if b.get('slot') == slot), None)
        if item is None and slot and 0 < slot <= len(boosters):
            item = boosters[slot - 1]
        return item

    def _format_action(self, action_id, params, prev_gs, next_gs=None):
        slot = params[0] if params else None
        next_gs = next_gs or {}
        shop_items = prev_gs.get('shop', {}).get('items', [])

        if action_id == 1:  # SELECT_HAND
            hand_cards = prev_gs.get('hand', {}).get('cards', [])
            selected = [hand_cards[i - 1] for i in params if 0 < i <= len(hand_cards)]
            if selected:
                return f"  select [{', '.join(_card_str(c) for c in selected)}]"
            return "  select"

        elif action_id == 2:  # PLAY_HAND
            hand_cards = prev_gs.get('hand', {}).get('cards', [])
            highlighted = [c for c in hand_cards if c.get('highlighted')]
            record = self._play_record(prev_gs, next_gs)
            hand_name = record.get('handname', '') if record else next_gs.get('current_hand', {}).get('handname', '')
            if hand_name in ('None', None):
                hand_name = ''
            feature = _hand_primary_feature(hand_name, highlighted) if hand_name and highlighted else ''
            feature_str = f" ({feature})" if feature else ""
            if highlighted:
                cards_str = ', '.join(_card_str(c) for c in highlighted)
                if hand_name:
                    return f"PLAY {hand_name}{feature_str} [{cards_str}]"
                return f"PLAY [{cards_str}]"
            return "PLAY_HAND"

        elif action_id == 3:  # DISCARD
            hand_cards = prev_gs.get('hand', {}).get('cards', [])
            highlighted = [c for c in hand_cards if c.get('highlighted')]
            if highlighted:
                return f"DISCARD [{', '.join(_card_str(c) for c in highlighted)}]"
            return "DISCARD"

        elif action_id == 5:  # SELECT_BLIND
            name = self._blind_on_deck_name(prev_gs, next_gs)
            target = (prev_gs.get('blind_choice_chips') or {}).get(prev_gs.get('blind_on_deck', ''), 0) \
                or next_gs.get('blind_chips', 0)
            return f"SELECT_BLIND \"{name}\"" + (f" ({target})" if target else "")

        elif action_id == 7:  # BUY_JOKER (jokers and consumables share the area)
            item = _shop_jokers_item(shop_items, slot)
            if item:
                return f"BUY \"{item['name']}\" ({item.get('item_set', 'Joker')}, slot {slot}, ${item.get('cost', 0)})"
            return f"BUY slot {slot}"

        elif action_id == 8:  # SELL_JOKER
            jokers = prev_gs.get('jokers', [])
            joker = jokers[slot - 1] if slot and 0 < slot <= len(jokers) else None
            if joker:
                return f"SELL_JOKER \"{joker['name']}\" (+${joker.get('sell_cost', 0)})"
            return f"SELL_JOKER slot:{slot}"

        elif action_id == 9:  # REROLL_SHOP
            cost = prev_gs.get('reroll_cost', 5)
            return f"REROLL_SHOP (${cost})"

        elif action_id == 10:  # SKIP_SHOP
            return "SKIP_SHOP"

        elif action_id == 12:  # BUY_VOUCHER
            item = self._voucher_item(shop_items)
            if item:
                return f"BUY_VOUCHER \"{item['name']}\" (${item.get('cost', 0)})"
            return "BUY_VOUCHER"

        elif action_id == 13:  # BUY_BOOSTER
            item = self._booster_item(shop_items, slot)
            if item:
                self._pending_pack_name = item['name']
                return f"BUY_BOOSTER \"{item['name']}\" (slot {slot}, ${item.get('cost', 0)})"
            return f"BUY_BOOSTER slot {slot}"

        elif action_id == 14:  # USE_CONSUMABLE
            consumables = prev_gs.get('consumables', [])
            item = consumables[slot - 1] if slot and 0 < slot <= len(consumables) else None
            if item:
                hand_cards = prev_gs.get('hand', {}).get('cards', [])
                targets = [c for c in hand_cards if c.get('highlighted')]
                on_str = f" on [{', '.join(_card_str(c) for c in targets)}]" if targets else ""
                return f"USE_CONSUMABLE \"{item['name']}\"{on_str}"
            return f"USE_CONSUMABLE slot:{slot}"

        elif action_id == 15:  # SELECT_PACK_CARD
            pack_cards = prev_gs.get('pack_cards', [])
            card = pack_cards[slot - 1] if slot and 0 < slot <= len(pack_cards) else None
            if card:
                if card.get('value') is not None and card.get('suit') is not None:
                    return f"TAKE slot {slot}: {_playing_card_str(card)}"
                return f"TAKE slot {slot}: \"{card['name']}\" ({card.get('item_set', 'Unknown')})"
            return f"TAKE slot {slot}"

        elif action_id == 16:  # SKIP_BOOSTER
            return "SKIP_BOOSTER"

        elif action_id == 17:  # SKIP_BLIND
            return f"SKIP_BLIND \"{self._blind_on_deck_name(prev_gs, next_gs)}\""

        elif action_id == 18:  # BUY_AND_USE_CONSUMABLE
            item = _shop_jokers_item(shop_items, slot)
            if item:
                return f"BUY_AND_USE \"{item['name']}\" ({item.get('item_set', 'Consumable')}, slot {slot}, ${item.get('cost', 0)})"
            return f"BUY_AND_USE slot {slot}"

        elif action_id == 19:  # SELL_CONSUMABLE
            consumables = prev_gs.get('consumables', [])
            item = consumables[slot - 1] if slot and 0 < slot <= len(consumables) else None
            if item:
                gain = item.get('sell_cost')
                if gain is None:
                    gain = next_gs.get('gold', prev_gs.get('gold', 0)) - prev_gs.get('gold', 0)
                return f"SELL_CONSUMABLE \"{item['name']}\" ({item.get('item_set', 'Consumable')}, +${gain})"
            return f"SELL_CONSUMABLE slot:{slot}"

        elif action_id == 20:  # REROLL_BOSS
            old = (prev_gs.get('blind_choices') or {}).get('Boss', '')
            new = (next_gs.get('blind_choices') or {}).get('Boss', '')
            cost = prev_gs.get('gold', 0) - next_gs.get('gold', prev_gs.get('gold', 0))
            cost = cost if cost > 0 else 10
            names = f" \"{old}\" → \"{new}\"" if old and new else ""
            return f"REROLL_BOSS{names} (${cost})"

        return ACTION_NAMES.get(action_id, f"ACTION_{action_id}")

    @staticmethod
    def _gained_str(prev_gs: dict, next_gs: dict) -> str:
        """' | jokers: a → b (+"X") | consumables: a → b (+"Y")' for any growth in the joker / consumable rows."""
        out = ""
        for key in ('jokers', 'consumables'):
            before = len(prev_gs.get(key, []))
            after = len(next_gs.get(key, []))
            if after > before:
                out += f" | {key}: {before} → {after} (+\"{next_gs[key][-1].get('name', 'item')}\")"
        return out

    def _format_effect(self, action_id, params, prev_gs, next_gs):
        if action_id == 2:  # PLAY_HAND — hand name is in action_str
            hands_left = next_gs.get('round', {}).get('hands_left', 0)
            target = prev_gs.get('blind_chips', 0) or next_gs.get('blind_chips', 0) or self._blind_target
            record = self._play_record(prev_gs, next_gs)
            if record:
                gain = record.get('score', 0)
                total = record.get('chips_before', 0) + gain
            else:
                gain = next_gs.get('chips', 0) - prev_gs.get('chips', 0)
                total = next_gs.get('chips', 0)
            if gain > 0:
                return f"+{gain} chips ({total}/{target}) | hands left: {hands_left}"
            return f"hands left: {hands_left}"

        elif action_id == 3:  # DISCARD
            discards_left = next_gs.get('round', {}).get('discards_left', 0)
            return f"discards left: {discards_left}"

        elif action_id == 14:  # USE_CONSUMABLE — detect failure
            prev_count = len(prev_gs.get('consumables', []))
            next_count = len(next_gs.get('consumables', []))
            if next_count >= prev_count:
                return "(FAIL)"
            return ""

        elif action_id == 17:  # SKIP_BLIND — tag gained
            before = self._tag_counts(prev_gs)
            gained = [k for k, v in self._tag_counts(next_gs).items()
                      if isinstance(v, (int, float)) and v > before.get(k, 0)]
            if gained:
                return "tag: " + ", ".join(k.removeprefix('tag_').replace('_', ' ').title() for k in gained)
            return ""

        elif action_id in (7, 13, 12, 18):  # BUY_JOKER, BUY_BOOSTER, BUY_VOUCHER, BUY_AND_USE
            gold_before = prev_gs.get('gold', 0)
            gold_after = next_gs.get('gold', 0)
            return f"${gold_before} → ${gold_after}" + self._gained_str(prev_gs, next_gs)

        elif action_id in (8, 19, 11):  # SELL_JOKER, SELL_CONSUMABLE, CASH_OUT
            gold_before = prev_gs.get('gold', 0)
            gold_after = next_gs.get('gold', 0)
            return f"${gold_before} → ${gold_after}"

        elif action_id == 9:  # REROLL_SHOP
            gold_before = prev_gs.get('gold', 0)
            gold_after = next_gs.get('gold', 0)
            new_items = [it['name'] for it in next_gs.get('shop', {}).get('items', [])]
            return f"${gold_before} → ${gold_after} | new items: {', '.join(new_items)}"

        elif action_id == 15:  # SELECT_PACK_CARD
            choices_left = next_gs.get('pack_choices', 0)
            return f"choices left: {choices_left}" + self._gained_str(prev_gs, next_gs)

        return ""

    @_never_raise
    def log_win(self, game_state, episode_num):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gs = game_state.get('game_state', {})
        jokers = gs.get('jokers', [])
        joker_names = [j.get('name', 'Joker') for j in jokers]
        ante = gs.get('ante', 0)
        chips = gs.get('chips', 0)
        gold = gs.get('gold', 0)
        round_count = gs.get('round_count', 0)
        deck = gs.get('deck_name') or self._ep_deck

        self._write(f"\n{'=' * 70}")
        self._write(f"  WIN! Episode {episode_num}  |  {ts}")
        self._write(f"  Ante: {ante}  |  Chips: {chips}  |  Gold: ${gold}  |  Rounds: {round_count}")
        self._write(f"  Deck: {deck}")
        self._write(f"  Jokers: [{', '.join(joker_names) if joker_names else 'None'}]")
        self._write(f"{'=' * 70}")

    @staticmethod
    def _blind_summary_line(cb: dict, defeated: bool = False) -> str:
        boss_tag = " [BOSS]" if cb['is_boss'] else ""
        hands = cb['hands']
        text = (f"    Ante {cb['ante']} {cb['name'] or 'Blind'}{boss_tag}  "
                f"{cb['chips']}/{cb['blind_chips']} chips  "
                f"{hands} hand{'s' if hands != 1 else ''}  "
                f"Round {cb['round_num']}")
        return text + ("  [DEFEATED ❌]" if defeated else "")

    def _patch_reward(self, total_reward: float) -> None:
        """Overwrite the EPISODE header's Reward placeholder in place (same byte length)."""
        if self._header_pos is None:
            return
        text = f"{total_reward:+.2f}"
        if len(text) > REWARD_FIELD_WIDTH:
            text = f"{total_reward:+.0f}"[:REWARD_FIELD_WIDTH]
        self._file.flush()
        self._file.seek(self._header_pos)
        self._file.write(self._episode_header_line(text) + '\n')
        self._file.flush()
        self._file.seek(0, os.SEEK_END)

    @_never_raise
    def log_episode_end(self, total_reward: float, reward_breakdown: list[str]) -> None:
        if self._info_pending:
            self._flush_episode_info({})

        # Cleared blinds TOC (written at end since steps are real-time)
        if self._cleared_blinds or self._defeat:
            self._blank()
            self._write("  CLEARED BLINDS:")
            for cb in self._cleared_blinds:
                self._write(self._blind_summary_line(cb))
            if self._defeat:
                self._write(self._blind_summary_line(self._defeat, defeated=True))

        # Money events summary
        if self._money_log:
            self._blank()
            self._write("  MONEY EVENTS:")
            for entry in self._money_log:
                self._write(entry)

        # Reward summary
        self._blank()
        self._write(f"  EPISODE REWARD: {total_reward:.2f}  |  Shops: {self._shop_count}")
        if reward_breakdown:
            self._write("  REWARD BREAKDOWN:")
            for entry in reward_breakdown:
                self._write(f"    {entry}")
        self._write("─" * 70)

        self._patch_reward(total_reward)
        self._maybe_rotate()

    def _maybe_rotate(self) -> None:
        """Continue the session in <name>_part_NN.log once the file exceeds ROTATE_BYTES."""
        self._file.flush()
        if self._file.tell() <= self.ROTATE_BYTES:
            return
        prev_name = os.path.basename(self._file.name)
        self._file.close()
        self._part += 1
        self._file = open(f"{self._base_path}_part_{self._part:02d}.log", 'w', buffering=1,
                          encoding='utf-8', newline='\n')
        # Header offset belongs to the previous part file
        self._header_pos = None
        self._last_line_blank = False
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"Training session continued (part {self._part:02d}) from {prev_name}  |  {ts}"
                    + (f"  |  Run: {self._run_label}" if self._run_label else ""))

    def close(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"\n{'='*70}")
        self._write(f"Session ended: {ts}")
        self._write(f"{'='*70}\n")
        self._file.close()
