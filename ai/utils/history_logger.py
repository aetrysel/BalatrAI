"""
History logger — writes a human-readable record of the bot's actions and their effects.
One log file per training session, stored in game_history_logs/.
Episode content is buffered in memory and flushed at episode end so the cleared-blinds
TOC at the top of each episode block can reference accurate line numbers.
"""

import os
from datetime import datetime
from pathlib import Path
from .constants import SHOP_STATE_ID

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
    18: "BUY_AND_USE_CONSUMABLE", 19: "SELL_CONSUMABLE",
}

SUIT_SYMBOLS = {"Spades": "♠", "Hearts": "♥", "Clubs": "♣", "Diamonds": "♦"}


def _card_str(card):
    suit = SUIT_SYMBOLS.get(card.get('suit', ''), card.get('suit', '')[:1])
    value = card.get('base', {}).get('value', '?')
    return f"{value}{suit}"


class HistoryLogger:
    def __init__(self, log_dir="game_history_logs", run_label: str = ""):
        Path(log_dir).mkdir(exist_ok=True)
        if run_label:
            log_path = os.path.join(log_dir, f"history_{run_label}.log")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(log_dir, f"history_{ts}.log")
        self._file = open(log_path, 'a', buffering=1)
        self.episode = 0
        self._file_line_count = 0

        # Per-episode buffer (flushed atomically at episode end)
        self._ep_buffer: list[str] = []
        self._cleared_blinds: list[dict] = []
        self._shop_count: int = 0
        self._money_log: list[str] = []
        self._ep_deck: str = "Unknown"
        self._ep_seed: str = ""
        self._ep_stake: str = "White"
        self._ep_ts: str = ""

        session_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"{'='*70}")
        self._write(f"Training session: {session_ts}" + (f"  |  Run: {run_label}" if run_label else ""))
        self._write(f"{'='*70}")

    def _write(self, line: str) -> None:
        self._file.write(line + '\n')
        self._file_line_count += 1

    def _ep_write(self, line: str) -> None:
        self._ep_buffer.append(line)

    def new_episode(self, episode_num: int, deck: str = "Unknown", seed: str = "", stake: str = "White") -> None:
        self.episode = episode_num
        self._ep_buffer = []
        self._cleared_blinds = []
        self._shop_count = 0
        self._money_log = []
        self._ep_deck = deck
        self._ep_seed = seed
        self._ep_stake = stake
        self._ep_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def log_step(self, prev_state, action_data, next_state):
        if not prev_state or not next_state or not action_data:
            return

        prev_gs = prev_state.get('game_state', {})
        next_gs = next_state.get('game_state', {})

        # Shop visit tracking: count transitions into SHOP state
        if prev_gs.get('state', 0) != SHOP_STATE_ID and next_gs.get('state', 0) == SHOP_STATE_ID:
            self._shop_count += 1

        action_id = action_data.get('action', 0)
        params = action_data.get('params', [])

        ante = prev_gs.get('ante', 1)
        blind_name = prev_gs.get('blind_name', '') or 'Unknown'
        gold = prev_gs.get('gold', 0)
        round_count = prev_gs.get('round_count', 0)

        action_str = self._format_action(action_id, params, prev_gs)
        effect_str = self._format_effect(action_id, params, prev_gs, next_gs)

        context = f"[A{ante} {blind_name} R{round_count} ${gold}]"
        line = f"  {context} {action_str}"
        if effect_str:
            line += f"  →  {effect_str}"
        self._ep_write(line)

        # Money event tracking
        prev_gold = prev_gs.get('gold', 0)
        next_gold = next_gs.get('gold', 0)
        delta = next_gold - prev_gold
        if delta != 0:
            if action_id == 7:  # BUY_JOKER
                shop_items = prev_gs.get('shop', {}).get('items', [])
                joker_items = [it for it in shop_items if it.get('type') == 'joker']
                item = joker_items[params[0] - 1] if params and 0 < params[0] <= len(joker_items) else {}
                self._money_log.append(f"  BUY_JOKER \"{item.get('name', '?')}\": {delta:+d} (${prev_gold} → ${next_gold})")
            elif action_id == 8:  # SELL_JOKER
                jokers_list = prev_gs.get('jokers', [])
                joker = jokers_list[params[0] - 1] if params and 0 < params[0] <= len(jokers_list) else {}
                self._money_log.append(f"  SELL_JOKER \"{joker.get('name', '?')}\": {delta:+d} (${prev_gold} → ${next_gold})")
            elif action_id == 9:  # REROLL_SHOP
                self._money_log.append(f"  REROLL: {delta:+d} (${prev_gold} → ${next_gold})")
            elif action_id == 11:  # CASH_OUT (end-of-round payout)
                self._money_log.append(f"  CASH_OUT (round payout): {delta:+d} (${prev_gold} → ${next_gold})")
            elif action_id == 12:  # BUY_VOUCHER
                shop_items = prev_gs.get('shop', {}).get('items', [])
                vouchers = [it for it in shop_items if it.get('type') == 'voucher']
                item = vouchers[params[0] - 1] if params and 0 < params[0] <= len(vouchers) else {}
                self._money_log.append(f"  BUY_VOUCHER \"{item.get('name', '?')}\": {delta:+d} (${prev_gold} → ${next_gold})")
            elif action_id == 13:  # BUY_BOOSTER
                shop_items = prev_gs.get('shop', {}).get('items', [])
                boosters = [it for it in shop_items if it.get('type') == 'booster']
                item = boosters[params[0] - 1] if params and 0 < params[0] <= len(boosters) else {}
                self._money_log.append(f"  BUY_BOOSTER \"{item.get('name', '?')}\": {delta:+d} (${prev_gold} → ${next_gold})")
            elif action_id == 18:  # BUY_AND_USE_CONSUMABLE
                shop_items = prev_gs.get('shop', {}).get('items', [])
                consumables = [it for it in shop_items if it.get('type') == 'consumable']
                item = consumables[params[0] - 1] if params and 0 < params[0] <= len(consumables) else {}
                self._money_log.append(f"  BUY_AND_USE \"{item.get('name', '?')}\": {delta:+d} (${prev_gold} → ${next_gold})")
            elif action_id == 19:  # SELL_CONSUMABLE
                consumables_list = prev_gs.get('consumables', [])
                item = consumables_list[params[0] - 1] if params and 0 < params[0] <= len(consumables_list) else {}
                self._money_log.append(f"  SELL_CONSUMABLE \"{item.get('name', '?')}\": {delta:+d} (${prev_gold} → ${next_gold})")

        # Blind clear detection — record after _ep_write so buf_pos reflects this line
        if next_gs.get('blind_cleared', 0) == 1:
            blind_nm = next_gs.get('blind_name', '') or prev_gs.get('blind_name', '')
            self._cleared_blinds.append({
                'name': blind_nm,
                'ante': next_gs.get('ante', ante),
                'chips': next_gs.get('chips', 0),
                'blind_chips': next_gs.get('blind_chips', 0),
                'round_num': next_gs.get('round_count', round_count),
                'is_boss': blind_nm not in ('Small Blind', 'Big Blind', 'None', ''),
                'buf_pos': len(self._ep_buffer),
            })

    def _format_action(self, action_id, params, prev_gs):
        slot = params[0] if params else None

        if action_id == 1:  # SELECT_HAND
            hand_cards = prev_gs.get('hand', {}).get('cards', [])
            selected = [hand_cards[i - 1] for i in params if 0 < i <= len(hand_cards)]
            if selected:
                return f"SELECT [{', '.join(_card_str(c) for c in selected)}]"
            return "SELECT_HAND"

        elif action_id == 2:  # PLAY_HAND
            hand_cards = prev_gs.get('hand', {}).get('cards', [])
            highlighted = [c for c in hand_cards if c.get('highlighted')]
            if highlighted:
                return f"PLAY [{', '.join(_card_str(c) for c in highlighted)}]"
            return "PLAY_HAND"

        elif action_id == 3:  # DISCARD
            hand_cards = prev_gs.get('hand', {}).get('cards', [])
            highlighted = [c for c in hand_cards if c.get('highlighted')]
            if highlighted:
                return f"DISCARD [{', '.join(_card_str(c) for c in highlighted)}]"
            return "DISCARD"

        elif action_id == 7:  # BUY_JOKER
            shop_items = prev_gs.get('shop', {}).get('items', [])
            joker_items = [it for it in shop_items if it.get('type') == 'joker']
            item = joker_items[slot - 1] if slot and 0 < slot <= len(joker_items) else None
            if item:
                return f"BUY_JOKER \"{item['name']}\" (${item.get('cost', '?')})"
            return f"BUY_JOKER slot:{slot}"

        elif action_id == 8:  # SELL_JOKER
            jokers = prev_gs.get('jokers', [])
            joker = jokers[slot - 1] if slot and 0 < slot <= len(jokers) else None
            if joker:
                return f"SELL_JOKER \"{joker['name']}\" (+${joker.get('sell_cost', '?')})"
            return f"SELL_JOKER slot:{slot}"

        elif action_id == 9:  # REROLL_SHOP
            cost = prev_gs.get('reroll_cost', 5)
            return f"REROLL_SHOP (${cost})"

        elif action_id == 10:  # SKIP_SHOP
            return "SKIP_SHOP"

        elif action_id == 12:  # BUY_VOUCHER
            shop_items = prev_gs.get('shop', {}).get('items', [])
            vouchers = [it for it in shop_items if it.get('type') == 'voucher']
            item = vouchers[slot - 1] if slot and 0 < slot <= len(vouchers) else None
            if item:
                return f"BUY_VOUCHER \"{item['name']}\" (${item.get('cost', '?')})"
            return f"BUY_VOUCHER slot:{slot}"

        elif action_id == 13:  # BUY_BOOSTER
            shop_items = prev_gs.get('shop', {}).get('items', [])
            boosters = [it for it in shop_items if it.get('type') == 'booster']
            item = boosters[slot - 1] if slot and 0 < slot <= len(boosters) else None
            if item:
                return f"BUY_BOOSTER \"{item['name']}\" (${item.get('cost', '?')})"
            return f"BUY_BOOSTER slot:{slot}"

        elif action_id == 14:  # USE_CONSUMABLE
            consumables = prev_gs.get('consumables', [])
            item = consumables[slot - 1] if slot and 0 < slot <= len(consumables) else None
            if item:
                return f"USE_CONSUMABLE \"{item['name']}\""
            return f"USE_CONSUMABLE slot:{slot}"

        elif action_id == 15:  # SELECT_PACK_CARD
            pack_cards = prev_gs.get('pack_cards', [])
            card = pack_cards[slot - 1] if slot and 0 < slot <= len(pack_cards) else None
            if card:
                return f"SELECT_PACK_CARD \"{card['name']}\" ({card.get('item_set', '?')})"
            return f"SELECT_PACK_CARD slot:{slot}"

        elif action_id == 16:  # SKIP_BOOSTER
            return "SKIP_BOOSTER"

        elif action_id == 17:  # SKIP_BLIND
            return "SKIP_BLIND"

        elif action_id == 18:  # BUY_AND_USE_CONSUMABLE
            shop_items = prev_gs.get('shop', {}).get('items', [])
            consumables = [it for it in shop_items if it.get('type') == 'consumable']
            item = consumables[slot - 1] if slot and 0 < slot <= len(consumables) else None
            if item:
                return f"BUY_AND_USE \"{item['name']}\" (${item.get('cost', '?')})"
            return f"BUY_AND_USE slot:{slot}"

        elif action_id == 19:  # SELL_CONSUMABLE
            consumables = prev_gs.get('consumables', [])
            item = consumables[slot - 1] if slot and 0 < slot <= len(consumables) else None
            if item:
                return f"SELL_CONSUMABLE \"{item['name']}\" (+${item.get('sell_cost', '?')})"
            return f"SELL_CONSUMABLE slot:{slot}"

        return ACTION_NAMES.get(action_id, f"ACTION_{action_id}")

    def _format_effect(self, action_id, params, prev_gs, next_gs):
        if action_id == 2:  # PLAY_HAND
            chip_gain = next_gs.get('chips', 0) - prev_gs.get('chips', 0)
            hand_name = next_gs.get('current_hand', {}).get('handname', '')
            blind_chips = next_gs.get('blind_chips', 0)
            total = next_gs.get('chips', 0)
            parts = []
            if hand_name:
                parts.append(hand_name)
            if chip_gain > 0:
                parts.append(f"+{chip_gain} chips ({total}/{blind_chips})")
            hands_left = next_gs.get('round', {}).get('hands_left', '?')
            parts.append(f"hands left: {hands_left}")
            return " | ".join(parts)

        elif action_id == 3:  # DISCARD
            discards_left = next_gs.get('round', {}).get('discards_left', '?')
            return f"discards left: {discards_left}"

        elif action_id in (7, 13, 12, 18):  # BUY_JOKER, BUY_BOOSTER, BUY_VOUCHER, BUY_AND_USE
            gold_before = prev_gs.get('gold', 0)
            gold_after = next_gs.get('gold', 0)
            parts = [f"${gold_before} → ${gold_after}"]
            prev_joker_count = len(prev_gs.get('jokers', []))
            next_joker_count = len(next_gs.get('jokers', []))
            if next_joker_count > prev_joker_count:
                new_joker = next_gs.get('jokers', [])[-1]
                parts.append(f"jokers: {prev_joker_count} → {next_joker_count} (+\"{new_joker.get('name', '?')}\")")
            return " | ".join(parts)

        elif action_id in (8, 19):  # SELL_JOKER, SELL_CONSUMABLE
            gold_before = prev_gs.get('gold', 0)
            gold_after = next_gs.get('gold', 0)
            return f"${gold_before} → ${gold_after}"

        elif action_id == 9:  # REROLL_SHOP
            gold_before = prev_gs.get('gold', 0)
            gold_after = next_gs.get('gold', 0)
            new_items = [it['name'] for it in next_gs.get('shop', {}).get('items', [])]
            return f"${gold_before} → ${gold_after} | new items: {', '.join(new_items)}"

        elif action_id == 15:  # SELECT_PACK_CARD
            prev_count = len(prev_gs.get('pack_cards', []))
            next_count = len(next_gs.get('pack_cards', []))
            choices_left = next_gs.get('pack_choices', 0)
            return f"pack cards: {prev_count} → {next_count} | choices left: {choices_left}"

        return ""

    def log_win(self, game_state, episode_num):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        gs = game_state.get('game_state', {})
        jokers = gs.get('jokers', [])
        joker_names = [j.get('name', 'Unknown') for j in jokers]
        ante = gs.get('ante', 0)
        chips = gs.get('chips', 0)
        gold = gs.get('gold', 0)
        round_count = gs.get('round_count', 0)
        deck = gs.get('deck_name', 'Unknown')

        self._ep_write(f"\n{'=' * 70}")
        self._ep_write(f"  WIN! Episode {episode_num}  |  {ts}")
        self._ep_write(f"  Ante: {ante}  |  Chips: {chips}  |  Gold: ${gold}  |  Rounds: {round_count}")
        self._ep_write(f"  Deck: {deck}")
        self._ep_write(f"  Jokers: [{', '.join(joker_names) if joker_names else 'None'}]")
        self._ep_write(f"{'=' * 70}")

    def log_episode_end(self, total_reward: float, reward_breakdown: list[str]) -> None:
        # Capture base before writing anything — used for TOC line number calculation
        base = self._file_line_count
        k = len(self._cleared_blinds)
        # toc_lines: "CLEARED BLINDS:" header + k entries + blank line after
        toc_lines = (2 + k) if k > 0 else 0
        # header (4) + meta (1) + toc = lines before buffer content
        buffer_start = base + 4 + 1 + toc_lines + 1  # 1-indexed first buffer line in file

        for cb in self._cleared_blinds:
            cb['file_line'] = buffer_start + cb['buf_pos'] - 1

        # Episode header
        self._write("")
        self._write("─" * 70)
        self._write(f"  EPISODE {self.episode}  |  {self._ep_ts}")
        self._write("─" * 70)
        self._write(f"  Deck: {self._ep_deck}  |  Seed: {self._ep_seed}  |  Stake: {self._ep_stake}  |  Shops: {self._shop_count}")

        # Cleared blinds TOC
        if self._cleared_blinds:
            self._write("  CLEARED BLINDS:")
            for cb in self._cleared_blinds:
                boss_tag = " [BOSS]" if cb['is_boss'] else ""
                self._write(
                    f"    Ante {cb['ante']} {cb['name']}{boss_tag}  "
                    f"{cb['chips']}/{cb['blind_chips']} chips  "
                    f"Round {cb['round_num']}  [line {cb['file_line']}]"
                )
            self._write("")

        # Episode body
        for line in self._ep_buffer:
            self._write(line)

        # Money events summary
        if self._money_log:
            self._write("")
            self._write("  MONEY EVENTS:")
            for entry in self._money_log:
                self._write(entry)

        # Reward summary
        self._write("")
        self._write(f"  EPISODE REWARD: {total_reward:.2f}")
        if reward_breakdown:
            self._write("  REWARD BREAKDOWN:")
            for entry in reward_breakdown:
                self._write(f"    {entry}")
        self._write("─" * 70)

    def close(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"\n{'='*70}")
        self._write(f"Session ended: {ts}")
        self._write(f"{'='*70}\n")
        self._file.close()
