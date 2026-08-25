"""
History logger — writes a human-readable record of the bot's actions and their effects.
One log file per training session, stored in logs/.
"""

import os
from datetime import datetime
from pathlib import Path

STATE_NAMES = {
    1: "SELECTING_HAND", 5: "SHOP", 7: "BLIND_SELECT",
    8: "ROUND_EVAL", 9: "TAROT_PACK", 10: "PLANET_PACK",
    15: "SPECTRAL_PACK", 17: "STANDARD_PACK", 18: "BUFFOON_PACK",
    6: "GAME_OVER", 4: "MENU",
}

ACTION_NAMES = {
    1: "SELECT_HAND", 2: "PLAY_HAND", 3: "DISCARD",
    4: "START_RUN", 5: "SELECT_BLIND", 6: "RESTART_RUN",
    7: "BUY_JOKER", 8: "SELL_JOKER", 9: "REROLL_SHOP",
    10: "SKIP_SHOP", 11: "CASH_OUT", 12: "BUY_VOUCHER",
    13: "BUY_BOOSTER", 14: "USE_CONSUMABLE",
    15: "SELECT_PACK_CARD", 16: "SKIP_PACK",
}

SUIT_SYMBOLS = {"Spades": "♠", "Hearts": "♥", "Clubs": "♣", "Diamonds": "♦"}


def _card_str(card):
    suit = SUIT_SYMBOLS.get(card.get('suit', ''), card.get('suit', '')[:1])
    value = card.get('base', {}).get('value', '?')
    return f"{value}{suit}"


class HistoryLogger:
    def __init__(self, log_dir="logs"):
        Path(log_dir).mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"history_{ts}.log")
        self._file = open(log_path, 'a', buffering=1)
        self.episode = 0
        session_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"{'='*70}")
        self._write(f"Training session: {session_ts}")
        self._write(f"{'='*70}")

    def new_episode(self, episode_num):
        self.episode = episode_num
        self._write(f"\n{'─'*70}")
        self._write(f"  EPISODE {episode_num}")
        self._write(f"{'─'*70}")

    def log_step(self, prev_state, action_data, next_state):
        if not prev_state or not next_state or not action_data:
            return

        prev_gs = prev_state.get('game_state', {})
        next_gs = next_state.get('game_state', {})

        action_id = action_data.get('action', 0)
        params = action_data.get('params', [])

        ante = prev_gs.get('ante', 1)
        blind_name = prev_gs.get('blind_name', '') or 'Unknown'
        gold = prev_gs.get('gold', 0)

        action_str = self._format_action(action_id, params, prev_gs)
        effect_str = self._format_effect(action_id, params, prev_gs, next_gs)

        context = f"[A{ante} {blind_name} ${gold}]"
        line = f"  {context} {action_str}"
        if effect_str:
            line += f"  →  {effect_str}"
        self._write(line)

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

        elif action_id == 16:  # SKIP_PACK
            return "SKIP_PACK"

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

        elif action_id in (7, 13, 12):  # BUY_JOKER, BUY_BOOSTER, BUY_VOUCHER
            gold_before = prev_gs.get('gold', 0)
            gold_after = next_gs.get('gold', 0)
            parts = [f"${gold_before} → ${gold_after}"]
            prev_joker_count = len(prev_gs.get('jokers', []))
            next_joker_count = len(next_gs.get('jokers', []))
            if next_joker_count > prev_joker_count:
                new_joker = next_gs.get('jokers', [])[-1]
                parts.append(f"jokers: {prev_joker_count} → {next_joker_count} (+\"{new_joker.get('name', '?')}\")")
            return " | ".join(parts)

        elif action_id == 8:  # SELL_JOKER
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

    def _write(self, line):
        self._file.write(line + '\n')

    def close(self):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._write(f"\n{'='*70}")
        self._write(f"Session ended: {ts}")
        self._write(f"{'='*70}\n")
        self._file.close()
