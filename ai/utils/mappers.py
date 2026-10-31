"""
Mappers for converting between Balatro game data and RL-compatible formats.
"""

import numpy as np
from typing import Dict, List, Any
from ..utils.validation import GameStateValidator, ResponseValidator
from ..utils.constants import PACK_STATE_IDS, SHOP_STATE_ID, TENURE_CAP
import logging


# ── Vocabularies ─────────────────────────────────────────────────────────────
# Item name → integer ID (0 = unknown/empty). Normalise by VOCAB_SIZE.

JOKER_NAMES = [
    "Joker", "Greedy Joker", "Lusty Joker", "Wrathful Joker", "Gluttonous Joker",
    "Jolly Joker", "Zany Joker", "Mad Joker", "Crazy Joker", "Droll Joker",
    "Sly Joker", "Wily Joker", "Clever Joker", "Devious Joker", "Crafty Joker",
    "Half Joker", "Joker Stencil", "Four Fingers", "Mime", "Credit Card",
    "Ceremonial Dagger", "Banner", "Mystic Summit", "Marble Joker", "Loyalty Card",
    "8 Ball", "Misprint", "Dusk", "Raised Fist", "Chaos the Clown",
    "Fibonacci", "Steel Joker", "Stone Joker", "Gold Joker", "Lucky Cat",
    "Baseball Card", "Bull", "Diet Cola", "Trading Card", "Flash Card",
    "Popcorn", "Spare Trousers", "Ancient Joker", "Ramen", "Walkie Talkie",
    "Seltzer", "Castle", "Smiley Face", "Campfire", "Ticket",
    "Mr. Bones", "Acrobat", "Sock and Buskin", "Swashbuckler", "Troubadour",
    "Certificate", "Smeared Joker", "Throwback", "Hanging Chad", "Rough Gem",
    "Bloodstone", "Arrowhead", "Onyx Agate", "Glass Joker", "Showman",
    "Flower Pot", "Blueprint", "Wee Joker", "Merry Andy", "Oops! All 6s",
    "The Idol", "Seeing Double", "Matador", "Hit the Road", "The Duo",
    "The Trio", "The Family", "The Order", "The Tribe", "Stuntman",
    "Invisible Joker", "Brainstorm", "Satellite", "Shoot the Moon", "Driver's License",
    "Cartomancer", "Astronomer", "Burnt Joker", "Bootstraps", "Caino",
    "Hologram", "Vagabond", "Baron", "Cloud 9", "Rocket",
    "Obelisk", "Midas Mask", "Luchador", "Photograph", "Gift Card",
    "Turtle Bean", "Erosion", "Reserved Parking", "Mail-In Rebate", "To the Moon",
    "Hallucination", "Fortune Teller", "Juggler", "Drunkard", "Swashbuckler",
    "Golden Joker", "Supernova", "Ride the Bus", "Space Joker", "Egg",
    "Burglar", "Blackboard", "Runner", "Ice Cream", "DNA",
    "Splash", "Blue Joker", "Sixth Sense", "Constellation", "Hiker",
    "Faceless Joker", "Green Joker", "Superposition", "To Do List", "Cavendish",
    "Card Sharp", "Red Card", "Madness", "Square Joker", "Seance",
    "Riff-Raff", "Vampire", "Shortcut", "Quasimodo", "Yorick",
    "Chicot", "Perkeo",
]

CONSUMABLE_NAMES = [
    # Tarot
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
    "The Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
    "The Devil", "The Tower", "The Star", "The Moon", "The Sun",
    "Judgement", "The World",
    # Planet
    "Mercury", "Venus", "Earth", "Mars", "Jupiter",
    "Saturn", "Uranus", "Neptune", "Pluto", "Planet X",
    "Ceres", "Eris",
    # Spectral
    "Familiar", "Grim", "Incantation", "Talisman", "Aura",
    "Wraith", "Sigil", "Ouija", "Ectoplasm", "Immolate",
    "Ankh", "Deja Vu", "Hex", "Trance", "Medium",
    "Cryptid", "The Soul", "Black Hole",
]

VOUCHER_NAMES = [
    "Overstock", "Overstock Plus", "Clearance Sale", "Liquidation",
    "Hone", "Glow Up", "Reroll Surplus", "Reroll Glut",
    "Crystal Ball", "Omen Globe", "Telescope", "Observatory",
    "Grabber", "Nacho Tong", "Wasteful", "Recyclomancy",
    "Tarot Merchant", "Tarot Tycoon", "Planet Merchant", "Planet Tycoon",
    "Seed Money", "Money Tree", "Blank", "Antimatter",
    "Magic Trick", "Illusion", "Hieroglyph", "Petroglyph",
    "Directors Cut", "Retcon",
]

BOOSTER_NAMES = [
    "Arcana Pack", "Jumbo Arcana Pack", "Mega Arcana Pack",
    "Celestial Pack", "Jumbo Celestial Pack", "Mega Celestial Pack",
    "Spectral Pack", "Jumbo Spectral Pack", "Mega Spectral Pack",
    "Standard Pack", "Jumbo Standard Pack", "Mega Standard Pack",
    "Buffoon Pack", "Jumbo Buffoon Pack", "Mega Buffoon Pack",
]

DECK_NAMES = [
    "Red Deck", "Blue Deck", "Yellow Deck", "Green Deck",
    "Black Deck", "Magic Trick", "Nebula Deck", "Ghost Deck",
    "Abandoned Deck", "Checkered Deck", "Zodiac Deck", "Painted Deck",
    "Anaglyph Deck", "Plasma Deck", "Erratic Deck", "Standard Deck",
]

_ALL_ITEMS = ["__unknown__"] + JOKER_NAMES + CONSUMABLE_NAMES + VOUCHER_NAMES + BOOSTER_NAMES
ITEM_VOCAB: Dict[str, int] = {name: idx for idx, name in enumerate(_ALL_ITEMS)}
VOCAB_SIZE = len(_ALL_ITEMS)

DECK_VOCAB: Dict[str, int] = {name: idx for idx, name in enumerate(DECK_NAMES)}
NUM_DECKS = len(DECK_NAMES)

EDITION_IDS = {"none": 0, "foil": 1, "holo": 2, "polychrome": 3, "negative": 4}
MAX_EDITION_ID = 4


def make_onehot(value: int, num_classes: int) -> List[float]:
    onehot = [0.0] * num_classes
    if 0 <= value < num_classes:
        onehot[value] = 1.0
    return onehot


def make_mask(available_items: List[int], total_slots: int) -> List[float]:
    mask = [0.0] * total_slots
    for item in available_items:
        if 0 <= item < total_slots:
            mask[item] = 1.0
    return mask


def normalize(value: float, max_value: float) -> float:
    return value / max_value if max_value > 0 else 0.0


def item_id_norm(name: str) -> float:
    return ITEM_VOCAB.get(name, 0) / VOCAB_SIZE


class BalatroStateMapper:
    """
    Converts raw Balatro game state JSON to a 440-dimensional RL observation.

    Layout (440 total):
      2  round (hands_left, discards_left)
      1  blind_chips
      1  chips
      1  ante (norm /8)
      1  round_count (norm /24)
      1  gold (norm /100)
      1  reroll_cost (norm /10)
     20  state one-hot
      1  game_over
      1  game_win
      1  retry_count
     16  deck name one-hot
     35  jokers (5 × 7: occupied, name_id, sell_cost_norm, edition_id, rounds_held_norm,
                 contribution, triggered_last_hand)
     20  consumables held (5 × 4: occupied, name_id, max_highlighted_norm, can_use)
     21  shop (4 joker × 3 + 1 voucher × 3 + 2 booster × 3)
     10  pack_cards (5 × 2: occupied, name_id)
     17  deck stats (size + 4 seals + 8 enhancements + 4 editions)
    254  hand features
     16  current_hand scoring
     20  available actions mask
    """

    OBSERVATION_SIZE = 440

    def __init__(self, observation_size: int, max_actions: int):
        self.observation_size = observation_size
        self.max_actions = max_actions
        self.logger = logging.getLogger(__name__)
        self.game_state_validator = GameStateValidator()

    def process_game_state(self, raw_state: Dict[str, Any] | None) -> np.ndarray:
        if not raw_state:
            return np.zeros(self.observation_size, dtype=np.float32)

        try:
            self.game_state_validator.validate_game_state(raw_state)
        except ValueError as e:
            self.logger.error(f"Invalid game state: {e}")

        features = []
        features.extend(self._extract_game_features(raw_state.get('game_state', {})))
        features.extend(self._extract_available_actions(raw_state.get('available_actions', [])))

        obs = np.array(features, dtype=np.float32)
        if len(obs) != self.observation_size:
            self.logger.warning(f"Obs size mismatch: got {len(obs)}, expected {self.observation_size}")
            padded = np.zeros(self.observation_size, dtype=np.float32)
            padded[:min(len(obs), self.observation_size)] = obs[:self.observation_size]
            return padded
        return obs

    def _extract_available_actions(self, available_actions: List[int]) -> List[float]:
        # Convert 1-based Balatro IDs to 0-based AI indices for the mask
        ai_indices = [a - 1 for a in available_actions if 1 <= a <= self.max_actions]
        return make_mask(ai_indices, self.max_actions)

    def _extract_game_features(self, state: Dict[str, Any]) -> List[float]:
        features = []
        features.extend(self._extract_round_features(state.get('round', {})))
        features.append(float(state.get('blind_chips', 0)))
        features.append(float(state.get('chips', 0)))
        features.append(normalize(float(state.get('ante', 1)), 8.0))
        features.append(normalize(float(state.get('round_count', 0)), 24.0))
        features.append(normalize(float(state.get('gold', 0)), 100.0))
        features.append(normalize(float(state.get('reroll_cost', 5)), 10.0))
        features.extend(make_onehot(state.get('state', 0), 20))
        features.append(float(state.get('game_over', 0)))
        features.append(float(state.get('game_win', 0)))
        features.append(float(state.get('retry_count', 0)))
        features.extend(self._extract_deck_name(state.get('deck_name', '')))
        features.extend(self._extract_joker_features(state.get('jokers', [])))
        features.extend(self._extract_consumable_features(state.get('consumables', [])))
        features.extend(self._extract_shop_features(state.get('shop', {}).get('items', [])))
        features.extend(self._extract_pack_features(state.get('pack_cards', [])))
        features.extend(self._extract_deck_stats(state.get('deck_cards', [])))
        features.extend(self._extract_hand_features(state.get('hand', {})))
        features.extend(self._extract_current_hand_scoring(state.get('current_hand', {})))
        return features

    def _extract_round_features(self, round_data: Dict[str, Any]) -> List[float]:
        return [
            float(round_data.get('hands_left', 0)),
            float(round_data.get('discards_left', 0)),
        ]

    def _extract_deck_name(self, deck_name: str) -> List[float]:
        idx = DECK_VOCAB.get(deck_name, NUM_DECKS - 1)
        return make_onehot(idx, NUM_DECKS)  # 16

    def _extract_joker_features(self, jokers: List[Dict[str, Any]]) -> List[float]:
        features = []
        for i in range(5):
            if i < len(jokers):
                j = jokers[i]
                features.append(1.0)
                features.append(item_id_norm(j.get('name', '')))
                features.append(normalize(float(j.get('sell_cost', 0)), 20.0))
                features.append(EDITION_IDS.get(j.get('edition', 'none'), 0) / MAX_EDITION_ID)
                features.append(normalize(float(j.get('rounds_held', 0)), float(TENURE_CAP)))
                features.append(float(j.get('contribution', 0.0)))
                features.append(1.0 if j.get('triggered_last_hand') else 0.0)
            else:
                features.extend([0.0] * 7)
        return features  # 5 × 7 = 35

    def _extract_consumable_features(self, consumables: List[Dict[str, Any]]) -> List[float]:
        features = []
        for i in range(5):
            if i < len(consumables):
                c = consumables[i]
                features.append(1.0)
                features.append(item_id_norm(c.get('name', '')))
                features.append(normalize(float(c.get('max_highlighted', 0)), 5.0))
                features.append(1.0 if c.get('can_use') else 0.0)
            else:
                features.extend([0.0] * 4)
        return features  # 5 × 4 = 20

    def _extract_shop_features(self, shop_items: List[Dict[str, Any]]) -> List[float]:
        joker_slots: List[Dict] = []
        voucher_slots: List[Dict] = []
        booster_slots: List[Dict] = []
        for item in shop_items:
            t = item.get('type', '')
            if t == 'joker':
                joker_slots.append(item)
            elif t == 'voucher':
                voucher_slots.append(item)
            elif t == 'booster':
                booster_slots.append(item)

        features = []
        for i in range(4):  # shop_jokers (4 slots)
            if i < len(joker_slots):
                item = joker_slots[i]
                features.extend([1.0, item_id_norm(item.get('name', '')), normalize(float(item.get('cost', 0)), 20.0)])
            else:
                features.extend([0.0, 0.0, 0.0])
        if voucher_slots:  # 1 voucher
            item = voucher_slots[0]
            features.extend([1.0, item_id_norm(item.get('name', '')), normalize(float(item.get('cost', 0)), 20.0)])
        else:
            features.extend([0.0, 0.0, 0.0])
        for i in range(2):  # boosters (2 slots)
            if i < len(booster_slots):
                item = booster_slots[i]
                features.extend([1.0, item_id_norm(item.get('name', '')), normalize(float(item.get('cost', 0)), 10.0)])
            else:
                features.extend([0.0, 0.0, 0.0])
        return features  # 12 + 3 + 6 = 21

    def _extract_pack_features(self, pack_cards: List[Dict[str, Any]]) -> List[float]:
        features = []
        for i in range(5):
            if i < len(pack_cards):
                c = pack_cards[i]
                features.extend([1.0, item_id_norm(c.get('name', ''))])
            else:
                features.extend([0.0, 0.0])
        return features  # 5 × 2 = 10

    def _extract_deck_stats(self, deck_cards: List[Dict[str, Any]]) -> List[float]:
        n = max(len(deck_cards), 1)
        seal_counts = {"Gold": 0, "Red": 0, "Blue": 0, "Purple": 0}
        enh_counts = {k: 0 for k in ("m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky")}
        edition_counts = {"foil": 0, "holo": 0, "polychrome": 0, "negative": 0}

        for card in deck_cards:
            seal = card.get('seal', 'None')
            if seal in seal_counts:
                seal_counts[seal] += 1
            enh = card.get('enhancement', 'Base')
            if enh in enh_counts:
                enh_counts[enh] += 1
            ed = card.get('edition', 'none')
            if ed in edition_counts:
                edition_counts[ed] += 1

        features = [normalize(len(deck_cards), 60.0)]
        for seal in ("Gold", "Red", "Blue", "Purple"):
            features.append(normalize(seal_counts[seal], n))
        for enh in ("m_bonus", "m_mult", "m_wild", "m_glass", "m_steel", "m_stone", "m_gold", "m_lucky"):
            features.append(normalize(enh_counts[enh], n))
        for ed in ("foil", "holo", "polychrome", "negative"):
            features.append(normalize(edition_counts[ed], n))
        return features  # 1 + 4 + 8 + 4 = 17

    def _extract_hand_features(self, hand: Dict[str, Any]) -> List[float]:
        features = []
        cards = hand.get('cards', [])
        features.append(float(hand.get('size', 0)))
        features.append(float(hand.get('highlighted_count', 0)))

        for card in cards:
            card_features = [float(card.get('highlighted', False))]
            suits_mapping = {"Hearts": 0, "Diamonds": 1, "Spades": 2, "Clubs": 3}
            suit = card.get('suit', 'Unknown')
            card_features.extend(make_onehot(suits_mapping.get(suit, 4), 5))
            base = card.get('base', {})
            values_mapping = {
                '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, '9': 7, '10': 8,
                'Jack': 9, 'Queen': 10, 'King': 11, 'Ace': 12
            }
            value = base.get('value')
            card_features.extend(make_onehot(values_mapping.get(value, 13), 14))
            card_features.append(base.get('nominal', 0.0))
            features.extend(card_features)

        max_cards = 12
        features_per_card = 21  # 1+5+14+1
        expected = 2 + max_cards * features_per_card  # 254
        features = features[:expected]
        if len(features) < expected:
            features.extend([0.0] * (expected - len(features)))
        return features  # 254

    def _extract_current_hand_scoring(self, current_hand: Dict[str, Any]) -> List[float]:
        features = []
        try:
            features.append(float(current_hand.get('chips', 0)))
        except (ValueError, TypeError):
            features.append(0.0)
        try:
            features.append(float(current_hand.get('mult', 0)))
        except (ValueError, TypeError):
            features.append(0.0)
        features.append(float(current_hand.get('score', 0)))

        hand_types = [
            "None", "High Card", "Pair", "Two Pair", "Three of a Kind",
            "Straight", "Flush", "Full House", "Four of a Kind",
            "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"
        ]
        hand_name = current_hand.get('handname', 'None')
        if not hand_name or '?' in hand_name:
            hand_name = "None"
        try:
            hand_index = hand_types.index(hand_name)
        except ValueError:
            hand_index = 0
        features.extend(make_onehot(hand_index, len(hand_types)))
        return features  # 3 + 13 = 16


class BalatroActionMapper:
    """Converts RL actions to Balatro command JSON."""

    def __init__(self, action_slices: Dict[str, slice]):
        self.slices = action_slices
        self.response_validator = ResponseValidator()
        self.logger = logging.getLogger(__name__)

    def process_action(self, rl_action: np.ndarray, state_id: int = 0) -> Dict[str, Any]:
        ai_action = rl_action[self.slices["action_selection"]].tolist()[0]

        ai_to_balatro_mapping = {
            0:  1,   # SELECT_HAND
            1:  2,   # PLAY_HAND
            2:  3,   # DISCARD_HAND
            3:  4,   # START_RUN
            4:  5,   # SELECT_BLIND
            5:  6,   # RESTART_RUN
            6:  7,   # BUY_JOKER
            7:  8,   # SELL_JOKER
            8:  9,   # REROLL_SHOP
            9:  10,  # SKIP_SHOP
            10: 11,  # CASH_OUT
            11: 12,  # BUY_VOUCHER
            12: 13,  # BUY_BOOSTER
            13: 14,  # USE_CONSUMABLE
            14: 15,  # SELECT_PACK_CARD
            15: 16,  # SKIP_BOOSTER
            16: 17,  # SKIP_BLIND
            17: 18,  # BUY_AND_USE_CONSUMABLE
            18: 19,  # SELL_CONSUMABLE
            19: 20,  # REROLL_BOSS
        }

        balatro_action_id = ai_to_balatro_mapping.get(ai_action, 1)

        if state_id in PACK_STATE_IDS:
            params = self._extract_pack_params(rl_action, balatro_action_id)
        elif state_id == SHOP_STATE_ID:
            params = self._extract_shop_params(rl_action, balatro_action_id)
        else:
            params = self._extract_play_params(rl_action, balatro_action_id)

        response_data = {"action": balatro_action_id, "params": params}
        try:
            self.response_validator.validate_response(response_data)
        except ValueError as e:
            self.logger.error(f"Invalid response: {e}")

        return response_data

    def _extract_play_params(self, raw_action: np.ndarray, balatro_action_id: int) -> List[int]:
        if balatro_action_id in (14, 19):  # USE_CONSUMABLE, SELL_CONSUMABLE
            consumable_slot = raw_action[self.slices["consumable_slot"]].tolist()[0]
            return [consumable_slot + 1]
        if balatro_action_id == 8:  # SELL_JOKER
            return [raw_action[self.slices["joker_slot"]].tolist()[0] + 1]
        card_indices = raw_action[self.slices["card_indices"]]
        return [i + 1 for i, val in enumerate(card_indices) if val == 1]

    def _extract_shop_params(self, rl_action: np.ndarray, balatro_action_id: int) -> List[int]:
        if balatro_action_id == 7:   # BUY_JOKER
            return [rl_action[self.slices["shop_slot"]].tolist()[0] + 1]
        elif balatro_action_id == 8:  # SELL_JOKER
            return [rl_action[self.slices["joker_slot"]].tolist()[0] + 1]
        elif balatro_action_id == 13:  # BUY_BOOSTER
            return [rl_action[self.slices["shop_slot"]].tolist()[0] + 1]
        elif balatro_action_id == 18:  # BUY_AND_USE_CONSUMABLE
            return [rl_action[self.slices["shop_slot"]].tolist()[0] + 1]
        elif balatro_action_id in (14, 19):  # USE_CONSUMABLE, SELL_CONSUMABLE
            return [rl_action[self.slices["consumable_slot"]].tolist()[0] + 1]
        return []

    def _extract_pack_params(self, rl_action: np.ndarray, balatro_action_id: int) -> List[int]:
        if balatro_action_id == 15:  # SELECT_PACK_CARD
            return [rl_action[self.slices["pack_card_slot"]].tolist()[0] + 1]
        elif balatro_action_id in (14, 19):  # USE_CONSUMABLE, SELL_CONSUMABLE
            return [rl_action[self.slices["consumable_slot"]].tolist()[0] + 1]
        elif balatro_action_id == 8:  # SELL_JOKER
            return [rl_action[self.slices["joker_slot"]].tolist()[0] + 1]
        elif balatro_action_id == 1:  # SELECT_HAND: target hand cards for a tarot/spectral
            card_indices = rl_action[self.slices["card_indices"]]
            return [i + 1 for i, val in enumerate(card_indices) if val == 1]
        return []
