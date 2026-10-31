from typing import Dict, Any
import numpy as np
from ..utils.constants import (PACK_STATE_IDS, SHOP_STATE_ID, SELECTING_HAND_STATE_ID,
                               PASSIVE_JOKERS, PASSIVE_CONTRIBUTION_FLOOR)

# Inventory potential weights: reward = Φ(s') - Φ(s), so buy→sell nets zero
# Joker potential grows with the joker's measured contribution (score share + value generated, 0..1)
JOKER_BASE_POTENTIAL = 4.0
JOKER_CONTRIBUTION_POTENTIAL = 8.0
# Deck potential: upgrades (enhancement/seal/edition, one each) as a fraction of deck size
DECK_QUALITY_POTENTIAL = 30.0
MID_BLIND_SELL_PENALTY = 2.0
FRESH_SELL_PENALTY = 1.0  # selling a joker before it has been held through a blind
SAVINGS_TIER_BONUS = 1.0  # per $5 interest tier held when leaving the shop
SAVINGS_TIER_CAP = 5
MONEY_EARNED_WEIGHT = 0.5
CONSUMABLE_USE_REWARD = {"Planet": 2.0, "Tarot": 4.0, "Spectral": 5.0}
PACK_PICK_REWARD = 1.5      # per card taken from a tarot/planet/spectral/standard pack
BOOSTER_SKIP_PENALTY = 1.5  # bought a pack and took nothing (waived with Red Card)
PB_HAND_BONUS = 3.0         # × min(1, improvement / blind requirement) on a new best single hand

class BalatroRewardCalculator:
    def __init__(self):
        self.previous_chips = 0
        self.previous_hand_played = "None"
        self.blind_already_defeated = False  # per-blind flag, reset on new blind
        self.current_ante = 1
        self.best_ante_reached = 1

        # Episode tracking
        self._started = False
        self.episode_count = 0
        self.wins = 0
        self.episode_total_reward = 0
        self.last_seen_hand_type = 'Unknown'
        self.winning_chips = 0

        # Per-blind tracking
        self.current_blind_name = ''
        self.current_blind_chips = 0       # requirement for current blind
        self.current_blind_ante = 1
        self.current_round_count = 0       # round_count when this blind started
        self.current_round_hands = []      # hands played in current blind
        self.rounds_played = []            # completed rounds: [{blind_name, blind_chips, ante, round_num, hands, cleared}]
        self.highest_chips_this_blind = 0  # peak chips seen this blind (for game_over reporting)
        self.pending_cards_removed = 0     # playing cards removed since last blind clear (paid on clear)

        # Game-over reporting
        self.game_over_chips = 0
        self.game_over_blind_chips = 0
        self.game_over_blind_name = ''
        self.game_over_ante = 1

        # Joker tracking
        self.episode_jokers: list = []

        # Shop vars
        self.previous_money = 0
        self.total_jokers = 0
        self.peak_money = 0

        # Episode stats for metrics
        self.fresh_joker_sells = 0
        self.max_joker_tenure = 0
        self.shop_exit_golds: list = []
        self.deck_stats: Dict[str, float] = {}
        self.max_deck_stats: Dict[str, float] = {}
        self.boosters_bought_skipped = 0
        self.pb_hand_bonuses = 0

        # Open-pack tracking (buy-then-skip penalty)
        self.pack_purchased = False
        self.pack_picks = 0

        self.episode_reward_breakdown: list[str] = []

        self.REWARD_THRESHOLDS = {
            "excellent": 75.0,
            "good": 40.0,
            "decent": 20.0
        }

    def calculate_reward(self, current_state, prev_state=None):
        self._started = True
        reward = 0.0
        reward += self.calculate_inventory_reward(current_state, prev_state)
        reward += self.calculate_shop_reward(current_state, prev_state)
        reward += self.calculate_pack_reward(current_state, prev_state)
        reward += self.calculate_play_reward(current_state, prev_state)
        return float(np.clip(reward, -30.0, 120.0))

    @staticmethod
    def _joker_contribution(joker) -> float:
        contribution = float(joker.get('contribution', 0.0))
        if joker.get('name') in PASSIVE_JOKERS:
            contribution = max(contribution, PASSIVE_CONTRIBUTION_FLOOR)
        return min(1.0, max(0.0, contribution))

    @classmethod
    def _joker_potential(cls, joker) -> float:
        return JOKER_BASE_POTENTIAL + JOKER_CONTRIBUTION_POTENTIAL * cls._joker_contribution(joker)

    @staticmethod
    def deck_quality(game_state) -> Dict[str, float]:
        """Upgraded-card counts in the full deck; upgrades counts each enhancement, seal and edition once."""
        cards = game_state.get('deck_cards') or []
        enhanced = sum(1 for c in cards if c.get('enhancement', 'c_base') not in ('c_base', 'Base', 'Default Base'))
        sealed = sum(1 for c in cards if c.get('seal', 'None') not in ('None', None))
        editioned = sum(1 for c in cards if c.get('edition', 'none') not in ('none', None))
        upgrades = enhanced + sealed + editioned
        return {
            "size": len(cards),
            "enhanced": enhanced,
            "sealed": sealed,
            "editioned": editioned,
            "upgrades": upgrades,
            "upgrade_pct": upgrades / len(cards) if cards else 0.0,
        }

    @classmethod
    def _inventory_potential(cls, game_state) -> float:
        return sum(cls._joker_potential(j) for j in game_state.get('jokers', []))

    def episode_stats(self) -> Dict[str, float]:
        """Per-episode reward-shaping stats for TensorBoard."""
        golds = self.shop_exit_golds
        return {
            "fresh_joker_sells": self.fresh_joker_sells,
            "max_joker_tenure": self.max_joker_tenure,
            "mean_shop_exit_gold": (sum(golds) / len(golds)) if golds else 0.0,
            "deck_enhanced": self.deck_stats.get("enhanced", 0),
            "deck_sealed": self.deck_stats.get("sealed", 0),
            "deck_editioned": self.deck_stats.get("editioned", 0),
            "deck_upgrades": self.deck_stats.get("upgrades", 0),
            "deck_upgrade_pct": self.deck_stats.get("upgrade_pct", 0.0),
            "max_deck_enhanced": self.max_deck_stats.get("enhanced", 0),
            "max_deck_sealed": self.max_deck_stats.get("sealed", 0),
            "max_deck_editioned": self.max_deck_stats.get("editioned", 0),
            "max_deck_upgrades": self.max_deck_stats.get("upgrades", 0),
            "max_deck_upgrade_pct": self.max_deck_stats.get("upgrade_pct", 0.0),
            "boosters_bought_skipped": self.boosters_bought_skipped,
            "pb_hand_bonuses": self.pb_hand_bonuses,
        }

    def calculate_inventory_reward(self, current_state, prev_state=None) -> float:
        """Contribution-weighted joker potential, deck-quality potential, fresh/mid-blind sell penalties, and shop-exit savings bonus."""
        if not prev_state:
            return 0.0
        reward = 0.0
        breakdown: list[str] = []
        current_game_state = current_state.get('game_state', {})
        prev_game_state = prev_state.get('game_state', {})
        if not prev_game_state:
            return 0.0

        potential_delta = self._inventory_potential(current_game_state) - self._inventory_potential(prev_game_state)
        if abs(potential_delta) > 1e-6:
            reward += potential_delta
            breakdown.append(f"joker_potential: {potential_delta:+.2f}")

        # Deck quality: skip frames where either deck is missing (menu/transition) so the potential doesn't swing
        curr_deck = self.deck_quality(current_game_state)
        prev_deck = self.deck_quality(prev_game_state)
        if curr_deck["size"] > 0:
            self.deck_stats = curr_deck
            for k, v in curr_deck.items():
                self.max_deck_stats[k] = max(self.max_deck_stats.get(k, 0), v)
        if curr_deck["size"] > 0 and prev_deck["size"] > 0:
            deck_delta = DECK_QUALITY_POTENTIAL * (curr_deck["upgrade_pct"] - prev_deck["upgrade_pct"])
            if abs(deck_delta) > 1e-6:
                reward += deck_delta
                breakdown.append(f"deck_quality {curr_deck['upgrades']}/{curr_deck['size']}: {deck_delta:+.2f}")

        curr_jokers = current_game_state.get('jokers', [])
        for j in curr_jokers:
            self.max_joker_tenure = max(self.max_joker_tenure, j.get('rounds_held', 0))

        # Selling a joker acquired this round (pack/tarot/shop flip)
        gold_gained = current_game_state.get('gold', 0) > prev_game_state.get('gold', 0)
        curr_ids = {j.get('id') for j in curr_jokers}
        sold = [j for j in prev_game_state.get('jokers', [])
                if j.get('id') is not None and j.get('id') not in curr_ids]
        fresh_sold = [j for j in sold if j.get('rounds_held', 0) == 0]
        if gold_gained and fresh_sold:
            penalty = FRESH_SELL_PENALTY * len(fresh_sold)
            reward -= penalty
            self.fresh_joker_sells += len(fresh_sold)
            breakdown.append(f"fresh_joker_sell x{len(fresh_sold)}: -{penalty:.2f}")

        # Savings: reward each interest tier held when leaving the shop
        if (prev_game_state.get('state', 0) == SHOP_STATE_ID
                and current_game_state.get('state', 0) != SHOP_STATE_ID
                and current_game_state.get('state', 0) not in PACK_STATE_IDS):
            exit_gold = current_game_state.get('gold', 0)
            self.shop_exit_golds.append(exit_gold)
            tiers = min(max(exit_gold, 0) // 5, SAVINGS_TIER_CAP)
            if tiers > 0:
                savings = SAVINGS_TIER_BONUS * tiers
                reward += savings
                breakdown.append(f"savings ${exit_gold} ({tiers} tiers): +{savings:.2f}")

        # Selling during a blind: allowed but discouraged (except clearing a full consumable area)
        if prev_game_state.get('state', 0) == SELECTING_HAND_STATE_ID:
            gold_increased = current_game_state.get('gold', 0) > prev_game_state.get('gold', 0)
            prev_consumables = prev_game_state.get('consumables', [])
            joker_sold = len(current_game_state.get('jokers', [])) < len(prev_game_state.get('jokers', []))
            consumable_sold = len(current_game_state.get('consumables', [])) < len(prev_consumables)
            consumables_full = len(prev_consumables) >= prev_game_state.get('consumable_slots_max', 2)
            if gold_increased and (joker_sold or (consumable_sold and not consumables_full)):
                reward -= MID_BLIND_SELL_PENALTY
                breakdown.append(f"mid_blind_sell: -{MID_BLIND_SELL_PENALTY:.2f}")

        if breakdown:
            self.episode_reward_breakdown.append("[INV] " + "; ".join(breakdown))
        return float(reward)

    def calculate_shop_reward(self, current_state, prev_state=None):
        reward = 0.0
        breakdown: list[str] = []
        current_game_state = current_state.get('game_state', {})
        prev_game_state = prev_state.get('game_state', {}) if prev_state else {}

        current_money = current_game_state.get('gold', 0)
        prev_money = prev_game_state.get('gold', self.previous_money)
        money_spent = prev_money - current_money

        current_jokers = current_game_state.get('jokers', [])

        prev_state_id = prev_game_state.get('state', 0)
        curr_state_id = current_game_state.get('state', 0)
        if prev_state_id == SHOP_STATE_ID and curr_state_id != SHOP_STATE_ID:
            max_joker_slots = 5
            utilization = len(current_jokers) / max_joker_slots
            reward += utilization * 1.0
            breakdown.append(f"joker_utilization_exit ({len(current_jokers)}/5): +{utilization:.2f}")

            if current_money == 0 and len(current_jokers) == 0:
                reward -= 3.0
                breakdown.append("broke_no_jokers: -3.00")

        # Update tracking
        self.previous_money = current_money
        self.total_jokers = len(current_jokers)
        self.episode_jokers = [j.get('name', 'Unknown') for j in current_jokers]

        prev_shop_items = prev_game_state.get('shop', {}).get('items', [])
        curr_shop_items = current_game_state.get('shop', {}).get('items', [])
        prev_had_voucher = any(it.get('type') == 'voucher' for it in prev_shop_items)
        curr_has_voucher = any(it.get('type') == 'voucher' for it in curr_shop_items)
        if prev_had_voucher and not curr_has_voucher and money_spent > 0:
            reward += 8.0
            breakdown.append("voucher_purchased: +8.00")

        scaled = float(np.tanh(reward / 5.0) * 5.0)
        if breakdown:
            self.episode_reward_breakdown.append(
                f"[SHOP] raw={reward:.2f} → scaled={scaled:.2f}: " + "; ".join(breakdown)
            )
        return scaled

    @staticmethod
    def _total_pack_picks(game_state) -> int:
        packs = (game_state.get('run_stats') or {}).get('packs') or {}
        if not isinstance(packs, dict):
            return 0
        return sum(int(p.get('picks', 0)) for p in packs.values() if isinstance(p, dict))

    @staticmethod
    def _consumable_uses_by_set(game_state) -> Dict[str, int]:
        usage = (game_state.get('run_stats') or {}).get('consumable_usage') or {}
        by_set: Dict[str, int] = {}
        if not isinstance(usage, dict):
            return by_set
        for entry in usage.values():
            if isinstance(entry, dict):
                s = entry.get('set', 'Unknown')
                by_set[s] = by_set.get(s, 0) + int(entry.get('count', 0))
        return by_set

    def calculate_pack_reward(self, current_state, prev_state=None) -> float:
        reward = 0.0
        if not prev_state:
            return reward

        current_game_state = current_state.get('game_state', {})
        prev_game_state = prev_state.get('game_state', {})

        prev_state_id = prev_game_state.get('state', 0)
        curr_state_id = current_game_state.get('state', 0)
        pack_names = {9: "TAROT_PACK", 10: "PLANET_PACK", 15: "SPECTRAL_PACK", 17: "STANDARD_PACK", 18: "BUFFOON_PACK"}
        pack_name = pack_names.get(prev_state_id, f'PACK_{prev_state_id}')

        # Pack opened: remember whether it was bought (gold dropped leaving the shop) or free (tag)
        if curr_state_id in PACK_STATE_IDS and prev_state_id not in PACK_STATE_IDS:
            self.pack_purchased = (prev_state_id == SHOP_STATE_ID
                                   and current_game_state.get('gold', 0) < prev_game_state.get('gold', 0))
            self.pack_picks = 0
            return reward

        if prev_state_id not in PACK_STATE_IDS:
            return reward

        # Picks counted by the Lua use_card hook (pack_choices also drops during skip cleanup)
        picks = self._total_pack_picks(current_game_state) - self._total_pack_picks(prev_game_state)
        if picks > 0:
            self.pack_picks += picks
            # Buffoon picks are paid through joker potential
            pack_cards = prev_game_state.get('pack_cards') or []
            is_buffoon = prev_state_id == 18 or (prev_state_id == 999 and bool(pack_cards)
                                                 and pack_cards[0].get('item_set') == 'Joker')
            if not is_buffoon:
                pick_reward = PACK_PICK_REWARD * picks
                reward += pick_reward
                self.episode_reward_breakdown.append(f"[PACK] {pack_name} pick: +{pick_reward:.2f}")

        # Pack closed: bought it but took nothing
        if curr_state_id not in PACK_STATE_IDS:
            holds_red_card = any(j.get('name') == 'Red Card' for j in current_game_state.get('jokers', []))
            if self.pack_purchased and self.pack_picks == 0 and not holds_red_card:
                reward -= BOOSTER_SKIP_PENALTY
                self.boosters_bought_skipped += 1
                self.episode_reward_breakdown.append(f"[PACK] {pack_name} bought then skipped: -{BOOSTER_SKIP_PENALTY:.2f}")
            self.pack_purchased = False
            self.pack_picks = 0

        return float(reward)

    def calculate_play_reward(self, current_state: Dict[str, Any],
                        prev_state: Dict[str, Any] = None) -> float:
        reward = 0.0
        reward_breakdown = []

        inner_game_state = current_state.get('game_state', {})
        prev_game_state = prev_state.get('game_state', {}) if prev_state else {}

        current_chips = inner_game_state.get('chips', 0)
        game_over = inner_game_state.get('game_over', 0)
        retry_count = inner_game_state.get('retry_count', 0)
        ante = inner_game_state.get('ante', 1)
        blind_chips = inner_game_state.get('blind_chips', 300)
        blind_name = inner_game_state.get('blind_name', '')
        blind_cleared_flag = inner_game_state.get('blind_cleared', 0) == 1
        game_won = inner_game_state.get('game_win', 0) == 1

        # --- Blind transition detection ---
        # When blind_name changes, a new blind has started
        if blind_name and blind_name not in ('None', '') and blind_name != self.current_blind_name:
            if self.current_blind_name:  # save the round we just finished
                self.rounds_played.append({
                    'blind_name': self.current_blind_name,
                    'blind_chips': self.current_blind_chips,
                    'ante': self.current_blind_ante,
                    'round_num': self.current_round_count,
                    'hands': list(self.current_round_hands),
                    'cleared': self.blind_already_defeated,
                })
            self.current_round_hands = []
            self.highest_chips_this_blind = 0
            self.previous_chips = 0
            self.blind_already_defeated = False
            self.current_blind_name = blind_name
            self.current_blind_chips = blind_chips
            self.current_blind_ante = ante
            self.current_round_count = inner_game_state.get('round_count', 0)

        # Track peak chips for accurate game-over reporting
        self.highest_chips_this_blind = max(self.highest_chips_this_blind, current_chips)

        # Deck thinning: count playing cards removed (additions ignored)
        prev_deck_size = (prev_game_state.get('run_stats') or {}).get('deck_size', 0)
        curr_deck_size = (inner_game_state.get('run_stats') or {}).get('deck_size', 0)
        if prev_deck_size > 0 and 0 < curr_deck_size < prev_deck_size:
            self.pending_cards_removed += prev_deck_size - curr_deck_size

        # === CONSUMABLE USE / SELL / MONEY EARNED ===
        if prev_state:
            prev_consumable_count = len(prev_game_state.get('consumables', []))
            curr_consumable_count = len(inner_game_state.get('consumables', []))
            consumable_decreased = curr_consumable_count < prev_consumable_count

            current_gold = inner_game_state.get('gold', 0)
            prev_gold = prev_game_state.get('gold', self.previous_money)
            gold_increased = current_gold > prev_gold

            # Sell = consumable gone AND gold went up
            is_consumable_sell = consumable_decreased and gold_increased
            # Joker sold (in any state) = joker count dropped AND gold went up
            joker_decreased = len(inner_game_state.get('jokers', [])) < len(prev_game_state.get('jokers', []))
            is_joker_sell = joker_decreased and gold_increased

            # Successful consumable use, by set (counts reset per run, so drops are ignored)
            prev_uses = self._consumable_uses_by_set(prev_game_state)
            for card_set, count in self._consumable_uses_by_set(inner_game_state).items():
                used = count - prev_uses.get(card_set, 0)
                if used > 0 and card_set in CONSUMABLE_USE_REWARD:
                    use_reward = CONSUMABLE_USE_REWARD[card_set] * used
                    reward += use_reward
                    reward_breakdown.append(f"use_{card_set.lower()} x{used}: +{use_reward:.2f}")

            # Personal best single hand this run, scaled to the blind so overkill isn't paid
            prev_best = (prev_game_state.get('run_stats') or {}).get('high_score', 0) or 0
            curr_best = (inner_game_state.get('run_stats') or {}).get('high_score', 0) or 0
            if prev_best > 0 and curr_best > prev_best and blind_chips > 0:
                pb_reward = PB_HAND_BONUS * min(1.0, (curr_best - prev_best) / blind_chips)
                reward += pb_reward
                self.pb_hand_bonuses += 1
                reward_breakdown.append(f"personal_best_hand {prev_best}→{curr_best}: +{pb_reward:.2f}")

            # Money earned from non-sell sources: gold cards, joker effects, blind payouts, interest
            if gold_increased and not is_consumable_sell and not is_joker_sell:
                money_reward = (current_gold - prev_gold) * MONEY_EARNED_WEIGHT
                reward += money_reward
                reward_breakdown.append(f"money_earned +{current_gold - prev_gold}: +{money_reward:.2f}")

        # === RETRY PENALTY ===
        if retry_count > 0:
            retry_penalty = max(-0.5, -0.05 * retry_count)
            reward += retry_penalty
            reward_breakdown.append(f"retry_penalty: {retry_penalty:.2f}")

        # === HAND TYPE tracking ===
        current_hand_info = inner_game_state.get('current_hand', {})
        current_hand_played = current_hand_info.get('last_handname', 'None')
        if current_hand_played and current_hand_played != 'None' and current_hand_played != '':
            self.last_seen_hand_type = current_hand_played

        # === CORE SCORING REWARDS ===
        chip_gain = current_chips - self.previous_chips
        if chip_gain > 0 and game_over == 0 and not self.blind_already_defeated and blind_chips > 0:
            hand_type_to_use = getattr(self, 'last_seen_hand_type', 'Unknown')
            self._track_hand_played(chip_gain, hand_type_to_use, current_chips, blind_chips)

            chip_percentage = (chip_gain / blind_chips) * 100
            chip_reward = float(np.clip(chip_percentage * 0.1, 0, 10))
            reward += chip_reward
            reward_breakdown.append(f"Chip gain: +{chip_gain} ({chip_percentage:.1f}% of blind): +{chip_reward:.2f}")

        # === BLIND COMPLETION (per-blind, not just game win) ===
        if (blind_cleared_flag or game_won) and not self.blind_already_defeated:
            base_reward = 15.0 + 5.0 * ante

            round_info = inner_game_state.get('round', {})
            hands_left = round_info.get('hands_left', 0)
            hands_used = 4 - hands_left

            if hands_used == 1:
                efficiency_bonus = 10.0
            elif hands_used == 2:
                efficiency_bonus = 6.0
            elif hands_used == 3:
                efficiency_bonus = 3.0
            else:
                efficiency_bonus = 0.0

            total_blind_reward = base_reward + efficiency_bonus
            reward += total_blind_reward
            reward_breakdown.append(f"BLIND CLEARED: +{base_reward:.1f} (base) +{efficiency_bonus:.1f} (efficiency)")

            # Deck thinning bonus: +1 per card removed this round
            if self.pending_cards_removed > 0:
                deck_bonus = 1.0 * self.pending_cards_removed
                reward += deck_bonus
                reward_breakdown.append(f"deck_thinning {self.pending_cards_removed} cards: +{deck_bonus:.1f}")
                self.pending_cards_removed = 0

            self.blind_already_defeated = True
            self.winning_chips = current_chips
            print(f"✅ Blind cleared! Ante {self.current_blind_ante} - {self.current_blind_name} "
                  f"| {self.highest_chips_this_blind}/{self.current_blind_chips} chips "
                  f"[round {self.current_round_count}]")

        # === PROGRESSION BONUS ===
        if ante > self.current_ante:
            ante_progress_bonus = 5.0 * ante
            reward += ante_progress_bonus
            reward_breakdown.append(f"ante_progression ante {ante}: +{ante_progress_bonus:.1f}")
            self.current_ante = ante

        # === BOSS BLIND BONUS ===
        blind_name_val = inner_game_state.get('blind_name', '')
        is_boss = blind_name_val not in ['Small Blind', 'Big Blind', 'None', '']
        if (blind_cleared_flag or game_won) and is_boss and self.blind_already_defeated:
            reward += 5.0 * ante
            reward_breakdown.append(f"boss_blind_bonus ante {ante}: +{5.0 * ante:.1f}")

        # === GAME OVER PENALTY ===
        if game_over == 1 and not hasattr(self, 'game_over_penalty_applied') and not self.blind_already_defeated:
            blind_req = self.current_blind_chips if self.current_blind_chips else blind_chips
            progress = min(1.0, self.highest_chips_this_blind / max(1, blind_req))
            progress_credit = 8.0 * progress
            reward -= 15.0
            reward += progress_credit
            reward_breakdown.append(f"Game over: -15.0 + progress_credit {progress:.1%}: +{progress_credit:.2f}")
            self.game_over_penalty_applied = True
            # Capture context for CLI output
            self.game_over_chips = self.highest_chips_this_blind
            self.game_over_blind_chips = blind_req
            self.game_over_blind_name = self.current_blind_name if self.current_blind_name else blind_name
            self.game_over_ante = self.current_blind_ante if self.current_blind_ante else ante

        if reward_breakdown:
            self.episode_reward_breakdown.extend([f"[PLAY] {e}" for e in reward_breakdown])

        self.episode_total_reward += reward
        self.previous_chips = current_chips
        self.previous_hand_played = current_hand_played

        return float(np.clip(reward, -25.0, 120.0))

    def reset(self):
        """Reset for new episode."""
        if not self._started:
            return

        self.episode_count += 1

        if self.current_ante > self.best_ante_reached:
            self.best_ante_reached = self.current_ante

        if self.blind_already_defeated:
            self.wins += 1
            self._log_episode_win()
            self._log_win_rate()
        else:
            self._log_episode()

        # Reset episode state
        self._started = False
        self.current_ante = 1
        self.previous_chips = 0
        self.previous_hand_played = "None"
        self.blind_already_defeated = False
        self.episode_total_reward = 0
        self.last_seen_hand_type = 'Unknown'
        self.winning_chips = 0

        # Reset per-blind state
        self.current_blind_name = ''
        self.current_blind_chips = 0
        self.current_blind_ante = 1
        self.current_round_count = 0
        self.current_round_hands = []
        self.rounds_played = []
        self.highest_chips_this_blind = 0
        self.pending_cards_removed = 0
        self.game_over_chips = 0
        self.game_over_blind_chips = 0
        self.game_over_blind_name = ''
        self.game_over_ante = 1

        # Reset jokers
        self.episode_jokers = []

        self.previous_money = 0
        self.total_jokers = 0
        self.peak_money = 0
        self.episode_reward_breakdown = []
        self.fresh_joker_sells = 0
        self.max_joker_tenure = 0
        self.shop_exit_golds = []
        self.deck_stats = {}
        self.max_deck_stats = {}
        self.boosters_bought_skipped = 0
        self.pb_hand_bonuses = 0
        self.pack_purchased = False
        self.pack_picks = 0

        if hasattr(self, 'game_over_penalty_applied'):
            delattr(self, 'game_over_penalty_applied')

    def _all_rounds_for_log(self):
        """Build full round list including in-progress round."""
        rounds = list(self.rounds_played)
        if self.current_round_hands:
            rounds.append({
                'blind_name': self.current_blind_name,
                'blind_chips': self.current_blind_chips,
                'ante': self.current_blind_ante,
                'round_num': self.current_round_count,
                'hands': list(self.current_round_hands),
                'cleared': self.blind_already_defeated,
            })
        return rounds

    def _log_rounds(self, rounds):
        for round_data in rounds:
            cleared_marker = ' [cleared!]' if round_data.get('cleared') else ''
            round_num = round_data.get('round_num', 0)
            print(f"   Ante {round_data['ante']} - {round_data['blind_name']} ({round_data['blind_chips']} needed) [round {round_num}]{cleared_marker}:")
            for i, hand in enumerate(round_data['hands'], 1):
                print(f"      Hand {i}: {hand['hand_type']} "
                      f"(+{hand['chips']} chips, total: {hand['total_chips']}/{hand['blind_chips']})")

    def _log_episode_win(self):
        final_chips = self.winning_chips
        print(f"🎯 EPISODE {self.episode_count} WON with {final_chips} chips | "
              f"reward: {self.episode_total_reward:.1f}")
        rounds = self._all_rounds_for_log()
        self._log_rounds(rounds)
        jokers_str = ', '.join(self.episode_jokers) if self.episode_jokers else 'None'
        print(f"   Jokers: [{jokers_str}]")

    def _log_win_rate(self):
        win_rate = (self.wins / self.episode_count) * 100
        print(f"📊 Episode {self.episode_count}: {self.wins} wins ({win_rate:.1f}%)")

    def _track_hand_played(self, chip_gain, hand_type, current_chips, blind_chips):
        self.current_round_hands.append({
            'chips': chip_gain,
            'hand_type': hand_type if hand_type != "None" else "Unknown",
            'total_chips': current_chips,
            'blind_chips': blind_chips,
        })

    def _log_episode(self):
        if self.game_over_blind_name:
            header = (f"Defeated by Ante {self.game_over_ante} {self.game_over_blind_name} "
                      f"({self.game_over_chips}/{self.game_over_blind_chips} chips)")
        else:
            header = f"{self.highest_chips_this_blind} chips"

        print(f"🎯 EPISODE {self.episode_count}: {header} | reward: {self.episode_total_reward:.1f}")
        rounds = self._all_rounds_for_log()
        self._log_rounds(rounds)
        jokers_str = ', '.join(self.episode_jokers) if self.episode_jokers else 'None'
        print(f"   Jokers: [{jokers_str}]\n\n")
