from typing import Dict, Any
import numpy as np

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

        self.episode_reward_breakdown: list[str] = []

        self.REWARD_THRESHOLDS = {
            "excellent": 75.0,
            "good": 40.0,
            "decent": 20.0
        }

    PACK_STATE_IDS = {9, 10, 15, 17, 18}

    def calculate_reward(self, current_state, prev_state=None):
        self._started = True
        reward = 0.0
        reward += self.calculate_shop_reward(current_state, prev_state)
        reward += self.calculate_pack_reward(current_state, prev_state)
        reward += self.calculate_play_reward(current_state, prev_state)
        return reward

    def calculate_shop_reward(self, current_state, prev_state=None):
        reward = 0.0
        breakdown: list[str] = []
        current_game_state = current_state.get('game_state', {})
        prev_game_state = prev_state.get('game_state', {}) if prev_state else {}

        current_money = current_game_state.get('gold', 0)
        prev_money = prev_game_state.get('gold', self.previous_money)
        money_spent = prev_money - current_money

        current_jokers = current_game_state.get('jokers', [])
        prev_joker_count = self.total_jokers
        jokers_gained = len(current_jokers) - prev_joker_count
        jokers_sold = max(0, prev_joker_count - len(current_jokers) - jokers_gained)

        if jokers_gained > 0:
            reward += 10.0 * jokers_gained
            breakdown.append(f"jokers_gained x{jokers_gained}: +{10.0 * jokers_gained:.1f}")

        if jokers_sold > 0:
            reward -= 1.0 * jokers_sold
            breakdown.append(f"jokers_sold x{jokers_sold}: -{1.0 * jokers_sold:.1f}")

        if current_money > prev_money and money_spent <= 0:
            money_gained = current_money - prev_money
            reward += money_gained * 0.5
            breakdown.append(f"money_earned +{money_gained}: +{money_gained * 0.5:.2f}")

        if money_spent > 0:
            if jokers_gained > 0:
                efficiency = min(1.0, money_spent / max(1, current_money + money_spent))
                reward += 0.5 * (1.0 - efficiency)
                breakdown.append(f"joker_efficiency: +{0.5 * (1.0 - efficiency):.2f}")
            else:
                penalty = 0.3 * np.log1p(money_spent)
                reward -= penalty
                breakdown.append(f"money_spent_penalty -{money_spent}: -{penalty:.2f}")

        max_joker_slots = 5
        utilization = len(current_jokers) / max_joker_slots
        reward += utilization * 0.5
        breakdown.append(f"joker_utilization ({len(current_jokers)}/5): +{utilization * 0.5:.2f}")

        if current_money == 0 and len(current_jokers) == 0:
            reward -= 3.0
            breakdown.append("broke_no_jokers: -3.00")

        retry_count = current_game_state.get('retry_count', 0)
        if retry_count > 0:
            reward -= 0.5 * retry_count
            breakdown.append(f"retry_penalty x{retry_count}: -{0.5 * retry_count:.2f}")

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

        prev_state_id = prev_game_state.get('state', 0)
        curr_state_id = current_game_state.get('state', 0)
        if prev_state_id == 5 and curr_state_id in self.PACK_STATE_IDS:
            reward += 3.0
            breakdown.append("entered_pack: +3.00")

        scaled = float(np.tanh(reward / 5.0) * 5.0)
        if breakdown:
            self.episode_reward_breakdown.append(
                f"[SHOP] raw={reward:.2f} → scaled={scaled:.2f}: " + "; ".join(breakdown)
            )
        return scaled

    def calculate_pack_reward(self, current_state, prev_state=None) -> float:
        reward = 0.0
        if not prev_state:
            return reward

        current_game_state = current_state.get('game_state', {})
        prev_game_state = prev_state.get('game_state', {})

        prev_state_id = prev_game_state.get('state', 0)
        curr_state_id = current_game_state.get('state', 0)

        if prev_state_id not in self.PACK_STATE_IDS:
            return reward

        prev_choices = prev_game_state.get('pack_choices', 0)
        curr_choices = current_game_state.get('pack_choices', 0)

        if curr_choices < prev_choices:
            prev_joker_count = len(prev_game_state.get('jokers', []))
            curr_joker_count = len(current_game_state.get('jokers', []))
            prev_consumable_count = len(prev_game_state.get('consumables', []))
            curr_consumable_count = len(current_game_state.get('consumables', []))

            new_jokers = curr_joker_count - prev_joker_count
            new_consumables = curr_consumable_count - prev_consumable_count

            if prev_state_id == 18:
                reward += 8.0 if new_jokers > 0 else 1.0
            elif prev_state_id == 10:
                if new_consumables > 0:
                    reward += 4.0
            elif prev_state_id in {9, 15}:
                if new_consumables > 0:
                    reward += 3.0
            elif prev_state_id == 17:
                reward += 1.0

        if reward != 0.0:
            pack_names = {9: "TAROT_PACK", 10: "PLANET_PACK", 15: "SPECTRAL_PACK", 17: "STANDARD_PACK", 18: "BUFFOON_PACK"}
            self.episode_reward_breakdown.append(
                f"[PACK] {pack_names.get(prev_state_id, f'PACK_{prev_state_id}')}: +{reward:.2f}"
            )
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

        # === USE_CONSUMABLE REWARD ===
        if prev_state:
            prev_consumable_count = len(prev_game_state.get('consumables', []))
            curr_consumable_count = len(inner_game_state.get('consumables', []))
            if curr_consumable_count < prev_consumable_count and inner_game_state.get('state', 0) == 1:
                reward += 2.0
                reward_breakdown.append("use_consumable: +2.00")

        # === RETRY PENALTY ===
        if retry_count > 0:
            retry_penalty = -0.1 * retry_count
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
            chip_reward = float(np.clip(chip_percentage**2 / 500, 0, 20))
            reward += chip_reward
            reward_breakdown.append(f"Chip gain: +{chip_gain} ({chip_percentage:.1f}% of blind): +{chip_reward:.2f}")

        # === BLIND COMPLETION (per-blind, not just game win) ===
        if (blind_cleared_flag or game_won) and not self.blind_already_defeated:
            ante_multiplier = 1.5 ** (ante - 1)
            base_reward = 30.0 * ante_multiplier

            round_info = inner_game_state.get('round', {})
            hands_left = round_info.get('hands_left', 0)
            hands_used = 4 - hands_left

            if hands_used == 1:
                efficiency_bonus = 25.0
            elif hands_used == 2:
                efficiency_bonus = 15.0
            elif hands_used == 3:
                efficiency_bonus = 8.0
            else:
                efficiency_bonus = max(0, 5.0 - hands_used)

            total_blind_reward = base_reward + efficiency_bonus
            reward += total_blind_reward
            reward_breakdown.append(f"BLIND CLEARED: +{base_reward:.1f} (base) +{efficiency_bonus:.1f} (efficiency)")
            self.blind_already_defeated = True
            self.winning_chips = current_chips
            print(f"✅ Blind cleared! Ante {self.current_blind_ante} - {self.current_blind_name} "
                  f"| {self.highest_chips_this_blind}/{self.current_blind_chips} chips "
                  f"[round {self.current_round_count}]")

        # === PROGRESSION BONUS ===
        if ante > self.current_ante:
            ante_progress_bonus = 10.0 * ante
            reward += ante_progress_bonus
            reward_breakdown.append(f"ante_progression ante {ante}: +{ante_progress_bonus:.1f}")
            self.current_ante = ante

        # === BOSS BLIND BONUS ===
        blind_name_val = inner_game_state.get('blind_name', '')
        is_boss = blind_name_val not in ['Small Blind', 'Big Blind', 'None', '']
        if (blind_cleared_flag or game_won) and is_boss and self.blind_already_defeated:
            reward += 15.0 * ante
            reward_breakdown.append(f"boss_blind_bonus ante {ante}: +{15.0 * ante:.1f}")

        # === GAME OVER PENALTY ===
        if game_over == 1 and not hasattr(self, 'game_over_penalty_applied') and not self.blind_already_defeated:
            reward -= 20.0
            reward_breakdown.append("Game over: -20.0")
            self.game_over_penalty_applied = True
            # Capture context for CLI output
            self.game_over_chips = self.highest_chips_this_blind
            self.game_over_blind_chips = self.current_blind_chips if self.current_blind_chips else blind_chips
            self.game_over_blind_name = self.current_blind_name if self.current_blind_name else blind_name
            self.game_over_ante = self.current_blind_ante if self.current_blind_ante else ante

        if reward_breakdown:
            self.episode_reward_breakdown.extend([f"[PLAY] {e}" for e in reward_breakdown])

        self.episode_total_reward += reward
        self.previous_chips = current_chips
        self.previous_hand_played = current_hand_played

        return float(np.clip(reward, -25.0, 500.0))

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
        print(f"   Jokers: [{jokers_str}]")
