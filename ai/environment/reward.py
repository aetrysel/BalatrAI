"""
Balatro Reward System - Ante 1 Focused

Simple reward system focused on teaching the AI to consistently beat
the 300-chip small blind in ante 1. No complex scaling, just core fundamentals.
"""

from typing import Dict, Any
import numpy as np

class BalatroRewardCalculator:
    """
    Focused reward calculator for ante 1 mastery.
    
    Goal: Teach AI to consistently score 300+ chips to beat small blind.
    Strategy: Reward consistent 75+ point hands, not risky high hands.
    """
    
    def __init__(self):
        self.previous_chips = 0
        self.previous_hand_played = "None"
        self.blind_already_defeated = False
        self.current_ante = 1
        self.best_ante_reached = 1
        
        # Episode tracking for win logging
        self.episode_count = 0
        self.wins = 0
        self.hands_played = []  # Track each hand in current episode
        self.episode_total_reward = 0
        self.last_seen_hand_type = 'Unknown'  # Store hand type when we see it
        self.winning_chips = 0  # Store chips when blind is defeated

        ## Shop vars
        self.previous_money = 0
        self.total_jokers = 0
        self.peak_money = 0
        
        # Percentage-based reward thresholds (% of blind requirement)
        # Updated to encourage bigger single hands
        self.REWARD_THRESHOLDS = {
            "excellent": 75.0,  # 75%+ of blind requirement (lowered from 80%)
            "good": 40.0,       # 40-74% of blind requirement (lowered from 50%)
            "decent": 20.0      # 20-39% of blind requirement (lowered from 25%)
        }
    PACK_STATE_IDS = {9, 10, 15, 17, 18}

    def calculate_reward(self, current_state, prev_state=None):
        reward = 0.0
        reward += self.calculate_shop_reward(current_state, prev_state)
        reward += self.calculate_pack_reward(current_state, prev_state)
        reward += self.calculate_play_reward(current_state, prev_state)
        return reward
    
    def calculate_shop_reward(self, current_state, prev_state=None):
        reward = 0.0
        current_game_state = current_state.get('game_state', {})
        prev_game_state = prev_state.get('game_state', {}) if prev_state else {}

        ante = current_game_state.get('ante', 1)
        current_money = current_game_state.get('gold', 0)
        prev_money = prev_game_state.get('gold', self.previous_money)
        money_spent = prev_money - current_money
        
        current_jokers = current_game_state.get('jokers', [])
        prev_joker_count = self.total_jokers
        jokers_gained = len(current_jokers) - prev_joker_count
        jokers_sold = max(0, prev_joker_count - len(current_jokers) - jokers_gained)

        # === JOKER PURCHASE ===
        if jokers_gained > 0:
            joker_value = 10.0
            reward += joker_value * jokers_gained

        # === JOKER SELL ===
        # Selling is sometimes correct (freeing slots, funding better joker)
        # Small penalty so agent doesn't sell randomly, but not too harsh
        if jokers_sold > 0:
            reward -= 1.0 * jokers_sold

        # === MONEY EARNED (interest + end of round income) ===
        # Reward gaining money — catches interest, blind rewards etc.
        # Only reward when money goes UP without a purchase context - TODO: refactor this when we allow tarot and spectral cards (Hermint, Temperance, Immolate, etc.)
        if current_money > prev_money and money_spent <= 0:
            money_gained = current_money - prev_money
            reward += money_gained * 0.5

        # === SMART SPENDING ===
        # Reward spending that resulted in a joker (value purchase)
        # Penalize spending with nothing to show for it (reroll fishing)
        if money_spent > 0:
            if jokers_gained > 0:
                # Good spend — bought something useful
                # Efficiency bonus for not overpaying relative to ante income
                efficiency = min(1.0, money_spent / max(1, current_money + money_spent))
                reward += 0.5 * (1.0 - efficiency)  # Reward cheaper purchases
            else:
                # Spent money but got nothing (reroll or bad buy)
                reward -= 0.3 * np.log1p(money_spent)

        # === JOKER SLOT UTILIZATION ===
        # Reward filling joker slots — empty slots are wasted potential
        max_joker_slots = 5
        utilization = len(current_jokers) / max_joker_slots
        reward += utilization * 0.5  # Small constant bonus per step in shop

        # === MONEY FLOOR PENALTY ===
        # Penalize leaving shop broke with no jokers
        # (spent everything and got nothing)
        if current_money == 0 and len(current_jokers) == 0:
            reward -= 3.0

        # === SKIP SHOP PENALTY ===
        # If agent skips shop with joker slots open and enough money to buy
        # This is tricky to detect directly, handle via utilization reward above

        # === RETRY PENALTY (carry over from play reward) ===
        retry_count = current_game_state.get('retry_count', 0)
        if retry_count > 0:
            reward -= 0.5 * retry_count

        # Update tracking
        self.previous_money = current_money
        self.total_jokers = len(current_jokers)

        # === VOUCHER PURCHASE ===
        # Detect: shop had a voucher before, doesn't now, and money decreased
        prev_shop_items = prev_game_state.get('shop', {}).get('items', [])
        curr_shop_items = current_game_state.get('shop', {}).get('items', [])
        prev_had_voucher = any(it.get('type') == 'voucher' for it in prev_shop_items)
        curr_has_voucher = any(it.get('type') == 'voucher' for it in curr_shop_items)
        if prev_had_voucher and not curr_has_voucher and money_spent > 0:
            reward += 8.0

        # === BOOSTER PURCHASE ===
        prev_state_id = prev_game_state.get('state', 0)
        curr_state_id = current_game_state.get('state', 0)
        if prev_state_id == 5 and curr_state_id in self.PACK_STATE_IDS:
            reward += 3.0

        return float(np.tanh(reward / 5.0) * 5.0)  # Soft clip shop rewards

    def calculate_pack_reward(self, current_state, prev_state=None) -> float:
        """Reward for pack interactions: selecting cards and skipping."""
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

        # A card was selected (choices decreased)
        if curr_choices < prev_choices:
            prev_joker_count = len(prev_game_state.get('jokers', []))
            curr_joker_count = len(current_game_state.get('jokers', []))
            prev_consumable_count = len(prev_game_state.get('consumables', []))
            curr_consumable_count = len(current_game_state.get('consumables', []))

            new_jokers = curr_joker_count - prev_joker_count
            new_consumables = curr_consumable_count - prev_consumable_count

            if prev_state_id == 18:  # BUFFOON_PACK — jokers
                if new_jokers > 0:
                    reward += 8.0
                else:
                    reward += 1.0  # still got something
            elif prev_state_id == 10:  # PLANET_PACK — levels up hands
                if new_consumables > 0:
                    reward += 4.0
            elif prev_state_id in {9, 15}:  # TAROT_PACK / SPECTRAL_PACK
                if new_consumables > 0:
                    reward += 3.0
            elif prev_state_id == 17:  # STANDARD_PACK — playing cards
                reward += 1.0

        # Skipped with choices remaining — small penalty
        if curr_state_id not in self.PACK_STATE_IDS and prev_choices > 0 and curr_choices == 0:
            # Choices ran out but we didn't select — skip was used
            # Only penalise if it wasn't the last card (prev_choices > 1 means we skipped early)
            pass  # Skip penalty intentionally light; skipping standard packs is fine

        return float(reward)

    def calculate_play_reward(self, current_state: Dict[str, Any],
                        prev_state: Dict[str, Any] = None) -> float:
        """Simple reward focused on ante 1 small blind success"""
        reward = 0.0
        reward_breakdown = []  # Track all reward components
        
        inner_game_state = current_state.get('game_state', {})
        
        # Extract key metrics
        current_chips = inner_game_state.get('chips', 0)
        game_over = inner_game_state.get('game_over', 0)
        retry_count = inner_game_state.get('retry_count', 0)
        ante = inner_game_state.get('ante', 1)
        
        # === USE_CONSUMABLE REWARD ===
        # Detect by consumable count decreasing while in SELECTING_HAND
        if prev_state:
            prev_inner = prev_state.get('game_state', {})
            prev_consumable_count = len(prev_inner.get('consumables', []))
            curr_consumable_count = len(inner_game_state.get('consumables', []))
            if curr_consumable_count < prev_consumable_count and inner_game_state.get('state', 0) == 1:
                reward += 2.0  # reward per consumable used

        # === RETRY PENALTY ===
        # Negative reward for invalid actions that require retries
        if retry_count > 0:
            retry_penalty = -0.1 * retry_count  # -0.1 per retry
            reward += retry_penalty
            reward_breakdown.append(f"retry_penalty: {retry_penalty:.2f} (retries: {retry_count})")
        
        # Check if blind is defeated by comparing chips to requirement
        blind_chips = inner_game_state.get('blind_chips', 300) # TODO 300 only focusing on first blind
        # Only consider blind defeated if we actually have chips AND a valid blind requirement
        blind_defeated = inner_game_state.get('game_win', 0)
        
        
        # Hand type info - use current hand scoring
        current_hand_info = inner_game_state.get('current_hand', {})
        current_hand_played = current_hand_info.get('handname', 'None')
        
        # Store hand type when we see it (for later use when chips change)
        if current_hand_played and current_hand_played != 'None' and current_hand_played != '':
            self.last_seen_hand_type = current_hand_played
        
        # === CORE SCORING REWARDS ===
        # Reward good hands (chip increases) based on % of blind requirement
        chip_gain = current_chips - self.previous_chips
        if chip_gain > 0 and game_over == 0 and not self.blind_already_defeated and blind_chips > 0:
            # Track this hand for episode logging (use stored hand type)
            # Only track if blind not already defeated
            hand_type_to_use = getattr(self, 'last_seen_hand_type', 'Unknown')
            self._track_hand_played(chip_gain, hand_type_to_use, current_chips)
            
            # Calculate percentage of blind requirement this hand achieved
            chip_percentage = (chip_gain / blind_chips) * 100
            
            chip_reward = float(np.clip(chip_percentage**2 / 500, 0, 20))  # Quadratic scaling which approximates old reward
            reward += chip_reward
            reward_breakdown.append(f"Chip gain: +{chip_gain} chips ({chip_percentage:.1f}% of blind): +{chip_reward:.2f}")
            # Monster hand bonus - overkill reward for beating blind in one shot
            # if chip_percentage >= 100:
            #     reward += 20.0  # Huge bonus for one-shot blind completion
            #     reward_breakdown.append(f"MONSTER HAND - One-shot blind kill (+{chip_gain} chips, {chip_percentage:.1f}%): +20.0")
            # elif chip_percentage >= self.REWARD_THRESHOLDS["excellent"]:
            #     reward += 10.0
            #     reward_breakdown.append(f"Excellent hand (+{chip_gain} chips, {chip_percentage:.1f}% of blind): +10.0")
            # elif chip_percentage >= self.REWARD_THRESHOLDS["good"]:
            #     reward += 4.0
            #     reward_breakdown.append(f"Good hand (+{chip_gain} chips, {chip_percentage:.1f}% of blind): +4.0")
            # elif chip_percentage >= self.REWARD_THRESHOLDS["decent"]:
            #     reward += 1.0
            #     reward_breakdown.append(f"Decent hand (+{chip_gain} chips, {chip_percentage:.1f}% of blind): +1.0")
            # else:
            #     reward += 0.1
            #     reward_breakdown.append(f"Small hand (+{chip_gain} chips, {chip_percentage:.1f}% of blind): +0.1")
        
        # === REMOVED HAND TYPE REWARDS ===
        # Hand type rewards removed - in Balatro, only chips matter!
        # A High Card scoring 300 chips beats a Full House scoring 100 chips
                
        # === BLIND COMPLETION ===
        # Main goal - beat the blind (only reward once per episode)
        if blind_defeated and not self.blind_already_defeated:
            # Exponential scaling — ante 8 boss worth ~16x ante 1 small blind
            ante_multiplier = 1.5 ** (ante - 1)
            base_reward = 30.0 * ante_multiplier
            reward += base_reward
            
            # Calculate hands used (starting hands - hands_left)
            round_info = inner_game_state.get('round', {})
            hands_left = round_info.get('hands_left', 0)
            # Assume we start with 4 hands in ante 1
            hands_used = 4 - hands_left
            
            # Efficiency bonus - reward fewer hands used
            if hands_used == 1:
                efficiency_bonus = 25.0
                reward_breakdown.append(f"ONE-HAND VICTORY BONUS: +{efficiency_bonus}")
            elif hands_used == 2:
                efficiency_bonus = 15.0
                reward_breakdown.append(f"Two-hand efficiency bonus: +{efficiency_bonus}")
            elif hands_used == 3:
                efficiency_bonus = 8.0
                reward_breakdown.append(f"Three-hand efficiency bonus: +{efficiency_bonus}")
            else:
                efficiency_bonus = max(0, 5.0 - hands_used)  # Diminishing returns
                if efficiency_bonus > 0:
                    reward_breakdown.append(f"Efficiency bonus: +{efficiency_bonus:.1f}")
                else:
                    efficiency_bonus = 0
            
            total_blind_reward = base_reward + efficiency_bonus
            reward += total_blind_reward
            reward_breakdown.append(f"BLIND DEFEATED: +{base_reward} (base)")
            self.blind_already_defeated = True
            self.winning_chips = current_chips  # Store winning chip count

        # === PROGRESSION BONUS ===
        # Detect ante increase
        ante = inner_game_state.get('ante', 1)
        if ante > self.current_ante:
            ante_progress_bonus = 10.0 * ante  # 20 for ante 2, 30 for ante 3...
            reward += ante_progress_bonus
            self.current_ante = ante

        # === BOSS BLIND BONUS ===
        # Boss blinds are checkpoints — reward them separately (may conflict with ante progression bonus and inflate rewards...)
        blind_name = inner_game_state.get('blind_name', '')
        is_boss = blind_name not in ['Small Blind', 'Big Blind', 'None']
        if blind_defeated and is_boss:
            reward += 15.0 * ante  # Extra signal for clearing bosses
            
        # === PENALTIES ===
        # Game over penalty - ONLY for actual losses (blind not defeated)
        if game_over == 1 and not hasattr(self, 'game_over_penalty_applied') and not self.blind_already_defeated:
            reward -= 20.0  # Normalized from -200 to -20
            reward_breakdown.append("Game over: -20.0")
            self.game_over_penalty_applied = True
            
        # Track episode total reward
        self.episode_total_reward += reward
        
        # Keep only essential logging - no step-by-step reward breakdown
        
        # Update tracking
        self.previous_chips = current_chips
        self.previous_hand_played = current_hand_played
        
        return float(np.clip(reward, -25.0, 500.0))
    
    def reset(self):
        """Reset for new episode - log win details if episode was won"""
        self.episode_count += 1

        # Update furthest ante-reached
        current_ante = self.current_ante  # track this
        if current_ante > self.best_ante_reached:
            self.best_ante_reached = current_ante
            # Bonus logged for curriculum tracking
        
        # Check if this episode was a win and log details
        if self.blind_already_defeated:
            self.wins += 1
            self._log_episode_win()
            
            # Show win rate on every win for immediate feedback
            self._log_win_rate()
        else:
            self._log_episode()
        
        # Reset for next episode
        self.current_ante = 1
        self.previous_chips = 0
        self.previous_hand_played = "None"
        self.blind_already_defeated = False
        self.hands_played = []
        self.episode_total_reward = 0
        self.last_seen_hand_type = 'Unknown'
        self.winning_chips = 0

        self.previous_money = 0
        self.total_jokers = 0
        self.peak_money = 0
     
        # Reset game over penalty flag
        if hasattr(self, 'game_over_penalty_applied'):
            delattr(self, 'game_over_penalty_applied')

    def _log_episode_win(self):
        """Log detailed breakdown of winning episode"""
        final_chips = self.winning_chips if self.winning_chips > 0 else (max([hand['total_chips'] for hand in self.hands_played]) if self.hands_played else 0)
        
        # Episode summary
        print(f"🎯 EPISODE {self.episode_count} COMPLETE: WON with {final_chips} chips in {len(self.hands_played)} hands | Total reward: {self.episode_total_reward:.1f}")
        # Hand-by-hand breakdown
        for i, hand in enumerate(self.hands_played, 1):
            print(f"   Hand {i}: {hand['hand_type']} (+{hand['chips']} chips, total: {hand['total_chips']})")
    
    def _log_win_rate(self):
        """Log win rate statistics"""
        win_rate = (self.wins / self.episode_count) * 100
        print(f"📊 Episode {self.episode_count}: {self.wins} wins ({win_rate:.1f}%) | Last 100: TBD")
    
    def _track_hand_played(self, chip_gain, current_hand_played, current_chips):
        """Track hand details for episode logging"""
        hand_info = {
            'chips': chip_gain,
            'hand_type': current_hand_played if current_hand_played != "None" else "Unknown",
            'total_chips': current_chips
        }
        self.hands_played.append(hand_info)

    def _log_episode(self):
        final_chips = self.winning_chips if self.winning_chips > 0 else (max([hand['total_chips'] for hand in self.hands_played]) if self.hands_played else 0)

        # Episode summary
        print(f"🎯 EPISODE {self.episode_count}: "
            # reached ante {self.current_ante}, "
            # f"cleared {self.blind_name} with "
            f"{self.winning_chips} chips | reward: {self.episode_total_reward:.1f}")  
        # Hand-by-hand breakdown
        for i, hand in enumerate(self.hands_played, 1):
            print(f"   Hand {i}: {hand['hand_type']} (+{hand['chips']} chips, total: {hand['total_chips']})")
    