from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict, deque


class BalatroMetricsCallback(BaseCallback):
    def __init__(self, log_freq=10, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq

        # All-time totals
        self.total_episodes = 0
        self.total_wins = 0
        self.total_blind_clears = 0
        self.total_blind_attempts = 0   # SELECT_BLIND (action 5) only
        self.total_blind_selections = 0  # attempts + skips (for overall skip rate)
        self.total_blind_skips = 0
        self.total_consumables_used = 0

        # Best-ever metrics
        self.best_ante = 0
        self.best_round = 0
        self.highest_chips = 0
        self.max_money_ever = 0

        # All-time hand type tracking
        self.hand_type_counts = defaultdict(int)
        self.hand_type_scores = defaultdict(list)

        # Sliding window for win rate (last 100 episodes)
        self.recent_window = 100
        self.recent_outcomes = []

        # Rolling last-100 blind outcomes (1=cleared, 0=failed)
        self.recent_blind_outcomes = deque(maxlen=100)

        # --- Per-episode accumulators (reset at episode end) ---
        self.current_run_length = 0
        self.episode_money_earned = 0.0
        self.episode_money_spent = 0.0
        self.episode_plays = 0
        self.episode_discards = 0
        self.episode_shop_count = 0
        self.episode_blind_attempts = 0
        self.episode_blind_skips = 0
        self.episode_jokers_gained = 0
        self.episode_jokers_sold = 0
        self.consumables_used_this_episode = 0

        # Previous-step deltas
        self._prev_chips = 0
        self._prev_gold = 0
        self._prev_consumable_count = 0
        self._prev_joker_count = 0

        # --- Rollout buffers (reset after each policy update) ---
        self._rollout_episodes = 0
        self._rollout_wins = 0
        self._rollout_blind_clears = 0
        self._rollout_blind_attempts = 0
        self._rollout_chips = []
        self._rollout_antes = []
        self._rollout_money_earned = []
        self._rollout_money_spent = []
        self._rollout_play_discard_ratios = []
        self._rollout_jokers_gained = []
        self._rollout_jokers_sold = []
        self._rollout_jokers_end = []
        self._rollout_consumables_used = []
        self._rollout_blind_skip_rates = []
        self._rollout_hand_counts = defaultdict(int)

    def _win_pct(self):
        if self.total_episodes == 0:
            return 0.0
        return round(100.0 * self.total_wins / self.total_episodes, 2)

    def _mean(self, lst):
        return sum(lst) / len(lst) if lst else 0.0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        needs_dump = False

        for done, info in zip(dones, infos):
            self.current_run_length += 1

            # Chip-gain based hand tracking
            curr_chips = info.get("chips", 0)
            chip_gain = curr_chips - self._prev_chips
            hand_type = info.get("hand_type", "None")
            if chip_gain > 0 and hand_type and hand_type != "None":
                self.hand_type_counts[hand_type] += 1
                self.hand_type_scores[hand_type].append(chip_gain)
                self._rollout_hand_counts[hand_type] += 1
            self._prev_chips = curr_chips

            # Consumable usage (count decrease = successful use)
            curr_consumables = info.get("consumable_count", 0)
            if curr_consumables < self._prev_consumable_count:
                used = self._prev_consumable_count - curr_consumables
                self.total_consumables_used += used
                self.consumables_used_this_episode += used
            self._prev_consumable_count = curr_consumables

            # Money tracking
            gold = info.get("gold", 0)
            if gold > self._prev_gold:
                self.episode_money_earned += gold - self._prev_gold
            elif gold < self._prev_gold:
                self.episode_money_spent += self._prev_gold - gold
            if gold > self.max_money_ever:
                self.max_money_ever = gold
            self._prev_gold = gold

            # Joker delta tracking
            curr_joker_count = len(info.get("jokers", []))
            joker_delta = curr_joker_count - self._prev_joker_count
            if joker_delta > 0:
                self.episode_jokers_gained += joker_delta
            elif joker_delta < 0:
                self.episode_jokers_sold += abs(joker_delta)
            self._prev_joker_count = curr_joker_count

            # Action tracking
            action_taken = info.get("action_taken", 0)
            if action_taken == 2:
                self.episode_plays += 1
            elif action_taken == 3:
                self.episode_discards += 1
            elif action_taken == 10:
                self.episode_shop_count += 1
            elif action_taken == 5:    # SELECT_BLIND (attempt)
                self.total_blind_attempts += 1
                self._rollout_blind_attempts += 1
                self.episode_blind_attempts += 1
                self.total_blind_selections += 1
            elif action_taken == 17:   # SKIP_BLIND
                self.total_blind_skips += 1
                self.total_blind_selections += 1
                self.episode_blind_skips += 1

            # Blind clear: capture chip score + rolling blind outcome
            if info.get("blind_cleared", False):
                self.total_blind_clears += 1
                self._rollout_blind_clears += 1
                clear_chips = info.get("chips", 0)  # cumulative blind score; chips only increases during a blind
                if clear_chips > self.highest_chips:
                    self.highest_chips = clear_chips
                self._rollout_chips.append(clear_chips)
                self.recent_blind_outcomes.append(1)

            if not done:
                continue

            # --- Episode ended ---
            self.total_episodes += 1
            won = bool(info.get("won", False))
            ante = info.get("ante", 0)
            round_ = info.get("round", 0)

            if won:
                self.total_wins += 1

            self.recent_outcomes.append(1 if won else 0)
            if len(self.recent_outcomes) > self.recent_window:
                self.recent_outcomes.pop(0)

            # Failed blind outcome (timeout is not a blind failure)
            if not won and not info.get("timeout", False):
                self.recent_blind_outcomes.append(0)

            if ante > self.best_ante or (ante == self.best_ante and round_ > self.best_round):
                self.best_ante = ante
                self.best_round = round_

            # Append to rollout buffers
            self._rollout_episodes += 1
            if won:
                self._rollout_wins += 1
            self._rollout_antes.append(ante)
            self._rollout_money_earned.append(self.episode_money_earned)
            self._rollout_money_spent.append(self.episode_money_spent)
            self._rollout_play_discard_ratios.append(
                self.episode_plays / max(1, self.episode_discards))
            self._rollout_jokers_gained.append(self.episode_jokers_gained)
            self._rollout_jokers_sold.append(self.episode_jokers_sold)
            self._rollout_jokers_end.append(len(info.get("jokers", [])))
            self._rollout_consumables_used.append(self.consumables_used_this_episode)
            ep_blind_total = self.episode_blind_attempts + self.episode_blind_skips
            self._rollout_blind_skip_rates.append(
                (self.episode_blind_skips / max(1, ep_blind_total)))

            # --- Live TensorBoard (best-ever / all-time / rolling) ---
            hand_types = [
                "High Card", "Pair", "Two Pair", "Three of a Kind",
                "Straight", "Flush", "Full House", "Four of a Kind",
                "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"
            ]

            self.logger.record("game/win_rate_overall",
                               self._win_pct())
            self.logger.record("game/win_rate_last_100",
                               round(100 * sum(self.recent_outcomes) / max(1, len(self.recent_outcomes)), 2))
            self.logger.record("game/blind_clear_rate_overall",
                               round(self.total_blind_clears / max(1, self.total_blind_attempts), 2))
            self.logger.record("game/blind_clear_rate_last_100",
                               round(100 * sum(self.recent_blind_outcomes) / max(1, len(self.recent_blind_outcomes)), 2))
            self.logger.record("game/highest_chips_ever",    self.highest_chips)
            self.logger.record("game/best_ante",             self.best_ante)
            self.logger.record("game/best_round",            self.best_round)
            self.logger.record("money/max_ever_held",        self.max_money_ever)
            self.logger.record("custom/total_wins",          self.total_wins)
            self.logger.record("custom/total_blind_clears",  self.total_blind_clears)
            self.logger.record("custom/total_consumables_used", self.total_consumables_used)
            self.logger.record("actions/blind_skip_rate",
                               round(self.total_blind_skips / max(1, self.total_blind_selections), 2))

            # All-time hand type distribution (played types only)
            total_hands = sum(self.hand_type_counts.values()) or 1
            for ht in hand_types:
                count = self.hand_type_counts[ht]
                if count == 0:
                    continue
                safe_name = ht.replace(" ", "_").lower()
                self.logger.record(f"hands/count_{safe_name}", count)
                self.logger.record(f"hands/pct_{safe_name}",   count / total_hands)
                if self.hand_type_scores[ht]:
                    avg_score = sum(self.hand_type_scores[ht]) / len(self.hand_type_scores[ht])
                    self.logger.record(f"hands/avg_score_{safe_name}", avg_score)

            # --- Per-episode metrics (commented out; restore if desired) ---
            # chips = info.get("chips", 0)
            # self.logger.record("game/chips_this_episode",         chips)
            # self.logger.record("game/run_length_this_episode",    self.current_run_length)
            # self.logger.record("game/rounds_this_episode",        round_)
            # self.logger.record("game/ante_this_episode",          ante)
            # self.logger.record("money/episode_earned",            self.episode_money_earned)
            # self.logger.record("money/episode_spent",             self.episode_money_spent)
            # self.logger.record("actions/play_discard_ratio",      self.episode_plays / max(1, self.episode_discards))
            # self.logger.record("custom/joker_count",              len(info.get("jokers", [])))
            # self.logger.record("custom/consumables_used_episode", self.consumables_used_this_episode)

            # Reset per-episode counters
            self.current_run_length = 0
            self._prev_chips = 0
            self._prev_gold = 0
            self._prev_consumable_count = 0
            self._prev_joker_count = 0
            self.episode_money_earned = 0.0
            self.episode_money_spent = 0.0
            self.episode_plays = 0
            self.episode_discards = 0
            self.episode_shop_count = 0
            self.episode_blind_attempts = 0
            self.episode_blind_skips = 0
            self.episode_jokers_gained = 0
            self.episode_jokers_sold = 0
            self.consumables_used_this_episode = 0

            needs_dump = True

        if needs_dump:
            self.logger.dump(self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        hand_types = [
            "High Card", "Pair", "Two Pair", "Three of a Kind",
            "Straight", "Flush", "Full House", "Four of a Kind",
            "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"
        ]

        if self._rollout_episodes > 0:
            n = self._rollout_episodes
            self.logger.record("rollout/win_rate",
                               round(100.0 * self._rollout_wins / n, 2))
            self.logger.record("rollout/blind_clear_rate",
                               round(self._rollout_blind_clears / max(1, self._rollout_blind_attempts), 2))
            self.logger.record("rollout/mean_ante",
                               self._mean(self._rollout_antes))
            self.logger.record("rollout/mean_money_earned",
                               self._mean(self._rollout_money_earned))
            self.logger.record("rollout/mean_money_spent",
                               self._mean(self._rollout_money_spent))
            self.logger.record("rollout/mean_play_discard_ratio",
                               self._mean(self._rollout_play_discard_ratios))
            self.logger.record("rollout/mean_jokers_gained",
                               self._mean(self._rollout_jokers_gained))
            self.logger.record("rollout/mean_jokers_sold",
                               self._mean(self._rollout_jokers_sold))
            self.logger.record("rollout/mean_jokers_end",
                               self._mean(self._rollout_jokers_end))
            self.logger.record("rollout/mean_consumables_used",
                               self._mean(self._rollout_consumables_used))
            self.logger.record("rollout/mean_blind_skip_rate",
                               self._mean(self._rollout_blind_skip_rates))
            if self._rollout_chips:
                self.logger.record("rollout/mean_chips",
                                   self._mean(self._rollout_chips))

            # Rollout hand type distribution (played types only)
            rollout_total = sum(self._rollout_hand_counts.values()) or 1
            for ht in hand_types:
                if self._rollout_hand_counts[ht] == 0:
                    continue
                safe_name = ht.replace(" ", "_").lower()
                self.logger.record(f"hands/pct_{safe_name}_rollout",
                                   self._rollout_hand_counts[ht] / rollout_total)

        # Reset rollout buffers
        self._rollout_episodes = 0
        self._rollout_wins = 0
        self._rollout_blind_clears = 0
        self._rollout_blind_attempts = 0
        self._rollout_chips = []
        self._rollout_antes = []
        self._rollout_money_earned = []
        self._rollout_money_spent = []
        self._rollout_play_discard_ratios = []
        self._rollout_jokers_gained = []
        self._rollout_jokers_sold = []
        self._rollout_jokers_end = []
        self._rollout_consumables_used = []
        self._rollout_blind_skip_rates = []
        self._rollout_hand_counts = defaultdict(int)

        self.logger.dump(self.num_timesteps)

        print(f"\n🔄 Policy update at {self.num_timesteps} steps | "
              f"Episodes: {self.total_episodes} | "
              f"Win rate: {self._win_pct():.1f}% | "
              f"Best ante: {self.best_ante} round: {self.best_round}")
