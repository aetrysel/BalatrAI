from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict, deque

# Per-run stats logged as run/episode_*, run/best_*, run/rollout_mean_*
RUN_STAT_NAMES = [
    "high_score", "max_gold", "rerolls", "planets_used", "tarots_used",
    "jokers_purchased", "hands_played", "discards_remaining", "final_deck_size",
]

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
        self.total_consumables_used = 0    # successful: action 14 (count decreased) + action 18
        self.total_consumables_failed = 0  # failed: action 14 but count didn't decrease
        self.total_consumables_sold = 0    # sold: action 19
        self.total_boss_blind_clears = 0
        self.total_boss_blind_encounters = 0  # clears + losses on boss blinds

        # Best-ever metrics
        self.best_ante = 0
        self.best_round = 0
        self.highest_blind_score = 0
        self.highest_single_hand_score = 0
        self.max_money_ever = 0
        self.best_run_stats = defaultdict(int)

        # All-time hand type tracking
        self.hand_type_counts = defaultdict(int)
        self.hand_type_scores = defaultdict(list)
        self.hand_type_max_scores = defaultdict(int)

        # Sliding window for win rate (last 100 episodes)
        self.recent_window = 100
        self.recent_outcomes = []

        # Rolling last-100 blind outcomes (1=cleared, 0=failed)
        self.recent_blind_outcomes = deque(maxlen=100)

        # Rolling last-100 boss blind outcomes (1=cleared, 0=failed)
        self.recent_boss_blind_outcomes = deque(maxlen=100)

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
        self.episode_consumables_used = 0
        self.episode_consumables_failed = 0
        self.episode_consumables_sold = 0
        self.episode_max_gold = 0
        self.episode_jokers_purchased = 0

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
        self._rollout_joker_buy_sell_ratios = []
        self._rollout_consumables_used = []
        self._rollout_consumables_failed = []
        self._rollout_consumables_sold = []
        self._rollout_blind_skip_rates = []
        self._rollout_hand_counts = defaultdict(int)
        self._rollout_run_stats = defaultdict(list)

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
                if chip_gain > self.hand_type_max_scores[hand_type]:
                    self.hand_type_max_scores[hand_type] = chip_gain
                if chip_gain > self.highest_single_hand_score:
                    self.highest_single_hand_score = chip_gain
            self._prev_chips = curr_chips

            # Consumable tracking split by action type
            curr_consumables = info.get("consumable_count", 0)
            action_taken = info.get("action_taken", 0)
            if action_taken == 14:  # USE_CONSUMABLE: success = count decreased
                if curr_consumables < self._prev_consumable_count:
                    self.total_consumables_used += 1
                    self.episode_consumables_used += 1
                else:
                    self.total_consumables_failed += 1
                    self.episode_consumables_failed += 1
            elif action_taken == 18:  # BUY_AND_USE_CONSUMABLE: used if money was spent
                if info.get("gold", 0) < self._prev_gold:
                    self.total_consumables_used += 1
                    self.episode_consumables_used += 1
                else:
                    self.total_consumables_failed += 1
                    self.episode_consumables_failed += 1
            elif action_taken == 19:  # SELL_CONSUMABLE
                self.total_consumables_sold += 1
                self.episode_consumables_sold += 1
            self._prev_consumable_count = curr_consumables

            # Money tracking
            gold = info.get("gold", 0)
            if gold > self._prev_gold:
                self.episode_money_earned += gold - self._prev_gold
            elif gold < self._prev_gold:
                self.episode_money_spent += self._prev_gold - gold
            if gold > self.max_money_ever:
                self.max_money_ever = gold
            if gold > self.episode_max_gold:
                self.episode_max_gold = gold
            self._prev_gold = gold

            # Joker delta tracking
            curr_joker_count = len(info.get("jokers", []))
            joker_delta = curr_joker_count - self._prev_joker_count
            if joker_delta > 0:
                self.episode_jokers_gained += joker_delta
                if action_taken == 7:  # BUY_JOKER
                    self.episode_jokers_purchased += joker_delta
            elif joker_delta < 0:
                self.episode_jokers_sold += abs(joker_delta)
            self._prev_joker_count = curr_joker_count

            # Action tracking
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
                if clear_chips > self.highest_blind_score:
                    self.highest_blind_score = clear_chips
                self._rollout_chips.append(clear_chips)
                self.recent_blind_outcomes.append(1)
                if info.get("blind_is_boss", False):
                    self.total_boss_blind_clears += 1
                    self.total_boss_blind_encounters += 1
                    self.recent_boss_blind_outcomes.append(1)

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
                if info.get("blind_is_boss", False):
                    self.total_boss_blind_encounters += 1
                    self.recent_boss_blind_outcomes.append(0)
                death_chips = info.get("chips", 0)
                if death_chips > self.highest_blind_score:
                    self.highest_blind_score = death_chips

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
            self._rollout_joker_buy_sell_ratios.append(
                self.episode_jokers_gained / max(1, self.episode_jokers_sold))
            self._rollout_consumables_used.append(self.episode_consumables_used)
            self._rollout_consumables_failed.append(self.episode_consumables_failed)
            self._rollout_consumables_sold.append(self.episode_consumables_sold)
            ep_blind_total = self.episode_blind_attempts + self.episode_blind_skips
            self._rollout_blind_skip_rates.append(
                (self.episode_blind_skips / max(1, ep_blind_total)))

            # Per-run stats (game counters from run_stats + Python accumulators)
            run_stats = info.get("run_stats", {}) or {}
            ep_stats = {
                "high_score":         run_stats.get("high_score", 0),
                "max_gold":           self.episode_max_gold,
                "rerolls":            run_stats.get("rerolls", 0),
                "planets_used":       run_stats.get("planets_used", 0),
                "tarots_used":        run_stats.get("tarots_used", 0),
                "jokers_purchased":   self.episode_jokers_purchased,
                "hands_played":       run_stats.get("hands_played", 0),
                "discards_remaining": run_stats.get("unused_discards", 0),
                "final_deck_size":    run_stats.get("deck_size", 0),
            }
            for name in RUN_STAT_NAMES:
                value = ep_stats[name]
                self._rollout_run_stats[name].append(value)
                if value > self.best_run_stats[name]:
                    self.best_run_stats[name] = value
                self.logger.record(f"run/episode_{name}", value)
                self.logger.record(f"run/best_{name}",    self.best_run_stats[name])

            # --- Live TensorBoard (best-ever / all-time / rolling) ---
            hand_types = [
                "High Card", "Pair", "Two Pair", "Three of a Kind",
                "Straight", "Flush", "Full House", "Four of a Kind",
                "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"
            ]

            self.logger.record("game/win_rate_overall",
                               self._win_pct())
            self.logger.record("game/win_rate_last_100",
                               sum(self.recent_outcomes) / max(1, len(self.recent_outcomes)))
            self.logger.record("game/blind_clear_rate_overall",
                               self.total_blind_clears / max(1, self.total_blind_attempts))
            self.logger.record("game/blind_clear_rate_last_100",
                               sum(self.recent_blind_outcomes) / max(1, len(self.recent_blind_outcomes)))
            self.logger.record("game/highest_blind_score",      self.highest_blind_score)
            self.logger.record("game/highest_single_hand_score", self.highest_single_hand_score)
            self.logger.record("game/best_ante",                  self.best_ante)
            self.logger.record("game/best_round",                 self.best_round)
            self.logger.record("money/max_ever_held",             self.max_money_ever)
            self.logger.record("custom/total_wins",               self.total_wins)
            self.logger.record("custom/total_blind_clears",       self.total_blind_clears)
            self.logger.record("custom/total_boss_blind_clears",  self.total_boss_blind_clears)
            self.logger.record("game/boss_blind_clear_rate_overall",
                               self.total_boss_blind_clears / max(1, self.total_boss_blind_encounters))
            self.logger.record("game/boss_blind_clear_rate_last_100",
                               sum(self.recent_boss_blind_outcomes) / max(1, len(self.recent_boss_blind_outcomes)))
            self.logger.record("actions/blind_skip_rate",
                               self.total_blind_skips / max(1, self.total_blind_selections))

            # Consumables (all-time totals + derived rates)
            total_c_attempts = self.total_consumables_used + self.total_consumables_failed
            self.logger.record("consumables/total_used",      self.total_consumables_used)
            self.logger.record("consumables/total_failed",    self.total_consumables_failed)
            self.logger.record("consumables/total_sold",      self.total_consumables_sold)
            self.logger.record("consumables/success_rate",
                               self.total_consumables_used / max(1, total_c_attempts))
            self.logger.record("consumables/use_sell_ratio",
                               self.total_consumables_used / max(1, self.total_consumables_sold))

            # Joker per-episode metrics
            self.logger.record("joker/episode_gained",         self.episode_jokers_gained)
            self.logger.record("joker/episode_sold",           self.episode_jokers_sold)
            self.logger.record("joker/episode_end_count",      len(info.get("jokers", [])))
            self.logger.record("joker/episode_buy_sell_ratio",
                               self.episode_jokers_gained / max(1, self.episode_jokers_sold))

            # All-time hand type stats (one category per hand type)
            total_hands = sum(self.hand_type_counts.values()) or 1
            for ht in hand_types:
                count = self.hand_type_counts[ht]
                if count == 0:
                    continue
                safe_name = ht.replace(" ", "_").lower()
                self.logger.record(f"hand_{safe_name}/count",     count)
                self.logger.record(f"hand_{safe_name}/pct",       count / total_hands)
                self.logger.record(f"hand_{safe_name}/avg_score", self._mean(self.hand_type_scores[ht]))
                self.logger.record(f"hand_{safe_name}/max_score", self.hand_type_max_scores[ht])

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
            self.episode_consumables_used = 0
            self.episode_consumables_failed = 0
            self.episode_consumables_sold = 0
            self.episode_max_gold = 0
            self.episode_jokers_purchased = 0

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
            self.logger.record("game_rollout/win_rate",
                               self._rollout_wins / n, 2)
            self.logger.record("game_rollout/blind_clear_rate",
                               self._rollout_blind_clears / max(1, self._rollout_blind_attempts))
            self.logger.record("game_rollout/mean_ante",
                               self._mean(self._rollout_antes))
            self.logger.record("game_rollout/mean_money_earned",
                               self._mean(self._rollout_money_earned))
            self.logger.record("game_rollout/mean_money_spent",
                               self._mean(self._rollout_money_spent))
            self.logger.record("game_rollout/mean_play_discard_ratio",
                               self._mean(self._rollout_play_discard_ratios))
            self.logger.record("game_rollout/mean_blind_skip_rate",
                               self._mean(self._rollout_blind_skip_rates))
            if self._rollout_chips:
                self.logger.record("game_rollout/mean_blind_score",
                                   self._mean(self._rollout_chips))

            # Joker rollout metrics
            self.logger.record("joker/mean_gained",
                               self._mean(self._rollout_jokers_gained))
            self.logger.record("joker/mean_sold",
                               self._mean(self._rollout_jokers_sold))
            self.logger.record("joker/mean_end_count",
                               self._mean(self._rollout_jokers_end))
            self.logger.record("joker/mean_buy_sell_ratio",
                               self._mean(self._rollout_joker_buy_sell_ratios))

            # Consumable rollout metrics
            self.logger.record("consumables/mean_used",
                               self._mean(self._rollout_consumables_used))
            self.logger.record("consumables/mean_failed",
                               self._mean(self._rollout_consumables_failed))
            self.logger.record("consumables/mean_sold",
                               self._mean(self._rollout_consumables_sold))

            # Per-run stat rollout means
            for name in RUN_STAT_NAMES:
                self.logger.record(f"run/rollout_mean_{name}",
                                   self._mean(self._rollout_run_stats[name]))

            # Rollout hand type distribution (one category per hand type)
            rollout_total = sum(self._rollout_hand_counts.values()) or 1
            for ht in hand_types:
                if self._rollout_hand_counts[ht] == 0:
                    continue
                safe_name = ht.replace(" ", "_").lower()
                self.logger.record(f"hand_{safe_name}/pct_rollout",
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
        self._rollout_joker_buy_sell_ratios = []
        self._rollout_consumables_used = []
        self._rollout_consumables_failed = []
        self._rollout_consumables_sold = []
        self._rollout_blind_skip_rates = []
        self._rollout_hand_counts = defaultdict(int)
        self._rollout_run_stats = defaultdict(list)

        self.logger.dump(self.num_timesteps)

        print(f"\n🔄 Policy update at {self.num_timesteps} steps | "
              f"Episodes: {self.total_episodes} | "
              f"Win rate: {self._win_pct():.1f}% | "
              f"Best ante: {self.best_ante} round: {self.best_round}")
