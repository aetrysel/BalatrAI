from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict, deque

class BalatroMetricsCallback(BaseCallback):
    def __init__(self, log_freq=10, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Episode metrics
        self.total_episodes = 0
        self.total_wins = 0
        self.total_blind_clears = 0
        self.all_chips = []
        self.all_run_lengths = []
        self.current_run_length = 0
        self.won = False

        # Best-ever metrics
        self.best_ante = 0
        self.best_round = 0
        self.highest_chips = 0

        # Hand type tracking (across all steps)
        self.hand_type_counts = defaultdict(int)
        self.hand_type_scores = defaultdict(list)

        # Sliding window for recent win rate (last N episodes)
        self.recent_window = 100
        self.recent_outcomes = []

        # Joker count (updated at episode end)
        self.joker_count = 0

        # Chip-gain and consumable tracking
        self._prev_chips = 0
        self._prev_consumable_count = 0
        self.total_consumables_used = 0
        self.consumables_used_this_episode = 0

        # Money tracking
        self._prev_gold = 0
        self.episode_money_earned = 0.0
        self.episode_money_spent = 0.0
        self.max_money_ever = 0

        # Play/Discard tracking
        self.episode_plays = 0
        self.episode_discards = 0

        # Blind skip tracking (cumulative across all episodes)
        self.total_blind_selections = 0
        self.total_blind_skips = 0

        # Rolling hand-type window (last 100 episodes)
        self.recent_hand_episodes = deque(maxlen=100)
        self._episode_hand_counts = defaultdict(int)

    def _win_pct(self):
        if self.total_episodes == 0:
            return 0.0
        return round(100.0 * self.total_wins / self.total_episodes, 2)

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
                self._episode_hand_counts[hand_type] += 1
            self._prev_chips = curr_chips

            # Consumable usage detection
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

            # Play/Discard + Blind skip tracking
            action_taken = info.get("action_taken", 0)
            if action_taken == 2:
                self.episode_plays += 1
            elif action_taken == 3:
                self.episode_discards += 1
            elif action_taken in {5, 17}:
                self.total_blind_selections += 1
                if action_taken == 17:
                    self.total_blind_skips += 1

            if info.get("blind_cleared", False):
                self.total_blind_clears += 1

            if not done:
                continue

            # --- Episode ended ---
            self.total_episodes += 1
            ep = self.total_episodes

            won = bool(info.get("won", False))
            ante = info.get("ante", 0)
            round_ = info.get("round", 0)
            chips = info.get("chips", 0)
            self.joker_count = len(info.get("jokers", []))

            if won:
                self.total_wins += 1

            self.all_chips.append(chips)
            self.all_run_lengths.append(self.current_run_length)
            self.current_run_length = 0

            self.recent_outcomes.append(1 if won else 0)
            if len(self.recent_outcomes) > self.recent_window:
                self.recent_outcomes.pop(0)

            if chips > self.highest_chips:
                self.highest_chips = chips
            if ante > self.best_ante or (ante == self.best_ante and round_ > self.best_round):
                self.best_ante = ante
                self.best_round = round_

            # --- TensorBoard logging ---
            self.logger.record("game/win_rate_overall",         self._win_pct())
            self.logger.record("game/win_rate_last_100",        round(100 * sum(self.recent_outcomes) / len(self.recent_outcomes), 2))
            self.logger.record("game/blind_clear_rate_overall", self.total_blind_clears / ep)
            self.logger.record("custom/total_wins",             self.total_wins)

            self.logger.record("game/chips_this_episode",       chips)
            self.logger.record("game/avg_chips",                round(sum(self.all_chips) / len(self.all_chips), 2))
            self.logger.record("game/highest_chips_ever",       self.highest_chips)

            self.logger.record("game/best_ante",                self.best_ante)
            self.logger.record("game/best_round",               self.best_round)
            self.logger.record("game/ante_this_episode",        ante)

            self.logger.record("game/avg_run_length",           sum(self.all_run_lengths) / len(self.all_run_lengths))
            self.logger.record("game/run_length_this_episode",  self.all_run_lengths[-1])

            hand_types = [
                "High Card", "Pair", "Two Pair", "Three of a Kind",
                "Straight", "Flush", "Full House", "Four of a Kind",
                "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"
            ]
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

            self.logger.record("custom/joker_count", self.joker_count)
            self.logger.record("game/rounds_this_episode",        round_)
            self.logger.record("custom/consumables_used_episode", self.consumables_used_this_episode)
            self.logger.record("custom/total_consumables_used",   self.total_consumables_used)

            # Money metrics
            self.logger.record("money/episode_earned", self.episode_money_earned)
            self.logger.record("money/episode_spent",  self.episode_money_spent)
            self.logger.record("money/max_ever_held",  self.max_money_ever)

            # Action ratio metrics
            self.logger.record("actions/play_discard_ratio",
                               self.episode_plays / max(1, self.episode_discards))
            self.logger.record("actions/blind_skip_rate",
                               self.total_blind_skips / max(1, self.total_blind_selections))

            # Rolling hand-type frequency (last 100 episodes, only played types)
            self.recent_hand_episodes.append(dict(self._episode_hand_counts))
            recent_totals = defaultdict(int)
            for ep_counts in self.recent_hand_episodes:
                for ht, cnt in ep_counts.items():
                    recent_totals[ht] += cnt
            recent_total = sum(recent_totals.values()) or 1
            for ht in hand_types:
                if recent_totals[ht] == 0:
                    continue
                safe_name = ht.replace(" ", "_").lower()
                self.logger.record(f"hands/pct_{safe_name}_last_100",
                                   recent_totals[ht] / recent_total)

            # Reset per-episode counters
            self._prev_chips = 0
            self._prev_consumable_count = 0
            self.consumables_used_this_episode = 0
            self.episode_money_earned = 0.0
            self.episode_money_spent = 0.0
            self.episode_plays = 0
            self.episode_discards = 0
            self._episode_hand_counts = defaultdict(int)
            self._prev_gold = 0

            needs_dump = True

        if needs_dump:
            self.logger.dump(self.num_timesteps)

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout, just before the policy update."""
        print(f"\n🔄 Policy update at {self.num_timesteps} steps | "
              f"Episodes: {self.total_episodes} | "
              f"Win rate: {self._win_pct():.1f}% | "
              f"Best ante: {self.best_ante} round: {self.best_round}")
