from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict, deque
import copy

# Per-run stats logged as run/episode_*, run/best_*, run/rollout_mean_*
RUN_STAT_NAMES = [
    "high_score", "max_gold", "rerolls",
    "jokers_purchased", "hands_played", "discards_remaining", "final_deck_size",
    "max_joker_tenure", "fresh_joker_sells", "mean_shop_exit_gold",
]
# Consumable type -> run_stats key, logged as consumable_<type>/*
CONSUMABLE_TYPE_STATS = {"tarot": "tarots_used", "planet": "planets_used", "spectral": "spectrals_used"}
# Run stats where lower is better, also logged as run/lowest_*
RUN_STAT_LOWER_IS_BETTER = ["final_deck_size"]

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
        self.total_a1r1_clears = 0  # episodes that cleared ante 1, round 1

        # Best-ever metrics
        self.best_ante = 0
        self.best_round = 0
        self.highest_blind_score = 0
        self.highest_single_hand_score = 0
        self.max_money_ever = 0
        self.best_run_stats = defaultdict(int)
        self.lowest_run_stats = {}

        # All-time per-consumable use counts, keyed by (type, name)
        self.consumable_use_counts = defaultdict(int)

        # All-time per-joker stats, keyed by sanitized joker name (logged as joker_by_name/*)
        self.joker_taken_counts = defaultdict(int)
        self.joker_sold_counts = defaultdict(int)
        self.joker_rounds_held = defaultdict(list)   # rounds_held when each copy left (sold/destroyed/run end)
        self.joker_max_rounds_held = defaultdict(int)
        self.all_joker_rounds_held = []              # every finished joker hold, all jokers

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
        self.episode_a1r1_cleared = False
        self.episode_joker_holds = []   # rounds_held of each joker that left this episode (incl. run end)

        # Previous-step deltas
        self._prev_chips = 0
        self._prev_gold = 0
        self._prev_consumable_count = 0
        self._prev_joker_count = 0
        self._prev_jokers_by_id = {}  # joker id → joker dict from the previous step

        # --- Rollout buffers (reset after each policy update) ---
        self._rollout_episodes = 0
        self._rollout_wins = 0
        self._rollout_blind_clears = 0
        self._rollout_blind_attempts = 0
        self._rollout_boss_blind_clears = 0
        self._rollout_boss_blind_encounters = 0
        self._rollout_a1r1_clears = 0
        self._rollout_chips = []
        self._rollout_antes = []
        self._rollout_rounds = []
        self._rollout_money_earned = []
        self._rollout_money_spent = []
        self._rollout_play_discard_ratios = []
        self._rollout_jokers_gained = []
        self._rollout_jokers_sold = []
        self._rollout_jokers_end = []
        self._rollout_joker_buy_sell_ratios = []
        self._rollout_joker_mean_holds = []
        self._rollout_consumables_used = []
        self._rollout_consumables_failed = []
        self._rollout_consumables_sold = []
        self._rollout_blind_skip_rates = []
        self._rollout_hand_counts = defaultdict(int)
        self._rollout_run_stats = defaultdict(list)
        self._rollout_consumable_type_used = defaultdict(list)

    @staticmethod
    def _joker_key(name):
        """TensorBoard-safe joker name, e.g. 'Mr. Bones' → 'mr_bones'."""
        key = "".join(c if c.isalnum() else "_" for c in str(name).lower())
        return "_".join(filter(None, key.split("_"))) or "unknown"

    def _record_joker_hold(self, joker, sold):
        key = self._joker_key(joker.get("name", "Unknown"))
        rounds = joker.get("rounds_held", 0)
        self.joker_rounds_held[key].append(rounds)
        self.joker_max_rounds_held[key] = max(self.joker_max_rounds_held[key], rounds)
        self.all_joker_rounds_held.append(rounds)
        self.episode_joker_holds.append(rounds)
        if sold:
            self.joker_sold_counts[key] += 1

    def _win_pct(self):
        if self.total_episodes == 0:
            return 0.0
        return round(100.0 * self.total_wins / self.total_episodes, 2)

    def _mean(self, lst):
        return sum(lst) / len(lst) if lst else 0.0

    def _per_env_fields(self):
        """Per-episode accumulators and previous-step deltas that belong to a single env's stream."""
        return [k for k in vars(self) if k == "current_run_length" or k.startswith(("episode_", "_prev_"))]

    def select_env(self, env_idx):
        """Swap in env_idx's per-episode state (async parallel collection calls this before each transition)."""
        if not hasattr(self, "_env_states"):
            self._env_states = {}
            self._env_state_defaults = {k: copy.deepcopy(getattr(self, k)) for k in self._per_env_fields()}
            self._active_env = env_idx
        if env_idx == self._active_env:
            return
        self._env_states[self._active_env] = {k: getattr(self, k) for k in self._env_state_defaults}
        state = self._env_states.get(env_idx) or copy.deepcopy(self._env_state_defaults)
        for k, v in state.items():
            setattr(self, k, v)
        self._active_env = env_idx

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
            gold_increased = gold > self._prev_gold
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

            # Per-joker lifecycle by id: new id = taken; missing id = sold (gold up) or destroyed
            if "jokers" in info:
                curr_jokers_by_id = {j["id"]: j for j in info["jokers"] if j.get("id") is not None}
                for jid, joker in curr_jokers_by_id.items():
                    if jid not in self._prev_jokers_by_id:
                        self.joker_taken_counts[self._joker_key(joker.get("name", "Unknown"))] += 1
                for jid, joker in self._prev_jokers_by_id.items():
                    if jid not in curr_jokers_by_id:
                        self._record_joker_hold(joker, sold=gold_increased)
                self._prev_jokers_by_id = curr_jokers_by_id

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
                    self._rollout_boss_blind_clears += 1
                    self._rollout_boss_blind_encounters += 1
                    self.recent_boss_blind_outcomes.append(1)
                if info.get("ante", 0) == 1 and info.get("round", 0) == 1:
                    self.episode_a1r1_cleared = True

            if not done:
                continue

            # --- Episode ended ---
            self.total_episodes += 1
            won = bool(info.get("won", False))
            ante = info.get("ante", 0)
            round_ = info.get("round", 0)

            if won:
                self.total_wins += 1
            if self.episode_a1r1_cleared:
                self.total_a1r1_clears += 1
                self._rollout_a1r1_clears += 1

            self.recent_outcomes.append(1 if won else 0)
            if len(self.recent_outcomes) > self.recent_window:
                self.recent_outcomes.pop(0)

            # Failed blind outcome (timeout is not a blind failure)
            if not won and not info.get("timeout", False):
                self.recent_blind_outcomes.append(0)
                if info.get("blind_is_boss", False):
                    self.total_boss_blind_encounters += 1
                    self._rollout_boss_blind_encounters += 1
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
            self._rollout_rounds.append(round_)
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
            reward_stats = info.get("reward_stats", {}) or {}
            ep_stats = {
                "high_score":         run_stats.get("high_score", 0),
                "max_gold":           self.episode_max_gold,
                "rerolls":            run_stats.get("rerolls", 0),
                "jokers_purchased":   self.episode_jokers_purchased,
                "hands_played":       run_stats.get("hands_played", 0),
                "discards_remaining": run_stats.get("unused_discards", 0),
                "final_deck_size":    run_stats.get("deck_size", 0),
                "max_joker_tenure":   reward_stats.get("max_joker_tenure", 0),
                "fresh_joker_sells":  reward_stats.get("fresh_joker_sells", 0),
                "mean_shop_exit_gold": reward_stats.get("mean_shop_exit_gold", 0.0),
            }
            for name in RUN_STAT_NAMES:
                value = ep_stats[name]
                self._rollout_run_stats[name].append(value)
                if value > self.best_run_stats[name]:
                    self.best_run_stats[name] = value
                self.logger.record(f"run/episode_{name}", value)
                self.logger.record(f"run/best_{name}",    self.best_run_stats[name])
                if name in RUN_STAT_LOWER_IS_BETTER and value > 0:
                    if name not in self.lowest_run_stats or value < self.lowest_run_stats[name]:
                        self.lowest_run_stats[name] = value
                if name in self.lowest_run_stats:
                    self.logger.record(f"run/lowest_{name}", self.lowest_run_stats[name])

            # Consumables used per type (logged as rollout means)
            for ctype, stat_key in CONSUMABLE_TYPE_STATS.items():
                self._rollout_consumable_type_used[ctype].append(run_stats.get(stat_key, 0))

            # Per-consumable all-time use counts (e.g. consumable_spectral/medium_used)
            for key, usage in (run_stats.get("consumable_usage") or {}).items():
                ctype = str(usage.get("set", "unknown")).lower()
                cname = key[2:] if key.startswith("c_") else key
                self.consumable_use_counts[(ctype, cname)] += usage.get("count", 0)
            for (ctype, cname), count in self.consumable_use_counts.items():
                self.logger.record(f"consumable_{ctype}/{cname}_used", count)

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
            self.logger.record("game/a1r1_clear_rate_overall",
                               self.total_a1r1_clears / max(1, self.total_episodes))
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

            # Joker hold metrics: jokers still held at run end count as held until the end
            end_holds = [j.get("rounds_held", 0) for j in self._prev_jokers_by_id.values()]
            for joker in self._prev_jokers_by_id.values():
                self._record_joker_hold(joker, sold=False)
            self._prev_jokers_by_id = {}
            ep_mean_hold = self._mean(self.episode_joker_holds)
            self._rollout_joker_mean_holds.append(ep_mean_hold)
            self.logger.record("joker/episode_mean_rounds_held",     ep_mean_hold)
            self.logger.record("joker/episode_max_rounds_held",      max(self.episode_joker_holds, default=0))
            self.logger.record("joker/episode_end_mean_rounds_held", self._mean(end_holds))
            self.logger.record("joker/alltime_mean_rounds_held",     self._mean(self.all_joker_rounds_held))

            # All-time per-joker stats (e.g. joker_by_name/blueprint_taken)
            for key, taken in self.joker_taken_counts.items():
                self.logger.record(f"joker_by_name/{key}_taken", taken)
                self.logger.record(f"joker_by_name/{key}_sold",  self.joker_sold_counts[key])
                if self.joker_rounds_held[key]:
                    self.logger.record(f"joker_by_name/{key}_mean_rounds_held",
                                       self._mean(self.joker_rounds_held[key]))
                    self.logger.record(f"joker_by_name/{key}_max_rounds_held",
                                       self.joker_max_rounds_held[key])

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
            self.episode_a1r1_cleared = False
            self.episode_joker_holds = []

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
            self.logger.record("game_rollout/mean_round",
                               self._mean(self._rollout_rounds))
            self.logger.record("game_rollout/a1r1_clear_rate",
                               self._rollout_a1r1_clears / n)
            if self._rollout_boss_blind_encounters > 0:
                self.logger.record("game_rollout/boss_blind_clear_rate",
                                   self._rollout_boss_blind_clears / self._rollout_boss_blind_encounters)
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
            self.logger.record("joker/rollout_mean_rounds_held",
                               self._mean(self._rollout_joker_mean_holds))

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

            # Per-type consumable rollout means
            for ctype in CONSUMABLE_TYPE_STATS:
                self.logger.record(f"consumable_{ctype}/rollout_mean_used",
                                   self._mean(self._rollout_consumable_type_used[ctype]))

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
        self._rollout_boss_blind_clears = 0
        self._rollout_boss_blind_encounters = 0
        self._rollout_a1r1_clears = 0
        self._rollout_chips = []
        self._rollout_antes = []
        self._rollout_rounds = []
        self._rollout_money_earned = []
        self._rollout_money_spent = []
        self._rollout_play_discard_ratios = []
        self._rollout_jokers_gained = []
        self._rollout_jokers_sold = []
        self._rollout_jokers_end = []
        self._rollout_joker_buy_sell_ratios = []
        self._rollout_joker_mean_holds = []
        self._rollout_consumables_used = []
        self._rollout_consumables_failed = []
        self._rollout_consumables_sold = []
        self._rollout_blind_skip_rates = []
        self._rollout_hand_counts = defaultdict(int)
        self._rollout_run_stats = defaultdict(list)
        self._rollout_consumable_type_used = defaultdict(list)

        self.logger.dump(self.num_timesteps)

        print(f"\n🔄 Policy update at {self.num_timesteps} steps | "
              f"Episodes: {self.total_episodes} | "
              f"Win rate: {self._win_pct():.1f}% | "
              f"Best ante: {self.best_ante} round: {self.best_round}")
