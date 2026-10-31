from stable_baselines3.common.callbacks import BaseCallback
from collections import Counter, defaultdict, deque

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

HAND_TYPES = [
    "High Card", "Pair", "Two Pair", "Three of a Kind",
    "Straight", "Flush", "Full House", "Four of a Kind",
    "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"
]
# Tag sources reported by RLBridge/stats.lua, logged as tags/via_<source>
TAG_SOURCES = ["skip", "double", "anaglyph", "diet_cola", "other"]
# Antes tracked in ante_reached/ (ante 1 is always reached)
ANTES_TRACKED = range(2, 9)
# Minimum attempts before a blind counts toward blinds/deadliest_death_rate
DEADLIEST_MIN_ATTEMPTS = 5
SHOP_STATE_ID = 5
# Per-pack counters from RLBridge/stats.lua (offered = rolled into a shop booster slot)
PACK_FIELDS = ("offered", "opened", "bought", "picks", "skips")
# Deck-quality fields from reward_stats (deck_<field> at run end, max_deck_<field> per-run peak)
DECK_STAT_FIELDS = ("enhanced", "sealed", "editioned", "upgrades", "upgrade_pct")

# Metric layout (all recorded once per rollout, at policy update):
#   <metric>          all-time value over the whole training process
#   <metric>_rollout  value over episodes that finished during the last rollout
# Rates and ratios are ratio-of-sums (Σnum / max(1, Σden)) over that window;
# mean_*_rollout are per-episode averages over the rollout.


class BalatroMetricsCallback(BaseCallback):
    def __init__(self, log_freq=10, verbose=1):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Summable counters: all-time and since the last policy update
        self.totals = Counter()
        self.rollout = Counter()
        # Per-episode values that don't sum (maxes, end-of-run snapshots), averaged over the rollout
        self.rollout_values = defaultdict(list)

        # Best-ever metrics
        self.best_ante = 0
        self.best_round = 0
        self.highest_blind_score = 0
        self.highest_single_hand_score = 0
        self.max_money_ever = 0
        self.max_joker_rounds_held = 0
        self.lowest_final_deck_size = None

        # All-time per-consumable use counts, keyed by (type, name)
        self.consumable_use_counts = defaultdict(int)

        # All-time per-joker stats, keyed by sanitized joker name (logged as joker_by_name/*)
        self.joker_taken_counts = defaultdict(int)
        self.joker_sold_counts = defaultdict(int)
        self.joker_rounds_held = defaultdict(list)   # rounds_held when each copy left (sold/destroyed/run end)
        self.joker_max_rounds_held = defaultdict(int)
        self.joker_contributions = defaultdict(list)  # contribution (0..1) when each copy left

        # Deck quality (reward_stats): all-time best per field
        self.best_deck_stats = Counter()

        # All-time hand type tracking
        self.hand_type_counts = defaultdict(int)
        self.hand_type_scores = defaultdict(list)
        self.hand_type_max_scores = defaultdict(int)
        self._rollout_hand_counts = defaultdict(int)

        # All-time per-blind outcomes, keyed by blind name (logged as blind_by_name/*)
        self.blind_clears_by_name = defaultdict(int)
        self.blind_deaths_by_name = defaultdict(int)

        # All-time per-object counters (logged as *_by_name/*, *_by_type/*)
        self.ante_reached = Counter()
        self.voucher_bought_by_name = Counter()
        self.voucher_offered_by_name = Counter()
        self.pack_stats_by_type = defaultdict(Counter)
        self.tag_counts_by_key = Counter()

        # Sliding window for win rate (last 100 episodes)
        # self.recent_window = 100
        # self.recent_outcomes = []

        # Rolling last-100 blind outcomes (1=cleared, 0=failed)
        # self.recent_blind_outcomes = deque(maxlen=100)

        # Rolling last-100 boss blind outcomes (1=cleared, 0=failed)
        # self.recent_boss_blind_outcomes = deque(maxlen=100)

        self._reset_episode()

    def _reset_episode(self):
        """Per-episode accumulators and previous-step values (reset at episode end)."""
        self.episode = Counter()
        self.episode_max_gold = 0
        self.episode_a1r1_cleared = False
        self.episode_joker_holds = []   # rounds_held of each joker that left this episode (incl. run end)
        self.episode_vouchers_seen = set()  # (ante, voucher name) offered this episode
        self.blind_plays = 0  # PLAY_HAND count since the current blind was selected

        # Previous-step deltas
        self._prev_chips = 0
        self._prev_gold = 0
        self._prev_consumable_count = 0
        self._prev_jokers_by_id = {}  # joker id → joker dict from the previous step
        self._prev_blind_cleared = False
        self._prev_shop_items = []

    def _add(self, key, amount=1):
        """Add to both the all-time and the rollout counter."""
        self.totals[key] += amount
        self.rollout[key] += amount

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
        self.max_joker_rounds_held = max(self.max_joker_rounds_held, rounds)
        self.joker_contributions[key].append(float(joker.get("contribution", 0.0)))
        self._add("joker_holds")
        self._add("joker_hold_rounds", rounds)
        self.episode_joker_holds.append(rounds)
        if sold:
            self.joker_sold_counts[key] += 1
            self._add("jokers_sold")
            self._add("sold_joker_contribution", float(joker.get("contribution", 0.0)))

    def _win_pct(self):
        if self.totals["episodes"] == 0:
            return 0.0
        return round(100.0 * self.totals["wins"] / self.totals["episodes"], 2)

    def _mean(self, lst):
        return sum(lst) / len(lst) if lst else 0.0

    def _record_ratio(self, name, num, den):
        """Log name (all-time) and name_rollout (last rollout) as Σnum / max(1, Σden)."""
        self.logger.record(name, self.totals[num] / max(1, self.totals[den]))
        if self.rollout["episodes"] > 0:
            self.logger.record(f"{name}_rollout", self.rollout[num] / max(1, self.rollout[den]))

    def _record_rollout_mean(self, name, key):
        """Log mean per episode of a summed counter over the last rollout."""
        if self.rollout["episodes"] > 0:
            self.logger.record(name, self.rollout[key] / self.rollout["episodes"])

    def _record_rollout_value_mean(self, name, key):
        """Log mean of a per-episode value over the last rollout."""
        if self.rollout_values[key]:
            self.logger.record(name, self._mean(self.rollout_values[key]))

    def _deadliest_blind(self):
        """(name, death_rate) of the blind with the highest deaths/attempts, or (None, 0)."""
        best = (None, 0.0)
        for name in set(self.blind_clears_by_name) | set(self.blind_deaths_by_name):
            attempts = self.blind_clears_by_name[name] + self.blind_deaths_by_name[name]
            if attempts < DEADLIEST_MIN_ATTEMPTS:
                continue
            rate = self.blind_deaths_by_name[name] / attempts
            if rate > best[1]:
                best = (name, rate)
        return best

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            self.episode["steps"] += 1

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
            gold = info.get("gold", 0)
            if action_taken == 14:  # USE_CONSUMABLE: success = count decreased
                if curr_consumables < self._prev_consumable_count:
                    self._add("consumables_used")
                else:
                    self._add("consumables_failed")
            elif action_taken == 18:  # BUY_AND_USE_CONSUMABLE: used if money was spent
                if gold < self._prev_gold:
                    self._add("consumables_used")
                else:
                    self._add("consumables_failed")
            elif action_taken == 19:  # SELL_CONSUMABLE
                self._add("consumables_sold")
            self._prev_consumable_count = curr_consumables

            # Voucher purchase (BUY_VOUCHER always buys G.shop_vouchers slot 1)
            if action_taken == 12 and gold < self._prev_gold:
                voucher = next((it for it in self._prev_shop_items if it.get("type") == "voucher"), None)
                self._add("vouchers_bought")
                self.voucher_bought_by_name[self._joker_key(voucher.get("name") if voucher else "unknown")] += 1

            # Money tracking
            gold_increased = gold > self._prev_gold
            if gold > self._prev_gold:
                self._add("money_earned", gold - self._prev_gold)
            elif gold < self._prev_gold:
                self._add("money_spent", self._prev_gold - gold)
            if gold > self.max_money_ever:
                self.max_money_ever = gold
            if gold > self.episode_max_gold:
                self.episode_max_gold = gold
            self._prev_gold = gold

            # Per-joker lifecycle by id: new id = taken; missing id = sold (gold up) or destroyed
            if "jokers" in info:
                curr_jokers_by_id = {j["id"]: j for j in info["jokers"] if j.get("id") is not None}
                for jid, joker in curr_jokers_by_id.items():
                    if jid not in self._prev_jokers_by_id:
                        self.joker_taken_counts[self._joker_key(joker.get("name", "Unknown"))] += 1
                        self._add("jokers_gained")
                        if action_taken == 7:  # BUY_JOKER
                            self._add("jokers_purchased")
                for jid, joker in self._prev_jokers_by_id.items():
                    if jid not in curr_jokers_by_id:
                        self._record_joker_hold(joker, sold=gold_increased)
                self._prev_jokers_by_id = curr_jokers_by_id

            # Vouchers on offer: each distinct (ante, voucher) counts once per episode
            if info.get("state") == SHOP_STATE_ID:
                for item in info.get("shop_items", []):
                    if item.get("type") != "voucher":
                        continue
                    seen_key = (info.get("ante", 0), item.get("name", "Unknown"))
                    if seen_key not in self.episode_vouchers_seen:
                        self.episode_vouchers_seen.add(seen_key)
                        self._add("vouchers_offered")
                        self.voucher_offered_by_name[self._joker_key(item.get("name", "Unknown"))] += 1
            self._prev_shop_items = info.get("shop_items", self._prev_shop_items)

            # Action tracking
            if action_taken == 2:
                self._add("plays")
                self.blind_plays += 1
            elif action_taken == 3:
                self._add("discards")
            elif action_taken == 10:
                self._add("shop_visits")
            elif action_taken == 5:    # SELECT_BLIND (attempt)
                self._add("blind_attempts")
                self._add("blind_selections")
                self.blind_plays = 0
            elif action_taken == 17:   # SKIP_BLIND
                self._add("blind_skips")
                self._add("blind_selections")

            # Blind clear (rising edge): chip score, hands needed, per-blind outcome
            blind_cleared = bool(info.get("blind_cleared", False))
            if blind_cleared and not self._prev_blind_cleared:
                self._add("blind_clears")
                self._add("hands_to_clear", self.blind_plays)
                clear_chips = info.get("chips", 0)  # cumulative blind score; chips only increases during a blind
                if clear_chips > self.highest_blind_score:
                    self.highest_blind_score = clear_chips
                self.rollout_values["blind_score"].append(clear_chips)
                self.blind_clears_by_name[info.get("blind_name", "None")] += 1
                if info.get("blind_is_boss", False):
                    self._add("boss_clears")
                    self._add("boss_encounters")
                if info.get("ante", 0) == 1 and info.get("round", 0) == 1:
                    self.episode_a1r1_cleared = True
                    if self.blind_plays == 1:
                        self._add("a1r1_one_shots")
            self._prev_blind_cleared = blind_cleared

            if done:
                self._end_episode(info)

        return True

    def _end_episode(self, info):
        """Fold a finished episode into the all-time and rollout counters."""
        self._add("episodes")
        won = bool(info.get("won", False))
        ante = info.get("ante", 0)
        round_ = info.get("round", 0)

        if won:
            self._add("wins")
        if self.episode_a1r1_cleared:
            self._add("a1r1_clears")

        # Failed blind outcome (timeout is not a blind failure)
        if not won and not info.get("timeout", False):
            self.blind_deaths_by_name[info.get("blind_name", "None")] += 1
            if info.get("blind_is_boss", False):
                self._add("boss_encounters")
            death_chips = info.get("chips", 0)
            if death_chips > self.highest_blind_score:
                self.highest_blind_score = death_chips

        if ante > self.best_ante or (ante == self.best_ante and round_ > self.best_round):
            self.best_ante = ante
            self.best_round = round_
        for a in ANTES_TRACKED:
            if ante >= a:
                self.ante_reached[a] += 1
        self.rollout_values["ante"].append(ante)
        self.rollout_values["round"].append(round_)
        self.rollout_values["max_gold"].append(self.episode_max_gold)

        # Jokers held at run end count as held until the end
        end_jokers = list(self._prev_jokers_by_id.values())
        self._add("jokers_end_count", len(info.get("jokers", [])))
        self.rollout_values["end_rounds_held"].append(self._mean([j.get("rounds_held", 0) for j in end_jokers]))
        for joker in end_jokers:
            self._record_joker_hold(joker, sold=False)

        # Per-run game counters (Lua run_stats) and reward stats
        run_stats = info.get("run_stats", {}) or {}
        reward_stats = info.get("reward_stats", {}) or {}
        self._add("rerolls", run_stats.get("rerolls", 0))
        self._add("unused_discards", run_stats.get("unused_discards", 0))
        self._add("fresh_joker_sells", reward_stats.get("fresh_joker_sells", 0))
        self._add("joker_upgrades", run_stats.get("joker_upgrades", 0))
        self._add("boss_rerolls", run_stats.get("boss_rerolls", 0))
        self._add("boosters_bought_skipped", reward_stats.get("boosters_bought_skipped", 0))
        self._add("pb_hand_bonuses", reward_stats.get("pb_hand_bonuses", 0))
        self.rollout_values["run_best_hand"].append(run_stats.get("high_score", 0))

        # Deck quality at run end (rollout mean) and per-run peak (all-time best)
        for field in DECK_STAT_FIELDS:
            self.rollout_values[f"deck_{field}"].append(reward_stats.get(f"deck_{field}", 0))
            self.best_deck_stats[field] = max(self.best_deck_stats[field], reward_stats.get(f"max_deck_{field}", 0))
        self.rollout_values["shop_exit_gold"].append(reward_stats.get("mean_shop_exit_gold", 0.0))
        deck_size = run_stats.get("deck_size", 0)
        self.rollout_values["final_deck_size"].append(deck_size)
        if deck_size > 0 and (self.lowest_final_deck_size is None or deck_size < self.lowest_final_deck_size):
            self.lowest_final_deck_size = deck_size

        # Consumables used per type (logged as consumable_<type>/rollout_mean_used)
        for ctype, stat_key in CONSUMABLE_TYPE_STATS.items():
            self._add(f"{ctype}_used", run_stats.get(stat_key, 0))

        # Per-consumable all-time use counts (e.g. consumable_spectral/medium_used)
        for key, usage in (run_stats.get("consumable_usage") or {}).items():
            ctype = str(usage.get("set", "unknown")).lower()
            cname = key[2:] if key.startswith("c_") else key
            self.consumable_use_counts[(ctype, cname)] += usage.get("count", 0)

        # Booster packs (per-run counters from RLBridge/stats.lua)
        packs = run_stats.get("packs") or {}
        for name, counts in (packs.items() if isinstance(packs, dict) else []):
            pack_key = self._joker_key(name)
            for field in PACK_FIELDS:
                value = counts.get(field, 0)
                self.pack_stats_by_type[pack_key][field] += value
                self._add(f"packs_{field}", value)

        # Tags acquired (per-run counters from RLBridge/stats.lua)
        tags = run_stats.get("tags") or {}
        by_key = tags.get("by_key") or {}
        for key, count in (by_key.items() if isinstance(by_key, dict) else []):
            self.tag_counts_by_key[key[4:] if key.startswith("tag_") else key] += count
            self._add("tags_acquired", count)
        by_source = tags.get("by_source") or {}
        for source, count in (by_source.items() if isinstance(by_source, dict) else []):
            self._add(f"tags_via_{source}", count)

        # --- Previous per-episode TensorBoard metrics (replaced by rollout-end logging in _on_rollout_end) ---
        # run_stats = info.get("run_stats", {}) or {}
        # reward_stats = info.get("reward_stats", {}) or {}
        # ep_stats = {
        #     "high_score":         run_stats.get("high_score", 0),
        #     "max_gold":           self.episode_max_gold,
        #     "rerolls":            run_stats.get("rerolls", 0),
        #     "jokers_purchased":   self.episode_jokers_purchased,
        #     "hands_played":       run_stats.get("hands_played", 0),
        #     "discards_remaining": run_stats.get("unused_discards", 0),
        #     "final_deck_size":    run_stats.get("deck_size", 0),
        #     "max_joker_tenure":   reward_stats.get("max_joker_tenure", 0),
        #     "fresh_joker_sells":  reward_stats.get("fresh_joker_sells", 0),
        #     "mean_shop_exit_gold": reward_stats.get("mean_shop_exit_gold", 0.0),
        # }
        # for name in RUN_STAT_NAMES:
        #     value = ep_stats[name]
        #     self._rollout_run_stats[name].append(value)
        #     if value > self.best_run_stats[name]:
        #         self.best_run_stats[name] = value
        #     self.logger.record(f"run/episode_{name}", value)
        #     self.logger.record(f"run/best_{name}",    self.best_run_stats[name])
        #     if name in RUN_STAT_LOWER_IS_BETTER and value > 0:
        #         if name not in self.lowest_run_stats or value < self.lowest_run_stats[name]:
        #             self.lowest_run_stats[name] = value
        #     if name in self.lowest_run_stats:
        #         self.logger.record(f"run/lowest_{name}", self.lowest_run_stats[name])

        # # Consumables used per type (logged as rollout means)
        # for ctype, stat_key in CONSUMABLE_TYPE_STATS.items():
        #     self._rollout_consumable_type_used[ctype].append(run_stats.get(stat_key, 0))

        # # Per-consumable all-time use counts (e.g. consumable_spectral/medium_used)
        # for key, usage in (run_stats.get("consumable_usage") or {}).items():
        #     ctype = str(usage.get("set", "unknown")).lower()
        #     cname = key[2:] if key.startswith("c_") else key
        #     self.consumable_use_counts[(ctype, cname)] += usage.get("count", 0)
        # for (ctype, cname), count in self.consumable_use_counts.items():
        #     self.logger.record(f"consumable_{ctype}/{cname}_used", count)

        # # --- Live TensorBoard (best-ever / all-time / rolling) ---
        # hand_types = [
        #     "High Card", "Pair", "Two Pair", "Three of a Kind",
        #     "Straight", "Flush", "Full House", "Four of a Kind",
        #     "Straight Flush", "Five of a Kind", "Flush House", "Flush Five"
        # ]

        # self.logger.record("game/win_rate_overall",
        #                    self._win_pct())
        # self.logger.record("game/win_rate_last_100",
        #                    sum(self.recent_outcomes) / max(1, len(self.recent_outcomes)))
        # self.logger.record("game/blind_clear_rate_overall",
        #                    self.total_blind_clears / max(1, self.total_blind_attempts))
        # self.logger.record("game/blind_clear_rate_last_100",
        #                    sum(self.recent_blind_outcomes) / max(1, len(self.recent_blind_outcomes)))
        # self.logger.record("game/highest_blind_score",      self.highest_blind_score)
        # self.logger.record("game/highest_single_hand_score", self.highest_single_hand_score)
        # self.logger.record("game/best_ante",                  self.best_ante)
        # self.logger.record("game/best_round",                 self.best_round)
        # self.logger.record("money/max_ever_held",             self.max_money_ever)
        # self.logger.record("custom/total_wins",               self.total_wins)
        # self.logger.record("custom/total_blind_clears",       self.total_blind_clears)
        # self.logger.record("custom/total_boss_blind_clears",  self.total_boss_blind_clears)
        # self.logger.record("game/boss_blind_clear_rate_overall",
        #                    self.total_boss_blind_clears / max(1, self.total_boss_blind_encounters))
        # self.logger.record("game/boss_blind_clear_rate_last_100",
        #                    sum(self.recent_boss_blind_outcomes) / max(1, len(self.recent_boss_blind_outcomes)))
        # self.logger.record("game/a1r1_clear_rate_overall",
        #                    self.total_a1r1_clears / max(1, self.total_episodes))
        # self.logger.record("actions/blind_skip_rate",
        #                    self.total_blind_skips / max(1, self.total_blind_selections))

        # # Consumables (all-time totals + derived rates)
        # total_c_attempts = self.total_consumables_used + self.total_consumables_failed
        # self.logger.record("consumables/total_used",      self.total_consumables_used)
        # self.logger.record("consumables/total_failed",    self.total_consumables_failed)
        # self.logger.record("consumables/total_sold",      self.total_consumables_sold)
        # self.logger.record("consumables/success_rate",
        #                    self.total_consumables_used / max(1, total_c_attempts))
        # self.logger.record("consumables/use_sell_ratio",
        #                    self.total_consumables_used / max(1, self.total_consumables_sold))

        # # Joker per-episode metrics
        # self.logger.record("joker/episode_gained",         self.episode_jokers_gained)
        # self.logger.record("joker/episode_sold",           self.episode_jokers_sold)
        # self.logger.record("joker/episode_end_count",      len(info.get("jokers", [])))
        # self.logger.record("joker/episode_buy_sell_ratio",
        #                    self.episode_jokers_gained / max(1, self.episode_jokers_sold))

        # # Joker hold metrics: jokers still held at run end count as held until the end
        # end_holds = [j.get("rounds_held", 0) for j in self._prev_jokers_by_id.values()]
        # for joker in self._prev_jokers_by_id.values():
        #     self._record_joker_hold(joker, sold=False)
        # self._prev_jokers_by_id = {}
        # ep_mean_hold = self._mean(self.episode_joker_holds)
        # self._rollout_joker_mean_holds.append(ep_mean_hold)
        # self.logger.record("joker/episode_mean_rounds_held",     ep_mean_hold)
        # self.logger.record("joker/episode_max_rounds_held",      max(self.episode_joker_holds, default=0))
        # self.logger.record("joker/episode_end_mean_rounds_held", self._mean(end_holds))
        # self.logger.record("joker/alltime_mean_rounds_held",     self._mean(self.all_joker_rounds_held))

        # # All-time per-joker stats (e.g. joker_by_name/blueprint_taken)
        # for key, taken in self.joker_taken_counts.items():
        #     self.logger.record(f"joker_by_name/{key}_taken", taken)
        #     self.logger.record(f"joker_by_name/{key}_sold",  self.joker_sold_counts[key])
        #     if self.joker_rounds_held[key]:
        #         self.logger.record(f"joker_by_name/{key}_mean_rounds_held",
        #                            self._mean(self.joker_rounds_held[key]))
        #         self.logger.record(f"joker_by_name/{key}_max_rounds_held",
        #                            self.joker_max_rounds_held[key])

        # # All-time hand type stats (one category per hand type)
        # total_hands = sum(self.hand_type_counts.values()) or 1
        # for ht in hand_types:
        #     count = self.hand_type_counts[ht]
        #     if count == 0:
        #         continue
        #     safe_name = ht.replace(" ", "_").lower()
        #     self.logger.record(f"hand_{safe_name}/count",     count)
        #     self.logger.record(f"hand_{safe_name}/pct",       count / total_hands)
        #     self.logger.record(f"hand_{safe_name}/avg_score", self._mean(self.hand_type_scores[ht]))
        #     self.logger.record(f"hand_{safe_name}/max_score", self.hand_type_max_scores[ht])

        self._reset_episode()

    def _on_rollout_end(self) -> None:
        n = self.rollout["episodes"]

        # --- Previous rollout metrics (game_rollout/, run/, joker/, consumables/; replaced by the categories below) ---
        # if self._rollout_episodes > 0:
        #     n = self._rollout_episodes
        #     self.logger.record("game_rollout/win_rate",
        #                        self._rollout_wins / n, 2)
        #     self.logger.record("game_rollout/blind_clear_rate",
        #                        self._rollout_blind_clears / max(1, self._rollout_blind_attempts))
        #     self.logger.record("game_rollout/mean_ante",
        #                        self._mean(self._rollout_antes))
        #     self.logger.record("game_rollout/mean_round",
        #                        self._mean(self._rollout_rounds))
        #     self.logger.record("game_rollout/a1r1_clear_rate",
        #                        self._rollout_a1r1_clears / n)
        #     if self._rollout_boss_blind_encounters > 0:
        #         self.logger.record("game_rollout/boss_blind_clear_rate",
        #                            self._rollout_boss_blind_clears / self._rollout_boss_blind_encounters)
        #     self.logger.record("game_rollout/mean_money_earned",
        #                        self._mean(self._rollout_money_earned))
        #     self.logger.record("game_rollout/mean_money_spent",
        #                        self._mean(self._rollout_money_spent))
        #     self.logger.record("game_rollout/mean_play_discard_ratio",
        #                        self._mean(self._rollout_play_discard_ratios))
        #     self.logger.record("game_rollout/mean_blind_skip_rate",
        #                        self._mean(self._rollout_blind_skip_rates))
        #     if self._rollout_chips:
        #         self.logger.record("game_rollout/mean_blind_score",
        #                            self._mean(self._rollout_chips))

        #     # Joker rollout metrics
        #     self.logger.record("joker/mean_gained",
        #                        self._mean(self._rollout_jokers_gained))
        #     self.logger.record("joker/mean_sold",
        #                        self._mean(self._rollout_jokers_sold))
        #     self.logger.record("joker/mean_end_count",
        #                        self._mean(self._rollout_jokers_end))
        #     self.logger.record("joker/mean_buy_sell_ratio",
        #                        self._mean(self._rollout_joker_buy_sell_ratios))
        #     self.logger.record("joker/rollout_mean_rounds_held",
        #                        self._mean(self._rollout_joker_mean_holds))

        #     # Consumable rollout metrics
        #     self.logger.record("consumables/mean_used",
        #                        self._mean(self._rollout_consumables_used))
        #     self.logger.record("consumables/mean_failed",
        #                        self._mean(self._rollout_consumables_failed))
        #     self.logger.record("consumables/mean_sold",
        #                        self._mean(self._rollout_consumables_sold))

        #     # Per-run stat rollout means
        #     for name in RUN_STAT_NAMES:
        #         self.logger.record(f"run/rollout_mean_{name}",
        #                            self._mean(self._rollout_run_stats[name]))

        #     # Per-type consumable rollout means
        #     for ctype in CONSUMABLE_TYPE_STATS:
        #         self.logger.record(f"consumable_{ctype}/rollout_mean_used",
        #                            self._mean(self._rollout_consumable_type_used[ctype]))

        #     # Rollout hand type distribution (one category per hand type)
        #     rollout_total = sum(self._rollout_hand_counts.values()) or 1
        #     for ht in hand_types:
        #         if self._rollout_hand_counts[ht] == 0:
        #             continue
        #         safe_name = ht.replace(" ", "_").lower()
        #         self.logger.record(f"hand_{safe_name}/pct_rollout",
        #                            self._rollout_hand_counts[ht] / rollout_total)

        # --- progress/ ---
        self.logger.record("progress/total_episodes",   self.totals["episodes"])
        self.logger.record("progress/ante_8_boss_wins", self.totals["wins"])
        self._record_ratio("progress/win_rate", "wins", "episodes")
        self.logger.record("progress/best_ante",        self.best_ante)
        self.logger.record("progress/best_round",       self.best_round)
        self._record_rollout_value_mean("progress/mean_ante_rollout",  "ante")
        self._record_rollout_value_mean("progress/mean_round_rollout", "round")

        # --- ante_reached/ (episodes that reached each ante) ---
        for a in ANTES_TRACKED:
            self.logger.record(f"ante_reached/ante_{a}", self.ante_reached[a])

        # --- blinds/ ---
        deadliest_name, deadliest_rate = self._deadliest_blind()
        self.logger.record("blinds/total_clears",      self.totals["blind_clears"])
        self.logger.record("blinds/total_boss_clears", self.totals["boss_clears"])
        self.logger.record("blinds/total_skips",       self.totals["blind_skips"])
        self._record_ratio("blinds/clear_rate",           "blind_clears", "blind_attempts")
        self._record_ratio("blinds/boss_clear_rate",      "boss_clears", "boss_encounters")
        self._record_ratio("blinds/skip_rate",            "blind_skips", "blind_selections")
        self._record_ratio("blinds/mean_hands_per_clear", "hands_to_clear", "blind_clears")
        self.logger.record("blinds/deadliest_death_rate", deadliest_rate)

        # --- first_blind/ (ante 1, round 1) ---
        self.logger.record("first_blind/a1r1_clears",    self.totals["a1r1_clears"])
        self.logger.record("first_blind/a1r1_one_shots", self.totals["a1r1_one_shots"])
        self._record_ratio("first_blind/a1r1_clear_rate",    "a1r1_clears", "episodes")
        self._record_ratio("first_blind/a1r1_one_shot_rate", "a1r1_one_shots", "episodes")

        # --- blind_by_name/ ---
        for name in set(self.blind_clears_by_name) | set(self.blind_deaths_by_name):
            key = self._joker_key(name)
            deaths = self.blind_deaths_by_name[name]
            attempts = self.blind_clears_by_name[name] + deaths
            self.logger.record(f"blind_by_name/{key}_attempts",   attempts)
            self.logger.record(f"blind_by_name/{key}_deaths",     deaths)
            self.logger.record(f"blind_by_name/{key}_death_rate", deaths / max(1, attempts))

        # --- scoring/ ---
        self.logger.record("scoring/highest_blind_score",       self.highest_blind_score)
        self.logger.record("scoring/highest_single_hand_score", self.highest_single_hand_score)
        self._record_rollout_value_mean("scoring/mean_blind_score_rollout",   "blind_score")
        self._record_rollout_value_mean("scoring/mean_run_best_hand_rollout", "run_best_hand")
        self.logger.record("scoring/total_pb_hand_bonuses", self.totals["pb_hand_bonuses"])
        self._record_rollout_mean("scoring/mean_pb_hand_bonuses_rollout", "pb_hand_bonuses")

        # --- deck/ (upgraded playing cards: run-end rollout mean + all-time per-run peak) ---
        for field in DECK_STAT_FIELDS:
            self._record_rollout_value_mean(f"deck/mean_{field}_rollout", f"deck_{field}")
            self.logger.record(f"deck/best_{field}", self.best_deck_stats[field])

        # --- hand_<type>/ (all-time + rollout distribution) ---
        total_hands = sum(self.hand_type_counts.values()) or 1
        rollout_total = sum(self._rollout_hand_counts.values()) or 1
        for ht in HAND_TYPES:
            count = self.hand_type_counts[ht]
            if count == 0:
                continue
            safe_name = ht.replace(" ", "_").lower()
            self.logger.record(f"hand_{safe_name}/count",     count)
            self.logger.record(f"hand_{safe_name}/pct",       count / total_hands)
            self.logger.record(f"hand_{safe_name}/avg_score", self._mean(self.hand_type_scores[ht]))
            self.logger.record(f"hand_{safe_name}/max_score", self.hand_type_max_scores[ht])
            if self._rollout_hand_counts[ht]:
                self.logger.record(f"hand_{safe_name}/pct_rollout",
                                   self._rollout_hand_counts[ht] / rollout_total)

        # --- hand_play/ ---
        self.logger.record("hand_play/total_plays",    self.totals["plays"])
        self.logger.record("hand_play/total_discards", self.totals["discards"])
        self._record_ratio("hand_play/play_discard_ratio", "plays", "discards")
        self._record_rollout_mean("hand_play/mean_plays_rollout",           "plays")
        self._record_rollout_mean("hand_play/mean_unused_discards_rollout", "unused_discards")
        self._record_rollout_value_mean("hand_play/mean_final_deck_size_rollout", "final_deck_size")
        if self.lowest_final_deck_size is not None:
            self.logger.record("hand_play/lowest_final_deck_size", self.lowest_final_deck_size)

        # --- money/ ---
        self.logger.record("money/max_ever_held", self.max_money_ever)
        self.logger.record("money/total_earned",  self.totals["money_earned"])
        self.logger.record("money/total_spent",   self.totals["money_spent"])
        self._record_rollout_mean("money/mean_earned_rollout", "money_earned")
        self._record_rollout_mean("money/mean_spent_rollout",  "money_spent")
        self._record_rollout_value_mean("money/mean_max_gold_rollout",       "max_gold")
        self._record_rollout_value_mean("money/mean_shop_exit_gold_rollout", "shop_exit_gold")

        # --- shop/ ---
        self.logger.record("shop/total_visits",     self.totals["shop_visits"])
        self.logger.record("shop/total_rerolls",    self.totals["rerolls"])
        self.logger.record("shop/vouchers_bought",  self.totals["vouchers_bought"])
        self.logger.record("shop/vouchers_offered", self.totals["vouchers_offered"])
        self._record_rollout_mean("shop/mean_visits_rollout",  "shop_visits")
        self._record_rollout_mean("shop/mean_rerolls_rollout", "rerolls")
        self._record_ratio("shop/voucher_purchase_rate", "vouchers_bought", "vouchers_offered")
        self.logger.record("shop/total_boss_rerolls", self.totals["boss_rerolls"])

        # --- voucher_by_name/ ---
        for key in set(self.voucher_offered_by_name) | set(self.voucher_bought_by_name):
            self.logger.record(f"voucher_by_name/{key}_bought",  self.voucher_bought_by_name[key])
            self.logger.record(f"voucher_by_name/{key}_offered", self.voucher_offered_by_name[key])

        # --- boosters/ ---
        self.logger.record("boosters/total_offered",        self.totals["packs_offered"])
        self.logger.record("boosters/total_bought",         self.totals["packs_bought"])
        self.logger.record("boosters/total_opened",         self.totals["packs_opened"])
        self.logger.record("boosters/total_skipped",        self.totals["packs_skips"])
        self.logger.record("boosters/total_picks",          self.totals["packs_picks"])
        self.logger.record("boosters/total_bought_skipped", self.totals["boosters_bought_skipped"])
        self._record_ratio("boosters/buy_rate",        "packs_bought", "packs_offered")
        self._record_ratio("boosters/pick_skip_ratio", "packs_picks", "packs_skips")
        self._record_ratio("boosters/skip_rate",       "packs_skips", "packs_opened")

        # --- booster_by_type/ ---
        for key, counts in self.pack_stats_by_type.items():
            for field in PACK_FIELDS:
                self.logger.record(f"booster_by_type/{key}_{field}", counts[field])
            self.logger.record(f"booster_by_type/{key}_buy_rate",
                               counts["bought"] / max(1, counts["offered"]))
            self.logger.record(f"booster_by_type/{key}_pick_skip_ratio",
                               counts["picks"] / max(1, counts["skips"]))

        # --- tags/ ---
        self.logger.record("tags/total_acquired", self.totals["tags_acquired"])
        self._record_rollout_mean("tags/mean_acquired_rollout", "tags_acquired")
        for source in TAG_SOURCES:
            self.logger.record(f"tags/via_{source}", self.totals[f"tags_via_{source}"])

        # --- tag_by_name/ ---
        for key, count in self.tag_counts_by_key.items():
            self.logger.record(f"tag_by_name/{key}_acquired", count)

        # --- jokers/ ---
        self.logger.record("jokers/total_gained",    self.totals["jokers_gained"])
        self.logger.record("jokers/total_purchased", self.totals["jokers_purchased"])
        self.logger.record("jokers/total_sold",      self.totals["jokers_sold"])
        self.logger.record("jokers/total_end_count", self.totals["jokers_end_count"])
        self._record_rollout_mean("jokers/mean_end_count_rollout", "jokers_end_count")
        self._record_ratio("jokers/buy_sell_ratio",  "jokers_purchased", "jokers_sold")
        self._record_ratio("jokers/gain_sell_ratio", "jokers_gained", "jokers_sold")
        self._record_ratio("jokers/mean_contribution_at_sell", "sold_joker_contribution", "jokers_sold")
        self.logger.record("jokers/total_upgrades",  self.totals["joker_upgrades"])

        # --- joker_tenure/ ---
        self._record_ratio("joker_tenure/mean_rounds_held", "joker_hold_rounds", "joker_holds")
        self.logger.record("joker_tenure/max_rounds_held",   self.max_joker_rounds_held)
        self.logger.record("joker_tenure/total_fresh_sells", self.totals["fresh_joker_sells"])
        self._record_rollout_mean("joker_tenure/mean_fresh_sells_rollout", "fresh_joker_sells")
        self._record_rollout_value_mean("joker_tenure/mean_end_rounds_held_rollout", "end_rounds_held")

        # --- joker_by_name/ (all-time) ---
        for key, taken in self.joker_taken_counts.items():
            self.logger.record(f"joker_by_name/{key}_taken", taken)
            self.logger.record(f"joker_by_name/{key}_sold",  self.joker_sold_counts[key])
            if self.joker_rounds_held[key]:
                self.logger.record(f"joker_by_name/{key}_mean_rounds_held",
                                   self._mean(self.joker_rounds_held[key]))
                self.logger.record(f"joker_by_name/{key}_max_rounds_held",
                                   self.joker_max_rounds_held[key])
            if self.joker_contributions[key]:
                self.logger.record(f"joker_by_name/{key}_mean_contribution",
                                   self._mean(self.joker_contributions[key]))

        # --- consumables/ ---
        self.logger.record("consumables/total_used",   self.totals["consumables_used"])
        self.logger.record("consumables/total_failed", self.totals["consumables_failed"])
        self.logger.record("consumables/total_sold",   self.totals["consumables_sold"])
        self.totals["consumable_attempts"] = self.totals["consumables_used"] + self.totals["consumables_failed"]
        self.rollout["consumable_attempts"] = self.rollout["consumables_used"] + self.rollout["consumables_failed"]
        self._record_ratio("consumables/success_rate",   "consumables_used", "consumable_attempts")
        self._record_ratio("consumables/use_sell_ratio", "consumables_used", "consumables_sold")
        self._record_rollout_mean("consumables/mean_used_rollout", "consumables_used")

        # --- consumable_<type>/ (per-type rollout means + per-consumable all-time counts) ---
        for ctype in CONSUMABLE_TYPE_STATS:
            self._record_rollout_mean(f"consumable_{ctype}/rollout_mean_used", f"{ctype}_used")
        for (ctype, cname), count in self.consumable_use_counts.items():
            self.logger.record(f"consumable_{ctype}/{cname}_used", count)

        # Reset rollout buffers
        self.rollout = Counter()
        self.rollout_values = defaultdict(list)
        self._rollout_hand_counts = defaultdict(int)

        self.logger.dump(self.num_timesteps)

        deadliest = f"{deadliest_name} ({deadliest_rate:.0%})" if deadliest_name else "n/a"
        print(f"\n🔄 Policy update at {self.num_timesteps} steps | "
              f"Episodes: {self.totals['episodes']} (+{n}) | "
              f"Win rate: {self._win_pct():.1f}% | "
              f"Best ante: {self.best_ante} round: {self.best_round} | "
              f"Deadliest blind: {deadliest}")
