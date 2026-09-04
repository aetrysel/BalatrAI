"""
Balatro RL Environment
Wraps the pipe-based communication with Balatro mod in a standard RL interface.
This acts as a translator between Balatro's JSON pipe communication and 
RL libraries that expect gym-style step()/reset() methods.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
import logging
import os
import gymnasium as gym
from gymnasium import spaces
from ..utils.communication import BalatroPipeIO, BalatroSocketIO
from .reward import BalatroRewardCalculator
from ..utils.mappers import BalatroStateMapper, BalatroActionMapper
from ..utils.replay import ReplaySystem
from ..utils.history_logger import HistoryLogger
from ..utils.constants import (
    PACK_STATE_IDS, SHOP_STATE_ID, ROUND_EVAL_STATE_ID, BLIND_SELECT_STATE_ID
)


class BalatroEnv(gym.Env):
    """
    Standard RL Environment wrapper for Balatro
    
    Translates between:
    - Balatro mod's JSON pipe communication (/tmp/balatro_request, /tmp/balatro_response)
    - Standard RL interface (step, reset, observation spaces)
    
    This allows RL libraries like Stable-Baselines3 to train on Balatro
    without knowing about the underlying pipe communication system.
    """
    
    def __init__(self, seed=None, deck="Red Deck", is_endless=False, stake: str = "White", run_label: str = ""):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.current_state = None
        self.prev_state = None
        self.game_over = False
        self.restart_pending = False
        self.seed = seed
        self.deck = deck
        self.is_endless = is_endless
        self.stake = stake

        # Initialize communication system — swap transport based on env var
        if os.getenv("BALATRO_COMM_MODE") == "socket":
            self.logger.info("Using socket communication")
            self.pipe_io = BalatroSocketIO(port=int(os.getenv("BALATRO_SOCKET_PORT", 9000)))
        else:
            self.logger.info("Using pipe communication")
            self.pipe_io = BalatroPipeIO()
        
        # Initialize reward systems
        self.reward_calculator = BalatroRewardCalculator()

        # Replay System
        self.replay_system = ReplaySystem()
        self.actions_taken = []

        # History logger
        self.history_logger = HistoryLogger(run_label=run_label)
        self._episode_count = 0

        # Action space constants
        self.MAX_ACTIONS = 19        # actions 0-18 (balatro IDs 1-19)
        self.MAX_CARDS = 8           # binary card selectors
        self.MAX_SHOP_SLOTS = 5      # shared by BUY_JOKER and BUY_BOOSTER
        self.MAX_JOKER_SLOTS = 5     # for SELL_JOKER
        self.MAX_CONSUMABLE_SLOTS = 5  # for USE_CONSUMABLE
        self.MAX_PACK_SLOTS = 5      # for SELECT_PACK_CARD

        ACTION_SLICE_LAYOUT = [
            ("action_selection",  1),
            ("card_indices",      self.MAX_CARDS),
            ("shop_slot",         1),
            ("joker_slot",        1),
            ("consumable_slot",   1),
            ("pack_card_slot",    1),
        ]
        slices = self._build_action_slices(ACTION_SLICE_LAYOUT)
        self.slices = slices

        self.action_space = spaces.MultiDiscrete(np.concatenate([
            np.array([self.MAX_ACTIONS]),
            np.array([2] * self.MAX_CARDS),
            np.array([self.MAX_SHOP_SLOTS]),
            np.array([self.MAX_JOKER_SLOTS]),
            np.array([self.MAX_CONSUMABLE_SLOTS]),
            np.array([self.MAX_PACK_SLOTS]),
        ]))

        # Observation space
        self.OBSERVATION_SIZE = 412
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.OBSERVATION_SIZE,),
            dtype=np.float32
        )

        # Initialize mappers
        # Observation action-mask stays 17 bits (actions 18/19 visible only via MultiDiscrete mask)
        self.state_mapper = BalatroStateMapper(observation_size=self.OBSERVATION_SIZE, max_actions=17)
        self.action_mapper = BalatroActionMapper(action_slices=slices)
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode
        
        In Balatro context, this means starting a new run.
        Communicates with Balatro mod via pipes to initiate reset.
        
        Returns:
            Initial observation/game state
        """
        self.current_state = None
        self.prev_state = None
        self.game_over = False
        self.actions_taken = []
        self._episode_accumulated_reward = 0.0
        
        if self.restart_pending:
            self.restart_pending = False
            restart_response = {"action": 6, "params": [], "seed": self.seed, "deck": self.deck}
            self.pipe_io.send_response(restart_response)

        # Reset reward tracking
        self.reward_calculator.reset()
        self._episode_count += 1

        # Wait for initial request from Balatro (game start)
        initial_request = self.pipe_io.wait_for_request()
        if not initial_request:
            raise RuntimeError("Failed to receive initial request from Balatro")

        while initial_request.get('game_state', {}).get('state') == 4:
            restart_response = {"action": 6, "params": [], "seed": self.seed, "deck": self.deck}

            self.pipe_io.send_response(restart_response)
            initial_request = self.pipe_io.wait_for_request()
            if not initial_request:
                raise RuntimeError("Failed to receive initial request from Balatro after restart")
        # Process initial state for SB3
        self.current_state = initial_request
        gs_for_ep = self.current_state.get('game_state', {})
        actual_seed = self.seed if self.seed else gs_for_ep.get('seed', '')
        self.history_logger.new_episode(
            episode_num=self._episode_count,
            deck=gs_for_ep.get('deck_name', self.deck),
            seed=str(actual_seed) if actual_seed else "",
            stake=self.stake,
        )
        initial_observation = self.state_mapper.process_game_state(self.current_state)
        
        # Create initial action mask
        initial_available_actions = initial_request.get('available_actions', [])
        initial_action_mask = self._create_action_mask(initial_available_actions, self.current_state)
        self._action_masks = initial_action_mask
        
        return initial_observation, {}
    
    def step(self, action):
        """
        Take an action in the Balatro environment
        Sends action to Balatro mod via JSON pipe, waits for response,
        calculates reward, and returns standard RL step format.
        
        Args:
            action: Action dictionary (e.g., {"action": 1, "params": {...}})
            
        Returns:
            Tuple of (observation, reward, done, info) where:
            - observation: Processed game state for neural network
            - reward: Calculated reward for this step
            - done: Whether episode is finished (game over)
            - info: Additional debug information
        """
        # Store previous state for reward calculation
        self.prev_state = self.current_state

        current_gs = self.current_state.get('game_state', {})
        state_id = current_gs.get('state', 0)
        is_game_win = (state_id == ROUND_EVAL_STATE_ID and current_gs.get('game_win', 0) == 1)

        if state_id == ROUND_EVAL_STATE_ID:  # ROUND_EVAL: force CASH_OUT (ai index 10 → balatro 11)
            action[self.slices["action_selection"]] = [10]

        # Send action response to Balatro mod
        response_data = self.action_mapper.process_action(rl_action=action, state_id=state_id)
        self.actions_taken.append(response_data)
        success = self.pipe_io.send_response(response_data)
        if not success:
            raise RuntimeError("Failed to send response to Balatro")

        # Win detected from ROUND_EVAL state (game_win=1 only set there)
        if is_game_win:
            win_obs = self.state_mapper.process_game_state(self.current_state)
            win_reward = self.reward_calculator.calculate_reward(
                current_state=self.current_state,
                prev_state=self.prev_state or {},
            )
            self._episode_accumulated_reward += win_reward
            self.history_logger.log_step(self.prev_state, response_data, self.current_state)
            self.history_logger.log_win(self.current_state, self._episode_count)
            self.history_logger.log_episode_end(self._episode_accumulated_reward, self.reward_calculator.episode_reward_breakdown)
            self.replay_system.try_save_replay(
                file_path=self.replay_system.REPLAY_FILE_PATH,
                seed=current_gs.get('seed', ''),
                actions=self.actions_taken,
                score=win_reward,
                chips=current_gs.get('chips', 0)
            )
            win_info = {
                "won": True,
                "ante": current_gs.get('ante', 0),
                "round": current_gs.get('round_count', 0),
                "chips": current_gs.get('chips', 0),
                "blind_chips": current_gs.get('blind_chips', 0),
                "blind_cleared": True,
                "blind_is_boss": current_gs.get('blind_is_boss', 0) == 1,
                "hand_type": current_gs.get('current_hand', {}).get('last_handname', 'None'),
                "hand_score": current_gs.get('current_hand', {}).get('last_score', 0),
                "jokers": current_gs.get('jokers', []),
                "consumable_count": len(current_gs.get('consumables', [])),
                "gold": current_gs.get('gold', 0),
                "action_taken": response_data.get('action', 0),
            }
            if not self.is_endless:
                # Non-endless: Lua WIN auto-skip handles 30s screen then restart
                # reset() will block in wait_for_request() until new run starts
                return win_obs, win_reward, True, False, win_info
            # Endless: count as win but continue — consume next state (SHOP after cash_out)
            next_request = self.pipe_io.wait_for_request()
            if not next_request:
                self.history_logger.log_episode_end(self._episode_accumulated_reward, self.reward_calculator.episode_reward_breakdown)
                return win_obs, win_reward, True, False, {**win_info, "timeout": True}
            self.current_state = next_request
            available_actions = next_request.get('available_actions', [])
            try:
                self._action_masks = self._create_action_mask(available_actions, self.current_state)
            except Exception as e:
                self.logger.error(f"Action mask error: {e}", exc_info=True)
                self._action_masks = np.ones(sum(self.action_space.nvec), dtype=bool)
            self.history_logger.log_step(self.prev_state, response_data, next_request)
            return self.state_mapper.process_game_state(self.current_state), win_reward, False, False, win_info

        # Wait for next request with new game state
        next_request = self.pipe_io.wait_for_request()
        if not next_request:
            self.game_over = True
            observation = self.state_mapper.process_game_state(self.current_state)
            reward = 0.0
            self.history_logger.log_episode_end(self._episode_accumulated_reward, self.reward_calculator.episode_reward_breakdown)
            return observation, reward, True, False, {"timeout": True}
        _ngs = next_request.get('game_state', {})
        self.logger.info(f"Received request, state: {_ngs.get('state', '?')}, round: {_ngs.get('round_count', 0)}, actions: {next_request.get('available_actions', [])}")
        
        #Check for Game Over
        game_state_early = next_request.get('game_state', {})
        if game_state_early.get('game_over', 0) == 1:
            observation = self.state_mapper.process_game_state(self.current_state)
            # Inject game_over=1 into current_state so reward.py fires the penalty and
            # captures game_over_blind_name/chips for CLI output, while keeping the real
            # chip/blind values (the game-over state itself has chips=0).
            game_over_state = {
                **self.current_state,
                'game_state': {**self.current_state.get('game_state', {}), 'game_over': 1}
            }
            reward = self.reward_calculator.calculate_reward(
                current_state=game_over_state,
                prev_state=self.prev_state or {},
            )
            self._episode_accumulated_reward += reward
            self.restart_pending = True
            self.history_logger.log_step(self.prev_state, response_data, next_request)
            self.history_logger.log_episode_end(self._episode_accumulated_reward, self.reward_calculator.episode_reward_breakdown)
            last_gs = self.current_state.get('game_state', {})
            return observation, reward, True, False, {
                "won": False,
                "ante": last_gs.get('ante', 0),
                "round": last_gs.get('round_count', 0),
                "chips": last_gs.get('chips', 0),
                "blind_chips": last_gs.get('blind_chips', 0),
                "blind_cleared": False,
                "blind_is_boss": last_gs.get('blind_is_boss', 0) == 1,
                "hand_type": last_gs.get('current_hand', {}).get('last_handname', 'None'),
                "hand_score": last_gs.get('current_hand', {}).get('last_score', 0),
                "jokers": last_gs.get('jokers', []),
                "consumable_count": len(last_gs.get('consumables', [])),
                "gold": last_gs.get('gold', 0),
                "action_taken": response_data.get('action', 0),
            }
        while next_request.get('game_state', {}).get('state') == 4:
            self.logger.info("Auto-handling START_RUN state, sending restart")
            restart_response = {"action": 6, "params": [], "seed": self.seed, "deck": self.deck}

            self.pipe_io.send_response(restart_response)
            next_request = self.pipe_io.wait_for_request()
            if not next_request:
                self.history_logger.log_episode_end(self._episode_accumulated_reward, self.reward_calculator.episode_reward_breakdown)
                return (
                    self.state_mapper.process_game_state(self.current_state),
                    0.0, True, False, {"timeout": True}
                )
            self.logger.info(
                f"Received request, state: {next_request.get('game_state', {}).get('state', '?')}, "
                f"actions: {next_request.get('available_actions', [])}"
            )
        # Update current state
        self.current_state = next_request
        game_state = self.current_state.get('game_state', {})

        # Process new state for SB3
        observation = self.state_mapper.process_game_state(self.current_state)
        # Calculate reward using expert reward calculator
        reward = self.reward_calculator.calculate_reward(
            current_state=self.current_state,
            prev_state=self.prev_state if self.prev_state else {}
        )
        self._episode_accumulated_reward += reward
        terminated = False
        truncated = False

        # Build info dict for every step (enables callback metrics)
        info = {
            "won": False,
            "ante": game_state.get('ante', 0),
            "round": game_state.get('round_count', 0),
            "chips": game_state.get('chips', 0),
            "blind_chips": game_state.get('blind_chips', 0),
            "blind_cleared": game_state.get('blind_cleared', 0) == 1,
            "blind_is_boss": game_state.get('blind_is_boss', 0) == 1,
            "hand_type": game_state.get('current_hand', {}).get('last_handname', 'None'),
            "hand_score": game_state.get('current_hand', {}).get('last_score', 0),
            "jokers": game_state.get('jokers', []),
            "consumable_count": len(game_state.get('consumables', [])),
            "gold": game_state.get('gold', 0),
            "action_taken": response_data.get('action', 0),
        }

        # Create action mask for MaskablePPO
        try:
            available_actions = next_request.get('available_actions', [])
            action_mask = self._create_action_mask(available_actions, self.current_state)
            self._action_masks = action_mask
        except Exception as e:
            self.logger.error(f"Action mask error: {e}", exc_info=True)
            self._action_masks = np.ones(sum(self.action_space.nvec), dtype=bool)

        self.history_logger.log_step(self.prev_state, response_data, next_request)
        return observation, reward, terminated, truncated, info

    def cleanup(self):
        """
        Clean up environment resources

        Call this when shutting down to clean up pipe communication.
        """
        self.pipe_io.cleanup()
        self.history_logger.close()

    # Action Masks for MaskablePPO and for ActionWrapper
    def action_masks(self):
        """Required method for MaskablePPO"""
        if hasattr(self, '_action_masks') and self._action_masks is not None:
            mask = np.array(self._action_masks, dtype=bool)
            expected = sum(self.action_space.nvec)
            assert len(mask) == expected, (
                f"Mask length {len(mask)} != expected {expected}. "
            )
            return mask
    
        # Safe fallback: allow all actions
        return np.ones(sum(self.action_space.nvec), dtype=bool)
    
    def _create_action_mask(self, available_actions, current_game_state):
        """Create action mask for MultiDiscrete space."""
        action_masks = []
        state_id = current_game_state.get('game_state', {}).get('state', 0)
        inner_state = current_game_state.get('game_state', {})
        money = inner_state.get('gold', 0)

        balatro_to_ai_mapping = {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5,
            7: 6, 8: 7, 9: 8, 10: 9, 11: 10,
            12: 11, 13: 12, 14: 13, 15: 14, 16: 15,
            17: 16, 18: 17, 19: 18,
        }

        action_selection_mask = [False] * self.MAX_ACTIONS
        if state_id == ROUND_EVAL_STATE_ID:  # ROUND_EVAL — force CASH_OUT (ai index 10)
            action_selection_mask[10] = True
        else:
            for action_id in available_actions:
                ai_idx = balatro_to_ai_mapping.get(action_id)
                if ai_idx is not None:
                    action_selection_mask[ai_idx] = True
        action_masks.append(action_selection_mask)

        if state_id == ROUND_EVAL_STATE_ID:
            # ROUND_EVAL: block all param slots
            for _ in range(self.MAX_CARDS):
                action_masks.append([True, False])
            action_masks.append([False] * self.MAX_SHOP_SLOTS)
            action_masks.append([False] * self.MAX_JOKER_SLOTS)
            action_masks.append([False] * self.MAX_CONSUMABLE_SLOTS)
            action_masks.append([False] * self.MAX_PACK_SLOTS)

        elif state_id in PACK_STATE_IDS:
            # Tarot/spectral packs let the agent highlight hand cards as consumable targets
            can_select_hand = state_id in {9, 15}  # TAROT_PACK=9, SPECTRAL_PACK=15 (vanilla)
            if state_id == 999:  # SMODS_BOOSTER_OPENED: detect type from pack card set
                pack_cards = inner_state.get('pack_cards', [])
                can_select_hand = bool(pack_cards) and pack_cards[0].get('item_set') in {'Tarot', 'Spectral'}
            for _ in range(self.MAX_CARDS):
                action_masks.append([True, True] if can_select_hand else [True, False])
            action_masks.append([False] * self.MAX_SHOP_SLOTS)
            action_masks.append([False] * self.MAX_JOKER_SLOTS)
            action_masks.append([False] * self.MAX_CONSUMABLE_SLOTS)
            # Pack card slots: enable slots that exist, block jokers if slots are full
            pack_cards = inner_state.get('pack_cards', [])
            jokers = inner_state.get('jokers', [])
            joker_slots_max = inner_state.get('joker_slots_max', 5)
            jokers_full = len(jokers) >= joker_slots_max
            pack_mask = [False] * self.MAX_PACK_SLOTS
            for i in range(min(len(pack_cards), self.MAX_PACK_SLOTS)):
                card = pack_cards[i]
                is_joker = card.get('item_set') == 'Joker'
                is_negative = card.get('edition') == 'negative'
                if is_joker and not is_negative and jokers_full:
                    pack_mask[i] = False
                else:
                    pack_mask[i] = True
            action_masks.append(pack_mask)

        elif state_id == SHOP_STATE_ID:
            # SHOP: disable card selection, enable shop/joker/booster slots
            for _ in range(self.MAX_CARDS):
                action_masks.append([True, False])

            shop_items = inner_state.get('shop', {}).get('items', [])
            joker_items = [it for it in shop_items if it.get('type') == 'joker']
            booster_items = [it for it in shop_items if it.get('type') == 'booster']
            consumable_shop_items = [it for it in shop_items if it.get('type') == 'consumable']

            shop_mask = [False] * self.MAX_SHOP_SLOTS
            # BUY_JOKER uses shop_slot for joker area slots
            for i, item in enumerate(joker_items):
                if i < self.MAX_SHOP_SLOTS and item.get('cost', 999) <= money:
                    shop_mask[i] = True
            # BUY_BOOSTER reuses shop_slot for booster area slots
            for i, item in enumerate(booster_items):
                if i < self.MAX_SHOP_SLOTS and item.get('cost', 999) <= money:
                    shop_mask[i] = True
            # BUY_AND_USE_CONSUMABLE reuses shop_slot for consumable items in shop_jokers
            for i, item in enumerate(consumable_shop_items):
                if i < self.MAX_SHOP_SLOTS and item.get('cost', 999) <= money:
                    shop_mask[i] = True
            action_masks.append(shop_mask)

            owned_jokers = inner_state.get('jokers', [])
            joker_mask = [False] * self.MAX_JOKER_SLOTS
            for i in range(min(len(owned_jokers), self.MAX_JOKER_SLOTS)):
                joker_mask[i] = True
            action_masks.append(joker_mask)

            # SELL_CONSUMABLE uses consumable_slot; enable slots with owned consumables
            owned_consumables = inner_state.get('consumables', [])
            consumable_mask = [False] * self.MAX_CONSUMABLE_SLOTS
            if 19 in available_actions:
                for i in range(min(len(owned_consumables), self.MAX_CONSUMABLE_SLOTS)):
                    consumable_mask[i] = True
            action_masks.append(consumable_mask)
            action_masks.append([False] * self.MAX_PACK_SLOTS)

        elif state_id == BLIND_SELECT_STATE_ID:  # BLIND_SELECT
            for _ in range(self.MAX_CARDS):
                action_masks.append([True, False])
            action_masks.append([False] * self.MAX_SHOP_SLOTS)
            action_masks.append([False] * self.MAX_JOKER_SLOTS)
            action_masks.append([False] * self.MAX_CONSUMABLE_SLOTS)
            action_masks.append([False] * self.MAX_PACK_SLOTS)

        else:
            # SELECTING_HAND and other states
            can_select = not any(a in [2, 3] for a in available_actions)
            for _ in range(self.MAX_CARDS):
                action_masks.append([True, True] if can_select else [True, False])
            action_masks.append([False] * self.MAX_SHOP_SLOTS)
            action_masks.append([False] * self.MAX_JOKER_SLOTS)

            consumables = inner_state.get('consumables', [])
            consumable_mask = [False] * self.MAX_CONSUMABLE_SLOTS
            if 14 in available_actions:  # USE_CONSUMABLE available
                for i in range(min(len(consumables), self.MAX_CONSUMABLE_SLOTS)):
                    consumable_mask[i] = True
            action_masks.append(consumable_mask)
            action_masks.append([False] * self.MAX_PACK_SLOTS)

        flat = [item for sublist in action_masks for item in sublist]
        return flat


    @staticmethod
    def _build_action_slices(layout: List[Tuple[str, int]]) -> Dict[str, slice]:
        """
        Create slices for our actions so that we can precisely extract the
        right params to send over to balatro
        
        Args:
            layout: Our ACTION_SLICE_LAYOUT that contains action name and size
        Return:
            A dictionary containing a key being our action space slice name, and  
            the slice 
        """
        slices = {}
        start = 0
        for action_name, size in layout:
            slices[action_name] = slice(start, start + size)
            start += size
        return slices

    def _build_info(self, game_state: dict, won: bool, blind_cleared: bool) -> dict:
        """Build a consistent info dict for all step returns."""
        return {
            "won": won,
            "blind_cleared": blind_cleared,
            "ante": game_state.get('ante', 0),
            "round": game_state.get('round_count', 0),
            "chips": game_state.get('chips', 0),
            "blind_chips": game_state.get('blind_chips', 300),
            "joker_count": len(game_state.get('jokers', [])),
            "hand_type": game_state.get('current_hand', {}).get('handname', 'None'),
            "hand_score": game_state.get('current_hand', {}).get('score', 0),
            "hands_left": game_state.get('round', {}).get('hands_left', 0),
            "discards_left": game_state.get('round', {}).get('discards_left', 0),
        }