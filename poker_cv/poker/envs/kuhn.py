import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict

# Kuhn Poker: 2 players, deck {J,Q,K}. Each antes 1. Single betting round.
# Actions: 0=check/fold (context-sensitive), 1=bet/call
# Observation: [private_card_id(0..2), history_bitmask]
# Simple reward: pot settlement at terminal.

CARD_IDS = {"J":0, "Q":1, "K":2}
ID_TO_CARD = {v:k for k,v in CARD_IDS.items()}

class KuhnPokerEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        # Observation: 2 dims (card_id, history mask 0..3)
        self.observation_space = spaces.MultiDiscrete([3, 4])
        self.action_space = spaces.Discrete(2)

        self.reset()

    def _deal(self):
        deck = [0,1,2]
        self.rng.shuffle(deck)
        self.p1_card, self.p2_card = deck[:2]

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # History encoding: bit0 = P1 bet, bit1 = P2 bet
        self.history = 0
        self.pot = 2  # antes
        self._deal()
        self.current_player = 0  # P1 to act
        obs = self._obs(self.current_player)
        return obs, {}

    def _obs(self, player: int):
        card = self.p1_card if player == 0 else self.p2_card
        return np.array([card, self.history], dtype=np.int64)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # action: 0=check/fold, 1=bet/call
        reward = 0.0
        terminated = False

        if self.current_player == 0:
            if self.history == 0:  # P1 first to act
                if action == 1:  # bet
                    self.history |= 1  # mark P1 bet
                    self.pot += 1
                    self.current_player = 1
                else:  # check
                    # P2 acts after check
                    self.current_player = 1
            elif self.history == 2:  # P2 bet, P1 to respond (shouldn't happen here)
                pass

        else:  # current_player == 1
            if self.history == 0:  # After P1 checked
                if action == 1:  # P2 bets
                    self.history |= 2
                    self.pot += 1
                    self.current_player = 0  # P1 must respond
                else:  # P2 checks -> showdown
                    reward = self._showdown_winner_reward()
                    terminated = True
            elif self.history == 1:  # P1 bet, P2 to respond
                if action == 1:  # call
                    self.pot += 1
                    reward = self._showdown_winner_reward()
                    terminated = True
                else:  # fold
                    reward = +1.0  # from P1's perspective, P2 folds; P1 wins 1 additional chip
                    terminated = True

            elif self.history == 2:  # P2 already bet and now P1 to act (handled when current_player==0)
                pass

        # If the game is not over, switch to the other player if needed
        if not terminated:
            obs = self._obs(self.current_player)
            return obs, 0.0, False, False, {}

        # Terminal: return reward from CURRENT player's perspective.
        return self._obs(self.current_player), float(reward), True, False, {}

    def _showdown_winner_reward(self) -> float:
        p1 = self.p1_card
        p2 = self.p2_card
        if p1 > p2:
            return -1.0 if self.current_player == 1 else +1.0
        else:
            return +1.0 if self.current_player == 1 else -1.0
