"""
Kuhn Poker Environment for Reinforcement Learning

Kuhn Poker is a simplified poker variant designed for research and learning:
- 2 players, 3-card deck: {J, Q, K} (Jack < Queen < King)
- Each player antes 1 chip at the start
- Single betting round with actions: 0=check/fold, 1=bet/call
- No ties possible (J < Q < K)

Game Flow:
1. Both players ante 1 chip (pot starts at 2)
2. Each player receives 1 private card
3. Player 1 acts first: check or bet
4. Player 2 responds: check, bet, call, or fold
5. Showdown or fold determines winner

State Representation:
- Observation: [private_card_id(0..2), history_bitmask]
  * card_id: 0=J, 1=Q, 2=K
  * history: bit0=P1 bet, bit1=P2 bet (0=no bets, 1=P1 bet, 2=P2 bet, 3=both bet)

Reward Structure (Real Chip Accounting):
- check/check showdown: winner +1, loser -1 (pot=2, each paid 1)
- bet/fold: bettor +1, folder -1 (pot=3, bettor paid 2, folder 1)
- bet/call showdown: winner +2, loser -2 (pot=4, each paid 2)

This environment is ideal for RL research due to its simplicity and clear optimal strategies.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Tuple, Dict

# Card mapping: J=0, Q=1, K=2 (numerical values for easy comparison)
CARD_IDS = {"J":0, "Q":1, "K":2}
ID_TO_CARD = {v:k for k,v in CARD_IDS.items()}

class KuhnPokerEnv(gym.Env):
    """
    Kuhn Poker environment implementing the Gymnasium interface.
    
    This environment simulates Kuhn Poker games and provides:
    - Standard gym interface (reset, step, observation_space, action_space)
    - Real chip accounting with proper pot management
    - Deterministic game logic with optional seeding
    - State encoding suitable for neural network input
    """
    metadata = {"render_modes": []}

    def __init__(self, seed: int | None = None):
        """
        Initialize the Kuhn Poker environment.
        
        Args:
            seed: Random seed for reproducible games (None for random)
        """
        super().__init__()
        self.rng = np.random.default_rng(seed)  # Random number generator for shuffling
        
        # Observation space: 2 dimensions
        # - card_id: 0-2 (J, Q, K)
        # - history: 0-3 (bitmask encoding betting history)
        self.observation_space = spaces.MultiDiscrete([3, 4])
        
        # Action space: 2 actions
        # - 0: check/fold (context-sensitive meaning)
        # - 1: bet/call
        self.action_space = spaces.Discrete(2)
        
        self.reset()  # Initialize game state

    def _deal(self):
        """
        Deal cards to players by shuffling the 3-card deck.
        
        Creates a random permutation of [0,1,2] representing [J,Q,K].
        First two cards go to players, third is unused.
        """
        deck = [0,1,2]  # J=0, Q=1, K=2
        self.rng.shuffle(deck)  # Randomly shuffle the deck
        self.p1_card, self.p2_card = deck[:2]  # Deal first two cards to players

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        """
        Reset the environment to start a new game.
        
        Args:
            seed: Optional new random seed
            options: Unused, kept for gym interface compatibility
            
        Returns:
            observation: Initial state [card_id, history]
            info: Empty dict (gym interface requirement)
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)  # Set new random seed if provided
        
        # Reset game state for new hand
        self.history = 0        # History encoding: bit0 = P1 bet, bit1 = P2 bet
        self.pot = 2            # Both players ante 1 chip each
        self.contribution = [1, 1]   # How much each player has contributed: [P1, P2]
        
        self._deal()            # Deal new cards
        self.current_player = 0 # Player 1 acts first
        
        obs = self._obs(self.current_player)  # Get observation for current player
        return obs, {}

    def _obs(self, player: int):
        """
        Generate observation for a specific player.
        
        Args:
            player: Player ID (0 for P1, 1 for P2)
            
        Returns:
            numpy array: [card_id, history] where card_id is the player's private card
        """
        card = self.p1_card if player == 0 else self.p2_card  # Get player's card
        return np.array([card, self.history], dtype=np.int64)

    def _winner_id(self) -> int:
        """
        Determine the winner at showdown based on card values.
        
        Returns:
            0 if P1 wins (P1's card > P2's card), else 1
            Note: No ties possible since J < Q < K
        """
        return 0 if self.p1_card > self.p2_card else 1

    def _terminal_reward_from_current_perspective(self, winner_id: int) -> float:
        """
        Calculate the reward from the CURRENT (acting) player's perspective.
        
        This is crucial for RL training - the reward must be from the perspective
        of the player who just took an action, not from a fixed player's view.
        
        Args:
            winner_id: 0 if P1 won, 1 if P2 won
            
        Returns:
            float: Reward from current player's perspective
            - If current player wins: pot - their contribution (positive)
            - If current player loses: -their contribution (negative)
        """
        cp = self.current_player  # Current player (who just acted)
        if winner_id == cp:
            return float(self.pot - self.contribution[cp])  # Won: get pot minus what they put in
        else:
            return float(-self.contribution[cp])  # Lost: lose what they put in

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Take an action in the environment.
        
        This is the core game logic that handles all possible game states and transitions.
        The action meaning depends on context:
        - 0: check (if no bet) or fold (if facing a bet)
        - 1: bet (if no bet) or call (if facing a bet)
        
        Args:
            action: 0 for check/fold, 1 for bet/call
            
        Returns:
            observation: New state after action
            reward: Reward from current player's perspective
            terminated: True if game ended
            truncated: Always False (episodes don't truncate)
            info: Empty dict
        """
        # action: 0=check/fold, 1=bet/call
        reward = 0.0
        terminated = False

        if self.current_player == 0:
            # ----- PLAYER 1'S TURN -----
            if self.history == 0:  # P1 first to act (no bets yet)
                if action == 1:  # P1 bets
                    self.history |= 1        # Set bit 0 (P1 bet)
                    self.pot += 1            # Add 1 chip to pot
                    self.contribution[0] += 1  # P1 contributed 1 more chip
                    self.current_player = 1  # Switch to P2
                else:  # P1 checks -> P2 to act
                    self.current_player = 1  # Switch to P2

            elif self.history == 2:
                # P2 has bet, now P1 must respond
                # This branch normally happens when current_player==0 after P2 bet
                # In this code path we don't expect to be here because we switch current_player accordingly above.
                pass

        else:
            # ----- PLAYER 2'S TURN -----
            if self.history == 0:  # After P1 checked (no bets yet)
                if action == 1:  # P2 bets
                    self.history |= 2        # Set bit 1 (P2 bet)
                    self.pot += 1            # Add 1 chip to pot
                    self.contribution[1] += 1  # P2 contributed 1 more chip
                    self.current_player = 0  # P1 must respond to the bet
                else:  # P2 checks -> showdown (pot=2, contrib=[1,1])
                    winner = self._winner_id()
                    reward = self._terminal_reward_from_current_perspective(winner)
                    terminated = True

            elif self.history == 1:  # P1 bet, P2 to respond
                if action == 1:  # P2 calls -> showdown (pot=4, contrib=[2,2])
                    self.pot += 1            # Add 1 chip to pot
                    self.contribution[1] += 1  # P2 contributed 1 more chip
                    winner = self._winner_id()
                    reward = self._terminal_reward_from_current_perspective(winner)
                    terminated = True
                else:  # P2 folds -> P1 wins immediately (pot=3, contrib P2=1)
                    # Winner is P1 (id=0). Acting player is P2 (id=1), who folded.
                    winner = 0
                    reward = self._terminal_reward_from_current_perspective(winner)
                    terminated = True

            elif self.history == 2:
                # P2 already bet, now P1 should act (handled when current_player==0)
                pass

        if not terminated:
            # Game continues - return observation for next player
            obs = self._obs(self.current_player)
            return obs, 0.0, False, False, {}

        # Terminal state reached - return reward from CURRENT player's perspective
        # NOTE: For folds, this is negative for the folding (acting) player.
        #       For showdowns, it is +/-(pot - contribution[cp]) or -contribution[cp].
        return self._obs(self.current_player), float(reward), True, False, {}
