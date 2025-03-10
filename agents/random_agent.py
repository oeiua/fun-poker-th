"""
Random baseline agent for poker environment.
"""
import numpy as np
from typing import List, Tuple, Optional

from agents.base_agent import BaseAgent
from game.state import GameState
from game.action import Action

class RandomAgent(BaseAgent):
    """Agent that makes random decisions for baseline comparison."""
    
    def __init__(self, name: Optional[str] = None, aggression: float = 0.5):
        """
        Initialize a random agent.
        
        Args:
            name: Name of the agent
            aggression: Probability of choosing bet/raise when available (0-1)
        """
        super().__init__(name=name or "RandomAgent")
        self.aggression = max(0.0, min(1.0, aggression))
    
    def act(self, state: GameState, valid_actions: List[int], valid_amounts: List[int]) -> Tuple[int, Optional[int]]:
        """
        Choose a random action from the valid actions.
        
        Args:
            state: Current game state
            valid_actions: List of valid action types
            valid_amounts: List of valid bet amounts
            
        Returns:
            Tuple of (action_type, bet_amount)
        """
        if not valid_actions:
            return Action.FOLD, None
        
        # Weight actions based on aggression
        if len(valid_actions) > 1 and Action.BET_RAISE in valid_actions:
            # If bet/raise is an option, use aggression to determine weights
            weights = []
            for action in valid_actions:
                if action == Action.FOLD:
                    weights.append(1.0 - self.aggression)
                elif action == Action.CHECK_CALL:
                    weights.append(1.0)
                elif action == Action.BET_RAISE:
                    weights.append(self.aggression * 2)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Select action based on weights
            action_type = np.random.choice(valid_actions, p=weights)
        else:
            # If bet/raise is not an option, choose uniformly
            action_type = np.random.choice(valid_actions)
        
        # Select random bet amount if needed
        bet_amount = None
        if action_type == Action.BET_RAISE and valid_amounts:
            # Prefer smaller bets with low aggression, larger bets with high aggression
            if self.aggression > 0.7:
                # More aggressive - prefer larger bets
                weights = np.linspace(0.5, 1.0, len(valid_amounts))
            elif self.aggression < 0.3:
                # More passive - prefer smaller bets
                weights = np.linspace(1.0, 0.5, len(valid_amounts))
            else:
                # Balanced - uniform distribution
                weights = np.ones(len(valid_amounts))
            
            # Normalize weights
            weights = weights / sum(weights)
            
            # Select bet amount based on weights
            bet_idx = np.random.choice(len(valid_amounts), p=weights)
            bet_amount = valid_amounts[bet_idx]
        
        return action_type, bet_amount