"""
Base agent class for poker AI.
"""
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np
import uuid
import time

from game.state import GameState
from game.action import Action

class BaseAgent:
    """Base class for all poker agents."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a base agent.
        
        Args:
            name: Name of the agent
        """
        self.name = name or f"Agent-{uuid.uuid4().hex[:8]}"
        self.player_idx = -1  # Will be set when added to a game
        self.stats = {
            "hands_played": 0,
            "hands_won": 0,
            "total_reward": 0.0,
            "actions": {
                Action.FOLD: 0,
                Action.CHECK_CALL: 0,
                Action.BET_RAISE: 0
            }
        }
    
    def set_player_index(self, idx: int) -> None:
        """
        Set the player index for this agent.
        
        Args:
            idx: Player index in the game
        """
        self.player_idx = idx
    
    def act(self, state: GameState, valid_actions: List[int], valid_amounts: List[int]) -> Tuple[int, Optional[int]]:
        """
        Choose an action based on the current game state.
        Must be implemented by subclasses.
        
        Args:
            state: Current game state
            valid_actions: List of valid action types
            valid_amounts: List of valid bet amounts (only relevant for BET_RAISE)
            
        Returns:
            Tuple of (action_type, bet_amount)
        """
        raise NotImplementedError("Subclasses must implement act()")
    
    def observe(self, state: GameState, action: Tuple[int, Optional[int]], reward: float, next_state: GameState, done: bool) -> None:
        """
        Observe the result of an action (for learning).
        
        Args:
            state: State before action
            action: Action taken (action_type, bet_amount)
            reward: Reward received
            next_state: State after action
            done: Whether the episode is done
        """
        # Update stats
        action_type, _ = action
        self.stats["actions"][action_type] += 1
        
        if done:
            self.stats["hands_played"] += 1
            if reward > 0:
                self.stats["hands_won"] += 1
            self.stats["total_reward"] += reward
    
    def reset(self) -> None:
        """Reset the agent's state between episodes if needed."""
        pass
    
    def save(self, path: str) -> None:
        """
        Save the agent to a file.
        
        Args:
            path: Path to save the agent
        """
        # Base implementation doesn't save anything
        pass
    
    def load(self, path: str) -> None:
        """
        Load the agent from a file.
        
        Args:
            path: Path to load the agent from
        """
        # Base implementation doesn't load anything
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get the agent's statistics.
        
        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()
    
    def debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the agent's decision making.
        Useful for analysis and visualization.
        
        Returns:
            Dictionary of debug information
        """
        return {
            "name": self.name,
            "player_idx": self.player_idx,
            "stats": self.stats.copy()
        }