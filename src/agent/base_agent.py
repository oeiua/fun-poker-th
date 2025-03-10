"""
Base agent class for poker AI.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

from src.environment.state import GameState


class BaseAgent(ABC):
    """
    Abstract base class for all poker agents.
    """
    
    def __init__(self, player_id: int):
        """
        Initialize the agent.
        
        Args:
            player_id: Unique identifier for this agent
        """
        self.player_id = player_id
        self.hand_history = []
        
    @abstractmethod
    def get_action(self, state: GameState, legal_actions: Dict[str, List[int]]) -> Tuple[str, int]:
        """
        Get the agent's action given the current game state.
        
        Args:
            state: Current game state
            legal_actions: Dictionary of legal actions with their parameters
            
        Returns:
            Tuple of (action_type, amount)
        """
        pass
    
    def record_action(self, state: GameState, action_type: str, amount: int) -> None:
        """
        Record an action taken in the game for later learning.
        
        Args:
            state: Game state before action
            action_type: Type of action taken
            amount: Amount (if applicable)
        """
        self.hand_history.append({
            'state': state,
            'action_type': action_type,
            'amount': amount
        })
    
    def reset(self) -> None:
        """Reset the agent's state for a new hand."""
        self.hand_history = []
        
    def update(self, reward: float) -> None:
        """
        Update the agent's policy based on the reward received.
        
        Args:
            reward: Reward value (positive or negative)
        """
        pass