"""
Action space definition for poker environment.
"""
from typing import List, Dict, Any, Optional, Tuple, Union

class Action:
    """Class defining poker actions and utilities for action handling."""
    
    # Action types (indices)
    FOLD = 0
    CHECK_CALL = 1
    BET_RAISE = 2
    
    # Action names for display/logging
    ACTION_NAMES = {
        FOLD: "Fold",
        CHECK_CALL: "Check/Call",
        BET_RAISE: "Bet/Raise"
    }
    
    @staticmethod
    def get_action_name(action_type: int) -> str:
        """
        Get the name of an action type.
        
        Args:
            action_type: Action type index
            
        Returns:
            Name of the action
        """
        return Action.ACTION_NAMES.get(action_type, "Unknown")
    
    @staticmethod
    def get_action_description(action_type: int, amount: Optional[int] = None) -> str:
        """
        Get a human-readable description of an action.
        
        Args:
            action_type: Action type index
            amount: Bet/raise amount if applicable
            
        Returns:
            Description string
        """
        if action_type == Action.FOLD:
            return "Fold"
        elif action_type == Action.CHECK_CALL:
            return "Check" if amount == 0 or amount is None else f"Call {amount}"
        elif action_type == Action.BET_RAISE:
            return "All-In" if amount == -1 else f"{'Bet' if amount == 0 or amount is None else 'Raise'} {amount}"
        else:
            return "Unknown action"
    
    @staticmethod
    def is_valid_bet(bet_amount: Union[int, float], min_bet: int, max_bet: int) -> bool:
        """
        Check if a bet amount is valid.
        
        Args:
            bet_amount: Amount to bet/raise
            min_bet: Minimum allowed bet
            max_bet: Maximum allowed bet (player's stack)
            
        Returns:
            True if valid, False otherwise
        """
        # Convert to int if float
        bet_amount = int(bet_amount) if isinstance(bet_amount, float) else bet_amount
        
        # All-in is always valid
        if bet_amount == -1:
            return True
        
        # Check if within valid range
        return min_bet <= bet_amount <= max_bet
    
    @staticmethod
    def standardize_bet_amount(
        bet_amount: Optional[Union[int, float]],
        min_bet: int,
        max_bet: int
    ) -> int:
        """
        Standardize a bet amount to ensure it's valid.
        
        Args:
            bet_amount: Amount to bet/raise
            min_bet: Minimum allowed bet
            max_bet: Maximum allowed bet (player's stack)
            
        Returns:
            Standardized bet amount
        """
        # Handle None or invalid input
        if bet_amount is None:
            return min_bet
        
        # Convert to int if float
        bet_amount = int(bet_amount) if isinstance(bet_amount, float) else bet_amount
        
        # Special case: -1 means all-in
        if bet_amount == -1:
            return max_bet
        
        # Clamp to valid range
        return max(min_bet, min(bet_amount, max_bet))
    
    @staticmethod
    def suggest_bet_sizes(pot_size: int, player_stack: int) -> List[int]:
        """
        Suggest common bet sizes based on the pot and player's stack.
        
        Args:
            pot_size: Current size of the pot
            player_stack: Player's current stack
            
        Returns:
            List of suggested bet amounts
        """
        suggestions = []
        
        # Standard bet sizes as percentages of the pot
        pot_percentages = [0.5, 0.75, 1.0, 1.5, 2.0]
        
        for percentage in pot_percentages:
            amount = int(pot_size * percentage)
            if amount <= player_stack and amount > 0:
                suggestions.append(amount)
        
        # Add all-in if not already in suggestions
        if player_stack > 0 and player_stack not in suggestions:
            suggestions.append(player_stack)
        
        return suggestions
    
    @staticmethod
    def encode_for_network(
        action_type: int,
        bet_amount: Optional[int],
        pot_size: int,
        player_stack: int
    ) -> List[float]:
        """
        Encode an action for neural network input.
        
        Args:
            action_type: Action type index
            bet_amount: Bet/raise amount if applicable
            pot_size: Current size of the pot
            player_stack: Player's current stack
            
        Returns:
            List of encoded action features
        """
        # One-hot encoding for action type
        action_encoding = [0.0] * 3
        if 0 <= action_type < 3:
            action_encoding[action_type] = 1.0
        
        # Normalize bet amount by pot size
        if action_type == Action.BET_RAISE and bet_amount is not None and bet_amount > 0:
            # Special case for all-in
            if bet_amount == -1:
                normalized_amount = 1.0
            else:
                normalized_amount = min(1.0, bet_amount / max(1, pot_size))
        else:
            normalized_amount = 0.0
        
        return action_encoding + [normalized_amount]
    
    @staticmethod
    def decode_from_network(
        network_output: List[float],
        valid_actions: List[int],
        pot_size: int,
        player_stack: int,
        min_bet: int
    ) -> Tuple[int, Optional[int]]:
        """
        Decode network output to a valid poker action.
        
        Args:
            network_output: Output from the neural network
            valid_actions: List of valid action types
            pot_size: Current size of the pot
            player_stack: Player's current stack
            min_bet: Minimum allowed bet
            
        Returns:
            Tuple of (action_type, bet_amount)
        """
        # Extract action probabilities and bet sizing
        action_probs = network_output[:3]
        bet_size_factor = network_output[3] if len(network_output) > 3 else 0.0
        
        # Filter valid actions
        valid_action_probs = [-float('inf')] * 3
        for action in valid_actions:
            valid_action_probs[action] = action_probs[action]
        
        # Select action with highest probability
        action_type = valid_actions[0] if valid_actions else Action.FOLD
        max_prob = valid_action_probs[action_type]
        
        for action in valid_actions:
            if valid_action_probs[action] > max_prob:
                max_prob = valid_action_probs[action]
                action_type = action
        
        # Calculate bet amount if bet/raise
        bet_amount = None
        if action_type == Action.BET_RAISE:
            # Scale by pot size
            raw_amount = int(pot_size * max(0.5, min(2.0, bet_size_factor * 2.0)))
            
            # Ensure it's at least the minimum bet
            bet_amount = max(min_bet, min(raw_amount, player_stack))
        
        return action_type, bet_amount