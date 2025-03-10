"""
Reward calculation for poker environment.
"""
from typing import List, Dict, Any, Optional, Union
import numpy as np

class RewardCalculator:
    """Class for calculating and shaping rewards in poker games."""
    
    def __init__(self, big_blind: int = 100):
        """
        Initialize the reward calculator.
        
        Args:
            big_blind: Big blind amount for normalization
        """
        self.big_blind = big_blind
    
    def calculate_hand_reward(
        self,
        initial_stack: int,
        final_stack: int,
        won_hand: bool,
        folded: bool = False,
        all_in: bool = False
    ) -> float:
        """
        Calculate reward for a player after a hand is complete.
        
        Args:
            initial_stack: Player's stack at the beginning of the hand
            final_stack: Player's stack after the hand
            won_hand: Whether the player won the hand
            folded: Whether the player folded
            all_in: Whether the player went all-in
            
        Returns:
            Calculated reward value
        """
        # Base reward: change in stack, normalized by big blind
        base_reward = (final_stack - initial_stack) / self.big_blind
        
        # Apply reward shaping based on actions and outcomes
        shaped_reward = base_reward
        
        # Extra incentive for winning hands
        if won_hand:
            shaped_reward *= 1.05  # Small bonus for winning
        
        # Slight penalty for folding too much
        if folded:
            shaped_reward -= 0.1  # Small penalty to discourage excessive folding
        
        # Small incentive for bold plays when successful
        if all_in and won_hand:
            shaped_reward *= 1.1  # Bonus for successful all-in
        
        return shaped_reward
    
    def calculate_action_reward(
        self,
        action_type: int,
        bet_amount: Optional[int],
        pot_size: int,
        player_stack: int,
        hand_strength: float
    ) -> float:
        """
        Calculate immediate reward for an action based on heuristics.
        This can be used for immediate feedback during training.
        
        Args:
            action_type: Type of action taken
            bet_amount: Amount bet/raised if applicable
            pot_size: Current pot size
            player_stack: Player's current stack
            hand_strength: Estimated strength of player's hand (0-1)
            
        Returns:
            Immediate reward value
        """
        # This is an optional shaping reward to guide learning
        # It should be much smaller than actual game outcomes
        
        # Initialize reward
        reward = 0.0
        
        # Fold
        if action_type == 0:
            # Reward for folding weak hands, penalize folding strong hands
            fold_threshold = 0.3  # Below this hand strength, folding is reasonable
            if hand_strength < fold_threshold:
                reward = 0.05  # Small reward for folding weak hands
            else:
                reward = -0.1 * hand_strength  # Penalty scales with hand strength
        
        # Check/Call
        elif action_type == 1:
            # Reasonable for medium-strength hands
            call_lower = 0.25
            call_upper = 0.7
            if call_lower <= hand_strength <= call_upper:
                reward = 0.05  # Small reward for appropriate calling
            elif hand_strength < call_lower:
                # Penalty for calling with very weak hands
                reward = -0.05
            # No penalty for calling with strong hands (might be slow-playing)
        
        # Bet/Raise
        elif action_type == 2 and bet_amount is not None:
            # Calculate bet size relative to pot
            pot_ratio = bet_amount / max(1, pot_size)
            
            # Reward for betting with strong hands
            if hand_strength >= 0.6:
                # Larger reward for appropriate sizing
                if 0.5 <= pot_ratio <= 1.0:
                    reward = 0.1
                else:
                    reward = 0.05  # Still okay but not optimal sizing
            
            # Penalty for betting too much with weak hands
            elif hand_strength < 0.3 and pot_ratio > 0.5:
                reward = -0.1
            
            # Small reward for bluffing with weak hands (occasional bluffing is good)
            elif hand_strength < 0.3 and pot_ratio <= 0.5:
                # Random small reward to encourage some bluffing
                if np.random.random() < 0.2:  # 20% chance
                    reward = 0.15
        
        return reward
    
    def calculate_hand_strength(
        self,
        hole_cards: List[str],
        community_cards: List[str],
        num_players: int
    ) -> float:
        """
        Calculate approximate hand strength given current cards.
        This is a simplified version - in a real implementation, 
        you would use a more sophisticated hand evaluator.
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Community cards
            num_players: Number of players in the hand
            
        Returns:
            Estimated hand strength (0-1)
        """
        # This is a very simplified model
        # In practice, you would use a proper poker hand evaluator
        
        # Count high cards (10, J, Q, K, A)
        high_cards = ['T', 'J', 'Q', 'K', 'A']
        num_high = sum(1 for card in hole_cards if card[0] in high_cards)
        
        # Initial strength based on high cards in hole cards
        if num_high == 2:
            base_strength = 0.7
        elif num_high == 1:
            base_strength = 0.5
        else:
            base_strength = 0.3
        
        # Check for pairs in hole cards
        if len(hole_cards) == 2 and hole_cards[0][0] == hole_cards[1][0]:
            if hole_cards[0][0] in high_cards:
                base_strength = 0.8  # High pair
            else:
                base_strength = 0.6  # Low pair
        
        # Adjust for suited cards
        if len(hole_cards) == 2 and hole_cards[0][1] == hole_cards[1][1]:
            base_strength += 0.1  # Suited bonus
        
        # Adjust for connected cards (simplified check)
        if len(hole_cards) == 2:
            rank_order = '23456789TJQKA'
            rank1_idx = rank_order.find(hole_cards[0][0])
            rank2_idx = rank_order.find(hole_cards[1][0])
            if abs(rank1_idx - rank2_idx) == 1:
                base_strength += 0.1  # Connected bonus
            elif abs(rank1_idx - rank2_idx) == 2:
                base_strength += 0.05  # One-gap bonus
        
        # Adjust for community cards (very simplified)
        if community_cards:
            # Check for pairs with hole cards
            for hole_card in hole_cards:
                for comm_card in community_cards:
                    if hole_card[0] == comm_card[0]:
                        base_strength += 0.15  # Pair with board
                        break
        
        # Adjust for number of players (more players = lower equity)
        player_factor = 1.0 - (num_players - 2) * 0.05  # Decrease by 5% per additional player
        player_factor = max(0.7, player_factor)  # Don't go below 70%
        
        # Apply adjustment
        final_strength = base_strength * player_factor
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, final_strength))
    
    def discounted_rewards(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """
        Calculate discounted rewards for an episode.
        
        Args:
            rewards: List of rewards for each step
            gamma: Discount factor
            
        Returns:
            List of discounted rewards
        """
        discounted = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted[t] = running_add
            
        # Normalize to reduce variance
        if len(discounted) > 1:
            discounted = (discounted - np.mean(discounted)) / (np.std(discounted) + 1e-8)
            
        return discounted.tolist()
    
    def calculate_expected_value(
        self,
        pot_size: int,
        to_call: int,
        hand_strength: float
    ) -> float:
        """
        Calculate simple expected value of a call.
        
        Args:
            pot_size: Current pot size
            to_call: Amount needed to call
            hand_strength: Estimated probability of winning (0-1)
            
        Returns:
            Expected value
        """
        # EV = (Probability of winning * Amount won) - (Probability of losing * Amount lost)
        win_amount = pot_size
        lose_amount = to_call
        
        win_probability = hand_strength
        lose_probability = 1.0 - win_probability
        
        ev = (win_probability * win_amount) - (lose_probability * lose_amount)
        
        # Normalize by big blind
        return ev / self.big_blind