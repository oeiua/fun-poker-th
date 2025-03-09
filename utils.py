"""
Utility functions for the Poker AI system.
"""
import os
import random
import numpy as np
import tensorflow as tf
import datetime
import json
from typing import List, Dict, Any, Tuple, Optional

from config import PokerConfig


def set_seeds(seed: int = PokerConfig.SEED):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_tf_memory_growth():
    """Configure TensorFlow to use memory growth for GPUs."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(f"Error setting memory growth: {e}")


def save_config_snapshot(config_obj, filepath: str):
    """
    Save a snapshot of the configuration.
    
    Args:
        config_obj: Configuration object.
        filepath (str): Path to save the configuration.
    """
    # Convert config to dictionary
    config_dict = {attr: getattr(config_obj, attr) for attr in dir(config_obj) 
                  if not attr.startswith('__') and not callable(getattr(config_obj, attr))}
    
    # Add timestamp
    config_dict['timestamp'] = datetime.datetime.now().isoformat()
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)


def load_training_history(filepath: str) -> Dict[str, Any]:
    """
    Load training history from file.
    
    Args:
        filepath (str): Path to the history file.
        
    Returns:
        Dict[str, Any]: Loaded history data.
    """
    try:
        data = np.load(filepath, allow_pickle=True)
        return {key: data[key] for key in data.files}
    except (FileNotFoundError, IOError):
        return {}


def calculate_hand_strength(hand: List[Any], community_cards: List[Any], num_players: int) -> float:
    """
    Calculate the approximate strength of a hand in the current game state.
    
    Args:
        hand (List[Any]): Player's hole cards.
        community_cards (List[Any]): Community cards on the table.
        num_players (int): Number of players in the hand.
        
    Returns:
        float: Hand strength score between 0 and 1.
    """
    from card import Card, Rank, Suit
    
    # This is a simplified implementation - for a real system, we would use a more
    # sophisticated approach like Monte Carlo simulation or a lookup table
    
    # Number of cards on the board determines the betting round
    num_community = len(community_cards)
    
    # Simplified scoring
    # Check for pairs in hole cards
    has_pocket_pair = hand[0].rank == hand[1].rank
    
    # Check for high cards
    high_card_value = max(card.rank.value for card in hand)
    
    # Check for suited cards
    suited = hand[0].suit == hand[1].suit
    
    # Base score based on hole cards
    base_score = 0.0
    
    # Pocket pairs
    if has_pocket_pair:
        rank_value = hand[0].rank.value
        base_score = 0.5 + 0.03 * (rank_value - 2)  # Higher pairs get better scores
    
    # High cards
    elif high_card_value >= Rank.JACK.value:
        base_score = 0.3 + 0.02 * (high_card_value - 10)
        
        # Suited high cards are better
        if suited:
            base_score += 0.1
    
    # Connected cards (within 3 ranks)
    elif abs(hand[0].rank.value - hand[1].rank.value) <= 3:
        base_score = 0.2 + 0.1 * (1 / (1 + abs(hand[0].rank.value - hand[1].rank.value)))
        
        # Suited connectors are better
        if suited:
            base_score += 0.1
    
    # Adjust for position (later positions are better)
    # position_factor = min(1.0, 0.8 + 0.2 * (position / (num_players - 1)))
    # base_score *= position_factor
    
    # Adjust for number of players (more players = need stronger hands)
    players_factor = max(0.5, 1.0 - 0.05 * (num_players - 2))
    base_score *= players_factor
    
    # If we have community cards, do a more detailed evaluation
    if community_cards:
        # Check for made hands with the community cards
        all_cards = hand + community_cards
        
        # Check for pairs, trips, etc. with the board
        rank_counts = {}
        for card in all_cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        
        # Count suits for flush draws
        suit_counts = {}
        for card in all_cards:
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1
        
        # Straight draws - just a simplification
        ranks = sorted(set(card.rank.value for card in all_cards))
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(ranks)):
            if ranks[i] == ranks[i-1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        # Calculate made hand bonuses
        made_hand_score = 0.0
        
        # Pairs, trips, quads
        if max(rank_counts.values()) == 4:  # Four of a kind
            made_hand_score = 0.95
        elif max(rank_counts.values()) == 3 and list(rank_counts.values()).count(2) >= 1:  # Full house
            made_hand_score = 0.9
        elif max(suit_counts.values()) >= 5:  # Flush
            made_hand_score = 0.85
        elif max_consecutive >= 5:  # Straight
            made_hand_score = 0.8
        elif max(rank_counts.values()) == 3:  # Three of a kind
            made_hand_score = 0.7
        elif list(rank_counts.values()).count(2) >= 2:  # Two pair
            made_hand_score = 0.6
        elif max(rank_counts.values()) == 2:  # One pair
            made_hand_score = 0.5
        
        # Calculate draw bonuses
        draw_score = 0.0
        
        # Flush draw
        for suit, count in suit_counts.items():
            if count == 4:
                draw_score = max(draw_score, 0.4)
        
        # Open-ended straight draw
        if max_consecutive == 4:
            draw_score = max(draw_score, 0.35)
        
        # Gutshot straight draw
        elif max_consecutive == 3 and len(ranks) >= 5:
            draw_score = max(draw_score, 0.2)
        
        # Combine base score with made hand and draw scores
        combined_score = max(base_score, made_hand_score, draw_score)
        
        # Scale based on the stage of the hand
        if num_community == 3:  # Flop
            return 0.3 * base_score + 0.7 * combined_score
        elif num_community == 4:  # Turn
            return 0.2 * base_score + 0.8 * combined_score
        elif num_community == 5:  # River
            return combined_score  # Only consider the final hand
        
    return base_score  # Pre-flop


def estimate_pot_odds(to_call: int, pot_size: int) -> float:
    """
    Calculate the pot odds.
    
    Args:
        to_call (int): Amount to call.
        pot_size (int): Current pot size.
        
    Returns:
        float: Pot odds as a ratio.
    """
    if to_call == 0:
        return float('inf')  # If we can check, the odds are infinite
    
    return pot_size / to_call


def optimal_bet_size(hand_strength: float, stack: int, pot: int) -> int:
    """
    Calculate an optimal bet size based on hand strength.
    
    Args:
        hand_strength (float): Strength of the hand (0-1).
        stack (int): Player's stack.
        pot (int): Current pot size.
        
    Returns:
        int: Optimal bet size.
    """
    # Base bet sizes on hand strength and pot size
    if hand_strength < 0.3:
        # Weak hands - small bets or bluffs
        bet_factor = random.uniform(0, 0.5) if random.random() < 0.3 else 0
    elif hand_strength < 0.6:
        # Medium hands - bet around 1/2 to 2/3 pot
        bet_factor = 0.5 + 0.15 * ((hand_strength - 0.3) / 0.3)
    else:
        # Strong hands - bet 2/3 to full pot
        bet_factor = 0.65 + 0.35 * ((hand_strength - 0.6) / 0.4)
    
    # Calculate bet size
    bet_size = int(pot * bet_factor)
    
    # Ensure bet is within stack limits
    return min(bet_size, stack)