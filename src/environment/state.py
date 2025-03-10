"""
Game state representation for the poker AI.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from pokerkit import Card, State


@dataclass
class GameState:
    """
    Representation of the game state for AI decision making.
    
    This class converts the pokerkit State to a format that's easier
    for neural networks to process.
    """
    # Game state
    hand_cards: np.ndarray  # One-hot encoded player's hand (52)
    board_cards: np.ndarray  # One-hot encoded board cards (52)
    pot: float  # Current pot size
    player_stacks: np.ndarray  # Stack sizes for all players
    player_bets: np.ndarray  # Current bets for all players
    player_position: np.ndarray  # One-hot encoded position (early, middle, late, etc.)
    blind_level: float  # Current big blind level
    
    # Action history for each player (last 3 actions)
    player_actions: np.ndarray  # [player_index, street, action_type, amount]
    
    # Additional information
    active_players: np.ndarray  # Boolean mask of active players
    player_index: int  # Current player's index
    street_index: int  # Current street index (preflop, flop, turn, river)
    
    @classmethod
    def from_pokerkit_state(cls, state: State, player_index: int) -> 'GameState':
        """
        Create a GameState instance from a pokerkit State.
        
        Args:
            state: pokerkit State object
            player_index: Index of the current player
            
        Returns:
            GameState object
        """
        player_count = len(state.stacks)
        
        # Encode player's hole cards
        hand_cards = np.zeros(52)
        for card in state.hole_cards[player_index]:
            if card:  # Check if card is not None
                card_index = cls._card_to_index(card)
                if card_index is not None:
                    hand_cards[card_index] = 1
        
        # Encode board cards
        board_cards = np.zeros(52)
        for card_list in state.board_cards:
            for card in card_list:
                if card:  # Check if card is not None
                    card_index = cls._card_to_index(card)
                    if card_index is not None:
                        board_cards[card_index] = 1
        
        # Get pot size
        pot = state.total_pot_amount
        
        # Get player stacks and bets
        player_stacks = np.array(state.stacks)
        player_bets = np.array(state.bets)
        
        # Encode player position
        player_position = np.zeros(player_count)
        player_position[player_index] = 1
        
        # Get blind level
        blind_level = max(state.blinds_or_straddles)
        
        # Create action history
        # For simplicity, we'll track the most recent action for each player
        player_actions = np.zeros((player_count, 1, 3))  # [player_index, action_type, amount]
        # TODO: Extract action history from state.operations
        
        # Get active players
        active_players = np.array(state.statuses)
        
        # Get current street
        street_index = state.street_index if state.street_index is not None else 0
        
        return cls(
            hand_cards=hand_cards,
            board_cards=board_cards,
            pot=pot,
            player_stacks=player_stacks,
            player_bets=player_bets,
            player_position=player_position,
            blind_level=blind_level,
            player_actions=player_actions,
            active_players=active_players,
            player_index=player_index,
            street_index=street_index
        )
    
    @staticmethod
    def _card_to_index(card: Card) -> Optional[int]:
        """
        Convert a pokerkit Card to a 0-51 index.
        
        Args:
            card: pokerkit Card object
            
        Returns:
            Index between 0-51, or None if the card is unknown or invalid
        """
        if card.unknown_status:
            return None
        
        # Get rank and suit values
        rank_value = {
            'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6,
            '8': 7, '9': 8, 'T': 9, 'J': 10, 'Q': 11, 'K': 12
        }
        
        suit_value = {'c': 0, 'd': 1, 'h': 2, 's': 3}
        
        rank = card.rank.value
        suit = card.suit.value
        
        if rank in rank_value and suit in suit_value:
            return rank_value[rank] * 4 + suit_value[suit]
        
        return None
    
    def to_input_tensor(self) -> np.ndarray:
        """
        Convert the game state to a tensor suitable for neural network input.
        
        Returns:
            Feature vector representing the game state
        """
        # Concatenate all features
        features = [
            self.hand_cards,
            self.board_cards,
            np.array([self.pot]),
            self.player_stacks,
            self.player_bets,
            self.player_position,
            np.array([self.blind_level]),
            self.player_actions.flatten(),
            self.active_players,
            np.array([self.player_index]),
            np.array([self.street_index])
        ]
        
        return np.concatenate([f.flatten() for f in features])