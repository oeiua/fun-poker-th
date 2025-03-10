"""
Game state representation for poker environment.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class GameState:
    """Class representing the state of a poker game."""
    
    def __init__(
        self,
        hole_cards: List[List[str]],
        community_cards: List[str],
        game_info: Dict[str, Any],
        hand_history: List[Dict[str, Any]]
    ):
        """
        Initialize a game state.
        
        Args:
            hole_cards: List of hole cards for each player
            community_cards: List of community cards
            game_info: Dictionary with game information
            hand_history: List of actions taken in the current hand
        """
        self.hole_cards = hole_cards
        self.community_cards = community_cards
        self.game_info = game_info
        self.hand_history = hand_history
    
    def vectorize(self, player_idx: int) -> np.ndarray:
        """
        Convert the game state to a vector representation for a neural network.
        
        Args:
            player_idx: Index of the player from whose perspective to vectorize
            
        Returns:
            Numpy array representing the state
        """
        # Encode all relevant game information into a vector
        features = []
        
        # 1. Encode player's hole cards (52 bits for each possible card)
        hole_card_features = self._encode_cards(self.hole_cards[player_idx])
        features.extend(hole_card_features)  # 52 features
        
        # 2. Encode community cards (52 bits)
        community_card_features = self._encode_cards(self.community_cards)
        features.extend(community_card_features)  # 52 features
        
        # 3. Encode player positions relative to button (one-hot, 10 features for 10 players)
        button_position = (self.game_info.get('button_position', 0) % len(self.hole_cards))
        position_features = [0] * len(self.hole_cards)
        for i in range(len(self.hole_cards)):
            # Calculate position relative to button (0 = button, 1 = small blind, etc.)
            rel_pos = (i - button_position) % len(self.hole_cards)
            position_features[rel_pos] = 1 if i == player_idx else 0
        features.extend(position_features)  # 10 features
        
        # 4. Encode pot size (normalized by big blind)
        pot_size = self.game_info.get('pot', 0)
        big_blind = self.game_info.get('big_blind', 100)
        features.append(pot_size / big_blind)  # 1 feature
        
        # 5. Encode player stacks (normalized by big blind)
        player_stacks = self.game_info.get('player_stacks', [0] * len(self.hole_cards))
        for i in range(len(player_stacks)):
            features.append(player_stacks[i] / big_blind)  # 10 features
        
        # 6. Encode current bets (normalized by big blind)
        player_bets = self.game_info.get('player_bets', [0] * len(self.hole_cards))
        for i in range(len(player_bets)):
            features.append(player_bets[i] / big_blind)  # 10 features
        
        # 7. Encode folded players (binary)
        folded_players = self.game_info.get('folded_players', [])
        for i in range(len(self.hole_cards)):
            features.append(1.0 if i in folded_players else 0.0)  # 10 features
        
        # 8. Encode street (one-hot, 4 features for preflop, flop, turn, river)
        street = self.game_info.get('street', 'preflop')
        street_features = [0] * 4
        street_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        if street in street_mapping:
            street_features[street_mapping[street]] = 1
        features.extend(street_features)  # 4 features
        
        # 9. Encode player's position (is current player, is next to act, etc.)
        is_current = 1.0 if player_idx == self.game_info.get('current_player', -1) else 0.0
        features.append(is_current)  # 1 feature
        
        # 10. Encode hand history (last 3 actions)
        # For each action we encode: player, action type, amount
        history_length = 3
        history = self.hand_history[-history_length:] if len(self.hand_history) >= history_length else self.hand_history
        
        # Pad history if needed
        while len(history) < history_length:
            history.insert(0, {'player': -1, 'action_type': -1, 'amount': 0})
        
        for action in history:
            # One-hot encoding for player (10 features)
            player_features = [0] * len(self.hole_cards)
            action_player = action.get('player', -1)
            if 0 <= action_player < len(player_features):
                player_features[action_player] = 1
            features.extend(player_features)  # 10 features
            
            # One-hot encoding for action type (3 features: fold, check/call, bet/raise)
            action_type = action.get('action_type', -1)
            action_features = [0] * 3
            if 0 <= action_type < len(action_features):
                action_features[action_type] = 1
            features.extend(action_features)  # 3 features
            
            # Normalize amount by big blind
            amount = action.get('amount', 0)
            features.append(amount / big_blind if amount is not None else 0)  # 1 feature
        
        # Convert to numpy array
        return np.array(features, dtype=np.float32)
    
    def _encode_cards(self, cards: List[str]) -> List[float]:
        """
        Encode cards as a 52-bit vector.
        
        Args:
            cards: List of card strings (e.g., ['As', 'Kh', '2c'])
            
        Returns:
            List of 52 binary values representing the cards
        """
        # Initialize 52-length vector (0 for each card not present)
        card_vector = [0.0] * 52
        
        # Card encoding: 13 ranks * 4 suits
        # Mapping: A, 2, 3, ..., Q, K for each suit (clubs, diamonds, hearts, spades)
        ranks = 'A23456789TJQK'
        suits = 'cdhs'
        
        for card in cards:
            if len(card) == 2:
                rank, suit = card[0], card[1]
                try:
                    rank_idx = ranks.index(rank)
                    suit_idx = suits.index(suit)
                    # Calculate card index in the vector
                    card_idx = rank_idx + (suit_idx * 13)
                    # Set card as present
                    if 0 <= card_idx < 52:
                        card_vector[card_idx] = 1.0
                except ValueError:
                    # Invalid card, ignore
                    pass
        
        return card_vector
    
    def get_player_view(self, player_idx: int) -> Dict[str, Any]:
        """
        Get the game state from a specific player's perspective (hiding other players' cards).
        
        Args:
            player_idx: Index of the player
            
        Returns:
            Dictionary with the visible game state information
        """
        # Create a copy of the game state with hidden information
        player_view = {
            'hole_cards': self.hole_cards[player_idx],
            'community_cards': self.community_cards,
            'game_info': self.game_info.copy(),
            'hand_history': self.hand_history.copy()
        }
        
        # Remove hidden information
        player_view['game_info']['all_hole_cards'] = None
        
        return player_view
    
    def is_terminal(self) -> bool:
        """
        Check if the current state is terminal (hand is over).
        
        Returns:
            True if terminal, False otherwise
        """
        return self.game_info.get('round', 0) >= self.game_info.get('max_rounds', 0)
    
    def get_current_player(self) -> int:
        """
        Get the index of the current player to act.
        
        Returns:
            Player index
        """
        return self.game_info.get('current_player', -1)