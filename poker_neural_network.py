import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PokerNeuralNet")

class PokerNN(nn.Module):
    """Neural network model for poker decision making with residual connections"""
    
    def __init__(self, input_size=17):
        super(PokerNN, self).__init__()
        
        # Initial layer
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        # Residual block 1
        self.res1_fc1 = nn.Linear(128, 128)
        self.res1_bn1 = nn.BatchNorm1d(128)
        self.res1_fc2 = nn.Linear(128, 128)
        self.res1_bn2 = nn.BatchNorm1d(128)
        
        # Residual block 2
        self.res2_fc1 = nn.Linear(128, 64)
        self.res2_bn1 = nn.BatchNorm1d(64)
        self.res2_fc2 = nn.Linear(64, 64)
        self.res2_bn2 = nn.BatchNorm1d(64)
        self.res2_downsample = nn.Linear(128, 64)
        
        # Final layers
        self.fc_final = nn.Linear(64, 32)
        self.bn_final = nn.BatchNorm1d(32)
        
        # Action head (fold, check/call, bet/raise)
        self.action_head = nn.Linear(32, 3)
        
        # Bet sizing head (percentage of pot)
        self.bet_head = nn.Linear(32, 1)
        
        # Confidence head (model certainty)
        self.confidence_head = nn.Linear(32, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Handle single input case
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Skip BatchNorm with batch size of 1 during inference
        if x.shape[0] == 1 and not self.training:
            # Initial layers
            x = F.relu(self.fc1(x))
            
            # Residual block 1
            identity = x
            x = F.relu(self.res1_fc1(x))
            x = self.dropout(x)
            x = self.res1_fc2(x)
            x = x + identity  # Residual connection
            x = F.relu(x)
            
            # Residual block 2
            identity = self.res2_downsample(x)
            x = F.relu(self.res2_fc1(x))
            x = self.dropout(x)
            x = self.res2_fc2(x)
            x = x + identity  # Residual connection
            x = F.relu(x)
            
            # Final layers
            x = F.relu(self.fc_final(x))
            x = self.dropout(x)
        else:
            # Initial layers with batch normalization
            x = F.relu(self.bn1(self.fc1(x)))
            
            # Residual block 1
            identity = x
            x = F.relu(self.res1_bn1(self.res1_fc1(x)))
            x = self.dropout(x)
            x = self.res1_bn2(self.res1_fc2(x))
            x = x + identity  # Residual connection
            x = F.relu(x)
            
            # Residual block 2
            identity = self.res2_downsample(x)
            x = F.relu(self.res2_bn1(self.res2_fc1(x)))
            x = self.dropout(x)
            x = self.res2_bn2(self.res2_fc2(x))
            x = x + identity  # Residual connection
            x = F.relu(x)
            
            # Final layers
            x = F.relu(self.bn_final(self.fc_final(x)))
            x = self.dropout(x)
        
        # Action probabilities with softmax
        action_output = F.softmax(self.action_head(x), dim=1)
        
        # Bet sizing with sigmoid (to get a value between 0 and 1)
        bet_output = torch.sigmoid(self.bet_head(x))
        
        # Confidence with sigmoid (to get a value between 0 and 1)
        confidence_output = torch.sigmoid(self.confidence_head(x))
        
        return action_output, bet_output, confidence_output


class PokerFeatureExtractor:
    """Extract features from poker game state for neural network input"""
    
    def __init__(self):
        self.logger = logging.getLogger("FeatureExtractor")
        
        # Card values for mapping
        self.card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                           '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        # Precomputed hand strength mappings for common hands
        self.preflop_hand_strength = self._initialize_preflop_strength()
    
    def _initialize_preflop_strength(self):
        """Initialize a basic mapping of preflop hand strength"""
        strength_map = {}
        
        # Pocket pairs
        for value in self.card_values:
            val = self.card_values[value]
            # Scale from 0.5 (22) to 1.0 (AA)
            strength = 0.5 + ((val - 2) / 24)
            key = f"{value}-{value}"
            strength_map[key] = strength
        
        # Suited connectors and high cards
        for v1 in self.card_values:
            for v2 in self.card_values:
                if v1 != v2:
                    val1 = self.card_values[v1]
                    val2 = self.card_values[v2]
                    
                    # Base strength from card values
                    base = (val1 + val2) / 28  # Simple normalization
                    
                    # Connected bonus
                    connected = 0.1 if abs(val1 - val2) == 1 else 0
                    
                    # Suited bonus (assume suited)
                    suited = 0.1
                    
                    # Combined strength (suited)
                    key_suited = f"{v1}s-{v2}s"
                    strength_map[key_suited] = min(0.95, base + connected + suited)
                    
                    # Combined strength (offsuit)
                    key_offsuit = f"{v1}o-{v2}o"
                    strength_map[key_offsuit] = min(0.85, base + connected)
        
        return strength_map
    
    def extract_features(self, state, player_position=0):
        """
        Extract features from the game state
        
        Args:
            state: Game state from rlcard
            player_position: Position of player (default 0 for rlcard)
            
        Returns:
            np.ndarray: Feature vector
        """
        try:
            # No need to extract raw_obs - rlcard is directly giving us the state
            obs = state
            
            # Initialize feature array
            features = []
            
            # 1. Hand strength features (2 features)
            hand = obs.get('hand', [])
            public_cards = obs.get('public_cards', [])
            
            # Convert from rlcard format (e.g., 'HJ' for Jack of Hearts) to our format
            hand_strength = self._calculate_hand_strength_rlcard(hand, public_cards)
            hand_potential = self._calculate_hand_potential_rlcard(hand, public_cards)
            
            features.append(hand_strength)
            features.append(hand_potential)
            
            # 2. Position feature (1 feature)
            # Normalize position to [0,1]
            position = obs.get('current_player', player_position)
            all_chips = obs.get('all_chips', [1, 1])
            num_players = len(all_chips)
            normalized_position = position / max(1, num_players - 1)
            features.append(normalized_position)
            
            # 3. Stack features (2 features)
            my_chips = obs.get('my_chips', all_chips[position] if position < len(all_chips) else 1)
            
            # Calculate average opponent stack
            opponent_stacks = []
            for i, chips in enumerate(all_chips):
                if i != position:
                    opponent_stacks.append(chips)
            
            avg_opponent_stack = sum(opponent_stacks) / max(1, len(opponent_stacks))
            
            # Normalize stack sizes
            max_stack = max(my_chips, avg_opponent_stack, 1)
            norm_player_stack = my_chips / max_stack
            norm_opponent_stack = avg_opponent_stack / max_stack
            
            features.append(norm_player_stack)
            features.append(norm_opponent_stack)
            
            # 4. Pot-related features (1 feature)
            pot = float(obs.get('pot', 0))
            pot_to_stack = pot / max(1, my_chips)
            features.append(min(1.0, pot_to_stack))
            
            # 5. Round features - one-hot encoding (4 features)
            # Convert Stage enum to index
            stage = obs.get('stage', 0)
            if hasattr(stage, 'value'):
                stage_idx = stage.value
            else:
                stage_idx = int(stage)
            
            # Ensure stage_idx is between 0-3
            stage_idx = min(3, max(0, stage_idx))
            
            round_features = [0, 0, 0, 0]
            round_features[stage_idx] = 1
            features.extend(round_features)
            
            # 6. Legal actions features (3 features)
            legal_actions = obs.get('legal_actions', [])
            
            # Check if specific actions are available
            can_fold = any("FOLD" in str(action) for action in legal_actions)
            can_check_call = any("CHECK_CALL" in str(action) for action in legal_actions)
            can_raise = any(("RAISE" in str(action) or "ALL_IN" in str(action)) for action in legal_actions)
            
            features.append(float(can_fold))
            features.append(float(can_check_call))
            features.append(float(can_raise))
            
            # 7. Betting history features (4 features)
            # Extract stakes information if available
            stakes = obs.get('stakes', [0, 0])
            
            # 1. Current stake to pot ratio
            my_stake = stakes[position] if position < len(stakes) else 0
            stake_to_pot = my_stake / max(1, pot)
            bet_history_features = [min(1.0, stake_to_pot)]
            
            # 2-4. Add placeholder features for now
            # In a full implementation, these would track more betting history
            bet_history_features.extend([0.0, 0.0, 0.0])
            
            # Ensure we have exactly 4 betting history features
            bet_history_features = bet_history_features[:4]
            features.extend(bet_history_features)
            
            # Ensure we have exactly 17 features
            if len(features) != 17:
                # Truncate or pad to get exactly 17 features
                if len(features) > 17:
                    features = features[:17]
                else:
                    features.extend([0.0] * (17 - len(features)))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            # Return zeros in case of error
            return np.zeros(17, dtype=np.float32)
    
    def _calculate_hand_strength_rlcard(self, hand, public_cards):
        """
        Calculate hand strength from RLCard format cards
        
        Args:
            hand: List of cards in RLCard format (e.g., 'HJ' for Jack of Hearts)
            public_cards: List of community cards in RLCard format
            
        Returns:
            float: Hand strength (0-1)
        """
        try:
            # If no cards, return minimal strength
            if not hand:
                return 0.1
            
            # Convert from RLCard format to our internal format
            converted_hand = []
            for card in hand:
                suit = card[0]  # First char is suit
                rank = card[1:] # Rest is rank
                
                # Map suit
                if suit == 'H':
                    suit_name = 'hearts'
                elif suit == 'D':
                    suit_name = 'diamonds'
                elif suit == 'C':
                    suit_name = 'clubs'
                elif suit == 'S':
                    suit_name = 'spades'
                else:
                    suit_name = 'unknown'
                
                # Map rank
                if rank == 'T':
                    rank = '10'
                
                converted_hand.append({'value': rank, 'suit': suit_name})
            
            # Convert public cards
            converted_public = []
            for card in public_cards:
                suit = card[0]
                rank = card[1:]
                
                # Map suit
                if suit == 'H':
                    suit_name = 'hearts'
                elif suit == 'D':
                    suit_name = 'diamonds'
                elif suit == 'C':
                    suit_name = 'clubs'
                elif suit == 'S':
                    suit_name = 'spades'
                else:
                    suit_name = 'unknown'
                
                # Map rank
                if rank == 'T':
                    rank = '10'
                
                converted_public.append({'value': rank, 'suit': suit_name})
            
            # Use existing method for actual calculation
            return self._calculate_hand_strength(converted_hand, converted_public)
            
        except Exception as e:
            self.logger.error(f"Error calculating hand strength from RLCard format: {str(e)}")
            return 0.3  # Default moderate strength
    
    def _calculate_hand_potential_rlcard(self, hand, public_cards):
        """Calculate hand potential from RLCard format cards"""
        # For simple implementation, reuse the hand strength with a similar conversion
        try:
            strength = self._calculate_hand_strength_rlcard(hand, public_cards)
            
            # Adjust potential based on number of community cards
            if not public_cards:  # Preflop
                return 0.7  # High potential
            elif len(public_cards) == 3:  # Flop
                return 0.5  # Medium potential
            elif len(public_cards) == 4:  # Turn
                return 0.3  # Low potential
            else:  # River
                return 0.0  # No more potential
                
        except Exception as e:
            self.logger.error(f"Error calculating hand potential from RLCard format: {str(e)}")
            return 0.3  # Default potential
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            # Return zeros in case of error
            return np.zeros(17, dtype=np.float32)
    
    def _calculate_hand_strength(self, hand, public_cards):
        """
        Calculate the strength of the current hand
        
        Args:
            hand: List of hole cards
            public_cards: List of community cards
            
        Returns:
            float: Hand strength (0-1)
        """
        try:
            # If no community cards (preflop), use precomputed strengths
            if not public_cards:
                # Sort cards by value
                card_values = [card[0] for card in hand]
                
                # Check if suited
                suited = hand[0][1] == hand[1][1] if len(hand) >= 2 else False
                
                # Create hand key for lookup
                if len(hand) == 2:
                    card1, card2 = sorted(card_values, reverse=True)
                    if card1 == card2:  # Pocket pair
                        key = f"{card1}-{card2}"
                    else:
                        suffix = 's' if suited else 'o'
                        key = f"{card1}{suffix}-{card2}{suffix}"
                        
                    # Look up strength
                    return self.preflop_hand_strength.get(key, 0.3)
            
            # For postflop, we'd use a more sophisticated hand evaluator
            # For this example, we'll use a simple placeholder function
            num_cards = len(hand) + len(public_cards)
            if num_cards == 2:  # Preflop
                return 0.3  # Default preflop strength
            elif num_cards == 5:  # Flop
                return 0.4  # Default flop strength
            elif num_cards == 6:  # Turn
                return 0.5  # Default turn strength
            else:  # River
                return 0.6  # Default river strength
                
        except Exception as e:
            self.logger.error(f"Error calculating hand strength: {str(e)}")
            return 0.3  # Default strength
    
    def _calculate_hand_potential(self, hand, public_cards):
        """
        Calculate potential of the hand to improve
        
        Args:
            hand: List of hole cards
            public_cards: List of community cards
            
        Returns:
            float: Hand potential (0-1)
        """
        # Simple potential calculation based on number of cards remaining
        if len(public_cards) == 0:  # Preflop
            return 0.7  # High potential
        elif len(public_cards) == 3:  # Flop
            return 0.5  # Medium potential
        elif len(public_cards) == 4:  # Turn
            return 0.3  # Low potential
        else:  # River
            return 0.0  # No more cards