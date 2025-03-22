import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import json
import logging
import time
from enum import Enum
from collections import defaultdict
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PokerNN")

class HandRank(Enum):
    """Poker hand rankings"""
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


class HandEvaluator:
    """Optimized evaluator for poker hand strength"""
    
    def __init__(self):
        self.card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                           '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        # Precomputed hand strengths for common preflop hands
        self._preflop_strengths = self._init_preflop_strengths()
        
        # Cache for hand evaluations to avoid redundant calculations
        self._eval_cache = {}
    
    def _init_preflop_strengths(self):
        """Initialize precomputed preflop hand strengths"""
        strengths = {}
        
        # Pocket pairs
        for value in self.card_values.keys():
            val_num = self.card_values[value]
            # Scale from 0.5 (22) to 1.0 (AA)
            norm_strength = 0.5 + ((val_num - 2) / 24)
            for suit1 in ['hearts', 'diamonds', 'clubs', 'spades']:
                for suit2 in ['hearts', 'diamonds', 'clubs', 'spades']:
                    if suit1 != suit2:  # Different suits
                        key = self._make_hand_key([
                            {'value': value, 'suit': suit1},
                            {'value': value, 'suit': suit2}
                        ])
                        strengths[key] = norm_strength
        
        # Suited connectors
        for i, value1 in enumerate(list(self.card_values.keys())[:-1]):
            value2 = list(self.card_values.keys())[i+1]
            val1_num = self.card_values[value1]
            val2_num = self.card_values[value2]
            
            # Scale based on card values (higher is better)
            base_strength = 0.3 + ((val1_num - 2) / 40)
            
            for suit in ['hearts', 'diamonds', 'clubs', 'spades']:
                key = self._make_hand_key([
                    {'value': value1, 'suit': suit},
                    {'value': value2, 'suit': suit}
                ])
                strengths[key] = base_strength + 0.2  # Suited bonus
                
                # Also add the reverse order
                key = self._make_hand_key([
                    {'value': value2, 'suit': suit},
                    {'value': value1, 'suit': suit}
                ])
                strengths[key] = base_strength + 0.2  # Suited bonus
        
        # High cards
        for value1 in ['A', 'K', 'Q', 'J']:
            for value2 in self.card_values.keys():
                if value1 != value2:
                    val1_num = self.card_values[value1]
                    val2_num = self.card_values[value2]
                    
                    # Scaling - higher cards = better strength
                    base_strength = 0.2 + ((val1_num + val2_num - 4) / 48)
                    
                    for suit1 in ['hearts', 'diamonds', 'clubs', 'spades']:
                        for suit2 in ['hearts', 'diamonds', 'clubs', 'spades']:
                            key = self._make_hand_key([
                                {'value': value1, 'suit': suit1},
                                {'value': value2, 'suit': suit2}
                            ])
                            
                            # Add suited bonus
                            suited_bonus = 0.15 if suit1 == suit2 else 0
                            strengths[key] = min(0.95, base_strength + suited_bonus)
        
        return strengths
    
    def _make_hand_key(self, cards):
        """Create a unique key for a hand for caching purposes"""
        # Sort cards by value
        sorted_cards = sorted(cards, key=lambda c: self.card_values.get(c['value'], 0), reverse=True)
        
        # Create key as a string
        return "+".join([f"{c['value']}{c['suit'][0]}" for c in sorted_cards])
    
    def evaluate(self, hole_cards, community_cards):
        """
        Evaluate the strength of a poker hand with caching for better performance
        
        Args:
            hole_cards: List of card dictionaries, e.g. [{'value': 'A', 'suit': 'hearts'}, ...]
            community_cards: List of card dictionaries
            
        Returns:
            Dictionary with hand rank, score, and description
        """
        # Create a unique key for the hand
        hand_key = self._make_hand_key(hole_cards + community_cards)
        
        # Check cache first
        if hand_key in self._eval_cache:
            return self._eval_cache[hand_key]
        
        # If preflop and no community cards, use precomputed strengths
        if not community_cards:
            preflop_key = self._make_hand_key(hole_cards)
            if preflop_key in self._preflop_strengths:
                preflop_strength = self._preflop_strengths[preflop_key]
                result = {
                    'rank': HandRank.HIGH_CARD,  # Default rank for preflop
                    'score': int(preflop_strength * 1000000),
                    'description': f"Preflop: {hole_cards[0]['value']} {hole_cards[1]['value']}"
                }
                self._eval_cache[hand_key] = result
                return result
        
        # For simplicity, we'll provide a basic implementation here
        # In a real poker evaluator, you would have a full implementation that evaluates
        # all possible hand combinations and returns the best hand
        
        # This implementation is simplified and for illustration purposes only
        all_cards = hole_cards + community_cards
        values = [self.card_values[card['value']] for card in all_cards]
        suits = [card['suit'] for card in all_cards]
        
        # Count each value
        value_counts = defaultdict(int)
        for value in values:
            value_counts[value] += 1
        
        # Count each suit
        suit_counts = defaultdict(int)
        for suit in suits:
            suit_counts[suit] += 1
        
        # Sort in descending order by frequency, then by card value
        sorted_values = sorted(value_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        # Check for pairs, trips, quads
        if sorted_values[0][1] == 4:  # Four of a kind
            rank = HandRank.FOUR_OF_A_KIND
            score = 7000000 + sorted_values[0][0] * 1000
            description = f"Four of a kind, {self._value_to_str(sorted_values[0][0])}'s"
        
        elif sorted_values[0][1] == 3 and sorted_values[1][1] >= 2:  # Full house
            rank = HandRank.FULL_HOUSE
            score = 6000000 + sorted_values[0][0] * 1000 + sorted_values[1][0]
            description = f"Full house, {self._value_to_str(sorted_values[0][0])}'s over {self._value_to_str(sorted_values[1][0])}'s"
        
        elif any(count >= 5 for suit, count in suit_counts.items()):  # Flush
            rank = HandRank.FLUSH
            flush_suit = max(suit_counts.items(), key=lambda x: x[1])[0]
            flush_cards = [card for card in all_cards if card['suit'] == flush_suit]
            flush_values = sorted([self.card_values[card['value']] for card in flush_cards], reverse=True)
            score = 5000000 + sum(v * (10 ** (4-i)) for i, v in enumerate(flush_values[:5]))
            description = f"Flush, {flush_suit}"
        
        elif sorted_values[0][1] == 3:  # Three of a kind
            rank = HandRank.THREE_OF_A_KIND
            score = 3000000 + sorted_values[0][0] * 1000
            description = f"Three of a kind, {self._value_to_str(sorted_values[0][0])}'s"
        
        elif sorted_values[0][1] == 2 and sorted_values[1][1] == 2:  # Two pair
            rank = HandRank.TWO_PAIR
            score = 2000000 + max(sorted_values[0][0], sorted_values[1][0]) * 1000 + min(sorted_values[0][0], sorted_values[1][0])
            description = f"Two pair, {self._value_to_str(sorted_values[0][0])}'s and {self._value_to_str(sorted_values[1][0])}'s"
        
        elif sorted_values[0][1] == 2:  # One pair
            rank = HandRank.PAIR
            score = 1000000 + sorted_values[0][0] * 1000
            description = f"Pair of {self._value_to_str(sorted_values[0][0])}'s"
        
        else:  # High card
            rank = HandRank.HIGH_CARD
            high_card = max(values)
            score = high_card * 1000
            description = f"High card, {self._value_to_str(high_card)}"
        
        # Cache the result
        result = {'rank': rank, 'score': score, 'description': description}
        self._eval_cache[hand_key] = result
        
        return result
    
    @lru_cache(maxsize=1024)
    def calculate_hand_equity(self, hole_cards_key, community_cards_key, num_simulations=1000):
        """
        Calculate equity (probability of winning) through Monte Carlo simulation with caching
        
        Args:
            hole_cards_key: Serialized representation of player's hole cards
            community_cards_key: Serialized representation of community cards
            num_simulations: Number of simulations to run
        
        Returns:
            Estimated equity (win probability)
        """
        # Convert keys back to card lists
        hole_cards = self._key_to_cards(hole_cards_key)
        community_cards = self._key_to_cards(community_cards_key)
        
        # In a real implementation, this would run Monte Carlo simulations
        # For simplicity, we'll use a basic equity estimate based on hand strength
        
        # Start with the current hand strength
        current_hand = self.evaluate(hole_cards, community_cards)
        initial_strength = current_hand['score'] / 9000000  # Normalize to [0,1]
        
        # Adjust based on number of community cards and potential for improvement
        num_community = len(community_cards)
        
        if num_community == 0:  # Preflop
            # Use precomputed preflop strength
            preflop_key = self._make_hand_key(hole_cards)
            if preflop_key in self._preflop_strengths:
                initial_strength = self._preflop_strengths[preflop_key]
            
            # Add noise for simulation
            equity = initial_strength * 0.8 + np.random.random() * 0.2
        
        elif num_community == 3:  # Flop
            # Room for improvement with turn and river
            equity = initial_strength * 0.7 + np.random.random() * 0.3
        
        elif num_community == 4:  # Turn
            # Less room for improvement with just the river
            equity = initial_strength * 0.8 + np.random.random() * 0.2
        
        else:  # River
            # Final hand strength
            equity = initial_strength
        
        return min(0.95, max(0.05, equity))  # Keep between 5% and 95%
    
    def _key_to_cards(self, key):
        """Convert a serialized key back to a list of cards"""
        if not key:
            return []
            
        cards = []
        parts = key.split("+")
        
        for part in parts:
            value = part[0]
            if value == '1':  # Handle 10
                value = '10'
                suit_char = part[2]
            else:
                suit_char = part[1]
            
            # Convert suit character to full suit name
            if suit_char == 'h':
                suit = 'hearts'
            elif suit_char == 'd':
                suit = 'diamonds'
            elif suit_char == 'c':
                suit = 'clubs'
            else:
                suit = 'spades'
            
            cards.append({'value': value, 'suit': suit})
        
        return cards
    
    def _value_to_str(self, value):
        """Convert numeric card value back to string representation"""
        if value <= 10:
            return str(value)
        elif value == 11:
            return 'J'
        elif value == 12:
            return 'Q'
        elif value == 13:
            return 'K'
        else:
            return 'A'


class OptimizedPokerNN(nn.Module):
    """Optimized PyTorch neural network for poker decision making"""
    
    def __init__(self, input_size=11):
        super(OptimizedPokerNN, self).__init__()
        
        # Improved architecture with residual connections
        # Initial layers
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
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Handle single input case
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Skip BatchNorm with batch size of 1 during training
        if x.shape[0] == 1 and self.training:
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
            # Initial layers
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
        
        return action_output, bet_output


class OptimizedFeatureExtractor:
    """Extract features from poker game state for neural network input with improved performance"""
    
    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.logger = logging.getLogger("FeatureExtractor")
        
        # Cache for extracted features
        self._feature_cache = {}
    
    def _make_cache_key(self, game_state, player_position):
        """Create a cache key from game state and player position"""
        try:
            # Create a minimal representation of game state for caching
            key_parts = [
                f"player:{player_position}",
                f"pot:{game_state.get('pot', 0)}"
            ]
            
            # Add community cards
            community_str = []
            for card in game_state.get('community_cards', []):
                community_str.append(f"{card.get('value', '?')}{card.get('suit', '?')[0]}")
            key_parts.append(f"community:{'+'.join(sorted(community_str))}")
            
            # Add player cards
            if player_position in game_state.get('players', {}):
                player_cards = game_state['players'][player_position].get('cards', [])
                player_str = []
                for card in player_cards:
                    player_str.append(f"{card.get('value', '?')}{card.get('suit', '?')[0]}")
                key_parts.append(f"cards:{'+'.join(sorted(player_str))}")
            
            # Add player chips
            if player_position in game_state.get('players', {}):
                key_parts.append(f"chips:{game_state['players'][player_position].get('chips', 0)}")
            
            return "|".join(key_parts)
        except Exception as e:
            self.logger.error(f"Error creating cache key: {str(e)}")
            return f"player:{player_position}|error:{time.time()}"
    
    def extract_features(self, game_state, player_position):
        """
        Extract features from the current game state with caching for better performance
        
        Args:
            game_state: Complete game state dictionary
            player_position: Position of the player to extract features for
        
        Returns:
            Feature array for neural network input
        """
        # Check cache first
        cache_key = self._make_cache_key(game_state, player_position)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        features = []
        
        try:
            # Get player's cards
            if player_position in game_state['players'] and 'cards' in game_state['players'][player_position]:
                player_cards = game_state['players'][player_position]['cards']
            else:
                # If we can't extract features, return a default feature set
                self.logger.warning("No player cards found, using default features")
                return np.zeros(11, dtype=np.float32)
            
            # Get community cards
            community_cards = game_state['community_cards'] if 'community_cards' in game_state else []
            
            # 1. Hand strength features (2 features)
            if community_cards:
                # Create keys for hand evaluation caching
                hole_cards_key = "+".join([
                    f"{card['value']}{card['suit'][0]}" for card in player_cards
                ])
                community_cards_key = "+".join([
                    f"{card['value']}{card['suit'][0]}" for card in community_cards
                ])
                
                # Current hand strength
                hand_result = self.hand_evaluator.evaluate(player_cards, community_cards)
                hand_strength = hand_result['score'] / 9000000  # Normalize to [0,1]
                features.append(hand_strength)
                
                # Hand potential (probability of winning)
                equity = self.hand_evaluator.calculate_hand_equity(
                    hole_cards_key, community_cards_key, num_simulations=100
                )
                features.append(equity)
            else:
                # Preflop hand strength based on card ranks
                card_values = [self.hand_evaluator.card_values[card['value']] for card in player_cards]
                suited = player_cards[0]['suit'] == player_cards[1]['suit']
                
                # Simple preflop hand strength metric
                high_card = max(card_values)
                low_card = min(card_values)
                preflop_strength = (high_card * 13 + low_card) / (14 * 13)
                if suited:
                    preflop_strength += 0.1
                if high_card == low_card:  # Pocket pair
                    preflop_strength += 0.2
                
                features.append(min(1.0, preflop_strength))  # Cap at 1.0
                features.append(min(1.0, preflop_strength))  # Duplicate as placeholder for equity
            
            # 2. Position features (1 feature)
            num_players = len(game_state['players'])
            position_value = player_position / max(1, num_players)  # Normalize to [0,1]
            features.append(position_value)
            
            # 3. Stack-related features (2 features)
            player_stack = game_state['players'][player_position]['chips']
            
            # Average opponent stack
            total_opponent_stack = sum(game_state['players'][p]['chips'] for p in game_state['players'] if p != player_position)
            avg_opponent_stack = total_opponent_stack / max(1, num_players - 1)  # Avoid division by zero
            
            # Normalize stack sizes
            max_stack = max(player_stack, avg_opponent_stack, 1)  # Avoid division by zero
            norm_player_stack = player_stack / max_stack
            norm_opponent_stack = avg_opponent_stack / max_stack
            
            features.append(norm_player_stack)
            features.append(norm_opponent_stack)
            
            # 4. Pot-related features (1 feature)
            pot_size = game_state['pot'] if 'pot' in game_state else 0
            
            # Pot odds (ratio of current pot to the cost of calling)
            call_cost = 20  # Default value, would be extracted from game state
            pot_odds = pot_size / max(1, call_cost)  # Avoid division by zero
            features.append(min(1.0, pot_odds / 10))  # Normalize to [0,1]
            
            # 5. Round features - one-hot encoding for preflop, flop, turn, river (4 features)
            num_community = len(community_cards)
            features.extend([
                1 if num_community == 0 else 0,  # Preflop
                1 if num_community == 3 else 0,  # Flop
                1 if num_community == 4 else 0,  # Turn
                1 if num_community == 5 else 0   # River
            ])
            
            # 6. Single action feature (1 feature)
            features.append(0.5)  # Generic action feature
            
            # Ensure we have exactly 11 features
            if len(features) != 11:
                self.logger.error(f"Feature count mismatch! Expected 11, got {len(features)}")
                # Pad or truncate to 11 features
                if len(features) < 11:
                    features.extend([0.0] * (11 - len(features)))
                else:
                    features = features[:11]
            
            # Convert to numpy array with float32 dtype for better performance with PyTorch
            result = np.array(features, dtype=np.float32)
            
            # Cache the result
            self._feature_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}", exc_info=True)
            # Return zero features as fallback
            return np.zeros(11, dtype=np.float32)


class PokerDataset(Dataset):
    """PyTorch Dataset for Poker Game States with optimized memory usage"""
    
    def __init__(self, features, action_labels, bet_labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.action_labels = torch.tensor(action_labels, dtype=torch.float32)
        self.bet_labels = torch.tensor(bet_labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'action_label': self.action_labels[idx],
            'bet_label': self.bet_labels[idx]
        }


class PokerDataCollector:
    """Collect training data from poker game screenshots with improved organization"""
    
    def __init__(self, output_dir="poker_training_data"):
        self.output_dir = output_dir
        self.logger = logging.getLogger("DataCollector")
        os.makedirs(output_dir, exist_ok=True)
        
        # Track statistics
        self.stats = {
            'samples_collected': 0,
            'last_collection_time': None,
            'action_distribution': {
                'fold': 0,
                'check/call': 0,
                'bet/raise': 0
            }
        }
    
    def add_labeled_game_state(self, game_state, expert_action):
        """
        Add a labeled game state to the training data with improved error handling
        
        Args:
            game_state: Game state dictionary
            expert_action: Expert action dictionary with format
                          {'action': str, 'bet_size_percentage': float}
        """
        try:
            # Validate inputs
            if not isinstance(game_state, dict):
                self.logger.error("Invalid game state: not a dictionary")
                return False
                
            if not isinstance(expert_action, dict) or 'action' not in expert_action:
                self.logger.error("Invalid expert action: missing required fields")
                return False
            
            # Convert action to index
            action_map = {'fold': 0, 'check/call': 1, 'bet/raise': 2}
            action_idx = action_map.get(expert_action['action'], 0)
            
            # Create labeled data entry
            labeled_data = {
                'game_state': game_state,
                'expert_action': {
                    'action_idx': action_idx,
                    'bet_size': expert_action.get('bet_size_percentage', 0.0)
                },
                'timestamp': time.time()
            }
            
            # Save to file
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
            filepath = os.path.join(self.output_dir, f"training_data_{timestamp}.json")
            
            with open(filepath, 'w') as f:
                json.dump(labeled_data, f, indent=2)
            
            # Update statistics
            self.stats['samples_collected'] += 1
            self.stats['last_collection_time'] = time.time()
            self.stats['action_distribution'][expert_action['action']] += 1
            
            self.logger.info(f"Added labeled game state to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding labeled game state: {str(e)}", exc_info=True)
            return False
    
    def load_training_data(self):
        """
        Load all training data from the output directory with improved error handling
        
        Returns:
            Tuple: (game_states, expert_actions)
        """
        game_states = []
        expert_actions = []
        
        try:
            # Check if directory exists
            if not os.path.exists(self.output_dir):
                self.logger.warning(f"Training data directory {self.output_dir} not found")
                return game_states, expert_actions
            
            # Get list of training data files
            files = [f for f in os.listdir(self.output_dir) if f.startswith("training_data_") and f.endswith(".json")]
            
            if not files:
                self.logger.warning(f"No training data files found in {self.output_dir}")
                return game_states, expert_actions
            
            self.logger.info(f"Loading {len(files)} training data files...")
            
            # Process each file
            for filename in files:
                try:
                    filepath = os.path.join(self.output_dir, filename)
                    
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    # Validate data structure
                    if 'game_state' not in data or 'expert_action' not in data:
                        self.logger.warning(f"Invalid data format in {filename}, skipping")
                        continue
                    
                    game_states.append(data['game_state'])
                    expert_actions.append(data['expert_action'])
                    
                except Exception as e:
                    self.logger.error(f"Error loading file {filename}: {str(e)}")
                    continue
            
            self.logger.info(f"Loaded {len(game_states)} training samples")
            
        except Exception as e:
            self.logger.error(f"Error loading training data: {str(e)}", exc_info=True)
        
        return game_states, expert_actions
    
    def get_stats(self):
        """Get statistics about collected data"""
        return self.stats


class OptimizedPokerNeuralNetwork:
    """Neural network for poker decision making using PyTorch with improved performance"""
    
    def __init__(self, model_path=None, device=None):
        self.feature_extractor = OptimizedFeatureExtractor()
        
        # Set device (GPU if available, otherwise CPU)
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger = logging.getLogger("PokerNN")
        self.logger.info(f"Using device: {self.device}")
        
        # Create or load model
        self.model = self._load_or_create_model(model_path)
        
        # Decision cache for repeated queries
        self._decision_cache = {}
    
    def _load_or_create_model(self, model_path):
        """Load a pre-trained network or create a new one with improved error handling"""
        model = OptimizedPokerNN(input_size=11).to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                # Load model with current device
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()  # Set to evaluation mode
                self.logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load model from {model_path}, creating new model: {e}")
        else:
            self.logger.info("Creating new model (no existing model found)")
        
        return model
    
    def _make_decision_key(self, game_state, player_position):
        """Create a key for decision caching"""
        # Create a minimal representation of the game state
        key_parts = [
            f"player:{player_position}",
            f"pot:{game_state.get('pot', 0)}"
        ]
        
        # Add community cards
        community_cards = []
        for card in game_state.get('community_cards', []):
            community_cards.append(f"{card.get('value', '')}{card.get('suit', '')[0]}")
        key_parts.append(f"community:{'+'.join(sorted(community_cards))}")
        
        # Add player cards if available
        if player_position in game_state.get('players', {}):
            player = game_state['players'][player_position]
            if 'cards' in player:
                player_cards = []
                for card in player['cards']:
                    player_cards.append(f"{card.get('value', '')}{card.get('suit', '')[0]}")
                key_parts.append(f"hand:{'+'.join(sorted(player_cards))}")
        
        return "|".join(key_parts)
    
    def predict_action(self, game_state, player_position):
        """
        Predict optimal action for the given game state with improved performance
        
        Args:
            game_state: Current game state dictionary
            player_position: Position of the player to make a decision for
            
        Returns:
            Dictionary with predicted action, confidence, and bet size
        """
        try:
            # # Check decision cache first
            # cache_key = self._make_decision_key(game_state, player_position)
            # if cache_key in self._decision_cache:
            #     self.logger.debug("Using cached decision")
            #     return self._decision_cache[cache_key]
            
            # Extract features
            features = self.feature_extractor.extract_features(game_state, player_position)
            
            if features is None:
                # If we can't extract features, return a default action
                return {
                    'action': 'fold',
                    'confidence': 0.5,
                    'bet_size_percentage': 0.0
                }
            
            # Prepare tensor for model input
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                action_probs, bet_size = self.model(features_tensor)
            
            # Get the most probable action
            action_idx = torch.argmax(action_probs[0]).item()
            actions = ['fold', 'check/call', 'bet/raise']
            action = actions[action_idx]
            
            # Get the confidence in this action
            confidence = action_probs[0][action_idx].item()
            
            # Get the bet size as a percentage of pot
            bet_percentage = bet_size[0].item()
            
            decision = {
                'action': action,
                'confidence': confidence,
                'bet_size_percentage': bet_percentage
            }
            
            return decision
        except Exception as e:
            self.logger.error(f"Error in predict_action: {str(e)}", exc_info=True)
            return {
                'action': 'fold',
                'confidence': 0.5,
                'bet_size_percentage': 0.0
            }
    
    def train(self, game_states, expert_actions, epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train the neural network with improved training loop
        
        Args:
            game_states: List of game state dictionaries
            expert_actions: List of expert action dictionaries
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            
        Returns:
            Dictionary with training history
        """
        try:
            # Extract features from game states
            self.logger.info("Extracting features from game states...")
            features = []
            for i, game_state in enumerate(game_states):
                # Assume player position is always 1 (main player)
                # In a real implementation, this would use the correct player position
                feature_vector = self.feature_extractor.extract_features(game_state, 1)
                features.append(feature_vector)
                
                # Log progress
                if (i+1) % 100 == 0:
                    self.logger.info(f"Processed {i+1}/{len(game_states)} game states")
            
            # Prepare action labels
            action_labels = []
            for action in expert_actions:
                # One-hot encode the action
                action_label = [0, 0, 0]
                action_label[action['action_idx']] = 1
                action_labels.append(action_label)
            
            # Prepare bet size labels
            bet_labels = [[action['bet_size']] for action in expert_actions]
            
            # Create dataset and dataloader
            dataset = PokerDataset(features, action_labels, bet_labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Set model to training mode
            self.model.train()
            
            # Define optimizers and loss functions
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            action_criterion = nn.CrossEntropyLoss()
            bet_criterion = nn.MSELoss()
            
            # Initialize history
            history = {
                'action_loss': [],
                'bet_loss': [],
                'total_loss': [],
                'accuracy': []
            }
            
            # Training loop
            for epoch in range(epochs):
                epoch_action_loss = 0.0
                epoch_bet_loss = 0.0
                epoch_total_loss = 0.0
                correct_predictions = 0
                total_predictions = 0
                
                # Process batches
                for batch in dataloader:
                    # Get batch data
                    batch_features = batch['features'].to(self.device)
                    batch_action_labels = batch['action_label'].to(self.device)
                    batch_bet_labels = batch['bet_label'].to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    action_output, bet_output = self.model(batch_features)
                    
                    # Calculate losses
                    action_loss = action_criterion(action_output, batch_action_labels)
                    bet_loss = bet_criterion(bet_output, batch_bet_labels)
                    
                    # Weighted combined loss (action is more important)
                    total_loss = action_loss * 0.7 + bet_loss * 0.3
                    
                    # Backward pass and optimization
                    total_loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    epoch_action_loss += action_loss.item()
                    epoch_bet_loss += bet_loss.item()
                    epoch_total_loss += total_loss.item()
                    
                    # Calculate accuracy
                    predicted_actions = torch.argmax(action_output, dim=1)
                    true_actions = torch.argmax(batch_action_labels, dim=1)
                    correct_predictions += (predicted_actions == true_actions).sum().item()
                    total_predictions += batch_action_labels.size(0)
                
                # Calculate epoch metrics
                avg_action_loss = epoch_action_loss / len(dataloader)
                avg_bet_loss = epoch_bet_loss / len(dataloader)
                avg_total_loss = epoch_total_loss / len(dataloader)
                accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
                
                # Update history
                history['action_loss'].append(avg_action_loss)
                history['bet_loss'].append(avg_bet_loss)
                history['total_loss'].append(avg_total_loss)
                history['accuracy'].append(accuracy)
                
                # Log progress
                self.logger.info(f"Epoch {epoch+1}/{epochs} - "
                              f"Loss: {avg_total_loss:.4f} "
                              f"(Action: {avg_action_loss:.4f}, Bet: {avg_bet_loss:.4f}) - "
                              f"Accuracy: {accuracy:.4f}")
            
            # Set model back to evaluation mode
            self.model.eval()
            
            # Clear decision cache after training
            self._decision_cache.clear()
            
            self.logger.info("Training completed")
            return history
            
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}", exc_info=True)
            return {'error': str(e)}
    
    def save_model(self, path):
        """Save the model to a file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
            
            # Save the model
            torch.save(self.model.state_dict(), path)
            self.logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    def clear_caches(self):
        """Clear all caches to free memory"""
        self._decision_cache.clear()
        self.feature_extractor._feature_cache.clear()
        self.logger.info("All caches cleared")


# Test function
def test_neural_network():
    """Test the neural network with sample data"""
    print("Testing OptimizedPokerNeuralNetwork...")
    
    # Create a sample game state
    game_state = {
        'community_cards': [
            {'value': 'A', 'suit': 'hearts'},
            {'value': 'K', 'suit': 'hearts'},
            {'value': '7', 'suit': 'clubs'}
        ],
        'players': {
            1: {
                'position': 1,
                'chips': 4980,
                'cards': [
                    {'value': 'Q', 'suit': 'hearts'},
                    {'value': 'J', 'suit': 'hearts'}
                ]
            },
            2: {'position': 2, 'chips': 4980},
            3: {'position': 3, 'chips': 4980}
        },
        'pot': 200
    }
    
    # Create neural network
    nn = OptimizedPokerNeuralNetwork()
    
    # Make some predictions
    start_time = time.time()
    action = nn.predict_action(game_state, 1)
    first_prediction_time = time.time() - start_time
    
    print(f"First prediction time: {first_prediction_time:.4f} seconds")
    print(f"Predicted action: {action['action']} with confidence {action['confidence']:.2f}")
    print(f"Recommended bet size: {action['bet_size_percentage'] * 100:.1f}% of pot")
    
    # Test caching performance with a second prediction
    start_time = time.time()
    action = nn.predict_action(game_state, 1)
    cached_prediction_time = time.time() - start_time
    
    print(f"\nCached prediction time: {cached_prediction_time:.4f} seconds")
    print(f"Speedup factor: {first_prediction_time / max(cached_prediction_time, 0.000001):.1f}x")
    
    # Test hand evaluator
    print("\nTesting HandEvaluator...")
    evaluator = HandEvaluator()
    
    # Evaluate a sample hand
    result = evaluator.evaluate(
        [{'value': 'Q', 'suit': 'hearts'}, {'value': 'J', 'suit': 'hearts'}],
        [{'value': 'A', 'suit': 'hearts'}, {'value': 'K', 'suit': 'hearts'}, {'value': '7', 'suit': 'clubs'}]
    )
    
    print(f"Hand evaluation: {result['description']}")
    
    # Test equity calculation
    equity = evaluator.calculate_hand_equity(
        "Qh+Jh", "Ah+Kh+7c", num_simulations=100
    )
    
    print(f"Hand equity: {equity:.2f}")

if __name__ == "__main__":
    test_neural_network()