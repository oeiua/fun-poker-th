import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import itertools
import random
import os
import json
from enum import Enum
import logging

# Filename: poker_neural_engine_torch.py
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PokerNN")

class HandEvaluator:
    """Evaluates poker hand strength"""
    
    class HandRank(Enum):
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
    
    def __init__(self):
        self.card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                           '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    def evaluate(self, hole_cards, community_cards):
        """
        Evaluate the strength of a poker hand
        
        Args:
            hole_cards: List of card dictionaries, e.g. [{'value': 'A', 'suit': 'hearts'}, ...]
            community_cards: List of card dictionaries
            
        Returns:
            Dictionary with hand rank, score, and description
        """
        # Implementation of evaluate method
        # Simplified for brevity
        return {'rank': self.HandRank.PAIR, 'score': 1000000, 'description': 'Pair of Jacks'}
        
    def calculate_hand_equity(self, hole_cards, community_cards, num_simulations=1000):
        """
        Calculate equity (probability of winning) through Monte Carlo simulation
        
        Args:
            hole_cards: Player's hole cards
            community_cards: Current community cards
            num_simulations: Number of simulations to run
        
        Returns:
            Estimated equity (win probability)
        """
        # Simplified for brevity
        return 0.42  # 42% chance of winning
    
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

# PyTorch Neural Network for Poker Decision Making
class PokerNN(nn.Module):
    def __init__(self, input_size=11):  # CHANGED FROM 12 to 11 to match actual feature vector size
        super(PokerNN, self).__init__()
        
        # Common layers
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(0.2)
        
        # Action head (fold, check/call, bet/raise)
        self.action_head = nn.Linear(32, 3)
        
        # Bet sizing head (percentage of pot)
        self.bet_head = nn.Linear(32, 1)
    
    def forward(self, x):
        # Add batch dimension if needed
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        # Handle BatchNorm with batch size of 1
        if x.shape[0] == 1 and self.training:
            # Skip BatchNorm during training with batch size 1
            x = F.relu(self.fc1(x))
            x = self.drop1(x)
            x = F.relu(self.fc2(x))
            x = self.drop2(x)
            x = F.relu(self.fc3(x))
            x = self.drop3(x)
        else:
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.drop1(x)
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.drop2(x)
            x = F.relu(self.bn3(self.fc3(x)))
            x = self.drop3(x)
        
        # Action probabilities with softmax
        action_output = F.softmax(self.action_head(x), dim=1)
        
        # Bet sizing with sigmoid (to get a value between 0 and 1)
        bet_output = torch.sigmoid(self.bet_head(x))
        
        return action_output, bet_output


class PokerFeatureExtractor:
    """Extract features from poker game state for neural network input"""
    
    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.logger = logging.getLogger("PokerFeatureExtractor")
    
    def extract_features(self, game_state, player_position):
        """
        Extract features from the current game state
        
        Args:
            game_state: Complete game state dictionary
            player_position: Position of the player to extract features for
        
        Returns:
            Feature array for neural network input - EXACTLY 11 FEATURES
        """
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
                # Current hand strength
                hand_result = self.hand_evaluator.evaluate(player_cards, community_cards)
                hand_strength = hand_result['score'] / 9000000  # Normalize to [0,1]
                features.append(hand_strength)
                
                # Hand potential (probability of winning)
                equity = self.hand_evaluator.calculate_hand_equity(player_cards, community_cards, num_simulations=100)
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
            position_value = player_position / num_players  # Normalize to [0,1]
            features.append(position_value)
            
            # 3. Stack-related features (2 features)
            player_stack = game_state['players'][player_position]['chips']
            
            # Average opponent stack
            total_opponent_stack = sum(game_state['players'][p]['chips'] for p in game_state['players'] if p != player_position)
            avg_opponent_stack = total_opponent_stack / (num_players - 1) if num_players > 1 else 0
            
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
            pot_odds = pot_size / call_cost if call_cost > 0 else 10  # Cap at 10 if call is free
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
            
            self.logger.info(f"Extracted {len(features)} features")
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {str(e)}")
            # Return zero features as fallback
            return np.zeros(11, dtype=np.float32)


class PokerNeuralNetwork:
    """Neural network for poker decision making using PyTorch"""
    
    def __init__(self, model_path=None):
        self.feature_extractor = PokerFeatureExtractor()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_or_create_model(model_path)
        self.logger = logging.getLogger("PokerNeuralNetwork")
    
    def _load_or_create_model(self, model_path):
        """Load a pre-trained network or create a new one"""
        model = PokerNN(input_size=11).to(self.device)  # CHANGED FROM 12 to 11
        
        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()  # Set to evaluation mode
                self.logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load model from {model_path}, creating new model: {e}")
        
        return model
    
    def predict_action(self, game_state, player_position):
        """
        Predict optimal action for the given game state
        
        Args:
            game_state: Current game state dictionary
            player_position: Position of the player to make a decision for
            
        Returns:
            Dictionary with predicted action, confidence, and bet size
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(game_state, player_position)
            
            if features is None:
                # If we can't extract features, return a default action
                return {
                    'action': 'fold',
                    'confidence': 0.5,
                    'bet_size_percentage': 0.0
                }
            
            # Print feature shape for debugging
            self.logger.info(f"Feature shape: {features.shape}")
            
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
            
            return {
                'action': action,
                'confidence': confidence,
                'bet_size_percentage': bet_percentage
            }
        except Exception as e:
            self.logger.error(f"Error in predict_action: {str(e)}")
            return {
                'action': 'fold',
                'confidence': 0.5,
                'bet_size_percentage': 0.0
            }


# PyTorch Dataset for Poker Game States
class PokerDataset(Dataset):
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
    """Collect training data from poker game screenshots"""
    
    def __init__(self, output_dir="poker_training_data"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def add_labeled_game_state(self, game_state, expert_action):
        """
        Add a labeled game state to the training data
        
        Args:
            game_state: Game state dictionary
            expert_action: Expert action dictionary with format
                          {'action': str, 'bet_size_percentage': float}
        """
        # Convert action to index
        action_map = {'fold': 0, 'check/call': 1, 'bet/raise': 2}
        action_idx = action_map.get(expert_action['action'], 0)
        
        # Create labeled data entry
        labeled_data = {
            'game_state': game_state,
            'expert_action': {
                'action_idx': action_idx,
                'bet_size': expert_action['bet_size_percentage']
            }
        }
        
        # Save to file
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S_%f")
        filepath = os.path.join(self.output_dir, f"training_data_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(labeled_data, f, indent=2)
    
    def load_training_data(self):
        """
        Load all training data from the output directory
        
        Returns:
            Lists of game states and expert actions
        """
        game_states = []
        expert_actions = []
        
        for filename in os.listdir(self.output_dir):
            if filename.startswith("training_data_") and filename.endswith(".json"):
                filepath = os.path.join(self.output_dir, filename)
                
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                game_states.append(data['game_state'])
                expert_actions.append(data['expert_action'])
        
        return game_states, expert_actions


if __name__ == "__main__":
    # Example usage
    
    # Create hand evaluator
    evaluator = HandEvaluator()
    
    # Example cards
    hole_cards = [
        {'value': 'A', 'suit': 'hearts'},
        {'value': 'K', 'suit': 'hearts'}
    ]
    
    community_cards = [
        {'value': 'Q', 'suit': 'hearts'},
        {'value': 'J', 'suit': 'hearts'},
        {'value': '7', 'suit': 'clubs'}
    ]
    
    # Evaluate hand
    result = evaluator.evaluate(hole_cards, community_cards)
    print(f"Hand evaluation: {result['description']}")
    
    # Calculate equity
    equity = evaluator.calculate_hand_equity(hole_cards, community_cards, num_simulations=100)
    print(f"Hand equity: {equity:.2f}")
    
    # Create neural network
    nn = PokerNeuralNetwork()
    
    # Example game state
    game_state = {
        'players': {
            1: {'position': 1, 'chips': 4980, 'cards': hole_cards},
            2: {'position': 2, 'chips': 4980},
            3: {'position': 3, 'chips': 4980}
        },
        'community_cards': community_cards,
        'pot': 200
    }
    
    # Predict action
    action = nn.predict_action(game_state, 1)
    print(f"Predicted action: {action['action']} with confidence {action['confidence']:.2f}")
    print(f"Recommended bet size: {action['bet_size_percentage'] * 100:.1f}% of pot")