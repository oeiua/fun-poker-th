import os
import logging
import torch
import numpy as np
import rlcard
from collections import defaultdict

# Import our neural network components
from poker_neural_network import PokerNN, PokerFeatureExtractor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PokerNNIntegration")

class RLCardAdapter:
    """Adapter to integrate rlcard-trained neural network with the existing poker assistant"""
    
    def __init__(self, model_path="models/poker_model.pt", device=None):
        self.feature_extractor = PokerFeatureExtractor()
        
        # Set device (GPU if available, otherwise CPU)
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Create or load model
        self.model = self._load_or_create_model(model_path)
        
        # For tracking decision statistics
        self.decision_stats = defaultdict(int)
        
        # Card value and suit mappings
        self.card_values_mapping = {
            '2': '2', '3': '3', '4': '4', '5': '5', '6': '6',
            '7': '7', '8': '8', '9': '9', '10': 'T', 'J': 'J', 
            'Q': 'Q', 'K': 'K', 'A': 'A'
        }
        
        self.card_suits_mapping = {
            'hearts': 'h', 'diamonds': 'd', 'clubs': 'c', 'spades': 's'
        }
    
    def _load_or_create_model(self, model_path):
        """Load a pre-trained network or create a new one"""
        model = PokerNN(input_size=17).to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                # Load model with current device
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()  # Set to evaluation mode
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {model_path}, creating new model: {str(e)}")
        else:
            logger.info("Creating new model (no existing model found)")
        
        return model
    
    def predict_action(self, game_state, player_position):
        """
        Predict optimal action for the given game state - compatible with the existing system
        
        Args:
            game_state: Current game state dictionary from the existing system
            player_position: Position of the player to make a decision for
            
        Returns:
            Dictionary with predicted action, confidence, and bet size
        """
        try:
            # Convert existing game state to rlcard format
            rlcard_state = self._convert_to_rlcard_state(game_state, player_position)
            
            # Extract features
            features = self.feature_extractor.extract_features(rlcard_state)
            
            # Prepare tensor for model input
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # Model prediction
            self.model.eval()
            with torch.no_grad():
                action_probs, bet_size, confidence = self.model(features_tensor)
            
            # Get the most probable action
            action_idx = torch.argmax(action_probs[0]).item()
            
            # Map index to action
            actions = ['fold', 'check/call', 'bet/raise']
            action = actions[min(action_idx, 2)]  # Ensure valid index
            
            # Get confidence score
            confidence_score = confidence[0].item()
            
            # Get bet size as percentage of pot
            bet_percentage = bet_size[0].item()
            
            # Update statistics
            self.decision_stats[action] += 1
            
            return {
                'action': action,
                'confidence': confidence_score,
                'bet_size_percentage': bet_percentage
            }
            
        except Exception as e:
            logger.error(f"Error predicting action: {str(e)}")
            # Return a safe default action
            return {
                'action': 'fold',
                'confidence': 0.5,
                'bet_size_percentage': 0.0
            }
    
    def _convert_to_rlcard_state(self, game_state, player_position):
        """Convert the existing game state to rlcard format"""
        try:
            # Extract state information
            community_cards = game_state.get('community_cards', [])
            players = game_state.get('players', {})
            pot = game_state.get('pot', 0)
            game_stage = game_state.get('game_stage', 'preflop')
            available_actions = game_state.get('available_actions', {})
            
            # Convert player cards to rlcard format
            player_cards = []
            if player_position in players and 'cards' in players[player_position]:
                for card in players[player_position]['cards']:
                    value = card.get('value', '')
                    suit = card.get('suit', '')
                    
                    # Map value and suit to rlcard format
                    rlcard_value = self.card_values_mapping.get(value, value)
                    rlcard_suit = self.card_suits_mapping.get(suit, suit[0].lower())
                    
                    player_cards.append(f"{rlcard_value}{rlcard_suit}")
            
            # Convert community cards to rlcard format
            rlcard_community_cards = []
            for card in community_cards:
                value = card.get('value', '')
                suit = card.get('suit', '')
                
                # Map value and suit to rlcard format
                rlcard_value = self.card_values_mapping.get(value, value)
                rlcard_suit = self.card_suits_mapping.get(suit, suit[0].lower())
                
                rlcard_community_cards.append(f"{rlcard_value}{rlcard_suit}")
            
            # Get player chips and positions
            all_chips = []
            for pos in sorted(players.keys()):
                all_chips.append(players[pos].get('chips', 0))
            
            # Determine legal actions
            legal_actions = []
            if available_actions.get('fold', False):
                legal_actions.append('fold')
            
            if available_actions.get('call', False):
                legal_actions.append('call')
            
            if available_actions.get('raise', False):
                legal_actions.append('raise')
            
            # Fall back to all actions if none specified
            if not legal_actions:
                legal_actions = ['fold', 'call', 'raise']
            
            # Create minimal rlcard state
            rlcard_state = {
                'raw_obs': {
                    'hand': player_cards,
                    'public_cards': rlcard_community_cards,
                    'all_chips': all_chips,
                    'current_player': player_position,
                    'pot': pot,
                    'legal_actions': legal_actions,
                    'current_round': game_stage
                },
                'legal_actions': legal_actions
            }
            
            return rlcard_state
            
        except Exception as e:
            logger.error(f"Error converting to rlcard state: {str(e)}")
            # Return a minimal valid state
            return {
                'raw_obs': {
                    'hand': [],
                    'public_cards': [],
                    'all_chips': [1000, 1000],
                    'current_player': 0,
                    'pot': 0,
                    'legal_actions': ['fold', 'call', 'raise'],
                    'current_round': 'preflop'
                },
                'legal_actions': ['fold', 'call', 'raise']
            }
    
    def get_stats(self):
        """Get decision statistics"""
        total = sum(self.decision_stats.values()) or 1  # Avoid division by zero
        
        return {
            'total_decisions': total,
            'fold_percentage': self.decision_stats['fold'] / total * 100,
            'check_call_percentage': self.decision_stats['check/call'] / total * 100,
            'bet_raise_percentage': self.decision_stats['bet/raise'] / total * 100,
            'decision_counts': dict(self.decision_stats)
        }
    
    def clear_stats(self):
        """Clear decision statistics"""
        self.decision_stats.clear()


# Example usage
def test_integration(model_path=None):
    """Test the integration with a sample game state"""
    # Create the adapter
    adapter = RLCardAdapter(model_path=model_path)
    
    # Create a sample game state based on the existing system's format
    sample_game_state = {
        'game_stage': 'flop',
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
                ],
                'current_bet': 20
            },
            2: {'position': 2, 'chips': 4980},
            3: {'position': 3, 'chips': 4980}
        },
        'pot': 200,
        'available_actions': {
            'fold': True,
            'call': True,
            'raise': True
        }
    }
    
    # Make a prediction
    player_position = 1
    decision = adapter.predict_action(sample_game_state, player_position)
    
    # Print the decision
    print(f"Decision for the sample game state:")
    print(f"Action: {decision['action']}")
    print(f"Confidence: {decision['confidence']:.2f}")
    print(f"Bet size: {decision['bet_size_percentage'] * 100:.1f}% of pot")
    
    return adapter, decision


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test the RLCard neural network integration')
    parser.add_argument('--model', type=str, default=None, help='Path to the model file')
    args = parser.parse_args()
    
    adapter, decision = test_integration(model_path=args.model)