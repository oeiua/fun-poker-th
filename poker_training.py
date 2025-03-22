import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed
import logging
import time
import traceback
import re
import argparse
from collections import deque, OrderedDict, namedtuple
import copy
import math

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ImprovedPokerTraining")

# Experience tuple for prioritized replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'td_error'])

class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer with importance sampling for more efficient learning"""
    
    def __init__(self, capacity=100000, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization to use (0 = none, 1 = full)
        self.beta_start = beta_start  # Start value of beta for importance sampling
        self.beta_frames = beta_frames  # Frames over which to anneal beta to 1
        self.frame = 1  # Current frame (used for beta annealing)
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done, td_error=None):
        """Add experience to buffer with priority based on TD error"""
        # Default priority is max priority if TD error is not provided
        if td_error is None:
            priority = max(0.00001, self.priorities.max()) if self.size > 0 else 1.0
        else:
            priority = max(0.00001, abs(td_error))  # Ensure positive priority
        
        # Create experience
        experience = Experience(state, action, reward, next_state, done, td_error)
        
        if self.size < self.capacity:
            self.buffer.append(experience)
            self.size += 1
        else:
            self.buffer[self.position] = experience
        
        # Update priority
        self.priorities[self.position] = priority
        
        # Update position
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        """Sample batch of experiences with importance sampling"""
        if self.size == 0:
            return [], [], [], [], [], []
        
        # Get current beta for importance sampling
        beta = min(1.0, self.beta_start + (self.frame / self.beta_frames) * (1.0 - self.beta_start))
        self.frame += 1
        
        # Calculate sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample experiences
        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False, p=probabilities)
        
        # Calculate importance sampling weights
        weights = (self.size * probabilities[indices]) ** -beta
        weights /= weights.max()  # Normalize
        
        # Extract sampled experiences
        states = [self.buffer[idx].state for idx in indices]
        actions = [self.buffer[idx].action for idx in indices]
        rewards = [self.buffer[idx].reward for idx in indices]
        next_states = [self.buffer[idx].next_state for idx in indices]
        dones = [self.buffer[idx].done for idx in indices]
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """Update priorities based on new TD errors"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = max(0.00001, abs(td_error))
    
    def __len__(self):
        return self.size


class ImprovedPokerNN(nn.Module):
    """Enhanced neural network model for poker decision making with advanced architecture"""
    
    def __init__(self, input_size=17, hidden_size=128, dropout_rate=0.3):
        super(ImprovedPokerNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Initial layer with layer normalization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        
        # Enhanced residual blocks with pre-activation
        # Block 1
        self.res1_ln1 = nn.LayerNorm(hidden_size)
        self.res1_fc1 = nn.Linear(hidden_size, hidden_size)
        self.res1_ln2 = nn.LayerNorm(hidden_size)
        self.res1_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Block 2
        self.res2_ln1 = nn.LayerNorm(hidden_size)
        self.res2_fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.res2_ln2 = nn.LayerNorm(hidden_size // 2)
        self.res2_fc2 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.res2_downsample = nn.Linear(hidden_size, hidden_size // 2)
        
        # Advanced attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size // 2, num_heads=4, batch_first=True)
        
        # Final layers with layer normalization
        self.fc_final = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.ln_final = nn.LayerNorm(hidden_size // 4)
        
        # Action head (fold, check/call, bet/raise) with dueling network architecture
        self.action_value = nn.Linear(hidden_size // 4, 1)  # Value stream
        self.action_advantage = nn.Linear(hidden_size // 4, 3)  # Advantage stream
        
        # Bet sizing head with distributional prediction
        self.bet_head = nn.Linear(hidden_size // 4, 10)  # 10 quantiles of bet size
        
        # Auxiliary heads for better learning
        self.value_head = nn.Linear(hidden_size // 4, 1)  # State value for baseline
        self.confidence_head = nn.Linear(hidden_size // 4, 1)  # Prediction confidence
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights using orthogonal initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.414)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass with improved architecture"""
        # Handle single input
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        if x.shape[0] == 1 and self.training:
            # Skip batch normalization during training with batch size 1
            initial_x = x
            
            # Initial layer
            x = F.relu(self.fc1(x))
            
            # Residual block 1 with pre-activation
            identity = x
            x = F.relu(self.res1_fc1(x))
            x = self.dropout(x)
            x = self.res1_fc2(x)
            x = x + identity
            x = F.relu(x)
            
            # Residual block 2 with pre-activation
            identity = self.res2_downsample(x)
            x = F.relu(self.res2_fc1(x))
            x = self.dropout(x)
            x = self.res2_fc2(x)
            x = x + identity
            x = F.relu(x)
            
            # No attention for single sample
            
            # Final layer
            x = F.relu(self.fc_final(x))
            x = self.dropout(x)
        else:
            # Full forward pass with batch normalization for batch size > 1
            # Input normalization
            x = self.input_norm(x)
            
            # Initial layer
            x = F.relu(self.ln1(self.fc1(x)))
            
            # Residual block 1 with pre-activation
            identity = x
            x = self.res1_ln1(x)
            x = F.relu(self.res1_fc1(x))
            x = self.dropout(x)
            x = self.res1_ln2(x)
            x = self.res1_fc2(x)
            x = x + identity
            x = F.relu(x)
            
            # Residual block 2 with pre-activation
            identity = self.res2_downsample(x)
            x = self.res2_ln1(x)
            x = F.relu(self.res2_fc1(x))
            x = self.dropout(x)
            x = self.res2_ln2(x)
            x = self.res2_fc2(x)
            x = x + identity
            
            # Attention mechanism for capturing card relationships
            if x.shape[0] > 1:  # Apply attention only for batch size > 1
                x_reshaped = x.unsqueeze(1)  # Add sequence dimension
                attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
                x = x + attn_output.squeeze(1)  # Residual connection
            
            # Final layer
            x = F.relu(self.ln_final(self.fc_final(x)))
            x = self.dropout(x)
        
        # Dueling network architecture for action selection
        value = self.action_value(x)
        advantage = self.action_advantage(x)
        
        # Q-values = value + (advantage - mean(advantage))
        action_output = value + advantage - advantage.mean(dim=1, keepdim=True)
        action_probs = F.softmax(action_output, dim=1)
        
        # Distributional bet sizing (10 quantiles)
        bet_output = F.softmax(self.bet_head(x), dim=1)
        
        # Calculate effective bet size as weighted average of quantiles
        bet_quantiles = torch.linspace(0.1, 1.0, 10, device=x.device).view(1, -1)
        bet_size = torch.sum(bet_output * bet_quantiles, dim=1, keepdim=True)
        
        # Value and confidence heads
        value_output = self.value_head(x)
        confidence_output = torch.sigmoid(self.confidence_head(x))
        
        return action_probs, bet_size, value_output, confidence_output


class EnhancedFeatureExtractor:
    """Improved feature extraction with more information about poker game state"""
    
    def __init__(self):
        # Card values for mapping
        self.card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                           '9': 9, '10': 10, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        # Precomputed hand strength mappings with detailed statistics
        self._initialize_hand_strength_data()
        
        # Feature normalization statistics for standardization
        self.feature_mean = None
        self.feature_std = None
        
        self.logger = logging.getLogger("FeatureExtractor")
    
    def _initialize_hand_strength_data(self):
        """Initialize detailed hand strength and equity data"""
        # Load precomputed strength data from file if available
        try:
            data_file = "hand_strength_data.npz"
            if os.path.exists(data_file):
                data = np.load(data_file)
                self.preflop_strength = data['preflop_strength'].item()
                self.preflop_equity = data['preflop_equity'].item()
                logger.info("Loaded hand strength data from file")
                return
        except Exception as e:
            logger.warning(f"Could not load hand strength data: {str(e)}")
        
        # Fallback to basic calculations if file not available
        self.preflop_strength = self._initialize_basic_strength()
        self.preflop_equity = self._initialize_basic_equity()
    
    def _initialize_basic_strength(self):
        """Initialize basic hand strength map as fallback"""
        strength_map = {}
        
        # Pocket pairs
        for value in self.card_values:
            val = self.card_values[value]
            # Scale from 0.5 (22) to 1.0 (AA)
            strength = 0.5 + ((val - 2) / 24)
            key = f"{value}-{value}"
            strength_map[key] = strength
        
        # Suited and unsuited cards
        for v1 in self.card_values:
            for v2 in self.card_values:
                if v1 != v2:
                    val1 = self.card_values[v1]
                    val2 = self.card_values[v2]
                    
                    # Base strength from card values
                    base = (val1 + val2) / 28  # Simple normalization
                    
                    # Connected bonus
                    connected = 0.1 if abs(val1 - val2) == 1 else 0
                    
                    # Suited bonus
                    suited = 0.1
                    
                    # Combined strength (suited)
                    key_suited = f"{v1}s-{v2}s"
                    strength_map[key_suited] = min(0.95, base + connected + suited)
                    
                    # Combined strength (offsuit)
                    key_offsuit = f"{v1}o-{v2}o"
                    strength_map[key_offsuit] = min(0.85, base + connected)
        
        return strength_map
    
    def _initialize_basic_equity(self):
        """Initialize basic hand equity map as fallback"""
        # Simple approximation of equity (probability of winning)
        # In a real implementation, this would be from simulation results
        equity_map = {}
        
        # Copy from strength with adjustments
        for hand, strength in self.preflop_strength.items():
            # Equity is usually lower than raw strength
            # More variance in equity based on hand type
            if "-" not in hand:  # Not a pair
                equity = strength * 0.8  # More penalty for non-pairs
            else:
                equity = strength * 0.9  # Less penalty for pairs
            equity_map[hand] = max(0.05, min(0.95, equity))
        
        return equity_map
    
    def update_normalization_stats(self, features_batch):
        """Update feature normalization statistics based on batch data"""
        features_array = np.array(features_batch)
        if self.feature_mean is None:
            # Initialize with first batch
            self.feature_mean = np.mean(features_array, axis=0)
            self.feature_std = np.std(features_array, axis=0) + 1e-8  # Avoid division by zero
        else:
            # Exponential moving average for ongoing updates
            alpha = 0.01  # Update rate
            new_mean = np.mean(features_array, axis=0)
            new_std = np.std(features_array, axis=0) + 1e-8
            
            self.feature_mean = (1 - alpha) * self.feature_mean + alpha * new_mean
            self.feature_std = (1 - alpha) * self.feature_std + alpha * new_std
    
    def normalize_features(self, features):
        """Normalize features using stored statistics"""
        if self.feature_mean is None:
            return features  # No normalization if stats not available
        
        features_np = np.array(features)
        normalized = (features_np - self.feature_mean) / self.feature_std
        return normalized
    
    def extract_features(self, state):
        """Extract enhanced features from game state"""
        try:
            # Get raw observation
            obs = state.get('raw_obs', {})
            
            # Initialize feature vector
            features = []
            
            # 1. Card features (4 features - expanded from 2)
            hole_cards = obs.get('hand', [])
            community_cards = obs.get('public_cards', [])
            
            # Extract hand strength (absolute power of the hand)
            hand_strength = self._estimate_enhanced_hand_strength(hole_cards, community_cards)
            features.append(hand_strength)
            
            # Extract equity (probability of winning against random hands)
            equity = self._estimate_equity(hole_cards, community_cards)
            features.append(equity)
            
            # Extract drawing potential (chances of improving the hand)
            drawing_potential = self._estimate_drawing_potential(hole_cards, community_cards)
            features.append(drawing_potential)
            
            # Extract hand vulnerability (how likely the hand is to be outdrawn)
            vulnerability = self._estimate_vulnerability(hole_cards, community_cards)
            features.append(vulnerability)
            
            # 2. Position and player features (2 features)
            position = obs.get('current_player', 0)
            num_players = len(obs.get('all_chips', [2]))
            
            # Relative position (distance from button, normalized)
            relative_position = float(position) / max(1, num_players - 1)
            features.append(relative_position)
            
            # Player count factor - strategy adjusts based on number of players
            player_count_factor = min(1.0, num_players / 9.0)  # Normalize to [0,1]
            features.append(player_count_factor)
            
            # 3. Stack features (3 features - expanded from 2)
            my_chips = obs.get('my_chips', 0)
            all_chips = obs.get('all_chips', [])
            
            # Calculate average opponent chips
            other_chips = [c for i, c in enumerate(all_chips) if i != position]
            avg_opp_chips = sum(other_chips) / max(1, len(other_chips))
            
            # Calculate stack-to-pot ratio (SPR)
            pot = float(obs.get('pot', 1))
            stack_to_pot = min(10.0, my_chips / max(1, pot)) / 10.0  # Normalize to [0,1]
            
            # Normalize my chips and opponent chips
            max_chips = max(my_chips, avg_opp_chips, 1)
            norm_my_chips = float(my_chips) / max_chips
            norm_opp_chips = float(avg_opp_chips) / max_chips
            
            features.append(norm_my_chips)
            features.append(norm_opp_chips)
            features.append(stack_to_pot)
            
            # 4. Betting features (4 features)
            # Pot odds
            call_amount = self._extract_call_amount(obs)
            pot_odds = min(1.0, call_amount / max(pot + call_amount, 1))
            features.append(pot_odds)
            
            # Aggression metrics
            num_raises = self._count_raises(obs)
            normalized_raises = min(1.0, num_raises / 5.0)  # Normalize to [0,1]
            features.append(normalized_raises)
            
            # Bet sizing relative to pot
            bet_to_pot = min(1.0, call_amount / max(1, pot))
            features.append(bet_to_pot)
            
            # Commitment ratio (chips invested / total chips)
            commitment = self._calculate_commitment(obs)
            features.append(commitment)
            
            # 5. Round features (4 features)
            stage = obs.get('stage', 0)
            if hasattr(stage, 'value'):
                stage_idx = stage.value
            else:
                stage_idx = int(stage)
            
            round_features = [0.0, 0.0, 0.0, 0.0]
            if 0 <= stage_idx < 4:
                round_features[stage_idx] = 1.0
            features.extend(round_features)
            
            # 6. Legal actions (3 features)
            legal_actions = obs.get('legal_actions', [])
            can_fold = 1.0 if any('FOLD' in str(a) for a in legal_actions) else 0.0
            can_call = 1.0 if any('CHECK_CALL' in str(a) for a in legal_actions) else 0.0
            can_raise = 1.0 if any(('RAISE' in str(a) or 'ALL_IN' in str(a)) for a in legal_actions) else 0.0
            
            features.append(can_fold)
            features.append(can_call)
            features.append(can_raise)
            
            # Ensure we have exactly 20 features
            if len(features) != 20:
                logger.warning(f"Feature count mismatch: {len(features)}, expected 20")
                if len(features) > 20:
                    features = features[:20]
                else:
                    features.extend([0.0] * (20 - len(features)))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros(20, dtype=np.float32)
    
    def _estimate_enhanced_hand_strength(self, hole_cards, community_cards):
        """Enhanced hand strength estimation with more detailed metrics"""
        try:
            if not hole_cards:
                return 0.0
            
            # Preflop strength (based on precomputed tables)
            if not community_cards:
                hand_key = self._create_hand_key(hole_cards)
                return self.preflop_strength.get(hand_key, 0.3)
            
            # Post-flop strength estimation
            # In a complete implementation, this would use a proper hand evaluator
            # Here we use a simplified version based on card values and patterns
            
            # Extract card values and suits
            all_cards = hole_cards + community_cards
            values = [self._get_card_value(card) for card in all_cards]
            suits = [card[0] for card in all_cards]
            
            # Count value frequencies for pairs, trips, quads
            value_counts = {}
            for val in values:
                value_counts[val] = value_counts.get(val, 0) + 1
            
            # Count suit frequencies for flush possibilities
            suit_counts = {}
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            # Check for pairs, trips, quads
            pairs = [v for v, count in value_counts.items() if count == 2]
            trips = [v for v, count in value_counts.items() if count == 3]
            quads = [v for v, count in value_counts.items() if count == 4]
            
            # Check for straight possibilities
            straight_potential = self._check_straight_potential(values)
            
            # Check for flush possibilities
            flush_potential = max(suit_counts.values()) if suit_counts else 0
            
            # Basic hand strength calculation
            strength = 0.0
            
            # High card
            high_card = max(values) if values else 0
            strength += 0.1 + (high_card - 2) * 0.01  # Base strength from high card
            
            # Pairs
            if pairs:
                strength = max(strength, 0.3 + max(pairs) * 0.01)
            
            # Two pair
            if len(pairs) >= 2:
                strength = max(strength, 0.5 + (sum(sorted(pairs)[-2:]) / 20))
            
            # Trips
            if trips:
                strength = max(strength, 0.6 + max(trips) * 0.01)
            
            # Straight
            if straight_potential >= 5:
                strength = max(strength, 0.7)
            
            # Flush
            if flush_potential >= 5:
                strength = max(strength, 0.75)
            
            # Full house
            if trips and pairs:
                strength = max(strength, 0.8 + max(trips) * 0.01)
            
            # Quads
            if quads:
                strength = max(strength, 0.9 + max(quads) * 0.01)
            
            # Straight flush (simplified)
            if straight_potential >= 5 and flush_potential >= 5:
                strength = max(strength, 0.95)
            
            return min(1.0, strength)
            
        except Exception as e:
            logger.error(f"Error estimating hand strength: {str(e)}")
            return 0.2
    
    def _estimate_equity(self, hole_cards, community_cards):
        """Estimate equity (probability of winning) against random hands"""
        try:
            if not hole_cards:
                return 0.0
            
            # Preflop equity from precomputed tables
            if not community_cards:
                hand_key = self._create_hand_key(hole_cards)
                return self.preflop_equity.get(hand_key, 0.3)
            
            # Post-flop equity approximation based on hand strength
            # In a complete implementation, this would use simulation
            hand_strength = self._estimate_enhanced_hand_strength(hole_cards, community_cards)
            
            # Adjust equity based on number of community cards
            if len(community_cards) == 3:  # Flop
                # More uncertainty on the flop
                return 0.3 + (hand_strength * 0.6)
            elif len(community_cards) == 4:  # Turn
                # Less uncertainty on the turn
                return 0.2 + (hand_strength * 0.7)
            else:  # River
                # No more cards to come
                return 0.1 + (hand_strength * 0.8)
            
        except Exception as e:
            logger.error(f"Error estimating equity: {str(e)}")
            return 0.3
    
    def _estimate_drawing_potential(self, hole_cards, community_cards):
        """Estimate potential for hand improvement"""
        try:
            if not hole_cards:
                return 0.0
            
            # No drawing potential on the river
            if len(community_cards) >= 5:
                return 0.0
            
            # Extract card values and suits
            all_cards = hole_cards + community_cards
            values = [self._get_card_value(card) for card in all_cards]
            suits = [card[0] for card in all_cards]
            
            # Count suit frequencies for flush draws
            suit_counts = {}
            for suit in suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            # Check for flush draw (4 cards of the same suit)
            flush_draw = any(count == 4 for count in suit_counts.values())
            
            # Check for straight draw
            straight_draw = self._check_straight_draw(values)
            
            # Check for pair draw (chance to improve to trips)
            value_counts = {}
            for val in values:
                value_counts[val] = value_counts.get(val, 0) + 1
            pair_draw = any(count == 2 for count in value_counts.values())
            
            # Calculate drawing potential
            potential = 0.0
            
            # Straight and flush draws have high potential
            if flush_draw:
                potential += 0.4  # ~19% chance to hit flush on next card
            
            if straight_draw:
                potential += 0.3  # ~17% chance to hit straight on next card
            
            # Pair draws have moderate potential
            if pair_draw:
                potential += 0.2  # ~8% chance to hit trips on next card
            
            # Two overcards have some potential
            hole_values = [self._get_card_value(card) for card in hole_cards]
            community_max = max([self._get_card_value(card) for card in community_cards]) if community_cards else 0
            overcard_potential = 0.1 if all(val > community_max for val in hole_values) else 0.0
            potential += overcard_potential
            
            # Adjust based on remaining streets
            if len(community_cards) == 3:  # Flop (2 cards to come)
                potential *= 1.0  # Full potential
            elif len(community_cards) == 4:  # Turn (1 card to come)
                potential *= 0.6  # Less potential with just one card
            
            return min(0.9, potential)
            
        except Exception as e:
            logger.error(f"Error estimating drawing potential: {str(e)}")
            return 0.1
    
    def _estimate_vulnerability(self, hole_cards, community_cards):
        """Estimate how vulnerable the hand is to being outdrawn"""
        try:
            if not hole_cards or not community_cards:
                return 0.5  # Default middle vulnerability
            
            # Extract card values
            hole_values = [self._get_card_value(card) for card in hole_cards]
            community_values = [self._get_card_value(card) for card in community_cards]
            all_values = hole_values + community_values
            
            # Count value frequencies
            value_counts = {}
            for val in all_values:
                value_counts[val] = value_counts.get(val, 0) + 1
            
            # Extract suits
            hole_suits = [card[0] for card in hole_cards]
            community_suits = [card[0] for card in community_cards]
            all_suits = hole_suits + community_suits
            
            # Count suit frequencies
            suit_counts = {}
            for suit in all_suits:
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
            
            # Check current hand type
            pairs = sum(1 for count in value_counts.values() if count == 2)
            trips = any(count == 3 for count in value_counts.values())
            flush_cards = max(suit_counts.values()) if suit_counts else 0
            straight_potential = self._check_straight_potential(all_values)
            
            # Calculate vulnerability
            vulnerability = 0.5  # Default middle vulnerability
            
            # High card hands are very vulnerable
            if pairs == 0 and trips == False:
                vulnerability = 0.8
            
            # Single pair is vulnerable
            if pairs == 1 and trips == False:
                pair_value = max(val for val, count in value_counts.items() if count == 2)
                # High pairs are less vulnerable
                vulnerability = 0.7 - (pair_value / 28.0)
            
            # Two pair is moderately vulnerable
            if pairs >= 2:
                vulnerability = 0.5
            
            # Trips and better are less vulnerable
            if trips:
                vulnerability = 0.3
            
            # Flush and straight draws on board increase vulnerability
            board_flush_draw = any(count >= 3 for suit, count in suit_counts.items() 
                                if sum(1 for s in community_suits if s == suit) >= 3)
            board_straight_draw = self._check_straight_potential(community_values) >= 4
            
            if board_flush_draw:
                vulnerability += 0.2
            
            if board_straight_draw:
                vulnerability += 0.2
            
            # Rainbow flop reduces vulnerability
            if len(set(community_suits)) == len(community_suits) and len(community_suits) >= 3:
                vulnerability -= 0.1
            
            # Unconnected board reduces vulnerability
            if not self._check_straight_draw(community_values):
                vulnerability -= 0.1
            
            return max(0.1, min(0.9, vulnerability))
            
        except Exception as e:
            logger.error(f"Error estimating vulnerability: {str(e)}")
            return 0.5
    
    def _extract_call_amount(self, obs):
        """Extract the amount needed to call from the state"""
        try:
            # This would depend on the specific structure of the observation
            # For example, in RLCard it might be extracted from the available actions
            legal_actions = obs.get('legal_actions', [])
            for action in legal_actions:
                action_str = str(action)
                if 'CHECK_CALL' in action_str:
                    # Extract call amount from action string, format might vary
                    call_match = re.search(r'CHECK_CALL:(\d+)', action_str)
                    if call_match:
                        return int(call_match.group(1))
                    return 0  # Check (no call required)
            
            # Default to a small value if not found
            return 10
            
        except Exception as e:
            logger.error(f"Error extracting call amount: {str(e)}")
            return 10
    
    def _count_raises(self, obs):
        """Count the number of raises in the current betting round"""
        try:
            # This would depend on the specific structure of the observation
            # For RLCard, we might need to track this externally
            # Return a default value for now
            return 1
            
        except Exception as e:
            logger.error(f"Error counting raises: {str(e)}")
            return 1
    
    def _calculate_commitment(self, obs):
        """Calculate player's commitment ratio"""
        try:
            # Get player's current chips and initial stack
            current_chips = obs.get('my_chips', 1000)
            initial_stack = 1000  # This should ideally be tracked from the game start
            
            # Calculate chips already invested
            invested = initial_stack - current_chips
            
            # Calculate commitment ratio
            commitment = invested / max(1, initial_stack)
            
            return min(1.0, commitment)
            
        except Exception as e:
            logger.error(f"Error calculating commitment: {str(e)}")
            return 0.1
    
    def _check_straight_potential(self, values):
        """Check for straight potential by counting consecutive cards"""
        if not values:
            return 0
        
        unique_values = sorted(set(values))
        max_consecutive = 1
        current_consecutive = 1
        
        for i in range(1, len(unique_values)):
            if unique_values[i] == unique_values[i-1] + 1:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1
        
        return max_consecutive
    
    def _check_straight_draw(self, values):
        """Check if there's a potential straight draw (open-ended or gutshot)"""
        if len(values) < 4:
            return False
        
        # Count distinct values around each possible position
        unique_values = sorted(set(values))
        
        # Open-ended straight draw: 4 consecutive cards
        for i in range(len(unique_values) - 3):
            if (unique_values[i+3] - unique_values[i] == 3):
                return True
        
        # Gutshot straight draw: 4 cards with one gap
        for i in range(len(unique_values) - 3):
            if (unique_values[i+3] - unique_values[i] == 4):
                return True
        
        return False
    
    def _get_card_value(self, card):
        """Get numeric value from card string (e.g., 'HT' -> 10)"""
        rank = card[1] if len(card) == 2 else card[1:3]
        
        # Convert T, J, Q, K, A to numeric values
        if rank == 'T':
            return 10
        elif rank == 'J':
            return 11
        elif rank == 'Q':
            return 12
        elif rank == 'K':
            return 13
        elif rank == 'A':
            return 14
        else:
            try:
                return int(rank)
            except ValueError:
                return 0
    
    def _create_hand_key(self, hole_cards):
        """Create a key for looking up precomputed hand strengths"""
        if len(hole_cards) < 2:
            return ""
        
        # Extract ranks and suits
        ranks = [card[1] if len(card) == 2 else card[1:3] for card in hole_cards]
        suits = [card[0] for card in hole_cards]
        
        # Convert to numeric values for sorting
        values = [self._get_card_value(card) for card in hole_cards]
        cards_with_values = list(zip(values, ranks, suits))
        
        # Sort by value (descending)
        sorted_cards = sorted(cards_with_values, reverse=True)
        
        # Check if suited
        suited = suits[0] == suits[1]
        
        # Create key
        if values[0] == values[1]:  # Pocket pair
            return f"{ranks[0]}-{ranks[1]}"
        else:
            suffix = "s" if suited else "o"
            return f"{sorted_cards[0][1]}{suffix}-{sorted_cards[1][1]}{suffix}"


class PPOAgent:
    """Poker agent implementing Proximal Policy Optimization (PPO) for better sample efficiency"""
    
    def __init__(self, model_path=None, device=None, epsilon=0.1, feature_dim=20):
        # Feature extractor
        self.feature_extractor = EnhancedFeatureExtractor()
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Create or load model
        self.model = self._load_or_create_model(model_path, feature_dim)
        
        # PPO hyperparameters
        self.clip_param = 0.2
        self.ppo_epochs = 4
        self.num_mini_batch = 4
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.lr = 3e-4
        self.eps = 1e-5
        
        # Exploration parameter with decay
        self.epsilon = epsilon
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.01
        
        # RLCard interface properties
        self.use_raw = True  # Use raw observations
        
        # Trajectory memory for PPO updates
        self.trajectories = []
        
        # Statistics tracking
        self.stats = {
            'decisions': 0,
            'actions_taken': {0: 0, 1: 0, 2: 0},  # Fold, Call, Raise
            'avg_value': 0,
            'avg_confidence': 0,
            'training_iterations': 0
        }
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, eps=self.eps)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)
    
    def _load_or_create_model(self, model_path, feature_dim):
        """Load existing model or create new one"""
        model = ImprovedPokerNN(input_size=feature_dim).to(self.device)
        
        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.eval()
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                logger.info("Creating new model")
        else:
            logger.info("Creating new model")
        
        return model
    
    def step(self, state):
        """Take action based on current state with exploration"""
        try:
            self.stats['decisions'] += 1
            
            # Extract legal actions
            if 'raw_legal_actions' in state:
                legal_actions = state['raw_legal_actions']
            elif 'raw_obs' in state and 'legal_actions' in state['raw_obs']:
                legal_actions = state['raw_obs']['legal_actions']
            elif 'legal_actions' in state:
                if isinstance(state['legal_actions'], (dict, OrderedDict)):
                    legal_actions = list(state['legal_actions'].keys())
                else:
                    legal_actions = state['legal_actions']
            else:
                logger.error("No legal actions found in state")
                legal_actions = []
            
            # Exploration with decaying epsilon-greedy
            if random.random() < self.epsilon:
                action = random.choice(legal_actions) if legal_actions else None
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                return action
            
            # Extract features
            features = self.feature_extractor.extract_features(state)
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # Get policy predictions from model
            self.model.eval()
            with torch.no_grad():
                action_probs, bet_size, value, confidence = self.model(features_tensor)
            
            # Update statistics
            self.stats['avg_value'] = (self.stats['avg_value'] * 0.95) + (value.item() * 0.05)
            self.stats['avg_confidence'] = (self.stats['avg_confidence'] * 0.95) + (confidence.item() * 0.05)
            
            # Map model outputs to legal actions
            # Model output: 0=fold, 1=check/call, 2=raise
            action_mapping = {}
            
            for legal_action in legal_actions:
                action_str = str(legal_action)
                
                if "FOLD" in action_str:
                    action_mapping[0] = legal_action
                elif "CHECK_CALL" in action_str:
                    action_mapping[1] = legal_action
                elif "RAISE" in action_str or "ALL_IN" in action_str:
                    action_mapping[2] = legal_action
            
            # Choose best available action
            action_values = action_probs[0].cpu().numpy()
            
            # Apply a mask for unavailable actions
            for i in range(3):
                if i not in action_mapping:
                    action_values[i] = float('-inf')
            
            if all(val == float('-inf') for val in action_values):
                # All actions are masked, use random
                return random.choice(legal_actions) if legal_actions else None
            
            best_action_idx = np.argmax(action_values)
            best_action = action_mapping.get(best_action_idx)
            
            # If no best action found, use random
            if best_action is None:
                best_action = random.choice(legal_actions) if legal_actions else None
            
            # Store chosen action info
            if best_action is not None:
                self.stats['actions_taken'][best_action_idx] = self.stats['actions_taken'].get(best_action_idx, 0) + 1
            
            # Save state and action probs for PPO update
            self.trajectories.append({
                'state': state,
                'features': features,
                'action_probs': action_probs.cpu().detach().numpy(),
                'action': best_action,
                'action_idx': best_action_idx if best_action in action_mapping.values() else None,
                'value': value.item()
            })
            
            return best_action
            
        except Exception as e:
            logger.error(f"Error in step: {str(e)}")
            logger.error(traceback.format_exc())
            return random.choice(legal_actions) if legal_actions else None
    
    def eval_step(self, state):
        """Evaluation step without exploration"""
        # Set epsilon to 0 temporarily
        old_epsilon = self.epsilon
        self.epsilon = 0
        
        action = self.step(state)
        
        # Restore epsilon
        self.epsilon = old_epsilon
        
        return action, {}
    
    def update(self, rewards):
        """Update policy using PPO based on collected trajectories"""
        if len(self.trajectories) == 0:
            return None
        
        try:
            self.stats['training_iterations'] += 1
            
            # Calculate returns and advantages
            returns, advantages = self._compute_returns_and_advantages(rewards)
            
            # Prepare training data
            states = [t['features'] for t in self.trajectories]
            actions = [t['action_idx'] for t in self.trajectories if t['action_idx'] is not None]
            old_action_probs = [t['action_probs'] for t in self.trajectories]
            old_values = [t['value'] for t in self.trajectories]
            
            # Filter out trajectories with None actions
            valid_indices = [i for i, t in enumerate(self.trajectories) if t['action_idx'] is not None]
            if not valid_indices:
                logger.warning("No valid action trajectories for training")
                self.trajectories = []
                return None
            
            # Filter data
            states = [states[i] for i in valid_indices]
            actions = [actions[i] for i in valid_indices]
            old_action_probs = [old_action_probs[i] for i in valid_indices]
            old_values = [old_values[i] for i in valid_indices]
            returns = [returns[i] for i in valid_indices]
            advantages = [advantages[i] for i in valid_indices]
            
            # Convert to tensors
            states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
            actions = torch.tensor(actions, dtype=torch.long).to(self.device)
            old_action_log_probs = torch.tensor(
                [np.log(old_action_probs[i][0][actions[i].item()]) for i in range(len(actions))],
                dtype=torch.float32
            ).to(self.device)
            returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)
            
            # Normalize advantages (only if we have enough samples)
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Mini-batch training
            batch_size = len(states)
            mini_batch_size = max(1, batch_size // self.num_mini_batch)
            
            for _ in range(self.ppo_epochs):
                # Generate random permutation
                permutation = torch.randperm(batch_size)
                
                for start_idx in range(0, batch_size, mini_batch_size):
                    # Get mini-batch indices
                    end_idx = min(start_idx + mini_batch_size, batch_size)
                    batch_indices = permutation[start_idx:end_idx]
                    
                    # Extract mini-batch data
                    mini_batch_states = states[batch_indices]
                    mini_batch_actions = actions[batch_indices]
                    mini_batch_old_action_log_probs = old_action_log_probs[batch_indices]
                    mini_batch_returns = returns[batch_indices]
                    mini_batch_advantages = advantages[batch_indices]
                    
                    # Forward pass
                    self.model.train()
                    action_probs, bet_sizes, values, _ = self.model(mini_batch_states)
                    
                    # Get log probabilities for the selected actions
                    action_log_probs = torch.log(
                        torch.gather(action_probs, 1, mini_batch_actions.unsqueeze(1)).squeeze(1)
                    )
                    
                    # Calculate entropy (for exploration)
                    entropy = -torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1).mean()
                    
                    # Calculate action loss (PPO objective)
                    ratio = torch.exp(action_log_probs - mini_batch_old_action_log_probs)
                    surr1 = ratio * mini_batch_advantages
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * mini_batch_advantages
                    action_loss = -torch.min(surr1, surr2).mean()
                    
                    # Calculate value loss (ensuring matching dimensions)
                    value_loss = F.mse_loss(values.view(-1), mini_batch_returns.view(-1))
                    
                    # Calculate bet sizing loss (simple regression to reward)
                    scaled_returns = (mini_batch_returns - mini_batch_returns.min()) / \
                                     (mini_batch_returns.max() - mini_batch_returns.min() + 1e-8)
                    bet_loss = F.mse_loss(bet_sizes.view(-1), scaled_returns.view(-1))
                    
                    # Total loss
                    loss = action_loss + self.value_loss_coef * value_loss + \
                           0.2 * bet_loss - self.entropy_coef * entropy
                    
                    # Backward pass and optimization
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # Gradient clipping
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()
            
            # Step the learning rate scheduler
            self.scheduler.step()
            
            # Clear trajectories
            self.trajectories = []
            
            return loss.item()
            
        except Exception as e:
            logger.error(f"Error in PPO update: {str(e)}")
            logger.error(traceback.format_exc())
            self.trajectories = []  # Clear to prevent re-using bad data
            return None
    
    def _compute_returns_and_advantages(self, rewards):
        """Compute discounted returns and advantages for PPO"""
        # Parameters
        gamma = 0.99  # Discount factor
        gae_lambda = 0.95  # GAE parameter
        
        # Extract values from trajectories
        values = [t['value'] for t in self.trajectories]
        
        # If rewards is a single value, duplicate it for all transitions
        if isinstance(rewards, (int, float)):
            rewards = [0] * (len(values) - 1) + [rewards]
        
        # Ensure rewards has the right length
        if len(rewards) == 1 and len(values) > 1:
            rewards = [0] * (len(values) - 1) + rewards
        
        # Calculate returns using GAE
        returns = []
        advantages = []
        gae = 0
        
        # For the last step, we don't have a next value
        next_value = 0
        
        # Compute returns and advantages from the end to the beginning
        for i in reversed(range(len(rewards))):
            # The reward is often given at the end, so distribute it
            if i == len(rewards) - 1:
                delta = rewards[i] - values[i]
            else:
                delta = rewards[i] + gamma * values[i+1] - values[i]
            
            gae = delta + gamma * gae_lambda * gae
            advantage = gae
            
            returns.insert(0, advantage + values[i])
            advantages.insert(0, advantage)
        
        return returns, advantages
    
    def save_model(self, path):
        """Save model to file"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.model.state_dict(), path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def get_stats(self):
        """Get training statistics"""
        action_counts = self.stats['actions_taken']
        total_actions = sum(action_counts.values()) or 1  # Avoid division by zero
        
        stats = {
            'decisions': self.stats['decisions'],
            'fold_percentage': action_counts.get(0, 0) / total_actions * 100,
            'call_percentage': action_counts.get(1, 0) / total_actions * 100,
            'raise_percentage': action_counts.get(2, 0) / total_actions * 100,
            'avg_value': self.stats['avg_value'],
            'avg_confidence': self.stats['avg_confidence'],
            'training_iterations': self.stats['training_iterations'],
            'learning_rate': self.scheduler.get_last_lr()[0],
            'exploration_rate': self.epsilon
        }
        
        return stats


class SelfPlayTrainer:
    """Enhanced poker training with self-play, curriculum learning, and opponent pools"""
    
    def __init__(self, args):
        # Configuration
        self.args = args
        self.seed = args.seed
        set_seed(self.seed)
        
        # Environment
        self.env = rlcard.make('no-limit-holdem', config={'seed': self.seed})
        
        # Create main agent with PPO
        self.agent = PPOAgent(
            model_path=args.load_model,
            device=torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'),
            epsilon=args.epsilon,
            feature_dim=20  # Enhanced features
        )
        
        # Create opponent pool
        self.opponent_pool = self._create_opponent_pool()
        
        # Experience buffer
        self.buffer = PrioritizedReplayBuffer(capacity=args.buffer_size)
        
        # Training stats
        self.wins = 0
        self.losses = 0
        self.ties = 0
        self.rewards = []
        self.evaluation_history = []
        
        # ELO ratings for agents
        self.elo_ratings = {'main': 1200}
        
        # Current curriculum level
        self.curriculum_level = 0
        self.curriculum_wins = 0
        self.curriculum_threshold = 100  # Wins needed to advance curriculum
        
        # Best model tracking
        self.best_win_rate = 0.0
        self.episodes_since_improvement = 0
        
        logger.info("Self-play trainer initialized")
    
    def _create_opponent_pool(self):
        """Create a diverse pool of opponents with different strategies"""
        opponents = {}
        
        # Add random agent
        opponents['random'] = RandomAgent(num_actions=self.env.num_actions)
        
        # Add rule-based agents with different strategies
        # These would normally be more sophisticated rule-based agents
        # For simplicity, we're using placeholder implementations
        opponents['tight'] = RandomAgent(num_actions=self.env.num_actions)  # Placeholder
        opponents['loose'] = RandomAgent(num_actions=self.env.num_actions)  # Placeholder
        opponents['aggressive'] = RandomAgent(num_actions=self.env.num_actions)  # Placeholder
        opponents['passive'] = RandomAgent(num_actions=self.env.num_actions)  # Placeholder
        
        # Add previous versions of our own agent (will be populated during training)
        # These are implemented as snapshots at different iterations
        opponents['snapshots'] = {}
        
        return opponents
    
    def _select_opponent(self):
        """Select an opponent based on curriculum level and learning progress"""
        # Select strategy based on curriculum level
        if self.curriculum_level == 0:
            # Beginner level: Play against random opponent
            opponent = self.opponent_pool['random']
            opponent_name = 'random'
        elif self.curriculum_level == 1:
            # Intermediate level: Mix of random and rule-based opponents
            strategies = ['random', 'tight', 'loose']
            opponent_name = random.choice(strategies)
            opponent = self.opponent_pool[opponent_name]
        elif self.curriculum_level == 2:
            # Advanced level: Rule-based and snapshot opponents
            strategies = ['tight', 'loose', 'aggressive', 'passive']
            
            # Include snapshots if available
            if self.opponent_pool['snapshots']:
                strategies.append('snapshot')
            
            opponent_name = random.choice(strategies)
            
            if opponent_name == 'snapshot':
                # Select a snapshot based on ELO ratings (prefer closer ratings for challenge)
                snapshot_keys = list(self.opponent_pool['snapshots'].keys())
                if snapshot_keys:
                    snapshot_key = random.choice(snapshot_keys)
                    opponent = self.opponent_pool['snapshots'][snapshot_key]
                    opponent_name = f"snapshot_{snapshot_key}"
                else:
                    # Fallback to rule-based if no snapshots
                    opponent_name = random.choice(['tight', 'loose', 'aggressive', 'passive'])
                    opponent = self.opponent_pool[opponent_name]
            else:
                opponent = self.opponent_pool[opponent_name]
        else:
            # Expert level: Mostly self-play with occasional rule-based
            if random.random() < 0.8 and self.opponent_pool['snapshots']:
                # Select a snapshot based on ELO
                snapshot_keys = list(self.opponent_pool['snapshots'].keys())
                if snapshot_keys:
                    # Prefer recent snapshots with higher ELO
                    weights = [1.0 + (int(k) / 10000) for k in snapshot_keys]
                    snapshot_key = random.choices(snapshot_keys, weights=weights, k=1)[0]
                    opponent = self.opponent_pool['snapshots'][snapshot_key]
                    opponent_name = f"snapshot_{snapshot_key}"
                else:
                    # Fallback to aggressive if no snapshots
                    opponent = self.opponent_pool['aggressive']
                    opponent_name = 'aggressive'
            else:
                # Occasionally use rule-based for diversity
                opponent_name = random.choice(['tight', 'loose', 'aggressive', 'passive'])
                opponent = self.opponent_pool[opponent_name]
        
        return opponent, opponent_name
    
    def _update_curriculum(self, win):
        """Update curriculum level based on performance"""
        if win:
            self.curriculum_wins += 1
            
            # Check if ready to advance to next level
            if (self.curriculum_wins >= self.curriculum_threshold and 
                self.curriculum_level < 3):
                self.curriculum_level += 1
                self.curriculum_wins = 0
                logger.info(f"Advanced to curriculum level {self.curriculum_level}")
                
                # Take a snapshot when advancing curriculum
                self._take_agent_snapshot()
        
        # Occasionally take a snapshot even if not advancing
        if self.curriculum_wins % 500 == 0 and self.curriculum_wins > 0:
            self._take_agent_snapshot()
    
    def _take_agent_snapshot(self):
        """Take a snapshot of the current agent for the opponent pool"""
        snapshot_id = str(len(self.opponent_pool['snapshots']))
        
        # Create a copy of the current model
        snapshot_model = ImprovedPokerNN(input_size=20).to(self.agent.device)
        snapshot_model.load_state_dict(copy.deepcopy(self.agent.model.state_dict()))
        
        # Create a new agent with the snapshot model
        snapshot_agent = PPOAgent(feature_dim=20)
        snapshot_agent.model = snapshot_model
        snapshot_agent.epsilon = 0.05  # Low exploration for opponent
        
        # Add to opponent pool
        self.opponent_pool['snapshots'][snapshot_id] = snapshot_agent
        
        # Initialize ELO rating (inherit from main agent)
        self.elo_ratings[f"snapshot_{snapshot_id}"] = self.elo_ratings['main']
        
        logger.info(f"Created agent snapshot {snapshot_id} with ELO {self.elo_ratings[f'snapshot_{snapshot_id}']}")
    
    def _update_elo_ratings(self, result, opponent_name):
        """Update ELO ratings based on game result"""
        if opponent_name not in self.elo_ratings:
            # Initialize new opponents with default rating
            self.elo_ratings[opponent_name] = 1200
        
        # K-factor determines how much ratings change
        k_factor = 32
        
        # Get current ratings
        rating_main = self.elo_ratings['main']
        rating_opponent = self.elo_ratings[opponent_name]
        
        # Calculate expected win probabilities
        expected_main = 1 / (1 + 10 ** ((rating_opponent - rating_main) / 400))
        expected_opponent = 1 - expected_main
        
        # Update ratings based on actual result
        if result > 0:  # Main agent won
            actual_main = 1
            actual_opponent = 0
        elif result < 0:  # Opponent won
            actual_main = 0
            actual_opponent = 1
        else:  # Tie
            actual_main = 0.5
            actual_opponent = 0.5
        
        # Calculate new ratings
        new_rating_main = rating_main + k_factor * (actual_main - expected_main)
        new_rating_opponent = rating_opponent + k_factor * (actual_opponent - expected_opponent)
        
        # Update ratings
        self.elo_ratings['main'] = new_rating_main
        self.elo_ratings[opponent_name] = new_rating_opponent
    
    def train(self):
        """Train an agent with enhanced techniques"""
        logger.info(f"Starting training for {self.args.num_episodes} episodes")
        
        # Pre-training setup
        best_model_path = f"{os.path.dirname(self.args.save_model)}/best_model.pt"
        
        for episode in range(self.args.num_episodes):
            try:
                # Select opponent for this episode
                opponent, opponent_name = self._select_opponent()
                
                # Register agents (main agent first)
                self.env.set_agents([self.agent, opponent])
                
                # Run one episode
                trajectories, payoffs = self.env.run(is_training=True)
                
                # Process episode results
                main_payoff = payoffs[0]
                self.rewards.append(main_payoff)
                
                # Update win/loss/tie counters
                if main_payoff > 0:
                    self.wins += 1
                    self._update_curriculum(True)
                elif main_payoff < 0:
                    self.losses += 1
                    self._update_curriculum(False)
                else:
                    self.ties += 1
                
                # Update ELO ratings
                self._update_elo_ratings(main_payoff, opponent_name)
                
                # Process trajectories for PPO update
                player_trajectory = trajectories[0]
                
                # Update agent with PPO
                loss = self.agent.update([main_payoff])
                
                # Periodically evaluate performance
                if (episode + 1) % self.args.eval_every == 0:
                    win_rate, avg_reward = self._evaluate()
                    self.evaluation_history.append((episode + 1, win_rate, avg_reward))
                    
                    # Track best model
                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate
                        self.episodes_since_improvement = 0
                        
                        # Save best model
                        self.agent.save_model(best_model_path)
                        logger.info(f"New best model saved with win rate {win_rate:.3f}")
                    else:
                        self.episodes_since_improvement += self.args.eval_every
                    
                    # Log training progress
                    recent_rewards = self.rewards[-self.args.eval_every:]
                    avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
                    
                    # Get agent stats
                    agent_stats = self.agent.get_stats()
                    
                    logger.info(
                        f"Episode {episode+1}/{self.args.num_episodes}, "
                        f"Win Rate: {win_rate:.3f}, "
                        f"Recent Reward: {avg_recent_reward:.3f}, "
                        f"ELO: {self.elo_ratings['main']:.1f}, "
                        f"Curriculum: {self.curriculum_level}, "
                        f"Fold: {agent_stats['fold_percentage']:.1f}%, "
                        f"Call: {agent_stats['call_percentage']:.1f}%, "
                        f"Raise: {agent_stats['raise_percentage']:.1f}%"
                    )
                    
                    # Save checkpoint
                    if (episode + 1) % (self.args.eval_every * 5) == 0:
                        checkpoint_path = f"{os.path.dirname(self.args.save_model)}/checkpoint_{episode+1}.pt"
                        self.agent.save_model(checkpoint_path)
                
                # Learning rate warm-up and decay
                if episode < 1000:
                    # Warm-up: gradually increase learning rate
                    for param_group in self.agent.optimizer.param_groups:
                        param_group['lr'] = self.agent.lr * min(1.0, episode / 1000)
                
            except Exception as e:
                logger.error(f"Error in episode {episode}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # Save final model
        self.agent.save_model(self.args.save_model)
        logger.info(f"Training completed. Final model saved to {self.args.save_model}")
        
        # Return trained agent and stats
        return self.agent, self.wins / self.args.num_episodes, self.evaluation_history
    
    def _evaluate(self, num_games=200):
        """Evaluate agent performance against RandomAgent"""
        # Create a fresh environment for evaluation
        eval_env = rlcard.make('no-limit-holdem', config={'seed': self.seed + 1000})
        
        # Create a random opponent for evaluation
        eval_opponent = RandomAgent(num_actions=eval_env.num_actions)
        
        # Register agents
        eval_env.set_agents([self.agent, eval_opponent])
        
        # Evaluation stats
        eval_wins = 0
        eval_payoffs = []
        
        # Run evaluation games
        for _ in range(num_games):
            _, payoffs = eval_env.run(is_training=False)
            
            # Track results
            if payoffs[0] > 0:
                eval_wins += 1
            
            eval_payoffs.append(payoffs[0])
        
        # Calculate metrics
        win_rate = eval_wins / num_games
        avg_reward = sum(eval_payoffs) / num_games
        
        return win_rate, avg_reward


def train_with_self_play(args):
    """Train a poker agent with enhanced self-play curriculum"""
    trainer = SelfPlayTrainer(args)
    agent, win_rate, history = trainer.train()
    
    logger.info(f"Training completed")
    logger.info(f"Final win rate against random agent: {win_rate:.3f}")
    logger.info(f"Final ELO rating: {trainer.elo_ratings['main']:.1f}")
    
    # Plot learning curve if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        episodes = [h[0] for h in history]
        win_rates = [h[1] for h in history]
        rewards = [h[2] for h in history]
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(episodes, win_rates, 'b-')
        plt.xlabel('Episodes')
        plt.ylabel('Win Rate')
        plt.title('Training Progress - Win Rate')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(episodes, rewards, 'r-')
        plt.xlabel('Episodes')
        plt.ylabel('Average Reward')
        plt.title('Training Progress - Reward')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        logger.info("Training progress plot saved to training_progress.png")
        
    except ImportError:
        logger.info("Matplotlib not available, skipping learning curve plot")
    
    return agent


def main():
    parser = argparse.ArgumentParser(description='Train poker agent with improved algorithms')
    parser.add_argument('--train', action='store_true', help='Train agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate agent')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model')
    parser.add_argument('--save-model', type=str, default='models/poker_model.pt', help='Path to save model')
    parser.add_argument('--num-episodes', type=int, default=100000, help='Number of training episodes')
    parser.add_argument('--eval-games', type=int, default=1000, help='Number of evaluation games')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=1000000, help='Replay buffer size')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Starting exploration rate')
    parser.add_argument('--eval-every', type=int, default=1000, help='Evaluation frequency')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    # Create model directory if it doesn't exist
    model_dir = os.path.dirname(args.save_model)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    if args.train:
        train_with_self_play(args)
    
    if args.evaluate:
        # Create agent for evaluation
        agent = PPOAgent(
            model_path=args.load_model,
            device=torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'),
            epsilon=0.0,  # No exploration during evaluation
            feature_dim=20
        )
        
        # Create environment
        env = rlcard.make('no-limit-holdem', config={'seed': args.seed})
        
        # Create random opponent
        opponent = RandomAgent(num_actions=env.num_actions)
        
        # Register agents
        env.set_agents([agent, opponent])
        
        # Evaluation stats
        wins = 0
        losses = 0
        ties = 0
        rewards = []
        
        logger.info(f"Evaluating agent for {args.eval_games} games")
        
        # Evaluation loop
        for game in range(args.eval_games):
            try:
                # Run one game
                _, payoffs = env.run(is_training=False)
                
                # Save game result
                rewards.append(payoffs[0])
                if payoffs[0] > 0:
                    wins += 1
                elif payoffs[0] < 0:
                    losses += 1
                else:
                    ties += 1
                
                # Periodically log progress
                if (game + 1) % 100 == 0:
                    logger.info(f"Completed {game+1}/{args.eval_games} games, "
                               f"Win rate so far: {wins/(game+1):.3f}")
                    
            except Exception as e:
                logger.error(f"Error in evaluation game {game}: {str(e)}")
                continue
        
        # Final stats
        win_rate = wins / args.eval_games
        loss_rate = losses / args.eval_games
        tie_rate = ties / args.eval_games
        avg_reward = sum(rewards) / args.eval_games
        
        logger.info(f"Evaluation completed. "
                   f"Win Rate: {win_rate:.3f} ({wins}/{args.eval_games}), "
                   f"Loss Rate: {loss_rate:.3f} ({losses}/{args.eval_games}), "
                   f"Tie Rate: {tie_rate:.3f} ({ties}/{args.eval_games}), "
                   f"Avg Reward: {avg_reward:.3f}")
        
    if not args.train and not args.evaluate:
        parser.print_help()


if __name__ == "__main__":
    main()