import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_seed
import logging
import time
import traceback
from collections import deque, OrderedDict
import copy

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PokerTraining")

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


class FeatureExtractor:
    """Extract features from poker game state for neural network input"""
    
    def __init__(self):
        # Card values for mapping
        self.card_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                           '9': 9, '10': 10, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    def extract_features(self, state):
        """Extract features from the RLCard state"""
        try:
            # Get raw observation
            obs = state.get('raw_obs', {})
            
            # Initialize feature vector
            features = []
            
            # 1. Cards features (2 features)
            hole_cards = obs.get('hand', [])
            community_cards = obs.get('public_cards', [])
            
            # Card strength - simple version using high card and pair detection
            card_strength = self._estimate_hand_strength(hole_cards, community_cards)
            features.append(card_strength)
            
            # Card potential - based on round and suited/connected status
            card_potential = self._estimate_hand_potential(hole_cards, community_cards)
            features.append(card_potential)
            
            # 2. Position feature (1 feature)
            position = obs.get('current_player', 0)
            num_players = len(obs.get('all_chips', [2]))
            position_value = float(position) / max(1, num_players - 1)
            features.append(position_value)
            
            # 3. Stack features (2 features)
            my_chips = obs.get('my_chips', 0)
            all_chips = obs.get('all_chips', [])
            
            # Calculate average opponent chips
            other_chips = [c for i, c in enumerate(all_chips) if i != position]
            avg_opp_chips = sum(other_chips) / max(1, len(other_chips))
            
            # Normalize
            max_chips = max(my_chips, avg_opp_chips, 1)
            norm_my_chips = float(my_chips) / max_chips
            norm_opp_chips = float(avg_opp_chips) / max_chips
            
            features.append(norm_my_chips)
            features.append(norm_opp_chips)
            
            # 4. Pot feature (1 feature)
            pot = float(obs.get('pot', 0))
            pot_to_stack = pot / max(1, my_chips)
            features.append(min(1.0, pot_to_stack))
            
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
            
            # 6. Action features (3 features)
            legal_actions = obs.get('legal_actions', [])
            can_fold = 1.0 if any('FOLD' in str(a) for a in legal_actions) else 0.0
            can_call = 1.0 if any('CHECK_CALL' in str(a) for a in legal_actions) else 0.0
            can_raise = 1.0 if any(('RAISE' in str(a) or 'ALL_IN' in str(a)) for a in legal_actions) else 0.0
            
            features.append(can_fold)
            features.append(can_call)
            features.append(can_raise)
            
            # 7. Betting features (4 features)
            stakes = obs.get('stakes', [0, 0])
            my_stake = stakes[position] if position < len(stakes) else 0
            stake_to_pot = my_stake / max(1, pot)
            
            features.append(min(1.0, stake_to_pot))
            features.extend([0.0, 0.0, 0.0])  # Placeholder for future features
            
            # Return fixed-length feature vector of 17 elements
            if len(features) != 17:
                # Ensure we have exactly 17 features
                if len(features) > 17:
                    features = features[:17]
                else:
                    features.extend([0.0] * (17 - len(features)))
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros(17, dtype=np.float32)
    
    def _estimate_hand_strength(self, hole_cards, community_cards):
        """Simple hand strength estimation"""
        try:
            if not hole_cards:
                return 0.0
                
            # Extract ranks
            hole_ranks = [c[1:] if c[0].isalpha() else c[1:] for c in hole_cards]
            
            # Replace T with 10 for consistency
            hole_ranks = ['10' if r == 'T' else r for r in hole_ranks]
            
            # Convert to values
            values = [self.card_values.get(r, 0) for r in hole_ranks]
            
            # Check for pairs
            if len(values) >= 2 and values[0] == values[1]:
                # Normalize pair strength: 22 = 0.5, AA = 1.0
                pair_value = values[0]
                return 0.5 + (pair_value - 2) * 0.025
            
            # High card
            high_card = max(values) if values else 0
            return 0.1 + (high_card - 2) * 0.03  # 2 = 0.1, A = 0.46
            
        except Exception as e:
            logger.error(f"Error estimating hand strength: {str(e)}")
            return 0.2  # Default moderate-low strength
    
    def _estimate_hand_potential(self, hole_cards, community_cards):
        """Estimate potential for hand improvement"""
        try:
            if not hole_cards:
                return 0.0
                
            # Higher potential with fewer community cards
            if not community_cards:
                base_potential = 0.7  # Preflop
            elif len(community_cards) == 3:
                base_potential = 0.5  # Flop
            elif len(community_cards) == 4:
                base_potential = 0.3  # Turn
            else:
                base_potential = 0.0  # River
            
            # Extract suits and ranks
            hole_suits = [c[0] for c in hole_cards]
            hole_ranks = [c[1:] if c[0].isalpha() else c[1:] for c in hole_cards]
            
            # Replace T with 10 for consistency
            hole_ranks = ['10' if r == 'T' else r for r in hole_ranks]
            
            # Convert to values
            values = [self.card_values.get(r, 0) for r in hole_ranks]
            
            # Suited bonus
            suited_bonus = 0.1 if len(set(hole_suits)) == 1 else 0.0
            
            # Connected bonus (within 3 positions)
            if len(values) >= 2:
                gap = abs(values[0] - values[1])
                if gap == 0:
                    connected_bonus = 0.0  # Already a pair
                elif gap == 1:
                    connected_bonus = 0.1  # Directly connected
                elif gap == 2:
                    connected_bonus = 0.05  # 1-gap
                elif gap == 3:
                    connected_bonus = 0.03  # 2-gap
                else:
                    connected_bonus = 0.0
            else:
                connected_bonus = 0.0
            
            return min(0.95, base_potential + suited_bonus + connected_bonus)
            
        except Exception as e:
            logger.error(f"Error estimating hand potential: {str(e)}")
            return 0.5  # Default moderate potential


class ReplayBuffer:
    """Experience replay buffer for training"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)), replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in indices])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class PokerAgent:
    """Neural network-based poker agent for RLCard"""
    
    def __init__(self, model_path=None, device=None, epsilon=0.1):
        # Feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Set device
        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        
        # Create or load model
        self.model = self._load_or_create_model(model_path)
        
        # Exploration parameter
        self.epsilon = epsilon
        
        # RLCard interface properties
        self.use_raw = True  # Use raw observations
        
        # Debug counter
        self.debug_count = 0
        
        # Statistics
        self.stats = {
            'decisions': 0,
            'wins': 0,
            'losses': 0
        }
    
    def _load_or_create_model(self, model_path):
        """Load existing model or create new one"""
        model = PokerNN(input_size=17).to(self.device)
        
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
        """Take action for the current state"""
        try:
            self.stats['decisions'] += 1
            
            # Extract legal actions
            # First priority: raw_legal_actions
            if 'raw_legal_actions' in state:
                legal_actions = state['raw_legal_actions']
            # Second priority: raw_obs.legal_actions
            elif 'raw_obs' in state and 'legal_actions' in state['raw_obs']:
                legal_actions = state['raw_obs']['legal_actions']
            # Third priority: legal_actions
            elif 'legal_actions' in state:
                if isinstance(state['legal_actions'], (dict, OrderedDict)):
                    legal_actions = list(state['legal_actions'].keys())
                else:
                    legal_actions = state['legal_actions']
            else:
                logger.error("No legal actions found in state")
                legal_actions = []  # Empty list as fallback
            
            # Debugging output (only for the first few states)
            if self.debug_count < 3:
                logger.debug(f"State shape: {state}")
                logger.debug(f"Legal actions: {legal_actions}")
                self.debug_count += 1
            
            # Random action with probability epsilon
            if random.random() < self.epsilon:
                action = random.choice(legal_actions) if legal_actions else None
                logger.debug(f"Random action: {action}")
                return action
            
            # Extract features
            features = self.feature_extractor.extract_features(state)
            
            # Convert to tensor
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # Get action probabilities from model
            self.model.eval()
            with torch.no_grad():
                action_probs, bet_size, confidence = self.model(features_tensor)
            
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
            best_value = -float('inf')
            best_action = None
            
            for model_idx, legal_action in action_mapping.items():
                if model_idx < len(action_values) and action_values[model_idx] > best_value:
                    best_value = action_values[model_idx]
                    best_action = legal_action
            
            # If no best action, use random
            if best_action is None:
                logger.warning("No best action found, using random")
                best_action = random.choice(legal_actions) if legal_actions else None
            
            logger.debug(f"Chosen action: {best_action}")
            return best_action
            
        except Exception as e:
            logger.error(f"Error in step: {str(e)}")
            logger.error(traceback.format_exc())
            # Fallback to random legal action
            try:
                return random.choice(legal_actions) if legal_actions else None
            except Exception:
                logger.error("Could not even choose random action!")
                return None
    
    def eval_step(self, state):
        """Evaluation step - same as step but returns info"""
        action = self.step(state)
        return action, {}  # No additional info
    
    def learn(self, batch):
        """Update model from batch of experiences"""
        if not batch or len(batch[0]) == 0:
            return None
            
        states, actions, rewards, next_states, dones = batch
        
        try:
            # Filter out state-action pairs where action is None
            valid_indices = []
            for i, action in enumerate(actions):
                if action is not None:
                    valid_indices.append(i)
            
            if not valid_indices:
                # If no valid transitions, return None
                return None
                
            # Filter the batch to only include valid transitions
            filtered_states = [states[i] for i in valid_indices]
            filtered_actions = [actions[i] for i in valid_indices]
            filtered_rewards = [rewards[i] for i in valid_indices]
            filtered_next_states = [next_states[i] for i in valid_indices]
            filtered_dones = [dones[i] for i in valid_indices]
            
            # Extract features from states
            state_features = []
            for state in filtered_states:
                features = self.feature_extractor.extract_features(state)
                state_features.append(features)
            
            # Convert to tensors
            state_tensor = torch.tensor(np.array(state_features), dtype=torch.float32).to(self.device)
            
            # Convert actions to indices
            action_indices = []
            for action in filtered_actions:
                # Handle different action formats
                action_str = str(action)
                if "FOLD" in action_str:
                    idx = 0
                elif "CHECK_CALL" in action_str:
                    idx = 1
                else:  # RAISE or ALL_IN
                    idx = 2
                action_indices.append(idx)
            
            action_tensor = torch.tensor(action_indices, dtype=torch.long).to(self.device)
            reward_tensor = torch.tensor(filtered_rewards, dtype=torch.float32).to(self.device)
            
            # Create one-hot action targets
            batch_size = len(filtered_states)
            action_targets = torch.zeros((batch_size, 3), dtype=torch.float32).to(self.device)
            for i, idx in enumerate(action_indices):
                action_targets[i, idx] = 1.0
            
            # Create optimizer
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Forward pass
            self.model.train()
            action_probs, bet_sizes, confidences = self.model(state_tensor)
            
            # Action loss
            action_criterion = nn.CrossEntropyLoss()
            action_loss = action_criterion(action_probs, action_targets)
            
            # Bet sizing loss
            bet_criterion = nn.MSELoss()
            
            # Normalize rewards to [0,1] for bet sizing
            if torch.max(reward_tensor) > torch.min(reward_tensor):
                norm_rewards = (reward_tensor - torch.min(reward_tensor)) / (torch.max(reward_tensor) - torch.min(reward_tensor))
            else:
                norm_rewards = torch.zeros_like(reward_tensor)
                
            bet_loss = bet_criterion(bet_sizes.squeeze(), norm_rewards)
            
            # Confidence loss - confidence should match correctness
            pred_actions = torch.argmax(action_probs, dim=1)
            correct_preds = (pred_actions == torch.argmax(action_targets, dim=1)).float()
            confidence_loss = bet_criterion(confidences.squeeze(), correct_preds)
            
            # Combined loss
            total_loss = action_loss + 0.3 * bet_loss + 0.3 * confidence_loss
            
            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            return total_loss.item()
            
        except Exception as e:
            logger.error(f"Learning error: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
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


def train(args):
    """Train a poker agent with RLCard"""
    # Set random seed
    set_seed(args.seed)
    
    # Create environment
    env = rlcard.make('no-limit-holdem', config={'seed': args.seed})
    
    # Create agent
    agent = PokerAgent(
        model_path=args.load_model,
        device=torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'),
        epsilon=args.epsilon
    )
    
    # Create opponent
    opponent = RandomAgent(num_actions=env.num_actions)
    
    # Register agents
    env.set_agents([agent, opponent])
    
    # Create replay buffer
    buffer = ReplayBuffer(capacity=args.buffer_size)
    
    # Training stats
    rewards = []
    wins = 0
    losses = 0
    ties = 0
    
    logger.info(f"Starting training for {args.num_episodes} episodes")
    
    # Training loop
    for episode in range(args.num_episodes):
        try:
            # Run one episode
            trajectories, payoffs = env.run(is_training=True)
            
            # Save episode result
            rewards.append(payoffs[0])
            if payoffs[0] > 0:
                wins += 1
            elif payoffs[0] < 0:
                losses += 1
            else:
                ties += 1
            
            # Process the trajectory for our agent (index 0)
            player_trajectory = trajectories[0]
            
            # Debug the trajectory format in the first episode
            if episode == 0:
                logger.debug(f"Number of trajectory steps: {len(player_trajectory)}")
                if player_trajectory:
                    logger.debug(f"First step type: {type(player_trajectory[0])}")
                    logger.debug(f"First step content: {player_trajectory[0]}")
            
            # Process each state in the trajectory
            prev_state = None
            
            for i, step in enumerate(player_trajectory):
                try:
                    # Check if this is a state
                    if isinstance(step, dict) and 'raw_obs' in step:
                        current_state = step
                        
                        # If we have a previous state and an action in between
                        if prev_state is not None and i > 0:
                            # Get the action (should be between prev_state and current_state)
                            if isinstance(player_trajectory[i-1], (int, float)) or hasattr(player_trajectory[i-1], 'value'):
                                action = player_trajectory[i-1]
                                # Add this transition to the replay buffer
                                buffer.add(prev_state, action, payoffs[0], current_state, False)
                        
                        # Update previous state
                        prev_state = current_state
                    
                    # Last state should be marked as terminal
                    if i == len(player_trajectory) - 1 and isinstance(step, dict) and 'raw_obs' in step:
                        # Mark the last state as terminal with the final reward
                        buffer.add(step, None, payoffs[0], step, True)
                
                except Exception as e:
                    logger.error(f"Error processing trajectory step {i}: {str(e)}")
                    logger.error(f"Step content: {step}")
                    continue
            
            # Learn from replay buffer
            if len(buffer) >= args.batch_size:
                batch = buffer.sample(args.batch_size)
                loss = agent.learn(batch)
                
                if loss is not None:
                    logger.debug(f"Episode {episode+1}, Loss: {loss:.4f}")
            
            # Periodically log progress
            if (episode + 1) % args.eval_every == 0:
                recent_rewards = rewards[-args.eval_every:]
                win_rate = wins / (episode + 1)
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                
                logger.info(f"Episode {episode+1}/{args.num_episodes}, "
                           f"Win Rate: {win_rate:.3f}, "
                           f"Recent Avg Reward: {avg_reward:.3f}, "
                           f"Buffer Size: {len(buffer)}")
                
                # Periodically save model
                if (episode + 1) % (args.eval_every * 2) == 0:
                    agent.save_model(args.save_model)
            
        except Exception as e:
            logger.error(f"Error in episode {episode}: {str(e)}")
            logger.error(traceback.format_exc())
            continue
    
    # Save final model
    agent.save_model(args.save_model)
    logger.info(f"Final model saved to {args.save_model}")
    
    # Final stats
    final_win_rate = wins / args.num_episodes
    final_loss_rate = losses / args.num_episodes
    final_tie_rate = ties / args.num_episodes
    final_avg_reward = sum(rewards) / args.num_episodes
    
    logger.info(f"Training completed. "
               f"Win Rate: {final_win_rate:.3f}, "
               f"Loss Rate: {final_loss_rate:.3f}, "
               f"Tie Rate: {final_tie_rate:.3f}, "
               f"Avg Reward: {final_avg_reward:.3f}")
    
    return agent, final_win_rate, final_avg_reward


def evaluate(args):
    """Evaluate a trained agent"""
    # Set random seed
    set_seed(args.seed)
    
    # Create environment
    env = rlcard.make('no-limit-holdem', config={'seed': args.seed})
    
    # Create agent with no exploration
    agent = PokerAgent(
        model_path=args.load_model,
        device=torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu'),
        epsilon=0.0
    )
    
    # Create opponent
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
    
    return win_rate, avg_reward


def main():
    parser = argparse.ArgumentParser(description='Train or evaluate a poker agent')
    parser.add_argument('--train', action='store_true', help='Train agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate agent')
    parser.add_argument('--load-model', type=str, default=None, help='Path to load model')
    parser.add_argument('--save-model', type=str, default='models/poker_model.pt', help='Path to save model')
    parser.add_argument('--num-episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--eval-games', type=int, default=1000, help='Number of evaluation games')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--buffer-size', type=int, default=100000, help='Replay buffer size')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate')
    parser.add_argument('--eval-every', type=int, default=500, help='Evaluation frequency')
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
        train(args)
    
    if args.evaluate:
        evaluate(args)
        
    if not args.train and not args.evaluate:
        parser.print_help()


if __name__ == "__main__":
    import torch.nn.functional as F
    main()