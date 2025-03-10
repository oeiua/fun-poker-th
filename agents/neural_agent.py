"""
Neural network-based agent for poker decision making.
"""
import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import time

from agents.base_agent import BaseAgent
from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
from game.state import GameState
from game.action import Action
from game.rewards import RewardCalculator

class NeuralAgent(BaseAgent):
    """Agent that uses neural networks for decision making."""
    
    def __init__(
        self,
        policy_network: Optional[PolicyNetwork] = None,
        value_network: Optional[ValueNetwork] = None,
        name: Optional[str] = None,
        exploration_rate: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Initialize a neural network agent.
        
        Args:
            policy_network: Policy network for action selection
            value_network: Value network for state evaluation
            name: Name of the agent
            exploration_rate: Probability of choosing a random action
            device: Device to run the networks on
        """
        super().__init__(name=name)
        
        self.policy_network = policy_network
        self.value_network = value_network
        self.exploration_rate = exploration_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Keep track of experiences for training
        self.episode_buffer = []
        self.reward_calculator = RewardCalculator()
        
        # Additional debug information
        self.last_state_value = 0.0
        self.last_action_probs = []
        self.training_stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": []
        }
    
    def act(self, state: GameState, valid_actions: List[int], valid_amounts: List[int]) -> Tuple[int, Optional[int]]:
        """
        Choose an action based on the current game state.
        
        Args:
            state: Current game state
            valid_actions: List of valid action types
            valid_amounts: List of valid bet amounts
            
        Returns:
            Tuple of (action_type, bet_amount)
        """
        # If no valid actions, default to fold
        if not valid_actions:
            return Action.FOLD, None
        
        # Random exploration
        if np.random.random() < self.exploration_rate:
            action_type = np.random.choice(valid_actions)
            if action_type == Action.BET_RAISE and valid_amounts:
                bet_amount = np.random.choice(valid_amounts)
            else:
                bet_amount = None
            
            # Save for debugging
            self.last_action_probs = [0.0] * 3
            if 0 <= action_type < 3:
                self.last_action_probs[action_type] = 1.0
                
            return action_type, bet_amount
        
        # Use policy network for decision
        if self.policy_network:
            # Convert state to vector
            state_vector = state.vectorize(self.player_idx)
            
            # Get action from policy network
            action_type, action_prob = self.policy_network.select_action(
                state=state_vector,
                valid_actions=valid_actions,
                explore=False  # Already handled exploration above
            )
            
            # Save action probabilities for debugging
            self.last_action_probs = self.policy_network.predict(state_vector).tolist()
            
            # Determine bet amount for bet/raise
            bet_amount = None
            if action_type == Action.BET_RAISE and valid_amounts:
                # If we have a value network, use it to evaluate different bet sizes
                if self.value_network:
                    # Evaluate each bet size
                    best_ev = float('-inf')
                    best_amount = valid_amounts[0] if valid_amounts else None
                    
                    game_info = state.game_info
                    pot_size = game_info.get('pot', 0)
                    
                    for amount in valid_amounts:
                        # For evaluation, we use a simple expected value calculation
                        # based on the value network's estimate of the resulting state
                        
                        # Get hand strength estimate
                        hand_strength = self.reward_calculator.calculate_hand_strength(
                            hole_cards=state.hole_cards[self.player_idx],
                            community_cards=state.community_cards,
                            num_players=len(state.hole_cards)
                        )
                        
                        # Calculate EV using pot odds and hand strength
                        ev = self.reward_calculator.calculate_expected_value(
                            pot_size=pot_size + amount,  # Potential pot after our bet
                            to_call=amount,
                            hand_strength=hand_strength
                        )
                        
                        if ev > best_ev:
                            best_ev = ev
                            best_amount = amount
                    
                    bet_amount = best_amount
                else:
                    # Without a value network, select a bet amount proportional to our confidence
                    confidence = action_prob
                    
                    if confidence > 0.8:
                        # High confidence - larger bet
                        bet_idx = min(len(valid_amounts) - 1, int(0.8 * len(valid_amounts)))
                    elif confidence > 0.6:
                        # Medium confidence - medium bet
                        bet_idx = min(len(valid_amounts) - 1, int(0.5 * len(valid_amounts)))
                    else:
                        # Low confidence - smaller bet
                        bet_idx = min(len(valid_amounts) - 1, int(0.3 * len(valid_amounts)))
                    
                    bet_amount = valid_amounts[bet_idx] if valid_amounts else None
            
            return action_type, bet_amount
        
        # Fallback to random action if no policy network
        action_type = np.random.choice(valid_actions)
        if action_type == Action.BET_RAISE and valid_amounts:
            bet_amount = np.random.choice(valid_amounts)
        else:
            bet_amount = None
            
        return action_type, bet_amount
    
    def observe(self, state: GameState, action: Tuple[int, Optional[int]], reward: float, next_state: GameState, done: bool) -> None:
        """
        Observe the result of an action for learning.
        
        Args:
            state: State before action
            action: Action taken (action_type, bet_amount)
            reward: Reward received
            next_state: State after action
            done: Whether the episode is done
        """
        # Call parent method to update stats
        super().observe(state, action, reward, next_state, done)
        
        # Store experience
        self.episode_buffer.append((
            state.vectorize(self.player_idx),
            action[0],  # action_type
            reward,
            next_state.vectorize(self.player_idx) if not done else None,
            done
        ))
        
        # If episode is complete, train on the collected experiences
        if done and len(self.episode_buffer) > 0:
            self._train_on_episode()
    
    def _train_on_episode(self) -> None:
        """Train networks on the collected episode data."""
        if not self.episode_buffer:
            return
        
        # Extract data from episode buffer
        states, actions, rewards, next_states, dones = zip(*self.episode_buffer)
        
        # Train policy network
        if self.policy_network:
            # Calculate discounted rewards
            discounted_rewards = self.reward_calculator.discounted_rewards(rewards)
            
            # Create valid action masks (simplified - assume all actions valid)
            valid_action_masks = np.ones((len(states), 3), dtype=np.float32)
            
            # Train policy network
            policy_loss = self.policy_network.train_on_episode(
                states=states,
                actions=actions,
                rewards=discounted_rewards,
                valid_action_masks=valid_action_masks
            )
            
            self.training_stats["policy_loss"].append(policy_loss)
        
        # Train value network
        if self.value_network:
            value_loss = self.value_network.train_on_episode(
                states=states,
                rewards=rewards
            )
            
            self.training_stats["value_loss"].append(value_loss)
        
        # Clear episode buffer
        self.episode_buffer = []
    
    def reset(self) -> None:
        """Reset the agent's state between episodes."""
        self.episode_buffer = []
        self.last_state_value = 0.0
        self.last_action_probs = []
    
    def save(self, path: str) -> None:
        """
        Save the agent's networks to files.
        
        Args:
            path: Directory path to save the agent
        """
        os.makedirs(path, exist_ok=True)
        
        if self.policy_network:
            policy_path = os.path.join(path, f"{self.name}_policy.pt")
            self.policy_network.save(policy_path)
        
        if self.value_network:
            value_path = os.path.join(path, f"{self.name}_value.pt")
            self.value_network.save(value_path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'NeuralAgent':
        """
        Load an agent from saved networks.
        
        Args:
            path: Directory path to load the agent from
            device: Device to load the networks on
            
        Returns:
            Loaded NeuralAgent
        """
        name = os.path.basename(path)
        
        # Load policy network if exists
        policy_path = os.path.join(path, f"{name}_policy.pt")
        policy_network = None
        if os.path.exists(policy_path):
            policy_network = PolicyNetwork.load(policy_path, device)
        
        # Load value network if exists
        value_path = os.path.join(path, f"{name}_value.pt")
        value_network = None
        if os.path.exists(value_path):
            value_network = ValueNetwork.load(value_path, device)
        
        # Create and return agent
        return cls(
            policy_network=policy_network,
            value_network=value_network,
            name=name,
            device=device
        )
    
    def debug_info(self) -> Dict[str, Any]:
        """
        Get debug information about the agent's decision making.
        
        Returns:
            Dictionary of debug information
        """
        info = super().debug_info()
        
        # Add neural network specific information
        info.update({
            "exploration_rate": self.exploration_rate,
            "last_state_value": self.last_state_value,
            "last_action_probs": self.last_action_probs,
            "training_stats": {
                "policy_loss": self.training_stats["policy_loss"][-10:] if self.training_stats["policy_loss"] else [],
                "value_loss": self.training_stats["value_loss"][-10:] if self.training_stats["value_loss"] else []
            }
        })
        
        return info