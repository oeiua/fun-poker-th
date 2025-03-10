"""
Policy network for deciding poker actions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional

from models.neural_network import PokerNeuralNetwork

class PolicyNetwork(PokerNeuralNetwork):
    """Policy network that determines which action to take."""
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        learning_rate: float = 0.0001,
        device: torch.device = None
    ):
        """
        Initialize the policy network.
        
        Args:
            input_size: Dimension of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of actions (typically 3: fold, call, raise)
            learning_rate: Learning rate for optimizer
            device: CPU or GPU device
        """
        super(PolicyNetwork, self).__init__(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=output_size,
            learning_rate=learning_rate,
            device=device
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute action probabilities.
        
        Args:
            x: Input tensor representing the state
            
        Returns:
            Action probability distribution
        """
        # Get raw outputs from base network
        raw_output = super().forward(x)
        
        # Apply softmax to get probabilities
        action_probs = F.softmax(raw_output, dim=-1)
        
        return action_probs
    
    def select_action(
        self,
        state: np.ndarray,
        valid_actions: List[int],
        explore: bool = True,
        temperature: float = 1.0
    ) -> Tuple[int, float]:
        """
        Select an action based on the current state.
        
        Args:
            state: State representation
            valid_actions: List of valid action indices
            explore: Whether to explore (True) or exploit (False)
            temperature: Temperature for exploration (higher = more random)
            
        Returns:
            Tuple of (selected action index, action probability)
        """
        self.eval()
        with torch.no_grad():
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action probabilities
            action_probs = self.forward(state_tensor).cpu().numpy()[0]
            
            # Set probabilities of invalid actions to 0
            if valid_actions:
                mask = np.zeros_like(action_probs)
                mask[valid_actions] = 1
                masked_probs = action_probs * mask
                
                # Renormalize
                if masked_probs.sum() > 0:
                    masked_probs = masked_probs / masked_probs.sum()
                else:
                    # If all valid actions have zero probability, use uniform distribution
                    masked_probs = mask / mask.sum()
            else:
                # Default to all actions valid if none specified
                masked_probs = action_probs
            
            if explore:
                # Apply temperature to control exploration/exploitation
                if temperature != 1.0:
                    # Higher temperature = more random, lower = more greedy
                    masked_probs = np.power(masked_probs, 1.0 / temperature)
                    masked_probs = masked_probs / masked_probs.sum()
                
                # Sample action from probability distribution
                action = np.random.choice(len(masked_probs), p=masked_probs)
            else:
                # Greedy selection
                action = np.argmax(masked_probs)
            
            return action, masked_probs[action]
    
    def train_on_episode(
        self,
        states: List[np.ndarray],
        actions: List[int],
        rewards: List[float],
        valid_action_masks: List[np.ndarray],
        batch_size: int = 64
    ) -> float:
        """
        Train the policy network using policy gradients.
        
        Args:
            states: List of states encountered in the episode
            actions: List of actions taken
            rewards: List of rewards received
            valid_action_masks: Masks of valid actions for each state
            batch_size: Size of training batches
            
        Returns:
            Loss value for the training
        """
        self.train()  # Set to training mode
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Calculate discounted rewards
        discounted_rewards = self._calculate_discounted_rewards(rewards)
        rewards_tensor = torch.FloatTensor(discounted_rewards).to(self.device)
        
        # Normalize rewards for stability
        if len(rewards_tensor) > 1:
            rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)
        
        # Process in batches
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(states), batch_size):
            # Get batch
            batch_states = states_tensor[i:i+batch_size]
            batch_actions = actions_tensor[i:i+batch_size]
            batch_rewards = rewards_tensor[i:i+batch_size]
            
            # Forward pass to get action probabilities
            action_probs = self.forward(batch_states)
            
            # Get probability of the actions that were taken
            selected_probs = action_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)
            
            # Calculate log probabilities
            log_probs = torch.log(selected_probs + 1e-10)
            
            # Policy gradient loss
            loss = -(log_probs * batch_rewards).mean()
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def _calculate_discounted_rewards(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """
        Calculate discounted rewards for an episode.
        
        Args:
            rewards: List of rewards
            gamma: Discount factor
            
        Returns:
            List of discounted rewards
        """
        discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        
        # Calculate discounted rewards in reverse
        for t in reversed(range(len(rewards))):
            running_add = running_add * gamma + rewards[t]
            discounted_rewards[t] = running_add
            
        return discounted_rewards