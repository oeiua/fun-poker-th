"""
Value network for estimating the expected value of a poker state.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Any, Optional

from models.neural_network import PokerNeuralNetwork

class ValueNetwork(PokerNeuralNetwork):
    """Value network that estimates the expected return from a given state."""
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        learning_rate: float = 0.0001,
        device: torch.device = None
    ):
        """
        Initialize the value network.
        
        Args:
            input_size: Dimension of input features
            hidden_layers: List of hidden layer sizes
            learning_rate: Learning rate for optimizer
            device: CPU or GPU device
        """
        # Value network has a single output (estimated value)
        super(ValueNetwork, self).__init__(
            input_size=input_size,
            hidden_layers=hidden_layers,
            output_size=1,  # Single value output
            learning_rate=learning_rate,
            device=device
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute the value estimate.
        
        Args:
            x: Input tensor representing the state
            
        Returns:
            Estimated value of the state
        """
        # Get raw outputs from base network
        value = super().forward(x)
        
        # No activation function on output - predicting raw value
        return value
    
    def estimate_value(self, state: np.ndarray) -> float:
        """
        Estimate the value of a state.
        
        Args:
            state: State representation
            
        Returns:
            Estimated value (float)
        """
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.forward(state_tensor)
            return value.item()
    
    def train_batch(
        self,
        states: List[np.ndarray],
        target_values: List[float],
        batch_size: int = 64
    ) -> float:
        """
        Train the value network on a batch of data.
        
        Args:
            states: List of states
            target_values: Target values for each state
            batch_size: Size of training batches
            
        Returns:
            Average loss value
        """
        self.train()  # Set to training mode
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        values_tensor = torch.FloatTensor(target_values).unsqueeze(1).to(self.device)
        
        # Process in batches
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(states), batch_size):
            # Get batch
            batch_states = states_tensor[i:i+batch_size]
            batch_values = values_tensor[i:i+batch_size]
            
            # Forward pass
            predicted_values = self.forward(batch_states)
            
            # Calculate MSE loss
            loss = F.mse_loss(predicted_values, batch_values)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def train_on_episode(
        self,
        states: List[np.ndarray],
        rewards: List[float],
        batch_size: int = 64,
        gamma: float = 0.99
    ) -> float:
        """
        Train the value network using observed rewards from an episode.
        
        Args:
            states: List of states encountered in the episode
            rewards: List of rewards received
            batch_size: Size of training batches
            gamma: Discount factor
            
        Returns:
            Loss value for the training
        """
        # Calculate discounted returns as target values
        discounted_rewards = self._calculate_discounted_rewards(rewards, gamma)
        
        # Train on the batch
        return self.train_batch(states, discounted_rewards, batch_size)
    
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
    
    def calculate_advantage(
        self, 
        states: List[np.ndarray], 
        rewards: List[float], 
        gamma: float = 0.99
    ) -> List[float]:
        """
        Calculate advantage values (actual return - predicted value).
        
        Args:
            states: List of states
            rewards: List of rewards
            gamma: Discount factor
            
        Returns:
            List of advantage values
        """
        self.eval()
        with torch.no_grad():
            # Calculate actual discounted returns
            discounted_rewards = self._calculate_discounted_rewards(rewards, gamma)
            
            # Get value predictions for each state
            values = []
            for state in states:
                value = self.estimate_value(state)
                values.append(value)
            
            # Calculate advantages
            advantages = [r - v for r, v in zip(discounted_rewards, values)]
            
            return advantages