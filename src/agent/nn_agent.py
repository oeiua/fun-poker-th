"""
Neural network-based poker agent.
"""

import logging
import numpy as np
import tensorflow as tf
from typing import Dict, List, Tuple, Any, Optional

from src.agent.base_agent import BaseAgent
from src.environment.state import GameState
from src.models.memory import ReplayMemory


class NNAgent(BaseAgent):
    """
    Neural network-based poker agent using TensorFlow.
    """
    
    def __init__(self, player_id: int, model: tf.keras.Model, config: Dict):
        """
        Initialize the neural network agent.
        
        Args:
            player_id: Unique identifier for this agent
            model: TensorFlow model for decision making
            config: Configuration dictionary
        """
        super().__init__(player_id)
        self.model = model
        self.config = config
        self.memory = ReplayMemory(config['model']['batch_size'])
        self.epsilon = 0.1  # Exploration rate
        
    def get_action(self, state: GameState, legal_actions: Dict[str, List[int]]) -> Tuple[str, int]:
        """
        Get the agent's action using the neural network.
        
        Args:
            state: Current game state
            legal_actions: Dictionary of legal actions with their parameters
            
        Returns:
            Tuple of (action_type, amount)
        """
        # Convert state to a tensor
        state_tensor = tf.convert_to_tensor(state.to_input_tensor())
        state_tensor = tf.expand_dims(state_tensor, 0)  # Add batch dimension
        
        # Epsilon-greedy exploration
        if np.random.random() < self.epsilon:
            return self._random_action(legal_actions)
        
        # Get action values from the model
        action_values = self.model(state_tensor, training=False).numpy()[0]
        
        # Select the best legal action
        return self._select_best_action(action_values, legal_actions)
    
    def _random_action(self, legal_actions: Dict[str, List[int]]) -> Tuple[str, int]:
        """
        Select a random legal action.
        
        Args:
            legal_actions: Dictionary of legal actions with their parameters
            
        Returns:
            Tuple of (action_type, amount)
        """
        # Choose a random action type
        action_types = list(legal_actions.keys())
        action_type = np.random.choice(action_types)
        
        # Choose a random amount for the selected action type
        amounts = legal_actions[action_type]
        amount = np.random.choice(amounts)
        
        return action_type, amount
    
    def _select_best_action(self, 
                           action_values: np.ndarray, 
                           legal_actions: Dict[str, List[int]]) -> Tuple[str, int]:
        """
        Select the best legal action based on the model's outputs.
        
        Args:
            action_values: Values from the model for each action
            legal_actions: Dictionary of legal actions with their parameters
            
        Returns:
            Tuple of (action_type, amount)
        """
        # Map action values to their respective types
        action_mapping = {
            0: ('fold', 0),
            1: ('check_call', 0),
            2: ('bet_raise', 0)
        }
        
        # Sort actions by their values (highest to lowest)
        sorted_actions = sorted(
            range(len(action_values)), 
            key=lambda i: action_values[i], 
            reverse=True
        )
        
        # Try actions in order of preference until we find a legal one
        for action_idx in sorted_actions:
            action_type, _ = action_mapping[action_idx]
            
            if action_type in legal_actions:
                # For bet/raise, select the bet amount with the best expected value
                if action_type == 'bet_raise' and legal_actions[action_type]:
                    # Simple heuristic: prefer larger bets/raises
                    # In a real implementation, you'd want a more sophisticated bet sizing strategy
                    amount = max(legal_actions[action_type])
                    return action_type, amount
                
                # For check/call, use the required amount
                elif action_type == 'check_call':
                    amount = legal_actions[action_type][0]
                    return action_type, amount
                
                # For fold, no amount needed
                elif action_type == 'fold':
                    return action_type, 0
        
        # If no valid action was found (shouldn't happen normally)
        logging.warning("No valid action found, defaulting to fold")
        return 'fold', 0
    
    def record_experience(self, state: GameState, action_type: str, amount: int, 
                         next_state: Optional[GameState], reward: float, done: bool) -> None:
        """
        Record an experience for later training.
        
        Args:
            state: Current state
            action_type: Action taken
            amount: Amount (for bet/raise)
            next_state: Resulting state
            reward: Reward received
            done: Whether this is a terminal state
        """
        # Convert action to index
        action_index = {'fold': 0, 'check_call': 1, 'bet_raise': 2}.get(action_type, 0)
        
        # Store experience in memory
        self.memory.add(
            state.to_input_tensor(),
            action_index,
            reward,
            next_state.to_input_tensor() if next_state else np.zeros_like(state.to_input_tensor()),
            done
        )
    
    def update(self, reward: float) -> None:
        """
        Update the agent's policy based on the reward received.
        
        Args:
            reward: Reward value (positive or negative)
        """
        # For immediate rewards, record them in the hand history
        if self.hand_history:
            last_action = self.hand_history[-1]
            self.record_experience(
                last_action['state'],
                last_action['action_type'],
                last_action['amount'],
                None,  # We don't have next_state here
                reward,
                True
            )
    
    def train(self, batch_size: int = None) -> float:
        """
        Train the model on a batch of experiences.
        
        Args:
            batch_size: Size of batch to train on, uses config default if None
            
        Returns:
            Loss value from training
        """
        if batch_size is None:
            batch_size = self.config['model']['batch_size']
        
        # Skip if we don't have enough experiences
        if len(self.memory) < batch_size:
            return 0.0
        
        # Sample a batch of experiences
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            # Current Q Values
            current_q = self.model(states, training=True)
            action_masks = tf.one_hot(actions, 3)  # 3 action types
            current_q_values = tf.reduce_sum(current_q * action_masks, axis=1)
            
            # Target Q Values (using target network would be better)
            next_q_values = tf.reduce_max(self.model(next_states, training=False), axis=1)
            target_q_values = rewards + (1 - dones) * 0.99 * next_q_values
            
            # Compute loss
            loss = tf.keras.losses.MSE(current_q_values, target_q_values)
        
        # Update model
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss.numpy()
    
    def save(self, path: str) -> None:
        """
        Save the agent's model.
        
        Args:
            path: Path to save the model
        """
        self.model.save(path)
    
    def load(self, path: str) -> None:
        """
        Load the agent's model.
        
        Args:
            path: Path to load the model from
        """
        self.model = tf.keras.models.load_model(path)