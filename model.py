"""
Neural network model for the Poker AI.
"""
import os
import copy
import numpy as np
import tensorflow as tf
from typing import List, Optional, Tuple, Dict, Any

from config import PokerConfig, Action

class PokerModel:
    """
    Neural network model for the Poker AI.
    """
    def __init__(self, input_size: int, output_size: int):
        """
        Initialize the poker model.
        
        Args:
            input_size (int): Size of the input state vector
            output_size (int): Size of the output action vector
        """
        self.input_size = input_size
        self.output_size = output_size
        self.model = self._build_model()
        
        # TF Function for prediction to avoid retracing
        self.predict_function = tf.function(
            self.model.call,
            input_signature=[tf.TensorSpec(shape=(None, input_size), dtype=tf.float32)]
        )
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Make a prediction with the model.
        
        Args:
            state (np.ndarray): The state vector
            
        Returns:
            np.ndarray: Action probabilities
        """
        # Ensure state has the right shape
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        
        # Use the compiled TF function for better performance
        return self.predict_function(tf.convert_to_tensor(state, dtype=tf.float32)).numpy()
    
    def get_action(self, state: np.ndarray, valid_actions: List[Action]) -> Action:
        """
        Get the best valid action based on the model prediction.
        
        Args:
            state (np.ndarray): The state vector
            valid_actions (List[Action]): List of valid actions
            
        Returns:
            Action: The selected action
        """
        # Get action probabilities
        action_probs = self.predict(state)[0]
        
        # Create a mask for valid actions
        valid_actions_indices = [action.value for action in valid_actions]
        mask = np.zeros_like(action_probs)
        mask[valid_actions_indices] = 1
        
        # Apply mask and normalize
        masked_probs = action_probs * mask
        if np.sum(masked_probs) > 0:
            masked_probs = masked_probs / np.sum(masked_probs)
        else:
            # If all probabilities are masked out, use uniform distribution
            masked_probs[valid_actions_indices] = 1.0 / len(valid_actions_indices)
        
        # Select action based on probability distribution
        action_index = np.random.choice(len(masked_probs), p=masked_probs)
        return Action(action_index)
    
    def save(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath (str): Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(filepath)
    
    def load(self, filepath: str):
        """
        Load the model from a file.
        
        Args:
            filepath (str): Path to load the model from
        """
        if os.path.exists(filepath):
            self.model.load_weights(filepath)
    
    def clone(self) -> 'PokerModel':
        """
        Create a clone of this model.
        
        Returns:
            PokerModel: A new model with the same weights
        """
        new_model = PokerModel(self.input_size, self.output_size)
        for target_layer, source_layer in zip(new_model.model.layers, self.model.layers):
            if hasattr(target_layer, 'kernel'):
                target_layer.kernel.assign(source_layer.kernel)
            if hasattr(target_layer, 'bias'):
                target_layer.bias.assign(source_layer.bias)
            if hasattr(target_layer, 'gamma'):
                target_layer.gamma.assign(source_layer.gamma)
            if hasattr(target_layer, 'beta'):
                target_layer.beta.assign(source_layer.beta)
            if hasattr(target_layer, 'moving_mean'):
                target_layer.moving_mean.assign(source_layer.moving_mean)
            if hasattr(target_layer, 'moving_variance'):
                target_layer.moving_variance.assign(source_layer.moving_variance)
        return new_model
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build the neural network model architecture.
        
        Returns:
            tf.keras.Model: The compiled model
        """
        inputs = tf.keras.layers.Input(shape=(self.input_size,))
        
        # First hidden layer with batch normalization
        x = tf.keras.layers.Dense(PokerConfig.HIDDEN_LAYERS[0])(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = tf.keras.layers.Dropout(PokerConfig.DROPOUT_RATE)(x)
        
        # Hidden layers
        for units in PokerConfig.HIDDEN_LAYERS[1:]:
            x = tf.keras.layers.Dense(units)(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = tf.keras.layers.Dropout(PokerConfig.DROPOUT_RATE)(x)
        
        # Output layer for actions (fold, check/call, bet/raise, all-in)
        # Using softmax activation for action probabilities
        outputs = tf.keras.layers.Dense(self.output_size, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=PokerConfig.LEARNING_RATE),
            loss='categorical_crossentropy'
        )
        
        return model
    
    def mutate(self, mutation_rate: float = PokerConfig.MUTATION_RATE):
        """
        Mutate the model's weights.
        
        Args:
            mutation_rate (float): Probability of mutation for each weight
        """
        for layer in self.model.layers:
            if hasattr(layer, 'kernel'):
                weights = layer.kernel.numpy()
                # Apply mutations with probability mutation_rate
                mask = np.random.random(weights.shape) < mutation_rate
                # Generate mutations with zero mean and small standard deviation
                mutations = np.random.normal(0, 0.1, size=weights.shape)
                weights = np.where(mask, weights + mutations, weights)
                layer.kernel.assign(weights)
            
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias = layer.bias.numpy()
                mask = np.random.random(bias.shape) < mutation_rate
                mutations = np.random.normal(0, 0.1, size=bias.shape)
                bias = np.where(mask, bias + mutations, bias)
                layer.bias.assign(bias)
    
    def crossover(self, other: 'PokerModel') -> 'PokerModel':
        """
        Perform crossover with another model.
        
        Args:
            other (PokerModel): The other model to crossover with
            
        Returns:
            PokerModel: A new model resulting from the crossover
        """
        child = PokerModel(self.input_size, self.output_size)
        
        for child_layer, self_layer, other_layer in zip(
            child.model.layers, self.model.layers, other.model.layers
        ):
            # Crossover weights
            if hasattr(child_layer, 'kernel'):
                self_weights = self_layer.kernel.numpy()
                other_weights = other_layer.kernel.numpy()
                
                # Crossover masks for weights
                mask = np.random.random(self_weights.shape) < 0.5
                new_weights = np.where(mask, self_weights, other_weights)
                child_layer.kernel.assign(new_weights)
            
            # Crossover biases
            if hasattr(child_layer, 'bias') and child_layer.bias is not None:
                self_bias = self_layer.bias.numpy()
                other_bias = other_layer.bias.numpy()
                
                # Crossover masks for biases
                mask = np.random.random(self_bias.shape) < 0.5
                new_bias = np.where(mask, self_bias, other_bias)
                child_layer.bias.assign(new_bias)
        
        return child