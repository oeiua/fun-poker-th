"""
Neural network architecture for the poker AI.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

class PokerNeuralNetwork(nn.Module):
    """Base neural network architecture for poker decision making."""
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        learning_rate: float = 0.0001,
        device: torch.device = None
    ):
        """
        Initialize the neural network.
        
        Args:
            input_size: Dimension of input features
            hidden_layers: List of hidden layer sizes
            output_size: Dimension of output
            learning_rate: Learning rate for optimizer
            device: CPU or GPU device
        """
        super(PokerNeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Build network layers
        self.layers = nn.ModuleList()
        
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_layers[0]))
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            self.layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.2)
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList()
        for hidden_size in hidden_layers:
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Move network to the specified device
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Process through hidden layers with activation and batch norm
        for i, (layer, batch_norm) in enumerate(zip(self.layers, self.batch_norms)):
            x = layer(x)
            # Apply batch norm for batches with size > 1
            if x.shape[0] > 1:
                x = batch_norm(x)
            x = F.leaky_relu(x, negative_slope=0.01)
            x = self.dropout(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Make a prediction for a single state.
        
        Args:
            state: Numpy array representing the state
            
        Returns:
            Numpy array with action probabilities/values
        """
        self.eval()  # Set to evaluation mode
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            output = self.forward(state_tensor)
            return output.cpu().numpy()[0]
    
    def save(self, path: str) -> None:
        """
        Save the model to disk.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'architecture': {
                'input_size': self.input_size,
                'hidden_layers': self.hidden_layers,
                'output_size': self.output_size,
                'learning_rate': self.learning_rate
            }
        }, path)
    
    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> 'PokerNeuralNetwork':
        """
        Load a model from disk.
        
        Args:
            path: Path to the saved model
            device: Device to load the model on
            
        Returns:
            Loaded PokerNeuralNetwork model
        """
        checkpoint = torch.load(path, map_location=device)
        architecture = checkpoint['architecture']
        
        model = cls(
            input_size=architecture['input_size'],
            hidden_layers=architecture['hidden_layers'],
            output_size=architecture['output_size'],
            learning_rate=architecture['learning_rate'],
            device=device
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return model