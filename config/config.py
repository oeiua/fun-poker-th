"""
Configuration parameters for the Poker AI system.
"""
import os
import torch
from typing import Dict, Any, List, Tuple

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR = os.path.join(BASE_DIR, "saved_models")
    LOGS_DIR = os.path.join(BASE_DIR, "logs")
    
    # Game parameters
    GAME_TYPE = "texas_holdem"  # Using pokerkit's game type
    NUM_PLAYERS = 10  # 9 AI + 1 human player for play mode
    STARTING_STACK = 10000  # Starting chips
    SMALL_BLIND = 50
    BIG_BLIND = 100
    MAX_ROUNDS = 1000  # Max rounds per game
    ACTION_TIMEOUT = 30  # Seconds before defaulting to fold
    
    # Neural Network parameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_SIZE = 520  # State representation size (cards, pot, history, etc.)
    HIDDEN_LAYERS = [512, 256, 128, 64]
    OUTPUT_SIZE = 3  # Fold, Call, Raise
    LEARNING_RATE = 0.0001
    
    # Evolution parameters
    POPULATION_SIZE = 100
    GENERATIONS = 1000
    TOURNAMENT_SIZE = 10
    MUTATION_RATE = 0.05
    CROSSOVER_RATE = 0.7
    ELITE_SIZE = 5  # Number of best agents to preserve
    
    # Training parameters
    BATCH_SIZE = 256
    EVAL_FREQUENCY = 50  # Evaluate every N generations
    CHECKPOINT_FREQUENCY = 100  # Save models every N generations
    NUM_EVAL_GAMES = 200
    
    # Parallel processing
    NUM_WORKERS = os.cpu_count() - 1 or 1  # Use all available CPUs minus one
    
    # Memory management
    MEMORY_CHECK_FREQUENCY = 20  # Check for memory leaks every N generations
    
    @classmethod
    def get_game_config(cls) -> Dict[str, Any]:
        """Returns game-specific configuration parameters."""
        return {
            "game_type": cls.GAME_TYPE,
            "num_players": cls.NUM_PLAYERS,
            "starting_stack": cls.STARTING_STACK,
            "small_blind": cls.SMALL_BLIND,
            "big_blind": cls.BIG_BLIND,
            "max_rounds": cls.MAX_ROUNDS,
            "action_timeout": cls.ACTION_TIMEOUT,
        }
    
    @classmethod
    def get_nn_config(cls) -> Dict[str, Any]:
        """Returns neural network configuration parameters."""
        return {
            "device": cls.DEVICE,
            "input_size": cls.INPUT_SIZE,
            "hidden_layers": cls.HIDDEN_LAYERS,
            "output_size": cls.OUTPUT_SIZE,
            "learning_rate": cls.LEARNING_RATE,
        }
    
    @classmethod
    def get_evolution_config(cls) -> Dict[str, Any]:
        """Returns evolutionary algorithm configuration parameters."""
        return {
            "population_size": cls.POPULATION_SIZE,
            "generations": cls.GENERATIONS,
            "tournament_size": cls.TOURNAMENT_SIZE,
            "mutation_rate": cls.MUTATION_RATE,
            "crossover_rate": cls.CROSSOVER_RATE,
            "elite_size": cls.ELITE_SIZE,
        }