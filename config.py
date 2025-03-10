"""
Configuration settings for the Poker AI system.
"""
import os
from enum import Enum, auto

class Action(Enum):
    """Enum for possible poker actions."""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5

class GamePhase(Enum):
    """Enum for game phases."""
    PREFLOP = 0
    FLOP = 1
    TURN = 2
    RIVER = 3
    SHOWDOWN = 4

class PokerConfig:
    """Configuration parameters for the poker AI system."""
    
    # General settings
    SEED = 42
    DEFAULT_PLAYERS = 10  # 9 AI + 1 human in play mode
    
    # Game settings
    SMALL_BLIND = 5
    BIG_BLIND = 10
    INITIAL_CHIPS = 100
    MAX_RAISES_PER_ROUND = 4
    TIMEOUT_SECONDS = 5  # Timeout for betting decisions
    
    # State representation
    NUM_PLAYERS = 10
    NUM_CARDS_IN_DECK = 52
    NUM_HOLE_CARDS = 2
    NUM_COMMUNITY_CARDS = 5
    
    # Neural network input size calculation:
    # - Player position (1)
    # - Player chips (1)
    # - Player pot commitment (1)
    # - Hole cards (2 cards * 4 features per card = 8)
    # - Community cards (5 cards * 4 features per card = 20)
    # - Pot size (1)
    # - Current bet to call (1)
    # - Game phase (5 one-hot encoded phases = 5)
    # - Number of players still in hand (1)
    # - Position relative to dealer (1)
    STATE_SIZE = 50
    
    # Neural network output size (possible actions)
    ACTION_SIZE = len(Action)
    
    # Model architecture
    HIDDEN_LAYERS = [128, 64, 32]
    LEARNING_RATE = 0.001
    DROPOUT_RATE = 0.2
    
    # Evolutionary algorithm settings
    POPULATION_SIZE = 200
    GENERATIONS = 1000
    ELITE_PERCENTAGE = 0.1  # Top percentage to keep unchanged
    MUTATION_RATE = 0.1
    CROSSOVER_RATE = 0.7
    TOURNAMENT_SIZE = 5
    GAMES_PER_EVALUATION = 100
    
    # File paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models")
    LOGS_PATH = os.path.join(BASE_DIR, "logs")
    CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, "checkpoints")
    
    # Betting parameters
    MIN_BET_MULTIPLIER = 1  # Minimum bet as multiplier of big blind
    MAX_BET_MULTIPLIER = 10  # Maximum bet as multiplier of big blind
    
    # Game visualization
    CARD_SYMBOLS = {
        'hearts': '♥',
        'diamonds': '♦',
        'clubs': '♣',
        'spades': '♠'
    }
