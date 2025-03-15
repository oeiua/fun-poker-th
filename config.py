import os
import torch

# Game configuration
NUM_PLAYERS = 10
STARTING_CHIPS = 2000
SMALL_BLIND = 5
BIG_BLIND = 10
MAX_ROUNDS = 1000
TOURNAMENT_TIMEOUT = 60  # seconds per betting round

# Calculate actual input size based on game parameters
# 52 cards + NUM_PLAYERS (position) + 2 (pot, current bet) + 1 (player chips)
# + (NUM_PLAYERS-1) (opponent chips) + 1 (player current bet) + 1 (amount to call)
# + 1 (active players ratio) + 1 (max raise)
INPUT_SIZE = 52 + NUM_PLAYERS + 2 + 1 + (NUM_PLAYERS-1) + 1 + 1 + 1 + 1

# Neural network configuration
HIDDEN_LAYERS = [128, 256, 64]
OUTPUT_SIZE = 3  # fold, call, raise percentage
LEARNING_RATE = 0.001

# Evolutionary algorithm parameters
POPULATION_SIZE = 10
TOURNAMENT_SIZE = 10
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.7
NUM_GENERATIONS = 5000
ELITISM_COUNT = 5

# Training configuration
CHECKPOINT_INTERVAL = 10
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

# UI configuration
CARD_SYMBOLS = {
    'clubs': '♣',
    'diamonds': '♦',
    'hearts': '♥',
    'spades': '♠'
}

CARD_COLORS = {
    'clubs': '\033[0;30m',    # Black
    'diamonds': '\033[0;31m',  # Red
    'hearts': '\033[0;31m',    # Red
    'spades': '\033[0;30m',    # Black
    'reset': '\033[0m'         # Reset
}

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)