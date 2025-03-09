"""
Main entry point for the Poker AI system.
Handles initialization, training, and game execution.
"""
import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from datetime import datetime

from config import PokerConfig
from game_engine import GameEngine
from player import AIPlayer, HumanPlayer
from model import PokerModel
from utils import set_seeds, setup_tf_memory_growth, save_config_snapshot
from evolutionary_trainer import EvolutionaryTrainer
from visualizer import TrainingVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("poker_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PokerAI")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Poker AI Training and Gameplay')
    parser.add_argument('--mode', choices=['train', 'play', 'evaluate'], default='train',
                        help='Mode to run: train, play with human, or evaluate')
    parser.add_argument('--generations', type=int, default=PokerConfig.GENERATIONS,
                        help='Number of generations for evolutionary training')
    parser.add_argument('--population', type=int, default=PokerConfig.POPULATION_SIZE,
                        help='Population size for evolutionary training')
    parser.add_argument('--games', type=int, default=PokerConfig.GAMES_PER_EVALUATION,
                        help='Number of games to play per evaluation')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output during gameplay')
    parser.add_argument('--model_path', type=str, default=PokerConfig.MODEL_SAVE_PATH,
                        help='Path to save/load models')
    parser.add_argument('--checkpoint', action='store_true',
                        help='Continue training from the latest checkpoint')
    parser.add_argument('--num_players', type=int, default=PokerConfig.DEFAULT_PLAYERS,
                        help='Number of players in the game')
    parser.add_argument('--initial_chips', type=int, default=PokerConfig.INITIAL_CHIPS,
                        help='Initial chips for each player')
    return parser.parse_args()

def setup_environment():
    """Set up the environment for training or gameplay."""
    # Set random seeds for reproducibility
    set_seeds(PokerConfig.SEED)
    
    # Configure TensorFlow for better performance
    setup_tf_memory_growth()
    
    # Create directories if they don't exist
    os.makedirs(PokerConfig.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(PokerConfig.LOGS_PATH, exist_ok=True)
    
    # Save config snapshot for reference
    config_snapshot_path = os.path.join(
        PokerConfig.LOGS_PATH, 
        f"config_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    save_config_snapshot(PokerConfig, config_snapshot_path)
    
    logger.info(f"Environment setup complete. Config saved to {config_snapshot_path}")

def train_model(args):
    """Train the poker AI models using evolutionary algorithms."""
    logger.info("Starting evolutionary training process")
    
    # Initialize training visualizer
    visualizer = TrainingVisualizer()
    
    # Create the evolutionary trainer
    trainer = EvolutionaryTrainer(
        population_size=args.population,
        state_size=PokerConfig.STATE_SIZE,
        action_size=PokerConfig.ACTION_SIZE,
        model_save_path=args.model_path,
        games_per_evaluation=args.games,
        verbose=args.verbose
    )
    
    # Load existing models if continuing from checkpoint
    if args.checkpoint:
        logger.info("Loading models from checkpoint")
        trainer.load_population()
    
    # Run the evolutionary training process
    trainer.train(
        generations=args.generations,
        visualizer=visualizer
    )
    
    logger.info("Training complete")

def play_with_human(args):
    """Play poker with a human player against trained AI models."""
    logger.info("Starting game with human player")
    
    # Load the best trained models
    ai_models = []
    for i in range(args.num_players - 1):  # -1 for the human player
        model_path = os.path.join(args.model_path, f"best_model_{i}.h5")
        if os.path.exists(model_path):
            model = PokerModel(PokerConfig.STATE_SIZE, PokerConfig.ACTION_SIZE)
            model.load(model_path)
            ai_models.append(model)
        else:
            # If not enough trained models, create a new one
            logger.warning(f"Model {model_path} not found. Creating a new model.")
            model = PokerModel(PokerConfig.STATE_SIZE, PokerConfig.ACTION_SIZE)
            ai_models.append(model)
    
    # Create players
    players = []
    for i in range(args.num_players - 1):
        players.append(AIPlayer(f"AI_{i+1}", ai_models[i], args.initial_chips))
    
    # Add human player (always at position 0)
    players.insert(0, HumanPlayer("Human", args.initial_chips))
    
    # Create and run the game engine
    game_engine = GameEngine(players, verbose=True)  # Always verbose with human
    game_engine.run_game(num_hands=10)  # Play 10 hands by default
    
    logger.info("Game with human player completed")

def evaluate_models(args):
    """Evaluate the performance of trained models."""
    logger.info("Starting model evaluation")
    
    # Load the best trained models
    ai_models = []
    for i in range(args.num_players):
        model_path = os.path.join(args.model_path, f"best_model_{i}.h5")
        if os.path.exists(model_path):
            model = PokerModel(PokerConfig.STATE_SIZE, PokerConfig.ACTION_SIZE)
            model.load(model_path)
            ai_models.append(model)
        else:
            logger.error(f"Model {model_path} not found. Evaluation cannot proceed.")
            return
    
    # Create players
    players = [AIPlayer(f"AI_{i+1}", ai_models[i], args.initial_chips) for i in range(args.num_players)]
    
    # Create and run the game engine
    game_engine = GameEngine(players, verbose=args.verbose)
    results = game_engine.run_game(num_hands=args.games)
    
    # Display evaluation results
    logger.info("Evaluation Results:")
    for player, chips in results.items():
        logger.info(f"{player}: {chips} chips")
    
    logger.info("Evaluation complete")

def main():
    """Main entry point."""
    args = parse_arguments()
    setup_environment()
    
    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'play':
        play_with_human(args)
    elif args.mode == 'evaluate':
        evaluate_models(args)

if __name__ == "__main__":
    main()
