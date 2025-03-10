"""
Main entry point for training the poker AI.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import torch
import numpy as np
import random
import time
from typing import Dict, Any

from config.config import Config
from training.trainer import Trainer
from utils.logger import Logger
from utils.memory_tracker import track_memory_usage, force_gc

def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train poker AI using evolutionary algorithms")
    
    parser.add_argument(
        "--generations",
        type=int,
        default=None,
        help="Number of generations to train"
    )
    
    parser.add_argument(
        "--population",
        type=int,
        default=None,
        help="Population size"
    )
    
    parser.add_argument(
        "--eval-games",
        type=int,
        default=None,
        help="Number of games for evaluation"
    )
    
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Checkpoint frequency (in generations)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel evaluation"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use (cpu or cuda)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint (start new training)"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run a test after training"
    )
    
    return parser.parse_args()

def main() -> None:
    """Main function."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Create configuration
    config = Config()
    
    # Override config with command-line arguments
    if args.generations is not None:
        config.GENERATIONS = args.generations
    
    if args.population is not None:
        config.POPULATION_SIZE = args.population
    
    if args.eval_games is not None:
        config.NUM_EVAL_GAMES = args.eval_games
    
    if args.checkpoint_freq is not None:
        config.CHECKPOINT_FREQUENCY = args.checkpoint_freq
    
    if args.num_workers is not None:
        config.NUM_WORKERS = args.num_workers
    
    if args.device is not None:
        config.DEVICE = torch.device(args.device)
    
    # Create logger
    logger = Logger(os.path.join(config.LOGS_DIR, "training.log"))
    
    # Log configuration
    logger.log_section("Configuration")
    logger.info(f"Generations: {config.GENERATIONS}")
    logger.info(f"Population size: {config.POPULATION_SIZE}")
    logger.info(f"Evaluation games: {config.NUM_EVAL_GAMES}")
    logger.info(f"Checkpoint frequency: {config.CHECKPOINT_FREQUENCY}")
    logger.info(f"Number of workers: {config.NUM_WORKERS}")
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Resume from checkpoint: {not args.no_resume}")
    
    # Check system
    logger.log_section("System Check")
    memory_info = track_memory_usage()
    logger.log_memory_usage(memory_info)
    
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Create trainer
    trainer = Trainer(config=config)
    
    # Start training
    logger.log_section("Training")
    training_history = trainer.train(
        num_generations=config.GENERATIONS,
        eval_games=config.NUM_EVAL_GAMES,
        checkpoint_frequency=config.CHECKPOINT_FREQUENCY,
        resume=not args.no_resume
    )
    
    # Log training results
    logger.log_section("Training Results")
    if training_history:
        last_gen = len(training_history['generation']) - 1
        logger.info(f"Generations completed: {training_history['generation'][-1]}")
        logger.info(f"Final average fitness: {training_history['avg_fitness'][-1]:.2f}")
        logger.info(f"Final maximum fitness: {training_history['max_fitness'][-1]:.2f}")
        logger.info(f"Best fitness achieved: {training_history['best_overall'][-1]:.2f}")
    
    # Run test if requested
    if args.test:
        logger.log_section("Testing Best Agent")
        test_results = trainer.test_best_agent(num_games=1000)
        
        logger.info(f"Test fitness: {test_results['fitness']:.2f}")
        logger.info(f"Hands played: {test_results['hands_played']}")
        logger.info(f"Hands won: {test_results['hands_won']}")
        logger.info(f"Win rate: {test_results['win_rate'] * 100:.2f}%")
        logger.info(f"Total reward: {test_results['total_reward']:.2f}")
    
    # Clean up
    force_gc()
    
    # Final memory usage
    memory_info = track_memory_usage()
    logger.log_memory_usage(memory_info)
    
    # Log elapsed time
    logger.log_elapsed_time("Total training time")

if __name__ == "__main__":
    main()