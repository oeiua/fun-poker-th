#!/usr/bin/env python
"""
Main entry point for the Poker AI application.
"""

import os
import argparse
import yaml
import logging
import tensorflow as tf
from datetime import datetime

from src.environment.game import PokerGame
from src.training.trainer import Trainer
from src.training.population import Population
from src.utils.resource import setup_resources


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Poker AI using evolutionary neural networks')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'play', 'evaluate'],
                        default='train', help='Mode to run the application')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint for continued training or playing')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with more verbose output')
    return parser.parse_args()


def setup_logging(config, debug=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if debug else getattr(logging, config['logging']['level'])
    
    # Create logs directory if it doesn't exist
    os.makedirs(config['logging']['save_path'], exist_ok=True)
    
    # Set up logging to file and console
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config['logging']['save_path'], f"poker_ai_{timestamp}.log")
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(config, checkpoint_path=None):
    """Train the Poker AI model."""
    logging.info("Starting training mode")
    
    # Initialize population and trainer
    population = Population(config)
    
    # Load from checkpoint if specified
    if checkpoint_path:
        logging.info(f"Loading population from checkpoint: {checkpoint_path}")
        population.load(checkpoint_path)
    
    trainer = Trainer(config, population)
    trainer.train()


def play(config, checkpoint_path):
    """Play against the AI."""
    logging.info("Starting play mode")
    
    if not checkpoint_path:
        logging.error("Checkpoint path must be specified for play mode")
        return
    
    # TODO: Implement play mode with human interface
    game = PokerGame(config)
    game.play_with_human(checkpoint_path)


def evaluate(config, checkpoint_path):
    """Evaluate the AI performance."""
    logging.info("Starting evaluation mode")
    
    if not checkpoint_path:
        logging.error("Checkpoint path must be specified for evaluation mode")
        return
    
    # TODO: Implement evaluation mode
    population = Population(config)
    population.load(checkpoint_path)
    population.evaluate()


def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config, args.debug)
    
    # Set up resources (CPU, GPU, memory)
    setup_resources(config)
    
    # Run in the specified mode
    if args.mode == 'train':
        train(config, args.checkpoint)
    elif args.mode == 'play':
        play(config, args.checkpoint)
    elif args.mode == 'evaluate':
        evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()