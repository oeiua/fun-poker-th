"""
Training loop for evolutionary poker AI.
"""

import os
import logging
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from src.training.population import Population
from src.training.metrics import TrainingMetrics
from src.utils.visualization import create_training_plots


class Trainer:
    """
    Trainer for the poker AI using evolutionary algorithms.
    """
    
    def __init__(self, config: Dict[str, Any], population: Population = None):
        """
        Initialize the trainer.
        
        Args:
            config: Configuration dictionary
            population: Optional pre-initialized population
        """
        self.config = config
        self.training_config = config['training']
        self.evolution_config = config['evolution']
        
        # Initialize or use provided population
        self.population = population or Population(config)
        if not self.population.individuals:
            self.population.initialize()
        
        # Setup training parameters
        self.generations = self.evolution_config['generations']
        self.episodes_per_generation = self.training_config['episodes_per_generation']
        self.hands_per_episode = self.training_config['hands_per_episode']
        self.checkpoint_frequency = self.training_config['checkpoint_frequency']
        self.eval_frequency = self.training_config['eval_frequency']
        
        # Initialize metrics
        self.metrics = TrainingMetrics()
        
        # Setup TensorBoard if enabled
        self.tensorboard = None
        if config['logging'].get('tensorboard', False):
            # Create logs directory
            log_dir = os.path.join(
                config['logging']['save_path'],
                'tensorboard',
                datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            os.makedirs(log_dir, exist_ok=True)
            self.tensorboard = TensorBoard(log_dir=log_dir)
            self.tensorboard.set_model(self.population.individuals[0].model.model)
            
            logging.info(f"TensorBoard logs will be saved to {log_dir}")
    
    def train(self) -> None:
        """Train the population over multiple generations."""
        logging.info(f"Starting training for {self.generations} generations")
        start_time = time.time()
        
        # Main training loop
        for generation in range(self.population.generation, self.generations):
            gen_start_time = time.time()
            logging.info(f"Generation {generation+1}/{self.generations}")
            
            # Evaluate population
            if generation % self.eval_frequency == 0:
                self.population.evaluate(self.episodes_per_generation)
                
                # Record metrics
                best_fitness = self.population.best_individual.fitness
                avg_fitness = np.mean([ind.fitness for ind in self.population.individuals])
                self.metrics.record_generation(generation, best_fitness, avg_fitness)
                
                # Log to TensorBoard if enabled
                if self.tensorboard:
                    logs = {
                        'best_fitness': best_fitness,
                        'avg_fitness': avg_fitness
                    }
                    self.tensorboard.on_epoch_end(generation, logs)
                
                logging.info(f"Evaluation - Best fitness: {best_fitness:.4f}, Avg fitness: {avg_fitness:.4f}")
            
            # Save checkpoint if needed
            if generation % self.checkpoint_frequency == 0:
                checkpoint_dir = os.path.join(
                    'checkpoints',
                    f'generation_{generation}'
                )
                self.population.save(checkpoint_dir)
                logging.info(f"Saved checkpoint to {checkpoint_dir}")
            
            # Evolve population to create next generation
            self.population.evolve()
            
            # Create visualization of training progress
            if self.metrics.has_data():
                plots_dir = os.path.join(self.config['logging']['save_path'], 'plots')
                os.makedirs(plots_dir, exist_ok=True)
                plot_path = os.path.join(plots_dir, f'training_progress_gen_{generation}.png')
                create_training_plots(self.metrics, plot_path)
                logging.info(f"Created training plots at {plot_path}")
            
            # Log generation time
            gen_time = time.time() - gen_start_time
            logging.info(f"Generation {generation+1} completed in {gen_time:.2f} seconds")
        
        # Training complete
        total_time = time.time() - start_time
        logging.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final population
        final_checkpoint_dir = os.path.join(
            'checkpoints',
            'final'
        )
        self.population.save(final_checkpoint_dir)
        logging.info(f"Saved final population to {final_checkpoint_dir}")
        
        # Create final visualization
        if self.metrics.has_data():
            plots_dir = os.path.join(self.config['logging']['save_path'], 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            plot_path = os.path.join(plots_dir, 'final_training_progress.png')
            create_training_plots(self.metrics, plot_path)
            logging.info(f"Created final training plots at {plot_path}")