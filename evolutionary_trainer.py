"""
Evolutionary algorithm trainer for poker AI.
"""
import os
import random
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
import tensorflow as tf
from datetime import datetime

from config import PokerConfig
from model import PokerModel
from player import AIPlayer
from game_engine import GameEngine

logger = logging.getLogger("PokerAI.EvolutionaryTrainer")

class EvolutionaryTrainer:
    """
    Class for training poker AI models using evolutionary algorithms.
    """
    def __init__(
        self,
        population_size: int,
        state_size: int,
        action_size: int,
        model_save_path: str,
        games_per_evaluation: int = 100,
        verbose: bool = False
    ):
        """
        Initialize the evolutionary trainer.
        
        Args:
            population_size (int): Size of the model population
            state_size (int): Size of the state input vector
            action_size (int): Size of the action output vector
            model_save_path (str): Path to save models
            games_per_evaluation (int): Number of games for each evaluation
            verbose (bool): Whether to output training progress details
        """
        self.population_size = population_size
        self.state_size = state_size
        self.action_size = action_size
        self.model_save_path = model_save_path
        self.games_per_evaluation = games_per_evaluation
        self.verbose = verbose
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        os.makedirs(os.path.join(model_save_path, "checkpoints"), exist_ok=True)
        
        # Initialize population
        self.population = []
        self.fitness_scores = []
        self.generation = 0
        self.best_fitness_history = []
        
        # For parallel training
        self.strategy = tf.distribute.MirroredStrategy()
        
        # Create initial population
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize the population of models."""
        logger.info(f"Initializing population of {self.population_size} models")
        
        # Create models in parallel where possible
        with self.strategy.scope():
            for i in range(self.population_size):
                model = PokerModel(self.state_size, self.action_size)
                self.population.append(model)
                self.fitness_scores.append(0)
    
    def train(self, generations: int, visualizer=None):
        """
        Run the evolutionary training process.
        
        Args:
            generations (int): Number of generations to train
            visualizer: Visualizer for training progress
        """
        logger.info(f"Starting evolutionary training for {generations} generations")
        
        for gen in range(self.generation, self.generation + generations):
            start_time = datetime.now()
            logger.info(f"Generation {gen+1}/{self.generation + generations}")
            
            # Evaluate all models
            self.fitness_scores = self._evaluate_population()
            
            # Save the best model
            best_idx = np.argmax(self.fitness_scores)
            best_fitness = self.fitness_scores[best_idx]
            self.best_fitness_history.append(best_fitness)
            
            # Save checkpoint
            self._save_checkpoint(gen, best_idx)
            
            # Display progress
            self._display_progress(gen, best_fitness)
            
            # Update visualizer if provided
            if visualizer:
                visualizer.update(gen, self.fitness_scores, self.best_fitness_history)
            
            # Create the next generation
            if gen < self.generation + generations - 1:  # Skip for last generation
                self._create_next_generation()
            
            # Log generation time
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Generation {gen+1} completed in {duration:.2f} seconds")
        
        self.generation += generations
        logger.info("Training completed")
    
    def _evaluate_population(self) -> List[float]:
        """
        Evaluate all models in the population.
        
        Returns:
            List[float]: Fitness scores for each model
        """
        fitness_scores = np.zeros(self.population_size)
        
        # Tournament style evaluation
        num_tournaments = self.games_per_evaluation
        players_per_tournament = min(PokerConfig.TOURNAMENT_SIZE, self.population_size)
        
        for _ in range(num_tournaments):
            # Select random models for this tournament
            tournament_indices = random.sample(range(self.population_size), players_per_tournament)
            
            # Create players using these models
            players = []
            for idx in tournament_indices:
                model = self.population[idx]
                players.append(AIPlayer(f"AI_{idx}", model, PokerConfig.INITIAL_CHIPS))
            
            # Run a game with these players
            game_engine = GameEngine(players, verbose=False)
            results = game_engine.run_game(num_hands=10)  # Play 10 hands per tournament
            
            # Update fitness scores based on final chip counts
            for idx, player_idx in enumerate(tournament_indices):
                player_name = f"AI_{player_idx}"
                final_chips = results.get(player_name, 0)
                
                # Calculate fitness as the relative performance
                relative_performance = final_chips / PokerConfig.INITIAL_CHIPS
                fitness_scores[player_idx] += relative_performance
        
        # Normalize fitness scores
        if num_tournaments > 0:
            fitness_scores = fitness_scores / num_tournaments
        
        return fitness_scores.tolist()
    
    def _create_next_generation(self):
        """Create the next generation of models using selection, crossover, and mutation."""
        new_population = []
        
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        # Elitism: Keep the best individuals unchanged
        num_elites = max(1, int(self.population_size * PokerConfig.ELITE_PERCENTAGE))
        for i in range(num_elites):
            elite_idx = sorted_indices[i]
            new_population.append(self.population[elite_idx])
        
        # Fill the rest with offspring
        while len(new_population) < self.population_size:
            # Select parents using tournament selection
            parent1_idx = self._tournament_selection(3)
            parent2_idx = self._tournament_selection(3)
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # Crossover with probability
            if random.random() < PokerConfig.CROSSOVER_RATE:
                child = parent1.crossover(parent2)
            else:
                # No crossover, just clone the better parent
                if self.fitness_scores[parent1_idx] > self.fitness_scores[parent2_idx]:
                    child = parent1.clone()
                else:
                    child = parent2.clone()
            
            # Mutation with probability
            if random.random() < PokerConfig.MUTATION_RATE:
                child.mutate()
            
            new_population.append(child)
        
        # Replace old population
        self.population = new_population
    
    def _tournament_selection(self, tournament_size: int) -> int:
        """
        Select a model index using tournament selection.
        
        Args:
            tournament_size (int): Number of individuals in the tournament
            
        Returns:
            int: Index of the selected model
        """
        tournament = random.sample(range(self.population_size), tournament_size)
        winner_idx = tournament[0]
        
        for idx in tournament:
            if self.fitness_scores[idx] > self.fitness_scores[winner_idx]:
                winner_idx = idx
        
        return winner_idx
    
    def _save_checkpoint(self, generation: int, best_idx: int):
        """
        Save checkpoint of the current generation.
        
        Args:
            generation (int): Current generation number
            best_idx (int): Index of the best model
        """
        # Save the best model
        best_model_path = os.path.join(self.model_save_path, f"best_model.h5")
        self.population[best_idx].save(best_model_path)
        
        # Also save as specific generation best
        gen_model_path = os.path.join(
            self.model_save_path, 
            "checkpoints", 
            f"best_model_gen_{generation}.h5"
        )
        self.population[best_idx].save(gen_model_path)
        
        # Save top models individually (for use in gameplay)
        top_k = min(10, self.population_size)  # Save top 10 or fewer models
        sorted_indices = np.argsort(self.fitness_scores)[::-1]
        
        for i in range(top_k):
            model_idx = sorted_indices[i]
            model_path = os.path.join(self.model_save_path, f"best_model_{i}.h5")
            self.population[model_idx].save(model_path)
        
        # Save training history
        history = {
            'generation': self.generation,
            'best_fitness_history': self.best_fitness_history,
            'latest_fitness_scores': self.fitness_scores
        }
        
        history_path = os.path.join(self.model_save_path, "training_history.npz")
        np.savez(history_path, **history)
    
    def load_population(self):
        """Load the population from saved models."""
        history_path = os.path.join(self.model_save_path, "training_history.npz")
        
        if os.path.exists(history_path):
            # Load training history
            history = np.load(history_path, allow_pickle=True)
            self.generation = history['generation'].item()
            self.best_fitness_history = history['best_fitness_history'].tolist()
            
            logger.info(f"Loaded training history from generation {self.generation}")
            
            # Try to load saved models
            for i in range(self.population_size):
                model_path = ""
                
                # Try loading specific model if available
                if i < 10:  # We save top 10 models
                    model_path = os.path.join(self.model_save_path, f"best_model_{i}.h5")
                
                # If specific model not found, use the overall best
                if not os.path.exists(model_path):
                    model_path = os.path.join(self.model_save_path, "best_model.h5")
                
                if os.path.exists(model_path):
                    self.population[i].load(model_path)
                    logger.info(f"Loaded model from {model_path}")
                else:
                    logger.warning(f"No saved model found at {model_path}")
        else:
            logger.warning("No training history found. Starting from scratch.")
    
    def _display_progress(self, generation: int, best_fitness: float):
        """
        Display training progress.
        
        Args:
            generation (int): Current generation
            best_fitness (float): Best fitness score
        """
        logger.info(f"Generation {generation+1} - Best fitness: {best_fitness:.4f}")
        
        if self.verbose:
            # Display more detailed statistics
            avg_fitness = np.mean(self.fitness_scores)
            median_fitness = np.median(self.fitness_scores)
            worst_fitness = np.min(self.fitness_scores)
            
            logger.info(f"  Average fitness: {avg_fitness:.4f}")
            logger.info(f"  Median fitness: {median_fitness:.4f}")
            logger.info(f"  Worst fitness: {worst_fitness:.4f}")
