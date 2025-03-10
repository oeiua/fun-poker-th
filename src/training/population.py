"""
Population management for evolutionary algorithms.
"""

import os
import logging
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import tensorflow as tf
import json
from datetime import datetime

from src.models.model import PokerModel
from src.agent.nn_agent import NNAgent
from src.environment.game import PokerGame


class Individual:
    """
    Individual in the population, representing a single poker agent model.
    """
    
    def __init__(self, model: PokerModel, config: Dict[str, Any], id: str = None):
        """
        Initialize an individual.
        
        Args:
            model: The neural network model
            config: Configuration dictionary
            id: Unique identifier for this individual
        """
        self.model = model
        self.config = config
        self.id = id or datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{random.randint(1000, 9999)}"
        self.fitness = 0.0
        self.wins = 0
        self.games_played = 0
        self.total_reward = 0.0
    
    def create_agent(self, player_id: int) -> NNAgent:
        """
        Create an agent using this individual's model.
        
        Args:
            player_id: Player ID to assign to the agent
            
        Returns:
            Neural network agent
        """
        return NNAgent(player_id, self.model.model, self.config)
    
    def update_fitness(self, reward: float, win: bool = False) -> None:
        """
        Update the fitness of this individual.
        
        Args:
            reward: Reward received in a game
            win: Whether the agent won the game
        """
        self.total_reward += reward
        self.games_played += 1
        if win:
            self.wins += 1
        
        # Update overall fitness (weighted combination of win rate and average reward)
        win_rate = self.wins / max(1, self.games_played)
        avg_reward = self.total_reward / max(1, self.games_played)
        
        # Normalize reward to be in [0, 1] range (assuming rewards can be negative)
        # This is just a simplistic approach; you might need to adjust based on your reward scale
        normalized_reward = max(0, min(1, (avg_reward + 10000) / 20000))
        
        # Weight win rate more heavily than reward
        self.fitness = 0.7 * win_rate + 0.3 * normalized_reward
    
    def save(self, directory: str) -> None:
        """
        Save the individual to disk.
        
        Args:
            directory: Directory to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        model_path = os.path.join(directory, f"{self.id}_model")
        self.model.save(model_path)
        
        # Save metadata
        metadata = {
            'id': self.id,
            'fitness': self.fitness,
            'wins': self.wins,
            'games_played': self.games_played,
            'total_reward': self.total_reward,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(directory, f"{self.id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        logging.info(f"Saved individual {self.id} to {directory}")
    
    @classmethod
    def load(cls, directory: str, id: str, config: Dict[str, Any]) -> 'Individual':
        """
        Load an individual from disk.
        
        Args:
            directory: Directory to load from
            id: ID of the individual to load
            config: Configuration dictionary
            
        Returns:
            Loaded individual
        """
        # Load model
        model_path = os.path.join(directory, f"{id}_model")
        model = PokerModel(config)
        model.load(model_path)
        
        # Create individual
        individual = cls(model, config, id)
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{id}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        individual.fitness = metadata['fitness']
        individual.wins = metadata['wins']
        individual.games_played = metadata['games_played']
        individual.total_reward = metadata['total_reward']
        
        logging.info(f"Loaded individual {id} from {directory}")
        return individual


class Population:
    """
    Population of poker AI individuals for evolutionary training.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the population.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.evolution_config = config['evolution']
        self.population_size = self.evolution_config['population_size']
        self.mutation_rate = self.evolution_config['mutation_rate']
        self.crossover_rate = self.evolution_config['crossover_rate']
        self.tournament_size = self.evolution_config['tournament_size']
        self.elitism_count = self.evolution_config['elitism_count']
        
        self.individuals = []
        self.generation = 0
        self.best_individual = None
    
    def initialize(self) -> None:
        """Initialize the population with random individuals."""
        logging.info(f"Initializing population with {self.population_size} individuals")
        
        for _ in range(self.population_size):
            model = PokerModel(self.config)
            individual = Individual(model, self.config)
            self.individuals.append(individual)
        
        self.generation = 0
        logging.info("Population initialized")
    
    def evaluate(self, num_games: int = 10) -> None:
        """
        Evaluate the fitness of all individuals by playing games.
        
        Args:
            num_games: Number of games to play for each individual
        """
        logging.info(f"Evaluating population fitness over {num_games} games per individual")
        
        game = PokerGame(self.config)
        
        # For each individual
        for idx, individual in enumerate(self.individuals):
            logging.info(f"Evaluating individual {idx+1}/{len(self.individuals)}")
            
            # Reset fitness metrics
            individual.wins = 0
            individual.games_played = 0
            individual.total_reward = 0.0
            
            # Play multiple games
            for game_num in range(num_games):
                # Create agents for this game
                agents = []
                
                # Add the individual being evaluated
                eval_position = random.randint(0, self.config['game']['player_count'] - 1)
                for i in range(self.config['game']['player_count']):
                    if i == eval_position:
                        agents.append(individual.create_agent(i))
                    else:
                        # Use a random opponent from the population
                        opponent = random.choice(self.individuals)
                        while opponent.id == individual.id:  # Ensure different opponent
                            opponent = random.choice(self.individuals)
                        agents.append(opponent.create_agent(i))
                
                # Play the game
                payoffs = game.play_hand(agents)
                
                # Update fitness
                reward = payoffs[eval_position]
                win = reward > 0
                individual.update_fitness(reward, win)
                
                logging.debug(f"Individual {idx+1} game {game_num+1}: Reward = {reward}, Win = {win}")
            
            logging.info(f"Individual {idx+1} fitness: {individual.fitness:.4f} "
                         f"(Win rate: {individual.wins/max(1, individual.games_played):.2f}, "
                         f"Avg reward: {individual.total_reward/max(1, individual.games_played):.2f})")
        
        # Update best individual
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = self.individuals[0]
        
        logging.info(f"Evaluation complete. Best fitness: {self.best_individual.fitness:.4f}")
    
    def evolve(self) -> None:
        """Evolve the population using genetic operators."""
        logging.info(f"Evolving population (generation {self.generation + 1})")
        
        # Sort population by fitness (descending)
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)
        
        new_population = []
        
        # Elitism: carry over the best individuals unchanged
        for i in range(min(self.elitism_count, len(self.individuals))):
            new_population.append(self.individuals[i])
            logging.debug(f"Elite {i+1}: fitness = {self.individuals[i].fitness:.4f}")
        
        # Generate the rest of the population through selection, crossover, and mutation
        while len(new_population) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child_model = self._crossover(parent1.model, parent2.model)
            else:
                # No crossover, just clone one parent
                child_model = parent1.model.clone()
            
            # Mutation
            if random.random() < self.mutation_rate:
                child_model.mutate()
            
            # Create new individual
            child = Individual(child_model, self.config)
            new_population.append(child)
        
        # Replace population
        self.individuals = new_population
        self.generation += 1
        
        logging.info(f"Evolution complete. New generation: {self.generation}")
    
    def _tournament_selection(self) -> Individual:
        """
        Select an individual using tournament selection.
        
        Returns:
            Selected individual
        """
        # Randomly select tournament_size individuals
        tournament = random.sample(self.individuals, min(self.tournament_size, len(self.individuals)))
        
        # Return the best one
        return max(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, model1: PokerModel, model2: PokerModel) -> PokerModel:
        """
        Perform crossover between two models.
        
        Args:
            model1: First parent model
            model2: Second parent model
            
        Returns:
            Child model
        """
        # Create a new model
        child_model = PokerModel(self.config)
        
        # Get weights from both parents
        weights1 = model1.model.get_weights()
        weights2 = model2.model.get_weights()
        
        # Combine weights
        child_weights = []
        for w1, w2 in zip(weights1, weights2):
            # For each layer, randomly select weights from either parent
            if len(w1.shape) > 1:  # Dense layer weights
                # Create a mask for selecting weights
                mask = np.random.randint(0, 2, size=w1.shape).astype(bool)
                child_w = np.where(mask, w1, w2)
            else:  # Bias terms - select from one parent
                child_w = w1 if random.random() < 0.5 else w2
            
            child_weights.append(child_w)
        
        # Set weights on child model
        child_model.model.set_weights(child_weights)
        
        return child_model
    
    def save(self, directory: str) -> None:
        """
        Save the population to disk.
        
        Args:
            directory: Directory to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save population metadata
        metadata = {
            'generation': self.generation,
            'population_size': self.population_size,
            'best_individual_id': self.best_individual.id if self.best_individual else None,
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(directory, "population_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        # Save all individuals
        for individual in self.individuals:
            individual.save(os.path.join(directory, "individuals"))
        
        # Save best individual separately
        if self.best_individual:
            self.best_individual.save(os.path.join(directory, "best"))
        
        logging.info(f"Saved population to {directory}")
    
    def load(self, directory: str) -> None:
        """
        Load the population from disk.
        
        Args:
            directory: Directory to load from
        """
        # Load population metadata
        metadata_path = os.path.join(directory, "population_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.generation = metadata['generation']
        
        # Load individuals
        individuals_dir = os.path.join(directory, "individuals")
        self.individuals = []
        
        for filename in os.listdir(individuals_dir):
            if filename.endswith("_metadata.json"):
                id = filename.split("_metadata.json")[0]
                try:
                    individual = Individual.load(individuals_dir, id, self.config)
                    self.individuals.append(individual)
                except Exception as e:
                    logging.error(f"Failed to load individual {id}: {e}")
        
        # Load best individual if available
        best_dir = os.path.join(directory, "best")
        if os.path.exists(best_dir) and metadata['best_individual_id']:
            try:
                self.best_individual = Individual.load(best_dir, metadata['best_individual_id'], self.config)
            except Exception as e:
                logging.error(f"Failed to load best individual: {e}")
                # Use the best from loaded individuals as fallback
                if self.individuals:
                    self.best_individual = max(self.individuals, key=lambda x: x.fitness)
        elif self.individuals:
            # Use the best from loaded individuals
            self.best_individual = max(self.individuals, key=lambda x: x.fitness)
        
        # If we couldn't load enough individuals, create new ones to fill the population
        while len(self.individuals) < self.population_size:
            model = PokerModel(self.config)
            individual = Individual(model, self.config)
            self.individuals.append(individual)
        
        logging.info(f"Loaded population from {directory} (generation {self.generation})")