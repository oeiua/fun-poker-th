"""
Population management for evolutionary algorithms.
"""
import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import time
import uuid
import shutil

from agents.neural_agent import NeuralAgent
from models.policy_network import PolicyNetwork
from models.value_network import ValueNetwork
from config.config import Config

class Population:
    """
    Manages a population of neural network agents for evolutionary training.
    """
    
    def __init__(
        self,
        population_size: int = 100,
        input_size: int = 520,
        hidden_layers: List[int] = [512, 256, 128, 64],
        output_size: int = 3,
        device: Optional[torch.device] = None,
        models_dir: str = "saved_models"
    ):
        """
        Initialize a population of neural network agents.
        
        Args:
            population_size: Number of agents in the population
            input_size: Input size for neural networks
            hidden_layers: Hidden layer sizes for neural networks
            output_size: Output size for neural networks
            device: Device to run the networks on
            models_dir: Directory to save models
        """
        self.population_size = population_size
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Initialize empty population
        self.population = []
        self.fitness_scores = []
        
        # Generation tracking
        self.current_generation = 0
        
        # Best agent tracking
        self.best_agent = None
        self.best_fitness = float('-inf')
        
        # Initialize population
        self._initialize_population()
    
    def _initialize_population(self) -> None:
        """Initialize the population with random agents."""
        self.population = []
        
        print(f"Initializing population of {self.population_size} agents...")
        
        for i in range(self.population_size):
            # Create policy network with random weights
            policy_network = PolicyNetwork(
                input_size=self.input_size,
                hidden_layers=self.hidden_layers,
                output_size=self.output_size,
                device=self.device
            )
            
            # Create value network with random weights
            value_network = ValueNetwork(
                input_size=self.input_size,
                hidden_layers=self.hidden_layers,
                device=self.device
            )
            
            # Create agent with the networks
            agent = NeuralAgent(
                policy_network=policy_network,
                value_network=value_network,
                name=f"Agent-Gen{self.current_generation}-{i}",
                exploration_rate=0.1,
                device=self.device
            )
            
            self.population.append(agent)
        
        # Initialize fitness scores
        self.fitness_scores = [0.0] * self.population_size
        
        print("Population initialized.")
    
    def evaluate_fitness(self, evaluate_func: Callable[[NeuralAgent], float]) -> List[float]:
        """
        Evaluate the fitness of each agent in the population.
        
        Args:
            evaluate_func: Function that takes an agent and returns its fitness score
            
        Returns:
            List of fitness scores
        """
        self.fitness_scores = []
        
        print(f"Evaluating fitness for generation {self.current_generation}...")
        
        # Evaluate each agent
        for i, agent in enumerate(self.population):
            start_time = time.time()
            
            # Get fitness score
            fitness = evaluate_func(agent)
            self.fitness_scores.append(fitness)
            
            # Update best agent if needed
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_agent = agent
                
                # Save best agent
                self._save_best_agent()
            
            # Log progress
            elapsed = time.time() - start_time
            print(f"Agent {i}/{self.population_size}: Fitness = {fitness:.2f}, Time = {elapsed:.2f}s")
        
        # Log generation stats
        avg_fitness = sum(self.fitness_scores) / len(self.fitness_scores)
        max_fitness = max(self.fitness_scores)
        min_fitness = min(self.fitness_scores)
        
        print(f"Generation {self.current_generation} stats:")
        print(f"  Average fitness: {avg_fitness:.2f}")
        print(f"  Max fitness: {max_fitness:.2f}")
        print(f"  Min fitness: {min_fitness:.2f}")
        print(f"  Best fitness overall: {self.best_fitness:.2f}")
        
        return self.fitness_scores
    
    def select_parents(self, selection_size: int, tournament_size: int = 5) -> List[NeuralAgent]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            selection_size: Number of parents to select
            tournament_size: Size of each tournament
            
        Returns:
            List of selected parent agents
        """
        parents = []
        
        for _ in range(selection_size):
            # Tournament selection
            tournament_indices = np.random.choice(len(self.population), tournament_size, replace=False)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            
            # Select the best agent from the tournament
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(self.population[winner_idx])
        
        return parents
    
    def create_offspring(
        self,
        parents: List[NeuralAgent],
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.7
    ) -> List[NeuralAgent]:
        """
        Create offspring from selected parents through crossover and mutation.
        
        Args:
            parents: List of parent agents
            mutation_rate: Probability of mutation for each parameter
            crossover_rate: Probability of crossover between parents
            
        Returns:
            List of offspring agents
        """
        offspring = []
        
        # Number of offspring to create
        num_offspring = self.population_size - len(parents)
        
        for i in range(num_offspring):
            # Select two random parents
            parent1, parent2 = np.random.choice(parents, 2, replace=False)
            
            # Create new networks
            policy_network = self._create_offspring_network(
                parent1.policy_network,
                parent2.policy_network,
                mutation_rate,
                crossover_rate
            )
            
            value_network = self._create_offspring_network(
                parent1.value_network,
                parent2.value_network,
                mutation_rate,
                crossover_rate
            )
            
            # Create new agent
            agent = NeuralAgent(
                policy_network=policy_network,
                value_network=value_network,
                name=f"Agent-Gen{self.current_generation + 1}-{i}",
                exploration_rate=max(0.01, parent1.exploration_rate * 0.99),  # Gradually reduce exploration
                device=self.device
            )
            
            offspring.append(agent)
        
        return offspring
    
    def _create_offspring_network(
        self,
        network1: torch.nn.Module,
        network2: torch.nn.Module,
        mutation_rate: float,
        crossover_rate: float
    ) -> torch.nn.Module:
        """
        Create a new network from two parent networks.
        
        Args:
            network1: First parent network
            network2: Second parent network
            mutation_rate: Probability of mutation for each parameter
            crossover_rate: Probability of crossover between parents
            
        Returns:
            New network
        """
        # Determine network type and create a new instance
        if isinstance(network1, PolicyNetwork):
            new_network = PolicyNetwork(
                input_size=network1.input_size,
                hidden_layers=network1.hidden_layers,
                output_size=network1.output_size,
                device=self.device
            )
        elif isinstance(network1, ValueNetwork):
            new_network = ValueNetwork(
                input_size=network1.input_size,
                hidden_layers=network1.hidden_layers,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported network type: {type(network1)}")
        
        # Get state dictionaries
        state_dict1 = network1.state_dict()
        state_dict2 = network2.state_dict()
        new_state_dict = new_network.state_dict()
        
        # Apply crossover and mutation
        for key in new_state_dict:
            # Crossover
            if np.random.random() < crossover_rate:
                # Handle different tensor types appropriately
                tensor = new_state_dict[key]
                if tensor.dtype == torch.long or tensor.dtype == torch.int64 or tensor.dtype == torch.int32:
                    # For integer tensors, use a numpy-based approach
                    mask = (np.random.rand(*tensor.shape) < 0.5)
                    tensor1 = state_dict1[key].cpu().numpy()
                    tensor2 = state_dict2[key].cpu().numpy()
                    result = np.where(mask, tensor1, tensor2)
                    new_state_dict[key] = torch.tensor(result, dtype=tensor.dtype, device=tensor.device)
                else:
                    # For float tensors, use the PyTorch approach
                    mask = torch.rand_like(tensor, dtype=torch.float) < 0.5
                    new_state_dict[key] = torch.where(mask, state_dict1[key], state_dict2[key])
            else:
                # No crossover, just inherit from first parent
                new_state_dict[key] = state_dict1[key].clone()
            
            # Mutation
            if np.random.random() < mutation_rate:
                tensor = new_state_dict[key]
                if tensor.dtype == torch.long or tensor.dtype == torch.int64 or tensor.dtype == torch.int32:
                    # For integer tensors, add small integer noise
                    noise = torch.randint_like(tensor, low=-1, high=2)  # -1, 0, or 1
                    new_state_dict[key] = torch.clamp(tensor + noise, min=0)  # Ensure non-negative
                else:
                    # For float tensors, add normal noise
                    noise = torch.randn_like(tensor) * 0.1
                    new_state_dict[key] += noise
        
        # Load the new state dictionary
        new_network.load_state_dict(new_state_dict)
        
        return new_network
    
    def evolve(
        self,
        evaluate_func: Callable[[NeuralAgent], float],
        tournament_size: int = 5,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.7,
        elite_size: int = 5
    ) -> Dict[str, Any]:
        """
        Perform one generation of evolution.
        
        Args:
            evaluate_func: Function to evaluate agent fitness
            tournament_size: Size of tournament for selection
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elite_size: Number of top agents to preserve unchanged
            
        Returns:
            Dictionary with generation statistics
        """
        # Evaluate fitness if not already done
        if not self.fitness_scores or len(self.fitness_scores) != len(self.population):
            self.evaluate_fitness(evaluate_func)
        
        # Sort population by fitness
        sorted_indices = np.argsort(self.fitness_scores)[::-1]  # Descending order
        sorted_population = [self.population[i] for i in sorted_indices]
        sorted_fitness = [self.fitness_scores[i] for i in sorted_indices]
        
        # Select elites
        elites = sorted_population[:elite_size]
        
        # Select parents for reproduction
        parents = self.select_parents(
            selection_size=max(2, self.population_size // 5),
            tournament_size=tournament_size
        )
        
        # Create offspring
        offspring = self.create_offspring(
            parents=parents,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )
        
        # Create new population with elites and offspring
        self.population = elites + offspring
        
        # Reset fitness scores
        self.fitness_scores = [0.0] * len(self.population)
        
        # Increment generation counter
        self.current_generation += 1
        
        # Return statistics
        return {
            "generation": self.current_generation - 1,
            "avg_fitness": sum(sorted_fitness) / len(sorted_fitness),
            "max_fitness": sorted_fitness[0],
            "min_fitness": sorted_fitness[-1],
            "best_overall": self.best_fitness,
            "population_size": len(self.population)
        }
    
    def _save_best_agent(self) -> None:
        """Save the best agent."""
        if self.best_agent:
            best_dir = os.path.join(self.models_dir, "best_agent")
            os.makedirs(best_dir, exist_ok=True)
            
            self.best_agent.save(best_dir)
            
            # Save metadata
            with open(os.path.join(best_dir, "metadata.txt"), "w") as f:
                f.write(f"Generation: {self.current_generation}\n")
                f.write(f"Fitness: {self.best_fitness}\n")
                f.write(f"Name: {self.best_agent.name}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def save_population(self, dir_name: Optional[str] = None) -> str:
        """
        Save the entire population.
        
        Args:
            dir_name: Directory name for saving, defaults to current generation
            
        Returns:
            Path where population was saved
        """
        if dir_name is None:
            dir_name = f"generation_{self.current_generation}"
        
        save_dir = os.path.join(self.models_dir, dir_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each agent
        for i, agent in enumerate(self.population):
            agent_dir = os.path.join(save_dir, f"agent_{i}")
            os.makedirs(agent_dir, exist_ok=True)
            agent.save(agent_dir)
        
        # Save fitness scores and metadata
        with open(os.path.join(save_dir, "fitness.txt"), "w") as f:
            for i, fitness in enumerate(self.fitness_scores):
                f.write(f"{i},{fitness}\n")
        
        with open(os.path.join(save_dir, "metadata.txt"), "w") as f:
            f.write(f"Generation: {self.current_generation}\n")
            f.write(f"Population size: {len(self.population)}\n")
            f.write(f"Input size: {self.input_size}\n")
            f.write(f"Hidden layers: {self.hidden_layers}\n")
            f.write(f"Output size: {self.output_size}\n")
            f.write(f"Best fitness: {self.best_fitness}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return save_dir
    
    def load_population(self, dir_path: str) -> bool:
        """
        Load a population from a directory.
        
        Args:
            dir_path: Path to directory containing saved population
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(dir_path):
            print(f"Directory not found: {dir_path}")
            return False
        
        try:
            # Load metadata
            with open(os.path.join(dir_path, "metadata.txt"), "r") as f:
                metadata = {}
                for line in f:
                    key, value = line.strip().split(":", 1)
                    metadata[key.strip()] = value.strip()
            
            # Set generation
            self.current_generation = int(metadata.get("Generation", 0))
            
            # Clear current population
            self.population = []
            
            # Load each agent
            agent_dirs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d)) and d.startswith("agent_")]
            
            for agent_dir in agent_dirs:
                agent_path = os.path.join(dir_path, agent_dir)
                agent = NeuralAgent.load(agent_path, device=self.device)
                self.population.append(agent)
            
            # Load fitness scores if available
            self.fitness_scores = [0.0] * len(self.population)
            fitness_path = os.path.join(dir_path, "fitness.txt")
            if os.path.exists(fitness_path):
                with open(fitness_path, "r") as f:
                    for line in f:
                        idx, fitness = line.strip().split(",")
                        idx = int(idx)
                        fitness = float(fitness)
                        if 0 <= idx < len(self.fitness_scores):
                            self.fitness_scores[idx] = fitness
            
            # Load best agent if available
            best_dir = os.path.join(self.models_dir, "best_agent")
            if os.path.exists(best_dir):
                self.best_agent = NeuralAgent.load(best_dir, device=self.device)
                
                # Get best fitness
                with open(os.path.join(best_dir, "metadata.txt"), "r") as f:
                    for line in f:
                        if line.startswith("Fitness:"):
                            self.best_fitness = float(line.split(":", 1)[1].strip())
                            break
            
            print(f"Loaded population of {len(self.population)} agents from generation {self.current_generation}")
            return True
            
        except Exception as e:
            print(f"Error loading population: {str(e)}")
            return False
    
    def get_best_agent(self) -> NeuralAgent:
        """
        Get the best agent in the current population.
        
        Returns:
            Best agent
        """
        if self.best_agent:
            return self.best_agent
        
        if not self.fitness_scores:
            return self.population[0]
        
        best_idx = np.argmax(self.fitness_scores)
        return self.population[best_idx]