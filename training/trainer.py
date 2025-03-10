"""
Main training loop for evolutionary poker AI with single-thread GPU usage.
"""
import os
import time
import torch
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue, Process
from typing import List, Dict, Any, Optional, Tuple, Union
import gc
import psutil
import copy
import pickle
from tqdm import tqdm

from config.config import Config
from game.environment import PokerEnvironment
from game.action import Action
from agents.neural_agent import NeuralAgent
from agents.random_agent import RandomAgent
from agents.base_agent import BaseAgent
from evolution.population import Population
from utils.visualization import plot_training_progress
from utils.memory_tracker import track_memory_usage
from utils.logger import Logger

class Trainer:
    """
    Manages the training process for poker AI using evolutionary algorithms.
    Restricts GPU usage to a single thread to prevent file handle exhaustion.
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        models_dir: Optional[str] = None,
        logs_dir: Optional[str] = None,
        num_workers: Optional[int] = None
    ):
        """Initialize the trainer."""
        # Initialize config
        self.config = config or Config()
        
        # Set up directories
        self.models_dir = models_dir or self.config.MODELS_DIR
        self.logs_dir = logs_dir or self.config.LOGS_DIR
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = Logger(os.path.join(self.logs_dir, "training.log"))
        
        # Set up device
        self.device = self.config.DEVICE
        self.logger.info(f"Using device: {self.device}")
        
        # Set up parallel processing
        self.num_workers = num_workers or self.config.NUM_WORKERS
        self.logger.info(f"Using {self.num_workers} worker processes for evaluation")
        
        # Initialize population
        self.population = Population(
            population_size=self.config.POPULATION_SIZE,
            input_size=self.config.INPUT_SIZE,
            hidden_layers=self.config.HIDDEN_LAYERS,
            output_size=self.config.OUTPUT_SIZE,
            device=self.device,
            models_dir=self.models_dir
        )
        
        # Training tracking
        self.current_generation = 0
        self.best_fitness = float('-inf')
        self.training_history = {
            'generation': [],
            'avg_fitness': [],
            'max_fitness': [],
            'min_fitness': [],
            'best_overall': []
        }
        
        # Memory usage tracking
        self.memory_usage = []
        
        # CPU-only evaluation flag
        self.use_cpu_for_eval = True if torch.cuda.is_available() else False
        
    def _cpu_compatible_agent(self, agent):
        """Create a CPU-compatible copy of an agent for evaluation."""
        # Create a simplified version of the agent that uses CPU only
        # We'll only copy the necessary data without the neural networks
        cpu_agent = copy.copy(agent)
        
        # Create simplified policy network
        if agent.policy_network:
            # Save policy network to a temporary file
            temp_policy_path = os.path.join(self.models_dir, f"temp_{id(agent)}_policy.pt")
            agent.policy_network.save(temp_policy_path)
            cpu_agent.policy_network_path = temp_policy_path
            cpu_agent.policy_network = None
            
        # Create simplified value network
        if agent.value_network:
            # Save value network to a temporary file
            temp_value_path = os.path.join(self.models_dir, f"temp_{id(agent)}_value.pt")
            agent.value_network.save(temp_value_path)
            cpu_agent.value_network_path = temp_value_path
            cpu_agent.value_network = None
            
        return cpu_agent
    
    def evaluate_agent_worker(self, agent_data, num_games, result_queue):
        """Worker process for agent evaluation."""
        try:
            # Set device to CPU for worker processes
            device = torch.device("cpu")
            
            # Recreate the agent if we have a simplified version
            if hasattr(agent_data, 'policy_network_path'):
                # This is a CPU-compatible version, load the actual networks
                from models.policy_network import PolicyNetwork
                from models.value_network import ValueNetwork
                
                policy_network = None
                if agent_data.policy_network_path and os.path.exists(agent_data.policy_network_path):
                    try:
                        policy_network = PolicyNetwork.load(agent_data.policy_network_path, device=device)
                    except Exception as e:
                        print(f"Error loading policy network: {str(e)}")
                        # Try loading with a direct approach
                        checkpoint = torch.load(agent_data.policy_network_path, map_location=device)
                        architecture = checkpoint.get('architecture', {})
                        
                        policy_network = PolicyNetwork(
                            input_size=architecture.get('input_size', 520),
                            hidden_layers=architecture.get('hidden_layers', [512, 256, 128, 64]),
                            output_size=architecture.get('output_size', 3),
                            learning_rate=architecture.get('learning_rate', 0.0001),
                            device=device
                        )
                        policy_network.load_state_dict(checkpoint['model_state_dict'])
                
                value_network = None
                if agent_data.value_network_path and os.path.exists(agent_data.value_network_path):
                    try:
                        value_network = ValueNetwork.load(agent_data.value_network_path, device=device)
                    except Exception as e:
                        print(f"Error loading value network: {str(e)}")
                        # Try loading with a direct approach - NOTE: ValueNetwork doesn't use output_size
                        checkpoint = torch.load(agent_data.value_network_path, map_location=device)
                        architecture = checkpoint.get('architecture', {})
                        
                        if 'output_size' in architecture:
                            del architecture['output_size']

                        value_network = ValueNetwork(
                            input_size=architecture.get('input_size', 520),
                            hidden_layers=architecture.get('hidden_layers', [512, 256, 128, 64]),
                            learning_rate=architecture.get('learning_rate', 0.0001),
                            device=device
                        )
                        value_network.load_state_dict(checkpoint['model_state_dict'])
                
                # Recreate the agent with loaded networks
                agent = NeuralAgent(
                    policy_network=policy_network,
                    value_network=value_network,
                    name=agent_data.name,
                    exploration_rate=agent_data.exploration_rate,
                    device=device
                )
                agent.set_player_index(agent_data.player_idx)
            else:
                # This is a normal agent, just use it directly
                agent = agent_data
            
            # Create default opponents (random agents)
            opponents = [
                RandomAgent(name=f"Random-{i}", aggression=np.random.uniform(0.3, 0.7))
                for i in range(self.config.NUM_PLAYERS - 1)
            ]
            
            # Set up environment
            env = PokerEnvironment(
                num_players=self.config.NUM_PLAYERS,
                starting_stack=self.config.STARTING_STACK,
                small_blind=self.config.SMALL_BLIND,
                big_blind=self.config.BIG_BLIND,
                max_rounds=num_games,
                action_timeout=self.config.ACTION_TIMEOUT
            )
            
            # Assign player indices
            agent.set_player_index(0)
            for i, opponent in enumerate(opponents):
                opponent.set_player_index(i + 1)
            
            # Reset environment
            state = env.reset()
            done = False
            
            # Play games
            while not done:
                # Get current player
                current_player_idx = state.get_current_player()
                
                # Get valid actions
                valid_actions = env.get_valid_actions(current_player_idx)
                valid_amounts = env.get_valid_bet_amounts(current_player_idx)
                
                # Get agent for current player
                current_agent = agent if current_player_idx == 0 else opponents[current_player_idx - 1]
                
                # Get action from agent
                action_type, bet_amount = current_agent.act(state, valid_actions, valid_amounts)
                
                # Execute action
                next_state, reward, done, info = env.step(current_player_idx, action_type, bet_amount)
                
                # Let agent observe the result
                current_agent.observe(state, (action_type, bet_amount), reward, next_state, done)
                
                # Update state
                state = next_state
            
            # Calculate fitness (agent's profit per hand)
            agent_stats = agent.get_stats()
            hands_played = max(1, agent_stats.get("hands_played", 1))
            total_reward = agent_stats.get("total_reward", 0.0)
            
            fitness = total_reward / hands_played
            
            # Put result in the queue
            result_queue.put((agent_data.player_idx if hasattr(agent_data, 'player_idx') else 0, fitness))
            
            # Clean up temporary files if used
            if hasattr(agent_data, 'policy_network_path') and agent_data.policy_network_path:
                if os.path.exists(agent_data.policy_network_path):
                    os.remove(agent_data.policy_network_path)
                    
            if hasattr(agent_data, 'value_network_path') and agent_data.value_network_path:
                if os.path.exists(agent_data.value_network_path):
                    os.remove(agent_data.value_network_path)
                    
        except Exception as e:
            # Log error and return a default fitness value
            print(f"Error in evaluation worker: {str(e)}")
            result_queue.put((agent_data.player_idx if hasattr(agent_data, 'player_idx') else 0, -1000.0))
    
    def evaluate_population_parallel(self, num_games: int = 100) -> List[float]:
        """
        Evaluate the entire population with GPU operations in main thread only.
        Worker processes use CPU only to prevent CUDA/file handle issues.
        
        Args:
            num_games: Number of games for each evaluation
            
        Returns:
            List of fitness scores
        """
        fitness_scores = [0.0] * len(self.population.population)
        
        if self.use_cpu_for_eval:
            # If using GPU, create CPU-compatible copies for workers
            cpu_agents = [self._cpu_compatible_agent(agent) for agent in self.population.population]
            for i, agent in enumerate(cpu_agents):
                agent.player_idx = i  # Set index for tracking results
        else:
            # If already on CPU, just use the agents directly
            cpu_agents = self.population.population
            for i, agent in enumerate(cpu_agents):
                agent.set_player_index(i)
        
        # Create a queue for results
        result_queue = mp.Queue()
        
        # Create worker processes
        processes = []
        for i, agent in enumerate(cpu_agents):
            p = mp.Process(
                target=self.evaluate_agent_worker,
                args=(agent, num_games, result_queue)
            )
            processes.append(p)
            p.start()
        
        # Collect results
        results_collected = 0
        with tqdm(total=len(cpu_agents), desc="Evaluating agents") as pbar:
            while results_collected < len(cpu_agents):
                if not result_queue.empty():
                    idx, fitness = result_queue.get()
                    fitness_scores[idx] = fitness
                    results_collected += 1
                    pbar.update(1)
                else:
                    time.sleep(0.1)
        
        # Wait for all processes to finish
        for p in processes:
            p.join()
        
        # Clean up
        for p in processes:
            if p.is_alive():
                p.terminate()
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return fitness_scores
    
    def train(
        self,
        num_generations: int = 100,
        eval_games: int = 100,
        checkpoint_frequency: int = 10,
        resume: bool = True
    ) -> Dict[str, Any]:
        """
        Train the population using evolutionary algorithms.
        
        Args:
            num_generations: Number of generations to train
            eval_games: Number of games for fitness evaluation
            checkpoint_frequency: Save checkpoint every N generations
            resume: Whether to resume from the last checkpoint
            
        Returns:
            Training history
        """
        # Try to resume from checkpoint if requested
        if resume:
            resume_success = self._resume_from_checkpoint()
            if resume_success:
                self.logger.info(f"Resumed training from generation {self.current_generation}")
            else:
                self.logger.info("Starting new training run")
        
        # Main training loop
        start_time = time.time()
        
        for generation in range(self.current_generation, num_generations):
            gen_start_time = time.time()
            self.logger.info(f"Generation {generation}/{num_generations}")
            
            # Evaluate population
            self.logger.info("Evaluating population...")
            fitness_scores = self.evaluate_population_parallel(num_games=eval_games)
            
            # Update population fitness
            self.population.fitness_scores = fitness_scores
            
            # Track best fitness
            max_fitness = max(fitness_scores)
            if max_fitness > self.best_fitness:
                self.best_fitness = max_fitness
                self.logger.info(f"New best fitness: {self.best_fitness}")
            
            # Calculate statistics
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            min_fitness = min(fitness_scores)
            
            # Update training history
            self.training_history['generation'].append(generation)
            self.training_history['avg_fitness'].append(avg_fitness)
            self.training_history['max_fitness'].append(max_fitness)
            self.training_history['min_fitness'].append(min_fitness)
            self.training_history['best_overall'].append(self.best_fitness)
            
            # Evolve population
            self.logger.info("Evolving population...")
            evolution_stats = self.population.evolve(
                evaluate_func=lambda agent: 0.0,  # Dummy function, we already evaluated
                tournament_size=self.config.TOURNAMENT_SIZE,
                mutation_rate=self.config.MUTATION_RATE,
                crossover_rate=self.config.CROSSOVER_RATE,
                elite_size=self.config.ELITE_SIZE
            )
            
            # Update generation counter
            self.current_generation = generation + 1
            
            # Check memory usage periodically
            if generation % self.config.MEMORY_CHECK_FREQUENCY == 0:
                memory_info = track_memory_usage()
                self.memory_usage.append(memory_info)
                self.logger.info(f"Memory usage: {memory_info['percent']}% ({memory_info['used'] / (1024 * 1024):.1f} MB)")
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save checkpoint periodically
            if generation % checkpoint_frequency == 0 or generation == num_generations - 1:
                self._save_checkpoint()
                self._plot_training_progress()
            
            # Calculate elapsed time
            gen_elapsed = time.time() - gen_start_time
            total_elapsed = time.time() - start_time
            if generation > self.current_generation:
                remaining = (total_elapsed / (generation - self.current_generation)) * (num_generations - generation)
            else:
                remaining = 0.0  # No time estimate for first generation
            
            self.logger.info(f"Generation {generation} completed in {gen_elapsed:.1f}s")
            self.logger.info(f"Avg fitness: {avg_fitness:.2f}, Max: {max_fitness:.2f}, Min: {min_fitness:.2f}")
            self.logger.info(f"Best overall: {self.best_fitness:.2f}")
            self.logger.info(f"Time elapsed: {total_elapsed:.1f}s, Estimated remaining: {remaining:.1f}s")
        
        self.logger.info(f"Training completed in {time.time() - start_time:.1f}s")
        self.logger.info(f"Best fitness achieved: {self.best_fitness:.2f}")
        
        # Final checkpoint
        self._save_checkpoint()
        self._plot_training_progress()
        
        return self.training_history
    
    def _save_checkpoint(self) -> None:
        """Save a checkpoint of the current training state."""
        # Save population
        checkpoint_dir = os.path.join(self.models_dir, f"checkpoint_gen{self.current_generation}")
        self.population.save_population(dir_name=f"checkpoint_gen{self.current_generation}")
        
        # Save training history
        history_path = os.path.join(checkpoint_dir, "training_history.npz")
        np.savez(
            history_path,
            generation=np.array(self.training_history['generation']),
            avg_fitness=np.array(self.training_history['avg_fitness']),
            max_fitness=np.array(self.training_history['max_fitness']),
            min_fitness=np.array(self.training_history['min_fitness']),
            best_overall=np.array(self.training_history['best_overall'])
        )
        
        self.logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def _resume_from_checkpoint(self) -> bool:
        """
        Resume training from the latest checkpoint.
        
        Returns:
            True if successful, False otherwise
        """
        # Find latest checkpoint
        checkpoints = [d for d in os.listdir(self.models_dir) if d.startswith("checkpoint_gen")]
        if not checkpoints:
            return False
        
        # Sort by generation number
        checkpoints.sort(key=lambda x: int(x.split("gen")[1]))
        latest_checkpoint = checkpoints[-1]
        
        # Load population
        checkpoint_dir = os.path.join(self.models_dir, latest_checkpoint)
        if not self.population.load_population(checkpoint_dir):
            return False
        
        # Load training history
        history_path = os.path.join(checkpoint_dir, "training_history.npz")
        if os.path.exists(history_path):
            history_data = np.load(history_path)
            self.training_history = {
                'generation': history_data['generation'].tolist(),
                'avg_fitness': history_data['avg_fitness'].tolist(),
                'max_fitness': history_data['max_fitness'].tolist(),
                'min_fitness': history_data['min_fitness'].tolist(),
                'best_overall': history_data['best_overall'].tolist()
            }
            
            # Update best fitness
            self.best_fitness = max(self.training_history['best_overall'])
        
        # Set current generation
        self.current_generation = self.population.current_generation
        
        return True
    
    def _plot_training_progress(self) -> None:
        """Plot and save training progress."""
        if not self.training_history['generation']:
            return
        
        # Create plots directory
        plots_dir = os.path.join(self.logs_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot fitness
        plot_path = os.path.join(plots_dir, f"fitness_gen{self.current_generation}.png")
        plot_training_progress(
            self.training_history,
            save_path=plot_path,
            title=f"Training Progress (Generation {self.current_generation})"
        )
        
        self.logger.info(f"Training progress plot saved to {plot_path}")
    
    def test_best_agent(self, num_games: int = 1000, against_random: bool = True) -> Dict[str, Any]:
        """
        Test the best agent's performance.
        
        Args:
            num_games: Number of games to play
            against_random: Whether to test against random agents
            
        Returns:
            Dictionary with test results
        """
        # Get best agent
        best_agent = self.population.get_best_agent()
        
        # Create opponents
        opponents = []
        if against_random:
            # Random opponents with varying aggression
            for i in range(self.config.NUM_PLAYERS - 1):
                opponents.append(RandomAgent(name=f"Random-{i}", aggression=np.random.uniform(0.3, 0.7)))
        else:
            # Use other agents from the population
            for i, agent in enumerate(self.population.population[:self.config.NUM_PLAYERS - 1]):
                if agent != best_agent:
                    agent_copy = NeuralAgent(
                        policy_network=agent.policy_network,
                        value_network=agent.value_network,
                        name=f"Agent-{i}",
                        exploration_rate=0.05,
                        device=self.device
                    )
                    opponents.append(agent_copy)
        
        # Create a queue for results
        result_queue = mp.Queue()
        
        # Create a CPU-compatible copy of the best agent
        cpu_agent = self._cpu_compatible_agent(best_agent) if self.use_cpu_for_eval else best_agent
        cpu_agent.player_idx = 0
        
        # Test in a separate process
        p = mp.Process(
            target=self.evaluate_agent_worker,
            args=(cpu_agent, num_games, result_queue)
        )
        p.start()
        
        # Get result
        self.logger.info(f"Testing best agent over {num_games} games...")
        idx, fitness = result_queue.get()
        p.join()
        
        # Get agent stats
        stats = best_agent.get_stats()
        
        # Create results
        results = {
            'fitness': fitness,
            'hands_played': stats.get('hands_played', 0),
            'hands_won': stats.get('hands_won', 0),
            'win_rate': stats.get('hands_won', 0) / max(1, stats.get('hands_played', 1)),
            'total_reward': stats.get('total_reward', 0.0),
            'actions': stats.get('actions', {})
        }
        
        # Log results
        self.logger.info(f"Test results:")
        self.logger.info(f"  Fitness: {results['fitness']:.2f}")
        self.logger.info(f"  Hands played: {results['hands_played']}")
        self.logger.info(f"  Hands won: {results['hands_won']}")
        self.logger.info(f"  Win rate: {results['win_rate'] * 100:.2f}%")
        self.logger.info(f"  Total reward: {results['total_reward']:.2f}")
        self.logger.info(f"  Actions: Fold: {results['actions'].get(Action.FOLD, 0)}, "
                         f"Check/Call: {results['actions'].get(Action.CHECK_CALL, 0)}, "
                         f"Bet/Raise: {results['actions'].get(Action.BET_RAISE, 0)}")
        
        return results