"""
Metrics tracking for poker AI training.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


class TrainingMetrics:
    """
    Class for tracking training metrics over time.
    """
    
    def __init__(self):
        """Initialize the metrics tracking."""
        self.generations = []
        self.best_fitness = []
        self.avg_fitness = []
        self.win_rates = []
        self.episode_rewards = []
        self.elapsed_times = []
    
    def record_generation(self, 
                        generation: int, 
                        best_fitness: float, 
                        avg_fitness: float,
                        win_rate: float = None,
                        avg_reward: float = None,
                        elapsed_time: float = None) -> None:
        """
        Record metrics for a generation.
        
        Args:
            generation: Generation number
            best_fitness: Best fitness in the population
            avg_fitness: Average fitness in the population
            win_rate: Optional win rate of the best individual
            avg_reward: Optional average reward of the best individual
            elapsed_time: Optional elapsed time for the generation
        """
        self.generations.append(generation)
        self.best_fitness.append(best_fitness)
        self.avg_fitness.append(avg_fitness)
        
        if win_rate is not None:
            self.win_rates.append(win_rate)
        
        if avg_reward is not None:
            self.episode_rewards.append(avg_reward)
        
        if elapsed_time is not None:
            self.elapsed_times.append(elapsed_time)
    
    def record_episode(self, 
                      episode: int, 
                      rewards: List[float],
                      win_status: List[bool],
                      elapsed_time: float = None) -> None:
        """
        Record metrics for a single episode.
        
        Args:
            episode: Episode number
            rewards: List of rewards for each agent
            win_status: List of win statuses for each agent
            elapsed_time: Optional elapsed time for the episode
        """
        self.episode_rewards.append((episode, np.mean(rewards)))
        self.win_rates.append((episode, np.mean(win_status)))
        
        if elapsed_time is not None:
            self.elapsed_times.append((episode, elapsed_time))
    
    def get_dataframe(self) -> pd.DataFrame:
        """
        Convert metrics to a pandas DataFrame.
        
        Returns:
            DataFrame containing all metrics
        """
        data = {
            'generation': self.generations,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
        }
        
        if len(self.win_rates) == len(self.generations):
            data['win_rate'] = self.win_rates
        
        if len(self.episode_rewards) == len(self.generations):
            data['avg_reward'] = self.episode_rewards
        
        if len(self.elapsed_times) == len(self.generations):
            data['elapsed_time'] = self.elapsed_times
        
        return pd.DataFrame(data)
    
    def save_to_csv(self, filepath: str) -> None:
        """
        Save metrics to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        df = self.get_dataframe()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
    
    def has_data(self) -> bool:
        """
        Check if metrics have data.
        
        Returns:
            True if metrics have data, False otherwise
        """
        return len(self.generations) > 0
    
    def get_best_generation(self) -> Tuple[int, float]:
        """
        Get the generation with the best fitness.
        
        Returns:
            Tuple of (generation, best_fitness)
        """
        if not self.has_data():
            return None, None
        
        best_idx = np.argmax(self.best_fitness)
        return self.generations[best_idx], self.best_fitness[best_idx]
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the training metrics.
        
        Returns:
            Dictionary containing summary statistics
        """
        if not self.has_data():
            return {}
        
        best_gen, best_fitness = self.get_best_generation()
        
        summary = {
            'generations': len(self.generations),
            'best_generation': best_gen,
            'best_fitness': best_fitness,
            'final_avg_fitness': self.avg_fitness[-1] if self.avg_fitness else None,
            'final_best_fitness': self.best_fitness[-1] if self.best_fitness else None,
        }
        
        if self.win_rates and len(self.win_rates) == len(self.generations):
            summary['final_win_rate'] = self.win_rates[-1]
        
        if self.episode_rewards and len(self.episode_rewards) == len(self.generations):
            summary['final_avg_reward'] = self.episode_rewards[-1]
        
        return summary