"""
Visualization tools for the poker AI training process.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime

from config import PokerConfig

logger = logging.getLogger("PokerAI.Visualizer")

class TrainingVisualizer:
    """
    Class for visualizing the poker AI training process.
    """
    def __init__(self, save_dir: str = None):
        """
        Initialize the training visualizer.
        
        Args:
            save_dir (str): Directory to save visualization files
        """
        if save_dir is None:
            self.save_dir = os.path.join(PokerConfig.LOGS_PATH, "visualizations")
        else:
            self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize plot data
        self.generations = []
        self.best_fitness = []
        self.avg_fitness = []
        self.median_fitness = []
        self.worst_fitness = []
        
        # Initialize figures
        self.fitness_fig, self.fitness_ax = plt.subplots(figsize=(10, 6))
        self.distribution_fig, self.distribution_ax = plt.subplots(figsize=(10, 6))
    
    def update(self, generation: int, fitness_scores: List[float], best_fitness_history: List[float]):
        """
        Update visualizations with new generation data.
        
        Args:
            generation (int): Current generation
            fitness_scores (List[float]): Fitness scores for the current generation
            best_fitness_history (List[float]): History of best fitness scores
        """
        # Update data
        self.generations.append(generation)
        self.best_fitness.append(np.max(fitness_scores))
        self.avg_fitness.append(np.mean(fitness_scores))
        self.median_fitness.append(np.median(fitness_scores))
        self.worst_fitness.append(np.min(fitness_scores))
        
        # Update fitness over time plot
        self._plot_fitness_over_time()
        
        # Update fitness distribution plot
        self._plot_fitness_distribution(fitness_scores)
        
        # Save plots
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.fitness_fig.savefig(os.path.join(self.save_dir, f"fitness_gen_{generation}.png"))
        self.distribution_fig.savefig(os.path.join(self.save_dir, f"distribution_gen_{generation}.png"))
        
        logger.info(f"Visualizations updated and saved for generation {generation}")
    
    def _plot_fitness_over_time(self):
        """Plot fitness metrics over generations."""
        self.fitness_ax.clear()
        
        self.fitness_ax.plot(self.generations, self.best_fitness, 'g-', label='Best Fitness', linewidth=2)
        self.fitness_ax.plot(self.generations, self.avg_fitness, 'b-', label='Average Fitness')
        self.fitness_ax.plot(self.generations, self.median_fitness, 'y-', label='Median Fitness')
        self.fitness_ax.plot(self.generations, self.worst_fitness, 'r-', label='Worst Fitness')
        
        self.fitness_ax.set_xlabel('Generation')
        self.fitness_ax.set_ylabel('Fitness')
        self.fitness_ax.set_title('Fitness Metrics Over Generations')
        self.fitness_ax.legend()
        self.fitness_ax.grid(True)
        
        # Ensure plot is updated
        self.fitness_fig.tight_layout()
        self.fitness_fig.canvas.draw_idle()
    
    def _plot_fitness_distribution(self, fitness_scores: List[float]):
        """
        Plot the distribution of fitness scores.
        
        Args:
            fitness_scores (List[float]): Current generation's fitness scores
        """
        self.distribution_ax.clear()
        
        # Create histogram
        self.distribution_ax.hist(fitness_scores, bins=20, alpha=0.7, color='skyblue')
        
        # Add lines for statistics
        self.distribution_ax.axvline(np.min(fitness_scores), color='r', linestyle='dashed', linewidth=1, label='Min')
        self.distribution_ax.axvline(np.max(fitness_scores), color='g', linestyle='dashed', linewidth=1, label='Max')
        self.distribution_ax.axvline(np.mean(fitness_scores), color='b', linestyle='dashed', linewidth=1, label='Mean')
        self.distribution_ax.axvline(np.median(fitness_scores), color='y', linestyle='dashed', linewidth=1, label='Median')
        
        self.distribution_ax.set_xlabel('Fitness Score')
        self.distribution_ax.set_ylabel('Count')
        self.distribution_ax.set_title(f'Fitness Distribution (Generation {self.generations[-1]})')
        self.distribution_ax.legend()
        
        # Ensure plot is updated
        self.distribution_fig.tight_layout()
        self.distribution_fig.canvas.draw_idle()
    
    def save_final_visualization(self, save_path: str = None):
        """
        Save final visualization summary.
        
        Args:
            save_path (str): Path to save the final visualization
        """
        if save_path is None:
            save_path = os.path.join(self.save_dir, "final_visualization.png")
        
        # Create a larger figure for the final visualization
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot fitness over time
        axes[0].plot(self.generations, self.best_fitness, 'g-', label='Best Fitness', linewidth=2)
        axes[0].plot(self.generations, self.avg_fitness, 'b-', label='Average Fitness')
        axes[0].plot(self.generations, self.median_fitness, 'y-', label='Median Fitness')
        axes[0].plot(self.generations, self.worst_fitness, 'r-', label='Worst Fitness')
        
        axes[0].set_xlabel('Generation')
        axes[0].set_ylabel('Fitness')
        axes[0].set_title('Fitness Metrics Over Generations')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot improvement rate (derivative of best fitness)
        if len(self.best_fitness) > 1:
            improvements = np.diff(self.best_fitness)
            axes[1].bar(self.generations[1:], improvements, color='green', alpha=0.7)
            axes[1].set_xlabel('Generation')
            axes[1].set_ylabel('Improvement')
            axes[1].set_title('Best Fitness Improvement Rate')
            axes[1].grid(True)
        
        # Finalize and save
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        
        logger.info(f"Final visualization saved to {save_path}")


class GameVisualizer:
    """
    Class for visualizing poker game results.
    """
    def __init__(self, save_dir: str = None):
        """
        Initialize the game visualizer.
        
        Args:
            save_dir (str): Directory to save visualization files
        """
        if save_dir is None:
            self.save_dir = os.path.join(PokerConfig.LOGS_PATH, "game_visualizations")
        else:
            self.save_dir = save_dir
        
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize tracking data
        self.hand_results = []
        self.player_chips = {}
    
    def update_hand_result(self, hand_num: int, players: List[Any], winners: List[Any]):
        """
        Update with results from a single hand.
        
        Args:
            hand_num (int): Hand number
            players (List[Any]): List of players
            winners (List[Any]): List of winning players
        """
        # Record chip counts
        hand_data = {
            'hand_num': hand_num,
            'player_chips': {},
            'winners': [winner.name for winner in winners]
        }
        
        for player in players:
            hand_data['player_chips'][player.name] = player.chips
            
            # Initialize player in overall tracking if needed
            if player.name not in self.player_chips:
                self.player_chips[player.name] = []
            
            # Add current chip count to player history
            self.player_chips[player.name].append(player.chips)
        
        self.hand_results.append(hand_data)
    
    def visualize_game(self, game_id: str = None):
        """
        Create visualizations for a complete game.
        
        Args:
            game_id (str): Identifier for the game
        """
        if not self.hand_results:
            logger.warning("No game data to visualize")
            return
        
        if game_id is None:
            game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create figure for chip progression
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot chip progression for each player
        hand_nums = list(range(1, len(next(iter(self.player_chips.values()))) + 1))
        
        for player_name, chip_history in self.player_chips.items():
            ax.plot(hand_nums, chip_history, marker='o', linewidth=2, label=player_name)
        
        ax.set_xlabel('Hand Number')
        ax.set_ylabel('Chips')
        ax.set_title('Player Chip Progression')
        ax.grid(True)
        ax.legend()
        
        # Save the plot
        save_path = os.path.join(self.save_dir, f"game_{game_id}_chip_progression.png")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        
        # Create a win distribution chart
        win_counts = {}
        for hand_data in self.hand_results:
            for winner in hand_data['winners']:
                win_counts[winner] = win_counts.get(winner, 0) + 1
        
        # Plot win distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        players = list(win_counts.keys())
        wins = [win_counts[player] for player in players]
        
        ax.bar(players, wins, color='gold')
        ax.set_xlabel('Player')
        ax.set_ylabel('Number of Hands Won')
        ax.set_title('Hand Win Distribution')
        
        # Save the plot
        save_path = os.path.join(self.save_dir, f"game_{game_id}_win_distribution.png")
        fig.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        
        logger.info(f"Game visualizations saved to {self.save_dir}")
    
    def save_game_summary(self, game_id: str = None):
        """
        Save a text summary of the game.
        
        Args:
            game_id (str): Identifier for the game
        """
        if not self.hand_results:
            logger.warning("No game data to summarize")
            return
        
        if game_id is None:
            game_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        
        # Create summary file
        summary_path = os.path.join(self.save_dir, f"game_{game_id}_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("==========================================\n")
            f.write(f"Poker Game Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("==========================================\n\n")
            
            # Initial and final chip counts
            f.write("Player Chip Summary:\n")
            f.write("--------------------\n")
            
            for player_name, chip_history in self.player_chips.items():
                initial_chips = chip_history[0]
                final_chips = chip_history[-1]
                net_change = final_chips - initial_chips
                
                f.write(f"{player_name}:\n")
                f.write(f"  Initial: {initial_chips}\n")
                f.write(f"  Final:   {final_chips}\n")
                f.write(f"  Net:     {net_change:+d}\n\n")
            
            # Win statistics
            win_counts = {}
            for hand_data in self.hand_results:
                for winner in hand_data['winners']:
                    win_counts[winner] = win_counts.get(winner, 0) + 1
            
            f.write("Win Statistics:\n")
            f.write("--------------\n")
            
            total_hands = len(self.hand_results)
            for player_name, wins in win_counts.items():
                win_percentage = (wins / total_hands) * 100
                f.write(f"{player_name}: {wins} wins ({win_percentage:.1f}%)\n")
        
        logger.info(f"Game summary saved to {summary_path}")