"""
Visualization utilities for poker AI training.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd

from src.training.metrics import TrainingMetrics


def create_training_plots(metrics: TrainingMetrics, filepath: str) -> None:
    """
    Create visualization plots for training progress.
    
    Args:
        metrics: Training metrics
        filepath: Path to save the plot
    """
    if not metrics.has_data():
        logging.warning("No metrics data available for plotting")
        return
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Plot fitness over generations
    generations = metrics.generations
    best_fitness = metrics.best_fitness
    avg_fitness = metrics.avg_fitness
    
    axs[0].plot(generations, best_fitness, 'b-', label='Best Fitness')
    axs[0].plot(generations, avg_fitness, 'r--', label='Average Fitness')
    axs[0].set_title('Fitness over Generations')
    axs[0].set_ylabel('Fitness')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot win rates if available
    if len(metrics.win_rates) == len(generations):
        ax2 = axs[0].twinx()
        ax2.plot(generations, metrics.win_rates, 'g-.', label='Win Rate')
        ax2.set_ylabel('Win Rate', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        ax2.legend(loc='upper right')
    
    # Plot rewards if available
    if len(metrics.episode_rewards) == len(generations):
        axs[1].plot(generations, metrics.episode_rewards, 'm-', label='Average Reward')
        axs[1].set_title('Rewards over Generations')
        axs[1].set_ylabel('Average Reward')
        axs[1].set_xlabel('Generation')
        axs[1].legend()
        axs[1].grid(True)
    else:
        axs[1].set_visible(False)
    
    # Add summary text
    summary = metrics.get_training_summary()
    summary_text = '\n'.join([
        f'Generations: {summary.get("generations", "N/A")}',
        f'Best Fitness: {summary.get("best_fitness", "N/A"):.4f} (Gen {summary.get("best_generation", "N/A")})',
        f'Final Avg Fitness: {summary.get("final_avg_fitness", "N/A"):.4f}',
    ])
    
    if "final_win_rate" in summary:
        summary_text += f'\nFinal Win Rate: {summary.get("final_win_rate", "N/A"):.2f}'
    
    if "final_avg_reward" in summary:
        summary_text += f'\nFinal Avg Reward: {summary.get("final_avg_reward", "N/A"):.2f}'
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.5))
    
    # Save figure
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close(fig)
    
    logging.info(f"Created training plots at {filepath}")


def create_reward_distribution_plot(rewards: List[float], filepath: str, title: str = "Reward Distribution") -> None:
    """
    Create a histogram of rewards.
    
    Args:
        rewards: List of rewards
        filepath: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.hist(rewards, bins=50, alpha=0.75, color='blue')
    plt.axvline(np.mean(rewards), color='red', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(rewards):.2f}')
    plt.axvline(np.median(rewards), color='green', linestyle='dashed', linewidth=1, label=f'Median: {np.median(rewards):.2f}')
    
    plt.title(title)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save figure
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    
    logging.info(f"Created reward distribution plot at {filepath}")


def create_action_distribution_plot(actions: List[str], filepath: str, title: str = "Action Distribution") -> None:
    """
    Create a bar chart of action frequencies.
    
    Args:
        actions: List of action strings ('fold', 'check_call', 'bet_raise')
        filepath: Path to save the plot
        title: Plot title
    """
    action_counts = pd.Series(actions).value_counts()
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(action_counts.index, action_counts.values, color=['red', 'blue', 'green'])
    
    plt.title(title)
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    
    # Add percentage labels
    total = sum(action_counts.values)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.,
                 height + 0.1,
                 f'{height/total*100:.1f}%',
                 ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    
    logging.info(f"Created action distribution plot at {filepath}")


def create_win_rate_comparison_plot(win_rates: Dict[str, float], filepath: str, 
                                   title: str = "Win Rate Comparison") -> None:
    """
    Create a bar chart comparing win rates of different agents.
    
    Args:
        win_rates: Dictionary mapping agent names to win rates
        filepath: Path to save the plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by win rate
    sorted_win_rates = {k: v for k, v in sorted(win_rates.items(), key=lambda item: item[1], reverse=True)}
    
    bars = plt.bar(sorted_win_rates.keys(), sorted_win_rates.values(), color='blue', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Agent')
    plt.ylabel('Win Rate')
    plt.axhline(1/len(win_rates), color='red', linestyle='--', label='Random')
    
    # Add labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.,
                 height + 0.01,
                 f'{height:.2f}',
                 ha='center', va='bottom')
    
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Save figure
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    
    logging.info(f"Created win rate comparison plot at {filepath}")


def create_performance_heatmap(performance_matrix: np.ndarray, 
                             agent_names: List[str],
                             filepath: str, 
                             title: str = "Performance Heatmap") -> None:
    """
    Create a heatmap showing performance of agents against each other.
    
    Args:
        performance_matrix: NxN matrix where cell [i,j] is agent i's win rate against agent j
        agent_names: List of agent names
        filepath: Path to save the plot
        title: Plot title
    """
    import seaborn as sns
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    ax = sns.heatmap(performance_matrix, annot=True, cmap="YlGnBu", 
                    xticklabels=agent_names, yticklabels=agent_names)
    
    plt.title(title)
    plt.xlabel('Opponent')
    plt.ylabel('Agent')
    
    # Save figure
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    
    logging.info(f"Created performance heatmap at {filepath}")


def create_dashboard(metrics: TrainingMetrics, performance_data: Dict[str, Any], 
                   filepath: str) -> None:
    """
    Create a comprehensive dashboard of training results.
    
    Args:
        metrics: Training metrics
        performance_data: Additional performance data
        filepath: Path to save the dashboard
    """
    if not metrics.has_data():
        logging.warning("No metrics data available for dashboard")
        return
    
    fig = plt.figure(figsize=(15, 12))
    grid = plt.GridSpec(3, 2, figure=fig)
    
    # Training progress
    ax1 = fig.add_subplot(grid[0, :])
    generations = metrics.generations
    ax1.plot(generations, metrics.best_fitness, 'b-', label='Best Fitness')
    ax1.plot(generations, metrics.avg_fitness, 'r--', label='Average Fitness')
    ax1.set_title('Training Progress')
    ax1.set_ylabel('Fitness')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Win rates by position
    if 'win_rates_by_position' in performance_data:
        ax2 = fig.add_subplot(grid[1, 0])
        win_rates = performance_data['win_rates_by_position']
        ax2.bar(win_rates.keys(), win_rates.values(), color='green', alpha=0.7)
        ax2.set_title('Win Rates by Position')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Win Rate')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Action distribution
    if 'action_distribution' in performance_data:
        ax3 = fig.add_subplot(grid[1, 1])
        action_dist = performance_data['action_distribution']
        ax3.pie(action_dist.values(), labels=action_dist.keys(), autopct='%1.1f%%', 
               shadow=True, startangle=90)
        ax3.axis('equal')
        ax3.set_title('Action Distribution')
    
    # Reward distribution
    if 'rewards' in performance_data:
        ax4 = fig.add_subplot(grid[2, 0])
        rewards = performance_data['rewards']
        ax4.hist(rewards, bins=30, alpha=0.75, color='blue')
        ax4.axvline(np.mean(rewards), color='red', linestyle='dashed', linewidth=1, 
                   label=f'Mean: {np.mean(rewards):.2f}')
        ax4.set_title('Reward Distribution')
        ax4.set_xlabel('Reward')
        ax4.legend()
    
    # Summary statistics
    ax5 = fig.add_subplot(grid[2, 1])
    ax5.axis('off')
    
    summary = metrics.get_training_summary()
    summary_text = '\n'.join([
        f'Generations: {summary.get("generations", "N/A")}',
        f'Best Fitness: {summary.get("best_fitness", "N/A"):.4f} (Gen {summary.get("best_generation", "N/A")})',
        f'Final Avg Fitness: {summary.get("final_avg_fitness", "N/A"):.4f}',
    ])
    
    if 'average_game_length' in performance_data:
        summary_text += f'\nAverage Game Length: {performance_data["average_game_length"]:.1f} actions'
    
    if 'average_decision_time' in performance_data:
        summary_text += f'\nAverage Decision Time: {performance_data["average_decision_time"]*1000:.1f} ms'
    
    ax5.text(0.1, 0.5, summary_text, fontsize=12)
    
    # Save dashboard
    plt.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
    
    logging.info(f"Created dashboard at {filepath}")