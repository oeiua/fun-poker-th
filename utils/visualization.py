"""
Visualization utilities for training progress.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple

def plot_training_progress(
    history: Dict[str, List], 
    save_path: Optional[str] = None,
    title: str = "Training Progress",
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot training progress metrics.
    
    Args:
        history: Dictionary with training history data
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot fitness curves
    plt.subplot(2, 2, 1)
    plt.plot(history['generation'], history['avg_fitness'], label='Average')
    plt.plot(history['generation'], history['max_fitness'], label='Maximum')
    plt.plot(history['generation'], history['min_fitness'], label='Minimum')
    plt.plot(history['generation'], history['best_overall'], label='Best Overall', linestyle='--')
    plt.title('Fitness Evolution')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot best fitness
    plt.subplot(2, 2, 2)
    plt.plot(history['generation'], history['best_overall'], color='green')
    plt.title('Best Fitness Overall')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot fitness improvement rate
    plt.subplot(2, 2, 3)
    if len(history['avg_fitness']) > 1:
        improvements = [history['avg_fitness'][i] - history['avg_fitness'][i-1] 
                       for i in range(1, len(history['avg_fitness']))]
        plt.bar(history['generation'][1:], improvements)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.title('Fitness Improvement Rate')
        plt.xlabel('Generation')
        plt.ylabel('Change in Average Fitness')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot fitness distribution
    plt.subplot(2, 2, 4)
    if len(history['generation']) > 0:
        latest_gen = history['generation'][-1]
        plt.errorbar(
            x=[latest_gen], 
            y=[history['avg_fitness'][-1]],
            yerr=[[history['avg_fitness'][-1] - history['min_fitness'][-1]], 
                 [history['max_fitness'][-1] - history['avg_fitness'][-1]]],
            fmt='o', capsize=10, capthick=2, markersize=8
        )
        plt.title('Current Fitness Distribution')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.xlim(latest_gen - 1, latest_gen + 1)
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Close plot to free memory
    plt.close()

def plot_agent_performance(
    agent_stats: Dict[str, Any],
    save_path: Optional[str] = None,
    title: str = "Agent Performance",
    figsize: Tuple[int, int] = (10, 8)
) -> None:
    """
    Plot performance statistics for an agent.
    
    Args:
        agent_stats: Dictionary with agent statistics
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Plot action distribution
    plt.subplot(2, 2, 1)
    actions = agent_stats.get('actions', {})
    action_labels = ['Fold', 'Check/Call', 'Bet/Raise']
    action_values = [actions.get(i, 0) for i in range(len(action_labels))]
    plt.bar(action_labels, action_values)
    plt.title('Action Distribution')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add win rate
    plt.subplot(2, 2, 2)
    hands_played = agent_stats.get('hands_played', 0)
    hands_won = agent_stats.get('hands_won', 0)
    win_rate = hands_won / max(1, hands_played)
    
    plt.pie([win_rate, 1 - win_rate], 
            labels=['Won', 'Lost'], 
            autopct='%1.1f%%',
            colors=['green', 'red'],
            startangle=90)
    plt.title(f'Win Rate ({hands_won}/{hands_played} hands)')
    
    # Plot reward progression
    if 'reward_history' in agent_stats:
        plt.subplot(2, 2, 3)
        reward_history = agent_stats['reward_history']
        plt.plot(reward_history)
        plt.title('Reward Progression')
        plt.xlabel('Hand')
        plt.ylabel('Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot cumulative reward
    if 'reward_history' in agent_stats:
        plt.subplot(2, 2, 4)
        reward_history = agent_stats['reward_history']
        cumulative_reward = np.cumsum(reward_history)
        plt.plot(cumulative_reward)
        plt.title('Cumulative Reward')
        plt.xlabel('Hand')
        plt.ylabel('Cumulative Reward')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Close plot to free memory
    plt.close()

def plot_game_state(
    state: Dict[str, Any],
    player_idx: int,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """
    Visualize a game state.
    
    Args:
        state: Dictionary representation of the game state
        player_idx: Index of the player from whose perspective to visualize
        save_path: Path to save the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    # Extract state information
    hole_cards = state.get('hole_cards', [])
    community_cards = state.get('community_cards', [])
    game_info = state.get('game_info', {})
    
    # Plot title and general info
    plt.title(f"Poker Game State - Player {player_idx}")
    
    # Plot layout
    plt.axis('off')
    
    # Display game info
    info_text = (
        f"Pot: {game_info.get('pot', 0)}\n"
        f"Round: {game_info.get('round', 0)}/{game_info.get('max_rounds', 0)}\n"
        f"Street: {game_info.get('street', 'preflop')}\n"
        f"Current Player: {game_info.get('current_player', -1)}"
    )
    plt.text(0.05, 0.9, info_text, fontsize=12, transform=plt.gca().transAxes)
    
    # Display community cards
    community_text = "Community Cards: " + (", ".join(community_cards) if community_cards else "None")
    plt.text(0.05, 0.75, community_text, fontsize=12, transform=plt.gca().transAxes)
    
    # Display player's hole cards
    hole_text = "Your Cards: " + (", ".join(hole_cards) if hole_cards else "None")
    plt.text(0.05, 0.65, hole_text, fontsize=12, transform=plt.gca().transAxes)
    
    # Display player stacks
    stacks_text = "Player Stacks:\n"
    player_stacks = game_info.get('player_stacks', [])
    for i, stack in enumerate(player_stacks):
        if i == player_idx:
            stacks_text += f"* Player {i} (YOU): {stack}\n"
        else:
            stacks_text += f"  Player {i}: {stack}\n"
    
    plt.text(0.05, 0.4, stacks_text, fontsize=10, transform=plt.gca().transAxes)
    
    # Save or show plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    # Close plot to free memory
    plt.close()