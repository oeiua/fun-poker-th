"""
Entry point for human vs AI play.
"""
import os
import argparse
import torch
import numpy as np
import random
import time
from typing import Dict, Any, List, Tuple, Optional

from config.config import Config
from game.environment import PokerEnvironment
from game.action import Action
from agents.neural_agent import NeuralAgent
from agents.human_agent import HumanAgent
from agents.random_agent import RandomAgent
from evolution.population import Population
from utils.logger import Logger
from utils.visualization import plot_game_state

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Play poker against AI agents")
    
    parser.add_argument(
        "--model-dir",
        type=str,
        default="saved_models/best_agent",
        help="Directory with trained model to play against"
    )
    
    parser.add_argument(
        "--num-ai",
        type=int,
        default=9,
        help="Number of AI opponents (1-9)"
    )
    
    parser.add_argument(
        "--stack",
        type=int,
        default=10000,
        help="Starting stack size"
    )
    
    parser.add_argument(
        "--blinds",
        type=str,
        default="50/100",
        help="Small/big blind amounts (format: small/big)"
    )
    
    parser.add_argument(
        "--hands",
        type=int,
        default=100,
        help="Maximum number of hands to play"
    )
    
    parser.add_argument(
        "--random-ai",
        action="store_true",
        help="Use random AI opponents instead of trained models"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use (cpu or cuda)"
    )
    
    return parser.parse_args()

def get_user_action(
    state_view: Dict[str, Any],
    valid_actions: List[int],
    valid_amounts: List[int]
) -> Tuple[int, Optional[int]]:
    """
    Get action from user via console input.
    
    Args:
        state_view: Game state from player's perspective
        valid_actions: List of valid action types
        valid_amounts: List of valid bet amounts
        
    Returns:
        Tuple of (action_type, bet_amount)
    """
    # Display game state
    print("\n" + "=" * 50)
    print("Your Turn")
    print("=" * 50)
    
    # Display hole cards
    print(f"Your Cards: {' '.join(state_view.get('hole_cards', []))}")
    
    # Display community cards
    community_cards = state_view.get('community_cards', [])
    print(f"Community Cards: {' '.join(community_cards) if community_cards else 'None'}")
    
    # Display game info
    game_info = state_view.get('game_info', {})
    print(f"Pot: {game_info.get('pot', 0)}")
    print(f"Your Stack: {game_info.get('player_stacks', [])[0]}")
    print(f"Street: {game_info.get('street', 'preflop')}")
    
    # Display player stacks
    player_stacks = game_info.get('player_stacks', [])
    print("\nPlayer Stacks:")
    for i, stack in enumerate(player_stacks):
        print(f"Player {i}: {stack}")
    
    # Display valid actions
    print("\nValid Actions:")
    for action in valid_actions:
        print(f"{action}: {Action.get_action_name(action)}")
    
    # Get action from user
    while True:
        try:
            action_input = input("\nEnter action number: ")
            action_type = int(action_input)
            
            if action_type not in valid_actions:
                print(f"Invalid action! Valid actions: {[Action.get_action_name(a) for a in valid_actions]}")
                continue
            
            break
        except ValueError:
            print("Please enter a valid number.")
    
    # Get bet amount if needed
    bet_amount = None
    if action_type == Action.BET_RAISE:
        # Display valid bet amounts
        if valid_amounts:
            print("\nValid Bet Amounts:")
            for i, amount in enumerate(valid_amounts):
                print(f"{i}: {amount}")
            
            while True:
                try:
                    amount_input = input("\nEnter bet amount index or specific value: ")
                    
                    # Try to interpret as index first
                    try:
                        amount_idx = int(amount_input)
                        if 0 <= amount_idx < len(valid_amounts):
                            bet_amount = valid_amounts[amount_idx]
                            break
                    except ValueError:
                        pass
                    
                    # Try to interpret as specific value
                    bet_amount = int(amount_input)
                    if bet_amount not in valid_amounts:
                        print(f"Invalid amount! Valid amounts: {valid_amounts}")
                        continue
                    
                    break
                except ValueError:
                    print("Please enter a valid number.")
    
    return action_type, bet_amount

def load_ai_agents(
    num_agents: int,
    model_dir: str,
    use_random: bool = False,
    device: Optional[torch.device] = None
) -> List[NeuralAgent]:
    """
    Load AI agents for playing against.
    
    Args:
        num_agents: Number of agents to load
        model_dir: Directory with trained models
        use_random: Whether to use random agents instead of trained models
        device: Device to load models on
        
    Returns:
        List of AI agents
    """
    agents = []
    
    if use_random:
        # Create random agents with different aggression levels
        for i in range(num_agents):
            aggression = 0.3 + (0.6 * i / (num_agents - 1)) if num_agents > 1 else 0.5
            agent = RandomAgent(name=f"RandomAI-{i}", aggression=aggression)
            agents.append(agent)
    else:
        # Try to load trained agent
        try:
            best_agent = NeuralAgent.load(model_dir, device=device)
            best_agent.name = "MasterAI"
            best_agent.exploration_rate = 0.05  # Low exploration for playing against humans
            
            # Create variations of the best agent
            for i in range(num_agents):
                if i == 0:
                    # Use the best agent directly
                    agents.append(best_agent)
                else:
                    # Create a slightly mutated version
                    agent = NeuralAgent(
                        policy_network=best_agent.policy_network,
                        value_network=best_agent.value_network,
                        name=f"AI-{i}",
                        exploration_rate=0.1 + (0.2 * i / num_agents),  # Varying exploration rates
                        device=device
                    )
                    agents.append(agent)
        except Exception as e:
            print(f"Error loading trained model: {str(e)}")
            print("Falling back to random agents.")
            
            # Fallback to random agents
            for i in range(num_agents):
                aggression = 0.3 + (0.6 * i / (num_agents - 1)) if num_agents > 1 else 0.5
                agent = RandomAgent(name=f"RandomAI-{i}", aggression=aggression)
                agents.append(agent)
    
    return agents

def main() -> None:
    """Main function for human vs AI play."""
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    
    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    
    # Parse blinds
    try:
        small_blind, big_blind = map(int, args.blinds.split('/'))
    except:
        print(f"Invalid blinds format: {args.blinds}, using default 50/100")
        small_blind, big_blind = 50, 100
    
    # Cap number of AI opponents
    num_ai = max(1, min(9, args.num_ai))
    
    # Create and configure environment
    env = PokerEnvironment(
        num_players=num_ai + 1,  # +1 for human player
        starting_stack=args.stack,
        small_blind=small_blind,
        big_blind=big_blind,
        max_rounds=args.hands,
        action_timeout=30
    )
    
    # Create human agent
    human_agent = HumanAgent(name="Human")
    human_agent.set_player_index(0)
    human_agent.set_callback(get_user_action)
    
    # Load AI agents
    ai_agents = load_ai_agents(
        num_agents=num_ai,
        model_dir=args.model_dir,
        use_random=args.random_ai,
        device=device
    )
    
    # Set player indices for AI agents
    for i, agent in enumerate(ai_agents):
        agent.set_player_index(i + 1)
    
    # Reset environment
    state = env.reset()
    done = False
    
    print("\n" + "=" * 50)
    print(f"Welcome to Poker AI - Playing {args.hands} hands")
    print(f"Your starting stack: {args.stack}")
    print(f"Blinds: {small_blind}/{big_blind}")
    print("=" * 50)
    
    # Main game loop
    try:
        while not done:
            # Get current player
            current_player_idx = state.get_current_player()
            
            # Get valid actions
            valid_actions = env.get_valid_actions(current_player_idx)
            valid_amounts = env.get_valid_bet_amounts(current_player_idx)
            
            # Get agent for current player
            if current_player_idx == 0:
                current_agent = human_agent
            else:
                current_agent = ai_agents[current_player_idx - 1]
            
            # Get action from agent
            action_type, bet_amount = current_agent.act(state, valid_actions, valid_amounts)
            
            # Display AI action
            if current_player_idx != 0:
                print(f"Player {current_player_idx} ({current_agent.name}) performs: {Action.get_action_description(action_type, bet_amount)}")
            
            # Execute action
            next_state, reward, done, info = env.step(current_player_idx, action_type, bet_amount)
            
            # Let agent observe the result
            current_agent.observe(state, (action_type, bet_amount), reward, next_state, done)
            
            # Display hand result if completed
            if info.get('hand_complete', False):
                rewards = info.get('rewards', {})
                print("\n" + "-" * 50)
                print("Hand Complete!")
                
                # Show all hole cards if hand is complete
                all_hole_cards = env.get_player_hands()
                print("\nFinal Hole Cards:")
                for i, cards in enumerate(all_hole_cards):
                    if i == 0:
                        print(f"Player {i} (YOU): {' '.join(cards)}")
                    else:
                        print(f"Player {i} ({ai_agents[i-1].name}): {' '.join(cards)}")
                
                # Show community cards
                print(f"\nCommunity Cards: {' '.join(next_state.community_cards)}")
                
                # Show rewards
                print("\nResults:")
                for player_idx, player_reward in rewards.items():
                    if player_idx == 0:
                        result = "won" if player_reward > 0 else "lost" if player_reward < 0 else "tied"
                        print(f"Player {player_idx} (YOU) {result}: {player_reward * big_blind:.0f} chips")
                    else:
                        result = "won" if player_reward > 0 else "lost" if player_reward < 0 else "tied"
                        print(f"Player {player_idx} ({ai_agents[player_idx-1].name}) {result}: {player_reward * big_blind:.0f} chips")
                
                print("-" * 50)
                
                # Show updated stacks
                print("\nCurrent Stacks:")
                player_stacks = next_state.game_info.get('player_stacks', [])
                for i, stack in enumerate(player_stacks):
                    if i == 0:
                        print(f"Player {i} (YOU): {stack}")
                    else:
                        print(f"Player {i} ({ai_agents[i-1].name}): {stack}")
                
                # Wait for user to continue
                input("\nPress Enter to continue to next hand...")
            
            # Update state
            state = next_state
    
    except KeyboardInterrupt:
        print("\nGame interrupted by user.")
    
    # Show final statistics
    print("\n" + "=" * 50)
    print("Game Over!")
    print("=" * 50)
    
    # Show human stats
    human_stats = human_agent.get_stats()
    print("\nYour Statistics:")
    print(f"Hands played: {human_stats.get('hands_played', 0)}")
    print(f"Hands won: {human_stats.get('hands_won', 0)}")
    if human_stats.get('hands_played', 0) > 0:
        win_rate = human_stats.get('hands_won', 0) / human_stats.get('hands_played', 0) * 100
        print(f"Win rate: {win_rate:.1f}%")
    print(f"Total profit: {human_stats.get('total_reward', 0) * big_blind:.0f} chips")
    
    # Show actions breakdown
    actions = human_stats.get('actions', {})
    total_actions = sum(actions.values())
    if total_actions > 0:
        print("\nYour Actions:")
        for action_type, count in actions.items():
            percentage = count / total_actions * 100
            print(f"{Action.get_action_name(action_type)}: {count} ({percentage:.1f}%)")
    
    print("\nThanks for playing!")

if __name__ == "__main__":
    main()