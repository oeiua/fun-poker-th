"""
Human player interface for poker environment.
"""
import time
from typing import List, Dict, Any, Optional, Tuple, Union

from agents.base_agent import BaseAgent
from game.state import GameState
from game.action import Action

class HumanAgent(BaseAgent):
    """Agent that allows a human player to interact with the poker environment."""
    
    def __init__(self, name: str = "Human"):
        """
        Initialize a human agent.
        
        Args:
            name: Name of the player
        """
        super().__init__(name=name)
        self.callback = None
        self.action_queue = []
    
    def set_callback(self, callback) -> None:
        """
        Set a callback function to handle user input.
        
        Args:
            callback: Function that will be called to get user input
        """
        self.callback = callback
    
    def add_action(self, action_type: int, bet_amount: Optional[int] = None) -> None:
        """
        Add an action to the queue.
        
        Args:
            action_type: Type of action
            bet_amount: Amount to bet/raise if applicable
        """
        self.action_queue.append((action_type, bet_amount))
    
    def act(self, state: GameState, valid_actions: List[int], valid_amounts: List[int]) -> Tuple[int, Optional[int]]:
        """
        Choose an action based on user input.
        
        Args:
            state: Current game state
            valid_actions: List of valid action types
            valid_amounts: List of valid bet amounts
            
        Returns:
            Tuple of (action_type, bet_amount)
        """
        # If we have queued actions, use the next one
        if self.action_queue:
            return self.action_queue.pop(0)
        
        # Use the callback to get user input, if available
        if self.callback:
            # Display information to the user
            player_view = state.get_player_view(self.player_idx)
            
            # Call the callback to get user input
            action_type, bet_amount = self.callback(player_view, valid_actions, valid_amounts)
            
            # Validate the action
            if action_type not in valid_actions:
                print(f"Invalid action! Valid actions: {[Action.get_action_name(a) for a in valid_actions]}")
                # Default to first valid action
                action_type = valid_actions[0] if valid_actions else Action.FOLD
            
            # Validate bet amount
            if action_type == Action.BET_RAISE and bet_amount is not None:
                if valid_amounts and bet_amount not in valid_amounts:
                    print(f"Invalid bet amount! Valid amounts: {valid_amounts}")
                    # Default to first valid amount
                    bet_amount = valid_amounts[0] if valid_amounts else None
            else:
                bet_amount = None
                
            return action_type, bet_amount
        
        # If no callback set, use console input
        return self._console_input(state, valid_actions, valid_amounts)
    
    def _console_input(self, state: GameState, valid_actions: List[int], valid_amounts: List[int]) -> Tuple[int, Optional[int]]:
        """
        Get user input from the console.
        
        Args:
            state: Current game state
            valid_actions: List of valid action types
            valid_amounts: List of valid bet amounts
            
        Returns:
            Tuple of (action_type, bet_amount)
        """
        # Display game state
        self._display_game_state(state)
        
        # Display valid actions
        print("\nValid actions:")
        for action in valid_actions:
            print(f"{action}: {Action.get_action_name(action)}")
        
        # Get action input
        while True:
            try:
                action_input = input("Enter action number: ")
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
                print("\nValid bet amounts:")
                for amount in valid_amounts:
                    print(f"- {amount}")
                
                while True:
                    try:
                        amount_input = input("Enter bet amount: ")
                        bet_amount = int(amount_input)
                        
                        if bet_amount not in valid_amounts:
                            print(f"Invalid amount! Valid amounts: {valid_amounts}")
                            continue
                        
                        break
                    except ValueError:
                        print("Please enter a valid number.")
        
        return action_type, bet_amount
    
    def _display_game_state(self, state: GameState) -> None:
        """
        Display the current game state to the console.
        
        Args:
            state: Current game state
        """
        print("\n" + "=" * 50)
        print(f"Round: {state.game_info.get('round', 0)}/{state.game_info.get('max_rounds', 0)}")
        print(f"Pot: {state.game_info.get('pot', 0)}")
        print(f"Street: {state.game_info.get('street', 'preflop')}")
        
        # Display community cards
        print("\nCommunity Cards:", end=" ")
        if state.community_cards:
            print(" ".join(state.community_cards))
        else:
            print("None")
        
        # Display player's hole cards
        print(f"\nYour Cards: {' '.join(state.hole_cards[self.player_idx])}")
        
        # Display player stacks
        print("\nPlayer Stacks:")
        player_stacks = state.game_info.get('player_stacks', [])
        for i, stack in enumerate(player_stacks):
            if i == self.player_idx:
                print(f"* Player {i} (YOU): {stack}")
            else:
                print(f"  Player {i}: {stack}")
        
        # Display current bets
        print("\nCurrent Bets:")
        player_bets = state.game_info.get('player_bets', [])
        for i, bet in enumerate(player_bets):
            if bet > 0:
                if i == self.player_idx:
                    print(f"* Player {i} (YOU): {bet}")
                else:
                    print(f"  Player {i}: {bet}")
        
        # Display folded players
        folded_players = state.game_info.get('folded_players', [])
        if folded_players:
            print("\nFolded Players:", ", ".join(f"Player {p}" for p in folded_players))
        
        # Display current player
        current_player = state.game_info.get('current_player', -1)
        if current_player >= 0:
            if current_player == self.player_idx:
                print("\nIt's YOUR turn!")
            else:
                print(f"\nIt's Player {current_player}'s turn.")
        
        print("=" * 50)
    
    def observe(self, state: GameState, action: Tuple[int, Optional[int]], reward: float, next_state: GameState, done: bool) -> None:
        """
        Observe the result of an action.
        
        Args:
            state: State before action
            action: Action taken (action_type, bet_amount)
            reward: Reward received
            next_state: State after action
            done: Whether the episode is done
        """
        super().observe(state, action, reward, next_state, done)
        
        # Display the result of the action if it's the player's action
        action_type, bet_amount = action
        if state.get_current_player() == self.player_idx:
            print(f"\nYou performed: {Action.get_action_description(action_type, bet_amount)}")
            
            if done:
                print(f"Hand complete! Your reward: {reward:.2f}")
                if reward > 0:
                    print("You won!")
                elif reward < 0:
                    print("You lost.")
                else:
                    print("Break even.")