"""
Poker game environment using pokerkit library.
"""
import pokerkit
import numpy as np
import time
import threading
from typing import List, Dict, Tuple, Any, Optional, Union, Callable

from config.config import Config
from game.state import GameState
from game.action import Action

class PokerEnvironment:
    """
    Poker game environment that wraps pokerkit functionality.
    """
    
    def __init__(
        self,
        num_players: int = 10,
        starting_stack: int = 10000,
        small_blind: int = 50,
        big_blind: int = 100,
        max_rounds: int = 1000,
        action_timeout: int = 30
    ):
        """
        Initialize the poker environment.
        
        Args:
            num_players: Number of players in the game
            starting_stack: Starting chips for each player
            small_blind: Small blind amount
            big_blind: Big blind amount
            max_rounds: Maximum number of rounds to play
            action_timeout: Timeout for actions in seconds
        """
        self.num_players = num_players
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_rounds = max_rounds
        self.action_timeout = action_timeout
        
        # Initialize game
        self.reset()
    
    def reset(self) -> GameState:
        """
        Reset the game to initial state.
        
        Returns:
            Initial game state
        """
        # Initialize pokerkit game
        self.game = pokerkit.FixedLimitTexasHoldem(
            num_players=self.num_players,
            starting_stacks=[self.starting_stack] * self.num_players,
            blinds=[self.small_blind, self.big_blind]
        )
        
        # Set up game state tracking
        self.current_round = 0
        self.done = False
        self.player_stats = {i: {"wins": 0, "chips_won": 0} for i in range(self.num_players)}
        
        # Track current hand information
        self.current_hand_history = []
        self.current_hand_actions = []
        self.current_community_cards = []
        self.current_hands = [[] for _ in range(self.num_players)]
        
        # Start a new hand
        return self._start_new_hand()
    
    def _start_new_hand(self) -> GameState:
        """
        Start a new poker hand.
        
        Returns:
            Initial state of the new hand
        """
        # Check if max rounds reached
        if self.current_round >= self.max_rounds:
            self.done = True
            return self._get_game_state()
        
        # Increment round counter
        self.current_round += 1
        
        # Reset hand-specific tracking
        self.current_hand_history = []
        self.current_hand_actions = []
        self.current_community_cards = []
        
        # Deal new hand
        self.game.deal()
        
        # Record player hands
        for player_idx in range(self.num_players):
            self.current_hands[player_idx] = self.game.hole_cards[player_idx].copy()
        
        # Return current game state
        return self._get_game_state()
    
    def step(self, player_idx: int, action_type: int, bet_amount: Optional[int] = None) -> Tuple[GameState, float, bool, Dict]:
        """
        Execute a player's action and advance the game.
        
        Args:
            player_idx: Index of the player making the action
            action_type: Type of action (fold, check/call, bet/raise)
            bet_amount: Amount to bet/raise (only used for bet/raise actions)
            
        Returns:
            Tuple of (new state, reward, done, info)
        """
        # Validate player index
        if player_idx != self.game.current_player:
            raise ValueError(f"Not player {player_idx}'s turn. Current player is {self.game.current_player}")
        
        # Convert action type to pokerkit action
        action = self._convert_action(action_type, bet_amount)
        
        # Record action in history
        self.current_hand_actions.append({
            'player': player_idx,
            'action_type': action_type,
            'amount': bet_amount,
            'street': self.game.street,
            'pot': self.game.pot
        })
        
        # Use timeout for action execution
        result = self._execute_action_with_timeout(action)
        
        # Check if hand is finished
        if self.game.is_hand_complete:
            # Update current community cards
            self.current_community_cards = self.game.board.copy()
            
            # Calculate rewards for all players
            rewards = self._calculate_rewards()
            
            # Start new hand for next round
            next_state = self._start_new_hand()
            
            # For the player who just acted
            player_reward = rewards[player_idx]
            
            return next_state, player_reward, self.done, {'rewards': rewards, 'hand_complete': True}
        else:
            # Update current community cards if they changed
            self.current_community_cards = self.game.board.copy()
            
            # Return updated state
            next_state = self._get_game_state()
            
            # No immediate reward during the hand
            return next_state, 0.0, self.done, {'hand_complete': False}
    
    def _execute_action_with_timeout(self, action: Any) -> Any:
        """
        Execute an action with a timeout to prevent long waits.
        
        Args:
            action: The action to execute
            
        Returns:
            Result of the action
        """
        result = None
        exception = None
        
        def target():
            nonlocal result, exception
            try:
                result = self.game.act(action)
            except Exception as e:
                exception = e
        
        # Start action in a thread
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        # Wait for the action to complete or timeout
        thread.join(self.action_timeout)
        
        # Check if action timed out
        if thread.is_alive():
            # Handle timeout - default to fold
            try:
                result = self.game.act(pokerkit.FoldAction())
            except Exception as e:
                # If fold fails (e.g., player can check), try check/call
                try:
                    result = self.game.act(pokerkit.CheckAction())
                except:
                    # Last resort - attempt smallest valid call
                    try:
                        result = self.game.act(pokerkit.CallAction())
                    except Exception as e:
                        # If all else fails, log the error
                        raise RuntimeError(f"Action timeout and couldn't execute default action: {str(e)}")
        
        # Re-raise any exception from the thread
        if exception:
            raise exception
        
        return result
    
    def _convert_action(self, action_type: int, bet_amount: Optional[int] = None) -> Any:
        """
        Convert internal action type to pokerkit action.
        
        Args:
            action_type: Action type (0=fold, 1=check/call, 2=bet/raise)
            bet_amount: Amount to bet/raise (only used for bet/raise)
            
        Returns:
            pokerkit action object
        """
        if action_type == Action.FOLD:
            return pokerkit.FoldAction()
        elif action_type == Action.CHECK_CALL:
            # Try check first, fallback to call if not possible
            try:
                return pokerkit.CheckAction()
            except:
                return pokerkit.CallAction()
        elif action_type == Action.BET_RAISE:
            # If bet amount specified, use it
            if bet_amount is not None:
                try:
                    return pokerkit.RaiseAction(bet_amount)
                except:
                    try:
                        return pokerkit.BetAction(bet_amount)
                    except:
                        # Fallback to standard raise if specific amount isn't valid
                        try:
                            return pokerkit.RaiseAction()
                        except:
                            return pokerkit.BetAction()
            else:
                # Default raise/bet
                try:
                    return pokerkit.RaiseAction()
                except:
                    return pokerkit.BetAction()
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def _get_game_state(self) -> GameState:
        """
        Get the current game state.
        
        Returns:
            Current game state
        """
        # Get basic game info
        game_info = {
            'round': self.current_round,
            'max_rounds': self.max_rounds,
            'pot': self.game.pot,
            'current_player': self.game.current_player,
            'street': self.game.street,
            'community_cards': self.current_community_cards.copy(),
            'player_stacks': self.game.stacks.copy(),
            'player_bets': self.game.bets.copy(),
            'folded_players': [p for p in range(self.num_players) if self.game.statuses[p] == 'F']
        }
        
        # Create GameState object
        return GameState(
            hole_cards=self.current_hands,
            community_cards=self.current_community_cards,
            game_info=game_info,
            hand_history=self.current_hand_actions.copy()
        )
    
    def _calculate_rewards(self) -> Dict[int, float]:
        """
        Calculate rewards for each player after a hand is complete.
        
        Returns:
            Dictionary mapping player indices to rewards
        """
        # Initialize rewards
        rewards = {i: 0.0 for i in range(self.num_players)}
        
        # Calculate change in chips for each player
        for player_idx in range(self.num_players):
            # Change in stack = current_stack - starting_stack
            net_chips = self.game.stacks[player_idx] - self.starting_stack
            
            # Update player stats
            if net_chips > 0:
                self.player_stats[player_idx]["wins"] += 1
                self.player_stats[player_idx]["chips_won"] += net_chips
            
            # Set reward as change in chips (normalized)
            rewards[player_idx] = float(net_chips) / self.big_blind
        
        return rewards
    
    def get_valid_actions(self, player_idx: int) -> List[int]:
        """
        Get list of valid actions for a player.
        
        Args:
            player_idx: Player index
            
        Returns:
            List of valid action types
        """
        valid_actions = []
        
        # Always add fold if player can act
        if player_idx == self.game.current_player:
            try:
                # Test if fold is valid
                test_action = pokerkit.FoldAction()
                # If no exception, add it to valid actions
                valid_actions.append(Action.FOLD)
            except:
                pass
            
            # Test check/call
            try:
                test_action = pokerkit.CheckAction()
                valid_actions.append(Action.CHECK_CALL)
            except:
                try:
                    test_action = pokerkit.CallAction()
                    valid_actions.append(Action.CHECK_CALL)
                except:
                    pass
            
            # Test bet/raise
            try:
                test_action = pokerkit.RaiseAction()
                valid_actions.append(Action.BET_RAISE)
            except:
                try:
                    test_action = pokerkit.BetAction()
                    valid_actions.append(Action.BET_RAISE)
                except:
                    pass
        
        return valid_actions
    
    def get_valid_bet_amounts(self, player_idx: int) -> List[int]:
        """
        Get list of valid bet amounts for a player.
        
        Args:
            player_idx: Player index
            
        Returns:
            List of valid bet amounts
        """
        # TODO: Implement based on pokerkit rules
        # For simplicity, return a range of standard bet sizes
        # In a real implementation, this should be based on the actual game rules
        
        stack = self.game.stacks[player_idx]
        pot = self.game.pot
        
        # Common bet sizes as multipliers of the pot
        pot_multipliers = [0.5, 0.75, 1.0, 1.5, 2.0]
        
        bet_amounts = []
        for mult in pot_multipliers:
            amount = int(pot * mult)
            if amount <= stack:
                bet_amounts.append(amount)
        
        # Add all-in
        if stack > 0 and stack not in bet_amounts:
            bet_amounts.append(stack)
        
        return bet_amounts
    
    def get_player_hands(self) -> List[List[str]]:
        """
        Get all player hole cards (for evaluation purposes).
        
        Returns:
            List of hole cards for each player
        """
        return self.current_hands.copy()
    
    def get_player_stats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get player statistics.
        
        Returns:
            Dictionary of player statistics
        """
        return self.player_stats.copy()