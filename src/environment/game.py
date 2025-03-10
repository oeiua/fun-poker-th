"""
Poker game environment using pokerkit.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

from pokerkit import NoLimitTexasHoldem, State, Mode, Automation
from src.environment.state import GameState
from src.utils.timeout import TimeoutHandler


class PokerGame:
    """
    Wrapper around pokerkit's NoLimitTexasHoldem to manage game flow and state.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the poker game environment.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.game_config = config['game']
        self.timeout_handler = TimeoutHandler(self.game_config['timeout_seconds'])
        self.current_state = None
        self.players = []
        self.human_player_index = None
        
        # Game parameters
        self.small_blind = self.game_config['small_blind']
        self.big_blind = self.game_config['big_blind']
        self.starting_stack = self.game_config['starting_stack']
        self.player_count = self.game_config['player_count']
        
        # Set up automations for pokerkit
        self.automations = (
            Automation.ANTE_POSTING,
            Automation.BET_COLLECTION,
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING,
        )
        
        logging.info(f"Initialized PokerGame with {self.player_count} players")

    def create_new_hand(self) -> State:
        """
        Create a new poker hand.
        
        Returns:
            pokerkit State object for the new hand
        """
        logging.debug("Creating new poker hand")
        
        # Create the game state
        state = NoLimitTexasHoldem.create_state(
            automations=self.automations,
            ante_trimming_status=True,
            raw_antes=0,
            raw_blinds_or_straddles=(self.small_blind, self.big_blind),
            min_bet=self.big_blind,
            raw_starting_stacks=self.starting_stack,
            player_count=self.player_count,
            mode=Mode.CASH_GAME
        )
        
        self.current_state = state
        return state

    def get_legal_actions(self, state: State = None) -> Dict[str, List[int]]:
        """
        Get legal actions for the current player.
        
        Args:
            state: pokerkit State object (uses current_state if None)
            
        Returns:
            Dictionary of legal actions with their parameters
        """
        if state is None:
            state = self.current_state
            
        actions = {}
        
        # Check if player can fold
        if state.can_fold():
            actions['fold'] = [0]  # No parameters needed
            
        # Check if player can check or call
        if state.can_check_or_call():
            call_amount = state.checking_or_calling_amount
            actions['check_call'] = [call_amount]
            
        # Check if player can bet or raise
        if state.can_complete_bet_or_raise_to():
            min_amount = state.min_completion_betting_or_raising_to_amount
            max_amount = state.max_completion_betting_or_raising_to_amount
            
            # Define standard raise sizes (e.g., 2x, 3x, pot)
            pot_size = state.total_pot_amount
            bets = [
                min_amount,  # min bet
                min(max_amount, min_amount * 2),  # 2x min bet
                min(max_amount, min_amount * 3),  # 3x min bet
                min(max_amount, pot_size + state.checking_or_calling_amount),  # pot size raise
                max_amount,  # all-in
            ]
            
            # Remove duplicates and sort
            bets = sorted(list(set(bets)))
            actions['bet_raise'] = bets
        
        return actions

    def process_action(self, action_type: str, amount: int = 0) -> None:
        """
        Process an action in the game.
        
        Args:
            action_type: Type of action ('fold', 'check_call', 'bet_raise')
            amount: Bet amount for 'bet_raise' actions
        """
        state = self.current_state
        
        # Apply the action
        if action_type == 'fold':
            state.fold()
        elif action_type == 'check_call':
            state.check_or_call()
        elif action_type == 'bet_raise':
            state.complete_bet_or_raise_to(amount)
        else:
            logging.error(f"Unknown action type: {action_type}")
            raise ValueError(f"Unknown action type: {action_type}")
        
        logging.debug(f"Processed action: {action_type}, amount: {amount}")

    def deal_cards(self) -> None:
        """Deal cards in the correct sequence for the current street."""
        state = self.current_state
        
        # Deal hole cards
        if state.street_index == 0:
            for player_index in range(self.player_count):
                if state.statuses[player_index]:
                    state.deal_hole('??')
        
        # Deal board cards
        elif state.can_deal_board():
            # Burn a card first
            if state.can_burn_card():
                state.burn_card('??')
            
            # Deal the appropriate number of cards
            if state.street_index == 1:  # Flop
                state.deal_board('???')
            elif state.street_index == 2:  # Turn
                state.deal_board('?')
            elif state.street_index == 3:  # River
                state.deal_board('?')

    def play_hand(self, players: List[Any]) -> Dict[int, float]:
        """
        Play a full hand of poker with the given players.
        
        Args:
            players: List of player agents
            
        Returns:
            Dictionary mapping player indices to their payoffs
        """
        self.players = players
        state = self.create_new_hand()
        self.deal_cards()  # Deal hole cards
        
        # Play until the hand is complete
        while state.status:
            # If the street has finished, deal the next set of cards
            if not state.actor_indices:
                # Collect bets if needed
                if state.can_collect_bets():
                    state.collect_bets()
                
                # Deal cards for the next street
                self.deal_cards()
                continue
            
            # Get the current player
            current_player_idx = state.actor_index
            if current_player_idx is None:
                continue
                
            current_player = players[current_player_idx]
            
            # Get the current game state
            game_state = GameState.from_pokerkit_state(state, current_player_idx)
            
            # Get legal actions
            legal_actions = self.get_legal_actions(state)
            
            # Get player action with timeout
            action_type, amount = self.timeout_handler.with_timeout(
                lambda: current_player.get_action(game_state, legal_actions)
            )
            
            # Process the action
            self.process_action(action_type, amount)
        
        # Hand is complete, handle showdown if needed
        self.handle_showdown()
        
        # Return payoffs
        return {i: state.payoffs[i] for i in range(self.player_count)}

    def handle_showdown(self) -> None:
        """Handle the showdown phase of the game."""
        state = self.current_state
        
        # Process all remaining operations to complete the hand
        while state.can_show_or_muck_hole_cards():
            for i in range(self.player_count):
                if state.can_show_or_muck_hole_cards(None, i):
                    state.show_or_muck_hole_cards(True, i)
        
        # Push chips to winners
        while state.can_push_chips():
            state.push_chips()
        
        # Players collect their winnings
        while state.can_pull_chips():
            for i in range(self.player_count):
                if state.can_pull_chips(i):
                    state.pull_chips(i)

    def play_with_human(self, ai_checkpoint_path: str) -> None:
        """
        Play a poker game with a human player against AI players.
        
        Args:
            ai_checkpoint_path: Path to the AI model checkpoint
        """
        # TODO: Implement play with human
        pass