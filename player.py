"""
Player classes for the poker game.
"""
import time
import random
import numpy as np
from typing import List, Dict, Optional, Tuple, Any

from card import Card
from model import PokerModel
from config import Action, GamePhase, PokerConfig


class Player:
    """Base class for poker players."""
    
    def __init__(self, name: str, chips: int):
        """
        Initialize a player.
        
        Args:
            name (str): Player name
            chips (int): Starting chips
        """
        self.name = name
        self.chips = chips
        self.hole_cards = []
        self.is_active = True
        self.is_all_in = False
        self.current_bet = 0
        self.round_bet = 0
        self.total_bet = 0
        self.position = 0
        
    def __str__(self) -> str:
        return f"{self.name} ({self.chips} chips)"
    
    def reset_for_new_hand(self):
        """Reset player state for a new hand."""
        self.hole_cards = []
        self.is_active = True
        self.is_all_in = False
        self.current_bet = 0
        self.round_bet = 0
        self.total_bet = 0
    
    def reset_for_new_round(self):
        """Reset player state for a new betting round."""
        self.current_bet = 0
    
    def receive_card(self, card: Card):
        """
        Receive a hole card.
        
        Args:
            card (Card): The card to receive
        """
        self.hole_cards.append(card)
    
    def place_bet(self, amount: int) -> int:
        """
        Place a bet.
        
        Args:
            amount (int): Amount to bet
            
        Returns:
            int: Actual amount bet (may be less if player doesn't have enough chips)
        """
        # Cap bet at available chips
        amount = min(amount, self.chips)
        
        # Update player state
        self.chips -= amount
        self.current_bet += amount
        self.round_bet += amount
        self.total_bet += amount
        
        # Check if player is now all-in
        if self.chips == 0:
            self.is_all_in = True
        
        return amount
    
    def fold(self):
        """Fold the current hand."""
        self.is_active = False
    
    def get_action(
        self, 
        game_state: Dict[str, Any], 
        valid_actions: List[Action]
    ) -> Tuple[Action, Optional[int]]:
        """
        Get the player's action.
        
        Args:
            game_state (Dict[str, Any]): Current game state
            valid_actions (List[Action]): List of valid actions
            
        Returns:
            Tuple[Action, Optional[int]]: The chosen action and bet amount (if applicable)
        """
        raise NotImplementedError("Subclasses must implement get_action()")


class AIPlayer(Player):
    """AI player controlled by a neural network model."""
    
    def __init__(self, name: str, model: PokerModel, chips: int):
        """
        Initialize an AI player.
        
        Args:
            name (str): Player name
            model (PokerModel): Neural network model for decision making
            chips (int): Starting chips
        """
        super().__init__(name, chips)
        self.model = model
    
    def get_action(
        self, 
        game_state: Dict[str, Any], 
        valid_actions: List[Action]
    ) -> Tuple[Action, Optional[int]]:
        """
        Get the AI player's action using its neural network model.
        
        Args:
            game_state (Dict[str, Any]): Current game state
            valid_actions (List[Action]): List of valid actions
            
        Returns:
            Tuple[Action, Optional[int]]: The chosen action and bet amount (if applicable)
        """
        # Convert game state to a state vector for the model
        state_vector = self._create_state_vector(game_state)
        
        # Get action from model
        action = self.model.get_action(state_vector, valid_actions)
        
        # Determine bet amount if applicable
        bet_amount = None
        if action in [Action.BET, Action.RAISE]:
            # Get hand strength and use it to determine bet size
            hand_strength = game_state.get('hand_strength', 0.5)
            pot_size = game_state.get('pot_size', 0)
            to_call = game_state.get('to_call', 0)
            
            # Use hand strength to determine bet size
            from utils import optimal_bet_size
            bet_amount = optimal_bet_size(hand_strength, self.chips, pot_size)
            
            # Ensure bet meets minimum raise requirement
            if to_call > 0:
                min_raise = to_call * 2
                if bet_amount < min_raise:
                    bet_amount = min_raise
        
        elif action == Action.ALL_IN:
            bet_amount = self.chips
        
        elif action == Action.CALL:
            bet_amount = game_state.get('to_call', 0)
        
        return action, bet_amount
    
    def _create_state_vector(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        Create a state vector from the game state for the neural network.
        
        Args:
            game_state (Dict[str, Any]): Current game state
            
        Returns:
            np.ndarray: State vector for the neural network
        """
        # Initialize state vector with zeros
        state = np.zeros(PokerConfig.STATE_SIZE)
        index = 0
        
        # Player position (normalized)
        state[index] = self.position / (PokerConfig.DEFAULT_PLAYERS - 1)
        index += 1
        
        # Player chips (normalized)
        state[index] = self.chips / PokerConfig.INITIAL_CHIPS
        index += 1
        
        # Player pot commitment (normalized)
        state[index] = self.total_bet / (PokerConfig.INITIAL_CHIPS * 0.5)  # Assuming 50% of initial chips is a reasonable upper bound
        index += 1
        
        # Hole cards (if any)
        for i in range(PokerConfig.NUM_HOLE_CARDS):
            if i < len(self.hole_cards):
                # Use 5-feature vector for each card 
                # (1 for normalized rank + 4 for one-hot encoded suit)
                features = self.hole_cards[i].to_feature_vector()
                state[index:index+4] = features[1:]  # Just use the suit features (4)
                state[index+4] = features[0]  # Rank
            index += 5
        
        # Community cards
        community_cards = game_state.get('community_cards', [])
        for i in range(PokerConfig.NUM_COMMUNITY_CARDS):
            if i < len(community_cards):
                # Use 4-feature vector for each card
                features = community_cards[i].to_feature_vector()
                state[index:index+4] = features[1:]  # Suit features
                state[index+4] = features[0]  # Rank
            index += 5
        
        # Pot size (normalized)
        pot_size = game_state.get('pot_size', 0)
        state[index] = pot_size / (PokerConfig.INITIAL_CHIPS * PokerConfig.DEFAULT_PLAYERS * 0.5)  # Assuming 50% of total initial chips is a reasonable upper bound
        index += 1
        
        # Current bet to call (normalized)
        to_call = game_state.get('to_call', 0)
        state[index] = to_call / PokerConfig.INITIAL_CHIPS
        index += 1
        
        # Game phase (one-hot encoded)
        phase = game_state.get('phase', GamePhase.PREFLOP)
        phase_index = phase.value if isinstance(phase, GamePhase) else phase
        if index + phase_index < len(state):
            state[index + phase_index] = 1
        index += 5
        
        # Number of players still in hand (normalized)
        active_players = game_state.get('active_players', PokerConfig.DEFAULT_PLAYERS)
        state[index] = active_players / PokerConfig.DEFAULT_PLAYERS
        index += 1
        
        # Position relative to dealer (normalized)
        dealer_position = game_state.get('dealer_position', 0)
        relative_position = (self.position - dealer_position) % PokerConfig.DEFAULT_PLAYERS
        state[index] = relative_position / PokerConfig.DEFAULT_PLAYERS
        
        return state


class HumanPlayer(Player):
    """Human player controlled via console input."""
    
    def get_action(
        self, 
        game_state: Dict[str, Any], 
        valid_actions: List[Action]
    ) -> Tuple[Action, Optional[int]]:
        """
        Get the human player's action via console input.
        
        Args:
            game_state (Dict[str, Any]): Current game state
            valid_actions (List[Action]): List of valid actions
            
        Returns:
            Tuple[Action, Optional[int]]: The chosen action and bet amount (if applicable)
        """
        # Display game state
        self._display_game_state(game_state)
        
        # Display valid actions
        self._display_valid_actions(valid_actions, game_state)
        
        # Get user input
        while True:
            try:
                action_input = input("Enter your action (number): ").strip()
                action_index = int(action_input)
                
                if 0 <= action_index < len(valid_actions):
                    action = valid_actions[action_index]
                    
                    # Handle bet amount if applicable
                    bet_amount = None
                    if action in [Action.BET, Action.RAISE]:
                        to_call = game_state.get('to_call', 0)
                        min_bet = max(PokerConfig.BIG_BLIND, to_call * 2)
                        
                        while True:
                            bet_input = input(f"Enter bet amount (min {min_bet}): ").strip()
                            try:
                                bet_amount = int(bet_input)
                                if bet_amount >= min_bet and bet_amount <= self.chips:
                                    break
                                elif bet_amount > self.chips:
                                    print(f"You only have {self.chips} chips.")
                                else:
                                    print(f"Minimum bet is {min_bet}.")
                            except ValueError:
                                print("Please enter a valid number.")
                    
                    elif action == Action.ALL_IN:
                        bet_amount = self.chips
                        print(f"Going all-in for {bet_amount} chips.")
                    
                    elif action == Action.CALL:
                        bet_amount = game_state.get('to_call', 0)
                        print(f"Calling {bet_amount}.")
                    
                    return action, bet_amount
                else:
                    print("Invalid action number.")
            
            except ValueError:
                print("Please enter a valid number.")
    
    def _display_game_state(self, game_state: Dict[str, Any]):
        """Display the current game state to the console."""
        print("\n" + "="*50)
        print(f"Your cards: {' '.join(str(card) for card in self.hole_cards)}")
        
        community_cards = game_state.get('community_cards', [])
        if community_cards:
            print(f"Community cards: {' '.join(str(card) for card in community_cards)}")
        else:
            print("Community cards: None")
        
        print(f"Your chips: {self.chips}")
        print(f"Your bet this round: {self.round_bet}")
        print(f"Pot size: {game_state.get('pot_size', 0)}")
        print(f"To call: {game_state.get('to_call', 0)}")
        
        # Display player information
        print("\nPlayers:")
        players = game_state.get('players', [])
        for i, player in enumerate(players):
            status = ""
            if not player.is_active:
                status = "(folded)"
            elif player.is_all_in:
                status = "(all-in)"
            
            if player.name == self.name:
                print(f"* {player.name}: {player.chips} chips, bet: {player.round_bet} {status}")
            else:
                print(f"  {player.name}: {player.chips} chips, bet: {player.round_bet} {status}")
        
        print("="*50)
    
    def _display_valid_actions(self, valid_actions: List[Action], game_state: Dict[str, Any]):
        """Display valid actions to the console."""
        print("\nValid actions:")
        for i, action in enumerate(valid_actions):
            if action == Action.CALL:
                to_call = game_state.get('to_call', 0)
                print(f"{i}: {action.name} ({to_call} chips)")
            elif action == Action.ALL_IN:
                print(f"{i}: {action.name} ({self.chips} chips)")
            else:
                print(f"{i}: {action.name}")