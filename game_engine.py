"""
Poker game engine that manages game flow and rules.
"""
import logging
import random
import time
from typing import List, Dict, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from card import Card, Deck
from player import Player
from config import Action, GamePhase, PokerConfig
from evaluator import HandEvaluator
from utils import calculate_hand_strength

logger = logging.getLogger("PokerAI.GameEngine")

class GameEngine:
    """
    Manages the poker game flow and rules.
    """
    def __init__(self, players: List[Player], verbose: bool = False):
        """
        Initialize the game engine.
        
        Args:
            players (List[Player]): List of players
            verbose (bool): Whether to output game progress details
        """
        self.players = players
        self.verbose = verbose
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.dealer_position = 0
        self.small_blind_position = 0
        self.big_blind_position = 0
        self.current_player_index = 0
        self.phase = GamePhase.PREFLOP
        
        # Assign initial positions
        for i, player in enumerate(players):
            player.position = i
    
    def run_game(self, num_hands: int = 1) -> Dict[str, int]:
        """
        Run the specified number of poker hands.
        
        Args:
            num_hands (int): Number of hands to play
            
        Returns:
            Dict[str, int]: Mapping of player names to final chip counts
        """
        eliminated_players = set()
        
        for hand_num in range(num_hands):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"Starting Hand #{hand_num + 1}")
                print(f"{'='*50}")
            
            # Check if we have enough players
            active_players = [p for p in self.players if p.chips > 0]
            if len(active_players) < 2:
                logger.info("Not enough players with chips to continue.")
                break
            
            self._play_hand()
            
            # Rotate dealer position (making sure we land on a player with chips)
            self.dealer_position = self._find_next_active_position(self.dealer_position - 1)
            
            # Check for eliminated players
            for player in self.players:
                if player.chips <= 0 and player.name not in eliminated_players:
                    eliminated_players.add(player.name)
                    if self.verbose:
                        print(f"{player.name} has been eliminated!")
        
        # Return final chip counts
        return {player.name: player.chips for player in self.players}
    
    def _play_hand(self):
        """Play a single hand of poker."""
        self._setup_hand()
        
        # Pre-flop betting round
        self.phase = GamePhase.PREFLOP
        if self.verbose:
            print(f"\n--- Pre-flop ---")
        self._betting_round()
        
        # Continue if at least 2 players are still active
        if self._count_active_players() >= 2:
            # Deal the flop
            self._deal_community_cards(3)
            self.phase = GamePhase.FLOP
            if self.verbose:
                print(f"\n--- Flop: {' '.join(str(card) for card in self.community_cards)} ---")
            self._betting_round()
        
        if self._count_active_players() >= 2:
            # Deal the turn
            self._deal_community_cards(1)
            self.phase = GamePhase.TURN
            if self.verbose:
                print(f"\n--- Turn: {' '.join(str(card) for card in self.community_cards)} ---")
            self._betting_round()
        
        if self._count_active_players() >= 2:
            # Deal the river
            self._deal_community_cards(1)
            self.phase = GamePhase.RIVER
            if self.verbose:
                print(f"\n--- River: {' '.join(str(card) for card in self.community_cards)} ---")
            self._betting_round()
        
        # Showdown if at least 2 players are still active
        if self._count_active_players() >= 2:
            self.phase = GamePhase.SHOWDOWN
            self._showdown()
        
    def _setup_hand(self):
        """Set up for a new hand."""
        # Reset deck and community cards
        self.deck = Deck()
        self.deck.shuffle()
        self.community_cards = []
        
        # Reset pot and bet tracking
        self.pot = 0
        self.current_bet = 0
        
        # Reset all players for the new hand
        for player in self.players:
            player.reset_for_new_hand()
        
        # Get a list of active players with chips
        active_players = [p for p in self.players if p.chips > 0]
        
        # Handle the case where there are too few active players
        if len(active_players) < 2:
            logger.warning("Not enough players with chips to play a hand.")
            return False  # Signal that hand setup failed
        
        # Determine blind positions (ensuring players have chips)
        small_blind_pos = self._find_next_active_position(self.dealer_position)
        big_blind_pos = self._find_next_active_position(small_blind_pos)
        
        # Update the actual positions
        self.small_blind_position = small_blind_pos
        self.big_blind_position = big_blind_pos
        
        # Post blinds
        small_blind_player = self.players[self.small_blind_position]
        small_blind_amount = min(PokerConfig.SMALL_BLIND, small_blind_player.chips)
        small_blind_player.place_bet(small_blind_amount)
        self.pot += small_blind_amount
        
        big_blind_player = self.players[self.big_blind_position]
        big_blind_amount = min(PokerConfig.BIG_BLIND, big_blind_player.chips)
        big_blind_player.place_bet(big_blind_amount)
        self.pot += big_blind_amount
        
        self.current_bet = big_blind_amount
        
        # Deal hole cards
        for _ in range(2):
            for player in active_players:
                player.receive_card(self.deck.deal())
        
        # Set starting player position (after big blind)
        self.current_player_index = self._find_next_active_position(self.big_blind_position)
        
        if self.verbose:
            print("\nBlinds posted:")
            print(f"{small_blind_player.name} posts small blind: {small_blind_amount}")
            print(f"{big_blind_player.name} posts big blind: {big_blind_amount}")
            print(f"Pot: {self.pot}")
            
        return True  # Signal successful hand setup

    def _find_next_active_position(self, start_pos: int) -> int:
        """
        Find the next position with a player who has chips.
        
        Args:
            start_pos (int): Starting position
            
        Returns:
            int: Next active player position
        """
        position = (start_pos + 1) % len(self.players)
        
        # Search until we find a player with chips
        while position != start_pos:
            if self.players[position].chips > 0:
                return position
            position = (position + 1) % len(self.players)
        
        # If we've gone full circle, return the start position
        # This shouldn't happen if we check for enough active players earlier
        return start_pos
    
    def _play_hand(self):
        """Play a single hand of poker."""
        # Setup the hand and check if we can proceed
        if not self._setup_hand():
            return  # Not enough active players to play
        
        # Pre-flop betting round
        self.phase = GamePhase.PREFLOP
        if self.verbose:
            print(f"\n--- Pre-flop ---")
        self._betting_round()
        
        # Continue if at least 2 players are still active
        if self._count_active_players() >= 2:
            # Deal the flop
            self._deal_community_cards(3)
            self.phase = GamePhase.FLOP
            if self.verbose:
                print(f"\n--- Flop: {' '.join(str(card) for card in self.community_cards)} ---")
            self._betting_round()
        
        if self._count_active_players() >= 2:
            # Deal the turn
            self._deal_community_cards(1)
            self.phase = GamePhase.TURN
            if self.verbose:
                print(f"\n--- Turn: {' '.join(str(card) for card in self.community_cards)} ---")
            self._betting_round()
        
        if self._count_active_players() >= 2:
            # Deal the river
            self._deal_community_cards(1)
            self.phase = GamePhase.RIVER
            if self.verbose:
                print(f"\n--- River: {' '.join(str(card) for card in self.community_cards)} ---")
            self._betting_round()
        
        # Showdown if at least 2 players are still active
        if self._count_active_players() >= 2:
            self.phase = GamePhase.SHOWDOWN
            self._showdown()

    def _deal_community_cards(self, count: int):
        """
        Deal community cards.
        
        Args:
            count (int): Number of cards to deal
        """
        for _ in range(count):
            self.community_cards.append(self.deck.deal())
    
    def _betting_round(self):
        """Execute a betting round."""
        # Get active players who can participate (have chips and haven't folded)
        players_to_act = [p for p in self.players if p.is_active and p.chips > 0]
        if not players_to_act:
            return
        
        # In pre-flop, we start after the big blind
        if self.phase == GamePhase.PREFLOP:
            start_idx = self._find_next_active_position(self.big_blind_position)
        else:
            # In other rounds, we start after the dealer
            start_idx = self._find_next_active_position(self.dealer_position)
        
        self.current_player_index = start_idx
        
        # For tracking when the betting round is complete
        players_acted = 0
        max_players = len(players_to_act)  # Only count players who can act
        last_raiser = None
        
        # Continue betting until everyone has called, folded, or gone all-in
        while players_acted < max_players:
            current_player = self.players[self.current_player_index]
            
            # Skip players who are not active, are all-in, or have 0 chips
            if not current_player.is_active or current_player.is_all_in or current_player.chips <= 0:
                self.current_player_index = self._find_next_active_position(self.current_player_index - 1)
                players_acted += 1
                continue
            
            # If everyone has acted and all active players have called or checked
            if players_acted >= max_players and (last_raiser is None or 
                self.current_player_index == last_raiser):
                break
            
            # Get the amount the player needs to call
            to_call = self.current_bet - current_player.current_bet
            
            # Determine valid actions
            valid_actions = self._get_valid_actions(current_player, to_call)
            
            # Create game state for the player's decision
            game_state = self._create_game_state(current_player, to_call)
            
            # Get player's action with timeout handling
            try:
                action, bet_amount = self._get_player_action_with_timeout(
                    current_player, game_state, valid_actions)
                
                # Process the action
                if action == Action.FOLD:
                    current_player.fold()
                    if self.verbose:
                        print(f"{current_player.name} folds")
                
                elif action == Action.CHECK:
                    if self.verbose:
                        print(f"{current_player.name} checks")
                
                elif action == Action.CALL:
                    # Call the current bet
                    call_amount = min(to_call, current_player.chips)
                    current_player.place_bet(call_amount)
                    self.pot += call_amount
                    if self.verbose:
                        print(f"{current_player.name} calls {call_amount}")
                
                elif action == Action.BET:
                    # Place a new bet
                    if bet_amount is None:
                        bet_amount = min(self.current_bet * 2, current_player.chips)
                    
                    bet_amount = min(bet_amount, current_player.chips)
                    current_player.place_bet(bet_amount)
                    self.pot += bet_amount
                    self.current_bet = current_player.current_bet
                    last_raiser = self.current_player_index
                    if self.verbose:
                        print(f"{current_player.name} bets {bet_amount}")
                
                elif action == Action.RAISE:
                    # Raise the current bet
                    if bet_amount is None:
                        bet_amount = min(self.current_bet * 2, current_player.chips)
                    
                    bet_amount = min(bet_amount, current_player.chips)
                    current_player.place_bet(bet_amount)
                    self.pot += bet_amount
                    self.current_bet = current_player.current_bet
                    last_raiser = self.current_player_index
                    if self.verbose:
                        print(f"{current_player.name} raises to {current_player.current_bet}")
                
                elif action == Action.ALL_IN:
                    # Go all-in
                    all_in_amount = current_player.chips
                    current_player.place_bet(all_in_amount)
                    self.pot += all_in_amount
                    
                    # Update current bet if this all-in is larger
                    if current_player.current_bet > self.current_bet:
                        self.current_bet = current_player.current_bet
                        last_raiser = self.current_player_index
                    
                    if self.verbose:
                        print(f"{current_player.name} goes all-in for {all_in_amount}")
            
            except TimeoutError:
                # Handle timeout (default to folding)
                current_player.fold()
                if self.verbose:
                    print(f"{current_player.name} folds (timeout)")
            
            # Move to the next player with chips
            self.current_player_index = self._find_next_active_position(self.current_player_index)
            players_acted += 1
        
        # Reset bets for the next round
        for player in self.players:
            player.reset_for_new_round()
        
    def _get_valid_actions(self, player: Player, to_call: int) -> List[Action]:
        """
        Determine valid actions for a player.
        
        Args:
            player (Player): The player
            to_call (int): Amount needed to call
            
        Returns:
            List[Action]: List of valid actions
        """
        valid_actions = []
        
        # Player can always fold
        valid_actions.append(Action.FOLD)
        
        # Check is valid if no bet to call
        if to_call == 0:
            valid_actions.append(Action.CHECK)
        else:
            # Call is valid if player has chips to call
            if player.chips > 0:
                valid_actions.append(Action.CALL)
        
        # Bet/Raise is valid if player has enough chips
        min_bet = max(PokerConfig.BIG_BLIND, to_call * 2)
        if player.chips >= min_bet:
            if to_call == 0:
                valid_actions.append(Action.BET)
            else:
                valid_actions.append(Action.RAISE)
        
        # All-in is valid if player has chips
        if player.chips > 0:
            valid_actions.append(Action.ALL_IN)
        
        return valid_actions
    
    def _create_game_state(self, player: Player, to_call: int) -> Dict[str, Any]:
        """
        Create a game state dictionary for player decisions.
        
        Args:
            player (Player): The current player
            to_call (int): Amount needed to call
            
        Returns:
            Dict[str, Any]: Game state information
        """
        # Calculate hand strength if community cards exist
        hand_strength = 0.0
        if player.hole_cards:
            hand_strength = calculate_hand_strength(
                player.hole_cards, 
                self.community_cards, 
                self._count_active_players()
            )
        
        return {
            'phase': self.phase,
            'pot_size': self.pot,
            'to_call': to_call,
            'community_cards': self.community_cards,
            'hand_strength': hand_strength,
            'active_players': self._count_active_players(),
            'dealer_position': self.dealer_position,
            'players': self.players,
            'current_bet': self.current_bet
        }
    
    def _get_player_action_with_timeout(
        self, 
        player: Player, 
        game_state: Dict[str, Any], 
        valid_actions: List[Action]
    ) -> Tuple[Action, Optional[int]]:
        """
        Get player action with timeout handling.
        
        Args:
            player (Player): The player
            game_state (Dict[str, Any]): Current game state
            valid_actions (List[Action]): List of valid actions
            
        Returns:
            Tuple[Action, Optional[int]]: Action and bet amount
            
        Raises:
            TimeoutError: If player takes too long to act
        """
        # Skip timeout for human players
        from player import HumanPlayer
        if isinstance(player, HumanPlayer):
            return player.get_action(game_state, valid_actions)
        
        # Use threading for AI players to enforce timeout
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(player.get_action, game_state, valid_actions)
            try:
                return future.result(timeout=PokerConfig.TIMEOUT_SECONDS)
            except TimeoutError:
                raise
    
    def _count_active_players(self) -> int:
        """
        Count number of active players.
        
        Returns:
            int: Number of active players
        """
        return sum(1 for p in self.players if p.is_active)
    
    def _showdown(self):
        """Handle the showdown phase."""
        if self.verbose:
            print("\n--- Showdown ---")
        
        # Evaluate hands for all active players
        player_hands = {}
        for player in self.players:
            if player.is_active:
                hand_rank, best_hand = HandEvaluator.evaluate_hand(
                    player.hole_cards, self.community_cards)
                player_hands[player] = (hand_rank, best_hand)
                
                if self.verbose:
                    hand_description = HandEvaluator.get_hand_description(hand_rank, best_hand)
                    print(f"{player.name}: {' '.join(str(card) for card in player.hole_cards)} - {hand_description}")
        
        # Determine winner(s)
        winners = HandEvaluator.compare_hands(player_hands)
        
        # Distribute pot
        if winners:
            pot_per_winner = self.pot // len(winners)
            remainder = self.pot % len(winners)
            
            for i, winner in enumerate(winners):
                # First winner gets any remainder chips
                winner_amount = pot_per_winner + (remainder if i == 0 else 0)
                winner.chips += winner_amount
                
                if self.verbose:
                    print(f"{winner.name} wins {winner_amount} chips")
        
        if self.verbose:
            print(f"\nChip counts after hand:")
            for player in self.players:
                print(f"{player.name}: {player.chips}")
