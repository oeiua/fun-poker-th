#!/usr/bin/env python
"""
Texas Hold'em Poker - Human vs AI Interface
A fancy interface for playing poker against AI opponents.
"""

import os
import sys
import time
import random
import torch
from poker_game import PokerGame, Player
from neural_network import PokerNet, create_state_tensor
from config import CARD_SYMBOLS, CARD_COLORS, CHECKPOINT_DIR

# ANSI color codes for terminal formatting
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GOLD = '\033[38;5;220m'
    SILVER = '\033[38;5;7m'
    BRONZE = '\033[38;5;172m'
    BG_BLACK = '\033[40m'
    BG_GREEN = '\033[42m'
    BG_BLUE = '\033[44m'
    BG_RED = '\033[41m'

class PokerTable:
    """Interactive poker table interface for human vs AI play."""
    
    def __init__(self, num_ai_opponents=4, starting_chips=1000):
        self.model = self._load_ai_model()
        self.game = PokerGame(num_players=num_ai_opponents+1, starting_chips=starting_chips)
        self.game.setup_game(ai_players=num_ai_opponents, human_players=1)
        self.human_player_id = num_ai_opponents
        self.human_index = self._find_human_index()
        self.num_opponents = num_ai_opponents
        self.chips_history = []
        self.round_num = 1
        self.hand_results = []
        
    def _load_ai_model(self):
        """Load the AI model if available."""
        model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
        
        if not os.path.exists(model_path):
            self._print_message("No trained model found. Using basic AI logic.", Colors.YELLOW)
            return None
        
        try:
            checkpoint = torch.load(model_path)
            model = PokerNet()
            model.load_state_dict(checkpoint['model_state_dict'])
            self._print_message(f"Loaded AI model: Gen {checkpoint['generation']}, Fitness {checkpoint['fitness']:.4f}", Colors.GREEN)
            return model
        except Exception as e:
            self._print_message(f"Error loading model: {e}", Colors.RED)
            self._print_message("Using basic AI logic instead.", Colors.YELLOW)
            return None
    
    def _find_human_index(self):
        """Find the index of the human player."""
        for i, p in enumerate(self.game.players):
            if p.player_id == self.human_player_id:
                return i
        raise ValueError("Human player not found in game!")
    
    def _print_message(self, message, color=Colors.END, delay=0.03):
        """Print a message with animation and color."""
        print(f"{color}", end="")
        for char in message:
            print(char, end='', flush=True)
            time.sleep(delay)
        print(f"{Colors.END}")
        time.sleep(0.2)
    
    def _print_header(self, text):
        """Print a fancy header."""
        width = 60
        print(f"\n{Colors.BG_BLUE}{Colors.BOLD}{' '*width}{Colors.END}")
        padding = (width - len(text)) // 2
        print(f"{Colors.BG_BLUE}{Colors.BOLD}{' '*padding}{text}{' '*(width-padding-len(text))}{Colors.END}")
        print(f"{Colors.BG_BLUE}{Colors.BOLD}{' '*width}{Colors.END}\n")
    
    def _print_separator(self):
        """Print a fancy separator line."""
        print(f"\n{Colors.CYAN}{'═'*60}{Colors.END}\n")
    
    def _format_card(self, card):
        """Format a card with color and symbol."""
        suit_symbol = CARD_SYMBOLS[card.suit]
        
        # Choose color based on suit
        if card.suit == 'hearts' or card.suit == 'diamonds':
            color = Colors.RED
        else:
            color = Colors.BLUE
            
        return f"{color}{card.value}{suit_symbol}{Colors.END}"
    
    def _display_pot(self):
        """Display the pot with fancy formatting."""
        chips = self.game.pot
        if chips > 500:
            color = Colors.GOLD
        elif chips > 100:
            color = Colors.SILVER
        else:
            color = Colors.BRONZE
            
        chips_display = f"{color}${chips}{Colors.END}"
        print(f"\n{Colors.BOLD}POT: {chips_display}{Colors.END}")
        
        # Draw a simple pot graphic
        if chips > 0:
            stack_height = min(3, chips // 100 + 1)
            for i in range(stack_height):
                offset = " " * (stack_height - i - 1)
                print(f"{offset}{color}[■■■■■]{Colors.END}")
    
    def _display_community_cards(self):
        """Display community cards with fancy formatting."""
        if not self.game.community_cards:
            print(f"\n{Colors.BOLD}COMMUNITY CARDS:{Colors.END} None yet")
            return
            
        print(f"\n{Colors.BOLD}COMMUNITY CARDS:{Colors.END}")
        
        # Table background
        width = len(self.game.community_cards) * 6 + 2
        print(f"{Colors.BG_GREEN}{' ' * width}{Colors.END}")
        
        # Cards
        print(f"{Colors.BG_GREEN} ", end="")
        for card in self.game.community_cards:
            print(f"{self._format_card(card)} ", end="")
        print(f"{Colors.BG_GREEN} {Colors.END}")
        
        # Bottom of table
        print(f"{Colors.BG_GREEN}{' ' * width}{Colors.END}")
    
    def _display_player_hand(self):
        """Display player's hand with fancy formatting."""
        human_player = self.game.players[self.human_index]
        
        print(f"\n{Colors.BOLD}YOUR HAND:{Colors.END}")
        
        # Hand background
        width = len(human_player.hand) * 6 + 2
        print(f"{Colors.BG_BLACK}{' ' * width}{Colors.END}")
        
        # Cards
        print(f"{Colors.BG_BLACK} ", end="")
        for card in human_player.hand:
            print(f"{self._format_card(card)} ", end="")
        print(f"{Colors.BG_BLACK} {Colors.END}")
        
        # Bottom of hand
        print(f"{Colors.BG_BLACK}{' ' * width}{Colors.END}")
    
    def _display_ai_players(self):
        """Display AI players with fancy formatting."""
        print(f"\n{Colors.BOLD}AI OPPONENTS:{Colors.END}")
        print(f"┌{'─'*6}┬{'─'*8}┬{'─'*8}┬{'─'*12}┐")
        print(f"│{'ID':^6}│{'CHIPS':^8}│{'BET':^8}│{'STATUS':^12}│")
        print(f"├{'─'*6}┼{'─'*8}┼{'─'*8}┼{'─'*12}┤")
        
        for i, p in enumerate(self.game.players):
            if i != self.human_index:
                status = "Folded" if p.folded else "All-In" if p.all_in else "Active"
                status_color = Colors.RED if p.folded else Colors.GOLD if p.all_in else Colors.GREEN
                print(f"│{i:^6}│{p.chips:^8}│{p.current_bet:^8}│{status_color}{status:^12}{Colors.END}│")
        
        print(f"└{'─'*6}┴{'─'*8}┴{'─'*8}┴{'─'*12}┘")
    
    def _display_game_state(self, stage="Current"):
        """Display the current game state with fancy formatting."""
        self._print_separator()
        
        # Header
        self._print_header(f"{stage.upper()} STATE")
        
        # Player info
        human_player = self.game.players[self.human_index]
        to_call = max(0, self.game.current_bet - human_player.current_bet)
        
        print(f"{Colors.BOLD}YOUR CHIPS:{Colors.END} {Colors.GREEN}${human_player.chips}{Colors.END}")
        print(f"{Colors.BOLD}CURRENT BET:{Colors.END} ${self.game.current_bet}")
        print(f"{Colors.BOLD}TO CALL:{Colors.END} ${to_call}")
        
        # Display pot
        self._display_pot()
        
        # Display community cards
        self._display_community_cards()
        
        # Display player's hand
        self._display_player_hand()
        
        # Display other players
        self._display_ai_players()
        
        self._print_separator()
    
    def _get_human_action(self, round_name):
        """Get action from human player with fancy UI."""
        human_player = self.game.players[self.human_index]
        to_call = max(0, self.game.current_bet - human_player.current_bet)
        
        self._print_message(f"\n[YOUR TURN - {round_name}]", Colors.BOLD)
        
        # Display options
        print(f"{Colors.YELLOW}What would you like to do?{Colors.END}")
        
        if to_call > 0:
            print(f"{Colors.RED}1: Fold{Colors.END}")
            print(f"{Colors.BLUE}2: Call ${to_call}{Colors.END}")
            print(f"{Colors.GREEN}3: Raise{Colors.END}")
        else:
            print(f"{Colors.BLUE}1: Check{Colors.END}")
            print(f"{Colors.GREEN}3: Bet{Colors.END}")
        
        while True:
            try:
                choice = input(f"{Colors.CYAN}Enter your choice (1-3): {Colors.END}").strip()
                
                if choice == "1":
                    if to_call > 0:
                        self._print_message("You fold.", Colors.RED)
                        return {"action": "fold"}
                    else:
                        self._print_message("You check.", Colors.BLUE)
                        return {"action": "call"}
                elif choice == "2" and to_call > 0:
                    self._print_message(f"You call ${to_call}.", Colors.BLUE)
                    return {"action": "call"}
                elif choice == "3":
                    min_raise = self.game.big_blind
                    if to_call > 0:
                        min_raise = to_call + self.game.big_blind
                    
                    max_raise = human_player.chips
                    
                    if min_raise > max_raise:
                        self._print_message("You don't have enough chips to raise.", Colors.RED)
                        if to_call > 0:
                            proceed = input(f"{Colors.YELLOW}Would you like to call instead? (y/n): {Colors.END}").strip().lower()
                            if proceed == 'y':
                                self._print_message(f"You call ${to_call}.", Colors.BLUE)
                                return {"action": "call"}
                            else:
                                self._print_message("You fold.", Colors.RED)
                                return {"action": "fold"}
                        else:
                            self._print_message("You check.", Colors.BLUE)
                            return {"action": "call"}
                    
                    print(f"{Colors.YELLOW}Min raise: ${min_raise}, Max raise: ${max_raise}{Colors.END}")
                    raise_input = input(f"{Colors.GREEN}Enter raise amount (${min_raise}-${max_raise}): {Colors.END}")
                    
                    try:
                        raise_amount = int(raise_input)
                        
                        if min_raise <= raise_amount <= max_raise:
                            action_name = "bet" if self.game.current_bet == 0 else "raise to"
                            self._print_message(f"You {action_name} ${raise_amount}.", Colors.GREEN)
                            return {"action": "raise", "amount": raise_amount}
                        else:
                            self._print_message(f"Invalid amount. Please enter a value between ${min_raise} and ${max_raise}.", Colors.RED)
                    except ValueError:
                        self._print_message("Please enter a valid number.", Colors.RED)
                else:
                    self._print_message("Invalid choice. Please try again.", Colors.RED)
            except ValueError:
                self._print_message("Please enter a valid number.", Colors.RED)
    
    def _handle_betting_round(self, round_name):
        """Handle a full betting round from start to finish."""
        self._print_header(f"{round_name.upper()} BETTING")
        
        # Reset bets for this round
        self.game.current_bet = 0
        for player in self.game.players:
            player.current_bet = 0
        
        # Display game state before betting
        self._display_game_state(f"Before {round_name}")
        
        # Find starting player (first after dealer button)
        starting_player = self.human_index
        if round_name == "Pre-Flop":
            # UTG position (first after big blind)
            sb_pos = (self.game.dealer_pos + 1) % len(self.game.players)
            bb_pos = (sb_pos + 1) % len(self.game.players)
            starting_player = (bb_pos + 1) % len(self.game.players)
        else:
            # First active after dealer
            starting_player = (self.game.dealer_pos + 1) % len(self.game.players)
        
        # Force an AI player to bet in this round?
        force_bet = False
        forced_bettor = None
        
        if round_name != "Pre-Flop" and random.random() < 0.8:
            # 80% chance to have a forced bettor in post-flop
            active_ai_players = [i for i, p in enumerate(self.game.players) 
                               if not p.folded and i != self.human_index]
            
            if active_ai_players:
                forced_bettor = random.choice(active_ai_players)
                force_bet = True
        
        # Betting loop
        current_player = starting_player
        last_raiser = None
        players_acted = 0
        max_iterations = len(self.game.players) * 4  # Safety
        iteration = 0
        
        self._print_message(f"Starting betting with Player {current_player}...", Colors.CYAN, delay=0.01)
        time.sleep(0.5)
        
        while iteration < max_iterations:
            iteration += 1
            
            # Skip folded or all-in players
            if self.game.players[current_player].folded or self.game.players[current_player].all_in:
                current_player = (current_player + 1) % len(self.game.players)
                continue
                
            # Get player action
            player = self.game.players[current_player]
            
            # Check if betting round is complete
            if player.current_bet == self.game.current_bet and (last_raiser is None or players_acted >= len(self.game.players)):
                # Everyone has called or checked
                break
            
            # Get action based on player type
            is_human = (current_player == self.human_index)
            should_force_bet = (force_bet and current_player == forced_bettor and self.game.current_bet == 0)
            
            if is_human:
                action_info = self._get_human_action(round_name)
            else:
                # AI action
                time.sleep(random.uniform(0.5, 1.5))  # Simulate AI "thinking"
                
                if should_force_bet:
                    # Force a bet from this AI
                    bet_size = max(self.game.big_blind * 2, min(player.chips // 10, player.chips))
                    action_info = {"action": "raise", "amount": bet_size}
                    force_bet = False  # Bet has been forced
                    self._print_message(f"Player {current_player} bets ${bet_size}.", Colors.GREEN)
                else:
                    # Regular AI logic
                    r = random.random()
                    to_call = max(0, self.game.current_bet - player.current_bet)
                    
                    if self.game.current_bet == 0:  # Can check
                        if r < 0.3 and round_name != "Pre-Flop":  # 30% chance to bet
                            # Make a bet (5-20% of stack)
                            bet_size = max(self.game.big_blind, min(player.chips // random.randint(5, 20), player.chips))
                            action_info = {"action": "raise", "amount": bet_size}
                            self._print_message(f"Player {current_player} bets ${bet_size}.", Colors.GREEN)
                        else:
                            # Check
                            action_info = {"action": "call"}
                            self._print_message(f"Player {current_player} checks.", Colors.BLUE)
                    else:  # Facing a bet
                        if r < 0.6:  # 60% call
                            action_info = {"action": "call"}
                            self._print_message(f"Player {current_player} calls ${to_call}.", Colors.BLUE)
                        elif r < 0.8 and player.chips >= to_call + self.game.big_blind:  # 20% raise
                            raise_to = to_call + min(player.chips - to_call, random.randint(1, 4) * self.game.big_blind)
                            action_info = {"action": "raise", "amount": raise_to}
                            self._print_message(f"Player {current_player} raises to ${raise_to}.", Colors.GREEN)
                        else:  # 20% fold
                            action_info = {"action": "fold"}
                            self._print_message(f"Player {current_player} folds.", Colors.RED)
            
            # Process the action
            action = action_info.get("action", "call")
            
            if action == "fold":
                player.folded = True
            elif action == "call":
                call_amount = min(self.game.current_bet - player.current_bet, player.chips)
                player.bet(call_amount)
                self.game.pot += call_amount
            elif action == "raise":
                amount = action_info.get("amount", 0)
                raise_to = min(amount, player.chips + player.current_bet)
                raise_amount = raise_to - player.current_bet
                
                player.bet(raise_amount)
                self.game.pot += raise_amount
                self.game.current_bet = player.current_bet
                
                last_raiser = current_player
                players_acted = 0
            
            # Move to next player
            players_acted += 1
            current_player = (current_player + 1) % len(self.game.players)
            
            # Check if everyone folded except one player
            active_players = [p for p in self.game.players if not p.folded]
            if len(active_players) <= 1:
                break
        
        # Display final state
        self._display_game_state(f"After {round_name}")
        
        self._print_header(f"{round_name.upper()} BETTING COMPLETE")
        
        # Return active players
        return [p for p in self.game.players if not p.folded]
    
    def play_hand(self):
        """Play a complete hand of poker with fancy UI."""
        self._print_header("NEW HAND")
        
        # Reset game state
        self.game.pot = 0
        self.game.current_bet = 0
        self.game.community_cards = []
        
        for player in self.game.players:
            player.reset_hand()
        
        # Set dealer button
        self._print_message(f"Dealer position: Player {self.game.dealer_pos}", Colors.YELLOW)
        
        # Post blinds
        sb_pos = (self.game.dealer_pos + 1) % len(self.game.players)
        bb_pos = (sb_pos + 1) % len(self.game.players)
        
        sb_amount = self.game.players[sb_pos].bet(self.game.small_blind)
        self.game.pot += sb_amount
        self._print_message(f"Player {sb_pos} posts small blind of ${sb_amount}.", Colors.BLUE)
        
        bb_amount = self.game.players[bb_pos].bet(self.game.big_blind)
        self.game.pot += bb_amount
        self._print_message(f"Player {bb_pos} posts big blind of ${bb_amount}.", Colors.BLUE)
        
        self.game.current_bet = self.game.big_blind
        
        # Deal hole cards
        self.game.deal_hands()
        self._print_message("Hole cards dealt to all players.", Colors.BLUE)
        time.sleep(0.5)
        
        # PRE-FLOP BETTING
        active_players = self._handle_betting_round("Pre-Flop")
        if len(active_players) <= 1:
            # Hand is over - one player wins uncontested
            if active_players:
                active_players[0].chips += self.game.pot
                self._print_message(f"Player {self.game.players.index(active_players[0])} wins ${self.game.pot} chips.", Colors.GOLD)
                
                # Record the result
                winner = "You" if self.game.players.index(active_players[0]) == self.human_index else f"Player {self.game.players.index(active_players[0])}"
                self.hand_results.append(f"Hand {self.round_num}: {winner} won ${self.game.pot} (uncontested)")
            return
        
        # FLOP
        self._print_header("FLOP")
        self.game.community_cards = self.game.deal_flop()
        cards_str = ", ".join([self._format_card(card) for card in self.game.community_cards])
        self._print_message(f"Flop cards: {cards_str}", Colors.BLUE)
        time.sleep(1)
        
        # FLOP BETTING
        active_players = self._handle_betting_round("Flop")
        if len(active_players) <= 1:
            if active_players:
                active_players[0].chips += self.game.pot
                self._print_message(f"Player {self.game.players.index(active_players[0])} wins ${self.game.pot} chips.", Colors.GOLD)
                
                # Record the result
                winner = "You" if self.game.players.index(active_players[0]) == self.human_index else f"Player {self.game.players.index(active_players[0])}"
                self.hand_results.append(f"Hand {self.round_num}: {winner} won ${self.game.pot} (after flop)")
            return
        
        # TURN
        self._print_header("TURN")
        turn_card = self.game.deal_turn()
        self.game.community_cards.extend(turn_card)
        self._print_message(f"Turn card: {self._format_card(turn_card[0])}", Colors.BLUE)
        time.sleep(1)
        
        # TURN BETTING
        active_players = self._handle_betting_round("Turn")
        if len(active_players) <= 1:
            if active_players:
                active_players[0].chips += self.game.pot
                self._print_message(f"Player {self.game.players.index(active_players[0])} wins ${self.game.pot} chips.", Colors.GOLD)
                
                # Record the result
                winner = "You" if self.game.players.index(active_players[0]) == self.human_index else f"Player {self.game.players.index(active_players[0])}"
                self.hand_results.append(f"Hand {self.round_num}: {winner} won ${self.game.pot} (after turn)")
            return
        
        # RIVER
        self._print_header("RIVER")
        river_card = self.game.deal_river()
        self.game.community_cards.extend(river_card)
        self._print_message(f"River card: {self._format_card(river_card[0])}", Colors.BLUE)
        time.sleep(1)
        
        # RIVER BETTING
        active_players = self._handle_betting_round("River")
        if len(active_players) <= 1:
            if active_players:
                active_players[0].chips += self.game.pot
                self._print_message(f"Player {self.game.players.index(active_players[0])} wins ${self.game.pot} chips.", Colors.GOLD)
                
                # Record the result
                winner = "You" if self.game.players.index(active_players[0]) == self.human_index else f"Player {self.game.players.index(active_players[0])}"
                self.hand_results.append(f"Hand {self.round_num}: {winner} won ${self.game.pot} (after river)")
            return
        
        # SHOWDOWN
        self._print_header("SHOWDOWN")
        self._print_message("Revealing cards for showdown...", Colors.CYAN)
        time.sleep(1)
        
        # Show all active players' cards
        for player in active_players:
            player_index = self.game.players.index(player)
            cards_str = ", ".join([self._format_card(card) for card in player.hand])
            if player_index == self.human_index:
                self._print_message(f"Your hand: {cards_str}", Colors.YELLOW)
            else:
                self._print_message(f"Player {player_index}'s hand: {cards_str}", Colors.BLUE)
            time.sleep(0.5)
        
        # Calculate winners
        winners = self.game.determine_winners()
        
        self._print_message("Community cards: " + ", ".join([self._format_card(card) for card in self.game.community_cards]), Colors.BLUE)
        time.sleep(0.5)
        
        for winner in winners:
            winner_index = self.game.players.index(winner)
            cards_str = ", ".join([self._format_card(card) for card in winner.hand])
            winner_text = "You win" if winner_index == self.human_index else f"Player {winner_index} wins"
            self._print_message(f"{winner_text} ${self.game.pot // len(winners)} chips with {cards_str}!", Colors.GOLD)
            
            # Record the result
            winner_name = "You" if winner_index == self.human_index else f"Player {winner_index}"
            self.hand_results.append(f"Hand {self.round_num}: {winner_name} won ${self.game.pot // len(winners)} (showdown with {cards_str})")
    
    def play_tournament(self):
        """Play a full tournament until human is eliminated or wins."""
        self._print_header("TEXAS HOLD'EM POKER")
        self._print_message("Welcome to the fancy poker table!", Colors.GOLD)
        self._print_message(f"You are playing against {self.num_opponents} AI opponents.", Colors.BLUE)
        self._print_message("Let's begin! Good luck!", Colors.GREEN)
        
        while True:
            # Remove players with no chips
            self.game.players = [p for p in game.players if p.chips > 0]
            
            # Update human index if players were removed
            human_exists = False
            for i, p in enumerate(self.game.players):
                if p.player_id == self.human_player_id:
                    self.human_index = i
                    human_exists = True
                    break
            
            if not human_exists:
                self._print_header("GAME OVER")
                self._print_message("You've been eliminated! Better luck next time.", Colors.RED)
                break
            
            # Check if human is the only player left
            if len(self.game.players) == 1 and self.game.players[0].player_id == self.human_player_id:
                self._print_header("CONGRATULATIONS")
                self._print_message("You've won the tournament!", Colors.GOLD)
                break
            
            # Save chips history
            self.chips_history.append({
                'round': self.round_num,
                'human_chips': self.game.players[self.human_index].chips,
                'ai_chips': [p.chips for i, p in enumerate(self.game.players) if i != self.human_index]
            })
            
            # Display round header
            self._print_header(f"HAND #{self.round_num}")
            
            # Advance dealer position
            self.game.dealer_pos = (self.game.dealer_pos + 1) % len(self.game.players)
            
            # Play a hand
            self.play_hand()
            
            # Show final game state
            self._display_game_state("End of hand")
            
            # Ask to continue
            input(f"{Colors.CYAN}Press Enter to continue to next hand...{Colors.END}")
            self.round_num += 1
        
        # Show game summary
        self.show_game_summary()
    
    def show_game_summary(self):
        """Show a summary of the game results."""
        self._print_header("GAME SUMMARY")
        
        self._print_message(f"Total hands played: {self.round_num}", Colors.YELLOW)
        
        # Show human player's final chip count
        human_player = next((p for p in self.game.players if p.player_id == self.human_player_id), None)
        if human_player:
            self._print_message(f"Your final chip count: ${human_player.chips}", Colors.GREEN)
        else:
            self._print_message("You were eliminated!", Colors.RED)
        
        # Show last 5 hand results
        self._print_message("\nLast hands:", Colors.BLUE)
        for result in self.hand_results[-5:]:
            self._print_message(f"  • {result}", Colors.CYAN, delay=0.01)
        
