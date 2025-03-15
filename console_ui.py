# Replace everything in console_ui.py with this code
import os
import torch
import time
import random
import numpy as np
from poker_game import PokerGame, Player
from neural_network import PokerNet, create_state_tensor
from config import CARD_SYMBOLS, CARD_COLORS, CHECKPOINT_DIR

def clear_screen():
    """Clear the console screen."""
    # os.system('cls' if os.name == 'nt' else 'clear')
    print("\n" + "="*80 + "\n")  # Just add a separator instead of clearing

def load_best_model():
    """Load the best trained model."""
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    
    if not os.path.exists(model_path):
        print("No trained model found. Using basic AI logic instead.")
        return None
    
    try:
        checkpoint = torch.load(model_path)
        model = PokerNet()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from generation {checkpoint['generation']} with fitness {checkpoint['fitness']:.4f}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using basic AI logic instead.")
        return None

def format_card(card):
    """Format a card with color and symbol."""
    suit_symbol = CARD_SYMBOLS[card.suit]
    color = CARD_COLORS[card.suit]
    reset = CARD_COLORS['reset']
    
    return f"{color}{card.value}{suit_symbol}{reset}"

def display_game_state(game, player_index, stage="Current"):
    """Display the current game state."""
    human_player = game.players[player_index]
    
    print(f"\n{'='*60}")
    print(f"[{stage.upper()} GAME STATE]")
    print(f"YOUR CHIPS: {human_player.chips} | POT: {game.pot} | CURRENT BET: {game.current_bet}")
    to_call = max(0, game.current_bet - human_player.current_bet)
    print(f"TO CALL: {to_call}")
    
    # Display community cards
    print(f"\nCOMMUNITY CARDS: ", end="")
    if game.community_cards:
        for card in game.community_cards:
            print(f"{format_card(card)} ", end="")
    else:
        print("None yet")
    
    # Display player's hand
    print(f"\n\nYOUR HAND: ", end="")
    for card in human_player.hand:
        print(f"{format_card(card)} ", end="")
    
    # Display other players
    print("\n\nOTHER PLAYERS:")
    print(f"{'ID':<4}{'CHIPS':<8}{'BET':<8}{'STATUS':<12}")
    print(f"{'-'*30}")
    
    for i, p in enumerate(game.players):
        if i != player_index:
            status = "Folded" if p.folded else "All-In" if p.all_in else "Active"
            print(f"{i:<4}{p.chips:<8}{p.current_bet:<8}{status:<12}")
    
    print(f"{'='*60}\n")

def get_human_action(game, player, round_name):
    """Get action from human player."""
    to_call = max(0, game.current_bet - player.current_bet)
    
    print(f"\n[YOUR TURN - {round_name}]")
    print(f"Your turn to act. What would you like to do?")
    
    if to_call > 0:
        print(f"1: Fold")
        print(f"2: Call {to_call}")
        print(f"3: Raise")
    else:
        print(f"1: Check")
        print(f"3: Bet")
    
    while True:
        try:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == "1":
                if to_call > 0:
                    return {"action": "fold"}
                else:
                    return {"action": "call"}  # Check is implemented as a call with 0 to call
            elif choice == "2" and to_call > 0:
                return {"action": "call"}
            elif choice == "3":
                min_raise = game.big_blind
                if to_call > 0:
                    min_raise = to_call + game.big_blind
                
                max_raise = player.chips
                
                if min_raise > max_raise:
                    print("You don't have enough chips to raise.")
                    if to_call > 0:
                        proceed = input("Would you like to call instead? (y/n): ").strip().lower()
                        if proceed == 'y':
                            return {"action": "call"}
                        else:
                            return {"action": "fold"}
                    else:
                        return {"action": "call"}  # Check
                
                print(f"Min raise: {min_raise}, Max raise: {max_raise}")
                raise_input = input(f"Enter raise amount ({min_raise}-{max_raise}): ")
                
                try:
                    raise_amount = int(raise_input)
                    
                    if min_raise <= raise_amount <= max_raise:
                        return {"action": "raise", "amount": raise_amount}
                    else:
                        print(f"Invalid amount. Please enter a value between {min_raise} and {max_raise}.")
                except ValueError:
                    print("Please enter a valid number.")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

def handle_betting_round(game, human_index, human_player_id, round_name):
    """Handle a full betting round from start to finish."""
    print(f"\n--- {round_name.upper()} BETTING ---")
    
    # Reset bets for this round
    game.current_bet = 0
    for player in game.players:
        player.current_bet = 0
    
    # Display game state before betting
    display_game_state(game, human_index, f"Before {round_name}")
    
    # Find starting player (first after dealer button)
    starting_player = human_index
    if round_name == "Pre-Flop":
        # UTG position (first after big blind)
        sb_pos = (game.dealer_pos + 1) % len(game.players)
        bb_pos = (sb_pos + 1) % len(game.players)
        starting_player = (bb_pos + 1) % len(game.players)
    else:
        # First active after dealer
        starting_player = (game.dealer_pos + 1) % len(game.players)
    
    # Force an AI player to bet in this round?
    force_bet = False
    forced_bettor = None
    
    if round_name != "Pre-Flop" and random.random() < 0.8:
        # 80% chance to have a forced bettor in post-flop
        active_ai_players = [i for i, p in enumerate(game.players) 
                           if not p.folded and i != human_index]
        
        if active_ai_players:
            forced_bettor = random.choice(active_ai_players)
            force_bet = True
            print(f"[DEBUG] Player {forced_bettor} will make a bet this round")
    
    # Betting loop
    current_player = starting_player
    last_raiser = None
    players_acted = 0
    max_iterations = len(game.players) * 4  # Safety
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        
        # Skip folded or all-in players
        if game.players[current_player].folded or game.players[current_player].all_in:
            current_player = (current_player + 1) % len(game.players)
            continue
            
        # Get player action
        player = game.players[current_player]
        
        # Check if betting round is complete
        if player.current_bet == game.current_bet and (last_raiser is None or players_acted >= len(game.players)):
            # Everyone has called or checked
            break
        
        # Get action based on player type
        is_human = (current_player == human_index)
        should_force_bet = (force_bet and current_player == forced_bettor and game.current_bet == 0)
        
        if is_human:
            action_info = get_human_action(game, player, round_name)
        else:
            # AI action
            if should_force_bet:
                # Force a bet from this AI
                bet_size = max(game.big_blind * 2, min(player.chips // 10, player.chips))
                action_info = {"action": "raise", "amount": bet_size}
                force_bet = False  # Bet has been forced
                print(f"Player {current_player} bets {bet_size}")
            else:
                # Regular AI logic
                r = random.random()
                to_call = max(0, game.current_bet - player.current_bet)
                
                if game.current_bet == 0:  # Can check
                    if r < 0.3 and round_name != "Pre-Flop":  # 30% chance to bet
                        # Make a bet (5-20% of stack)
                        bet_size = max(game.big_blind, min(player.chips // random.randint(5, 20), player.chips))
                        action_info = {"action": "raise", "amount": bet_size}
                        print(f"Player {current_player} bets {bet_size}")
                    else:
                        # Check
                        action_info = {"action": "call"}
                        print(f"Player {current_player} checks")
                else:  # Facing a bet
                    if r < 0.6:  # 60% call
                        action_info = {"action": "call"}
                        print(f"Player {current_player} calls {to_call}")
                    elif r < 0.8 and player.chips >= to_call + game.big_blind:  # 20% raise
                        raise_to = to_call + min(player.chips - to_call, random.randint(1, 4) * game.big_blind)
                        action_info = {"action": "raise", "amount": raise_to}
                        print(f"Player {current_player} raises to {raise_to}")
                    else:  # 20% fold
                        action_info = {"action": "fold"}
                        print(f"Player {current_player} folds")
        
        # Process the action
        action = action_info.get("action", "call")
        
        if action == "fold":
            player.folded = True
        elif action == "call":
            call_amount = min(game.current_bet - player.current_bet, player.chips)
            player.bet(call_amount)
            game.pot += call_amount
        elif action == "raise":
            amount = action_info.get("amount", 0)
            raise_to = min(amount, player.chips + player.current_bet)
            raise_amount = raise_to - player.current_bet
            
            player.bet(raise_amount)
            game.pot += raise_amount
            game.current_bet = player.current_bet
            
            last_raiser = current_player
            players_acted = 0
        
        # Move to next player
        players_acted += 1
        current_player = (current_player + 1) % len(game.players)
        
        # Check if everyone folded except one player
        active_players = [p for p in game.players if not p.folded]
        if len(active_players) <= 1:
            break
    
    # Display final state
    display_game_state(game, human_index, f"After {round_name}")
    
    print(f"\n{'='*60}")
    print(f" {round_name.upper()} BETTING COMPLETE")
    print(f"{'='*60}\n")
    
    # Return active players
    return [p for p in game.players if not p.folded]

def play_hand(game, human_index, human_player_id):
    """Play a complete hand of poker."""
    print("\n=== NEW HAND ===")
    
    # Reset game state
    game.pot = 0
    game.current_bet = 0
    game.community_cards = []
    
    for player in game.players:
        player.reset_hand()
    
    # Set dealer button
    print(f"Dealer position: Player {game.dealer_pos}")
    
    # Post blinds
    sb_pos = (game.dealer_pos + 1) % len(game.players)
    bb_pos = (sb_pos + 1) % len(game.players)
    
    sb_amount = game.players[sb_pos].bet(game.small_blind)
    game.pot += sb_amount
    print(f"Player {sb_pos} posts small blind of {sb_amount}")
    
    bb_amount = game.players[bb_pos].bet(game.big_blind)
    game.pot += bb_amount
    print(f"Player {bb_pos} posts big blind of {bb_amount}")
    
    game.current_bet = game.big_blind
    
    # Deal hole cards
    game.deal_hands()
    print("Hole cards dealt to all players.")
    
    # PRE-FLOP BETTING
    active_players = handle_betting_round(game, human_index, human_player_id, "Pre-Flop")
    if len(active_players) <= 1:
        # Hand is over - one player wins uncontested
        if active_players:
            active_players[0].chips += game.pot
            print(f"Player {game.players.index(active_players[0])} wins {game.pot} chips.")
        return
    
    # FLOP
    print("\n--- FLOP ---")
    game.community_cards = game.deal_flop()
    print(f"Flop cards: {game.community_cards}")
    
    # FLOP BETTING
    active_players = handle_betting_round(game, human_index, human_player_id, "Flop")
    if len(active_players) <= 1:
        if active_players:
            active_players[0].chips += game.pot
            print(f"Player {game.players.index(active_players[0])} wins {game.pot} chips.")
        return
    
    # TURN
    print("\n--- TURN ---")
    turn_card = game.deal_turn()
    game.community_cards.extend(turn_card)
    print(f"Turn card: {turn_card[0]}")
    
    # TURN BETTING
    active_players = handle_betting_round(game, human_index, human_player_id, "Turn")
    if len(active_players) <= 1:
        if active_players:
            active_players[0].chips += game.pot
            print(f"Player {game.players.index(active_players[0])} wins {game.pot} chips.")
        return
    
    # RIVER
    print("\n--- RIVER ---")
    river_card = game.deal_river()
    game.community_cards.extend(river_card)
    print(f"River card: {river_card[0]}")
    
    # RIVER BETTING
    active_players = handle_betting_round(game, human_index, human_player_id, "River")
    if len(active_players) <= 1:
        if active_players:
            active_players[0].chips += game.pot
            print(f"Player {game.players.index(active_players[0])} wins {game.pot} chips.")
        return
    
    # SHOWDOWN
    print("\n--- SHOWDOWN ---")
    winners = game.determine_winners()
    
    print(f"Final community cards: {game.community_cards}")
    for winner in winners:
        winner_index = game.players.index(winner)
        print(f"Winner: Player {winner_index} with hand: {winner.hand}")

def play_vs_ai():
    """Play a game against AI opponents."""
    model = load_best_model()
    
    # Create a game with one human player and 4 AI opponents
    game = PokerGame(num_players=5, starting_chips=1000)
    
    # Set up the game with 4 AI players and 1 human player
    game.setup_game(ai_players=4, human_players=1)
    
    # Human player is the last player (player_id = 4)
    human_player_id = 4
    
    # Find human player index
    human_index = None
    for i, p in enumerate(game.players):
        if p.player_id == human_player_id:
            human_index = i
            break
    
    if human_index is None:
        print("Error: Human player not found. Exiting.")
        return
    
    # Store original play_hand method
    original_play_hand = game.play_hand
    
    try:
        # Play tournament until human is eliminated or wins
        round_num = 1
        
        print("\nWelcome to Texas Hold'em Poker!")
        print("You are playing against 4 AI opponents.")
        print("Let's begin!\n")
        
        while True:
            # Remove players with no chips
            game.players = [p for p in game.players if p.chips > 0]
            
            # Update human index if players were removed
            human_exists = False
            for i, p in enumerate(game.players):
                if p.player_id == human_player_id:
                    human_index = i
                    human_exists = True
                    break
            
            if not human_exists:
                print("\nYou've been eliminated! Game over.")
                break
            
            # Check if human is the only player left
            if len(game.players) == 1 and game.players[0].player_id == human_player_id:
                print("\nCongratulations! You've won the tournament!")
                break
            
            print(f"\n{'='*60}")
            print(f" HAND #{round_num}")
            print(f"{'='*60}\n")
            
            # Advance dealer position
            game.dealer_pos = (game.dealer_pos + 1) % len(game.players)
            
            # Play a hand with our custom implementation
            play_hand(game, human_index, human_player_id)
            
            # Show final game state
            display_game_state(game, human_index, "End of hand")
            
            input("\nPress Enter to continue to next hand...")
            round_num += 1
            
    finally:
        # Restore original methods
        game.play_hand = original_play_hand

if __name__ == "__main__":
    print("Welcome to Poker AI")
    print("1: Play against AI")
    print("2: Exit")
    
    choice = input("Enter your choice: ").strip()
    
    if choice == "1":
        play_vs_ai()
    else:
        print("Goodbye!")
        exit(0)