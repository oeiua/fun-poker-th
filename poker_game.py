import random
import numpy as np
import threading
import time
import platform
import sys
from collections import Counter, defaultdict
from config import SMALL_BLIND, BIG_BLIND, TOURNAMENT_TIMEOUT

# Use signal for timeout only on Unix systems
if platform.system() != "Windows":
    import signal

class Card:
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        
    def __repr__(self):
        return f"{self.value}{self.suit[0]}"

class Deck:
    def __init__(self):
        self.suits = ['hearts', 'diamonds', 'clubs', 'spades']
        self.values = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        self.reset()
        
    def reset(self):
        self.cards = [Card(suit, value) for suit in self.suits for value in self.values]
        random.shuffle(self.cards)
        
    def deal(self, n=1):
        if len(self.cards) < n:
            self.reset()
        return [self.cards.pop() for _ in range(n)]
    
    def burn_and_deal(self, n=1):
        """Burn a card and then deal n cards."""
        if len(self.cards) < n + 1:
            self.reset()
        
        # Burn one card
        self.cards.pop()
        
        # Deal n cards
        return [self.cards.pop() for _ in range(n)]

class Player:
    def __init__(self, player_id, chips=1000, is_ai=True):
        self.player_id = player_id
        self.chips = chips
        self.hand = []
        self.is_ai = is_ai
        self.folded = False
        self.all_in = False
        self.current_bet = 0
        self.total_bet = 0
        
    def reset_hand(self):
        self.hand = []
        self.folded = False
        self.all_in = False
        self.current_bet = 0
        self.total_bet = 0
        
    def bet(self, amount):
        amount = min(amount, self.chips)
        self.chips -= amount
        self.current_bet += amount
        self.total_bet += amount
        if self.chips == 0:
            self.all_in = True
        return amount

class PokerGame:
    def __init__(self, num_players=10, starting_chips=1000):
        self.num_players = num_players
        self.starting_chips = starting_chips
        self.players = []
        self.deck = Deck()
        self.community_cards = []
        self.pot = 0
        self.current_bet = 0
        self.dealer_pos = 0
        self.active_player = 0
        self.small_blind = SMALL_BLIND
        self.big_blind = BIG_BLIND
        self.hand_history = []
        self.side_pots = []
        self.timeout_occurred = False
        
    class TimeoutThread(threading.Thread):
        def __init__(self, timeout, callback):
            threading.Thread.__init__(self)
            self.timeout = timeout
            self.callback = callback
            self.daemon = True
            self._stop_event = threading.Event()
            
        def run(self):
            start_time = time.time()
            while not self._stop_event.is_set() and time.time() - start_time < self.timeout:
                time.sleep(0.1)
            
            if not self._stop_event.is_set():
                self.callback()
                
        def stop(self):
            self._stop_event.set()
        
    def setup_game(self, ai_players=10, human_players=0):
        self.players = []
        for i in range(ai_players):
            self.players.append(Player(i, self.starting_chips, True))
        for i in range(human_players):
            self.players.append(Player(ai_players + i, self.starting_chips, False))
        random.shuffle(self.players)
        self.dealer_pos = random.randint(0, len(self.players) - 1)
        
    def get_next_active_player(self, start_pos):
        # Check if start_pos is None
        if start_pos is None:
            # Return the first active player if any, otherwise None
            for i in range(len(self.players)):
                if not self.players[i].folded and self.players[i].chips > 0:
                    return i
            return None
            
        pos = start_pos
        while True:
            pos = (pos + 1) % len(self.players)
            if pos == start_pos:
                return None
            if not self.players[pos].folded and self.players[pos].chips > 0:
                return pos
    
    def handle_timeout(self):
        self.timeout_occurred = True
        
    def deal_hands(self):
        """Deal two cards to each player."""
        self.deck.reset()
        for player in self.players:
            player.reset_hand()
            player.hand = self.deck.deal(2)
    
    def deal_flop(self):
        """Burn a card and deal the flop (3 community cards)."""
        return self.deck.burn_and_deal(3)
    
    def deal_turn(self):
        """Burn a card and deal the turn (1 community card)."""
        return self.deck.burn_and_deal(1)
    
    def deal_river(self):
        """Burn a card and deal the river (1 community card)."""
        return self.deck.burn_and_deal(1)
    
    def post_blinds(self):
        """Post small and big blinds."""
        sb_pos = self.get_next_active_player(self.dealer_pos)
        if sb_pos is None:
            return False
        
        bb_pos = self.get_next_active_player(sb_pos)
        if bb_pos is None:
            return False
        
        # Post small blind
        sb_amount = self.players[sb_pos].bet(self.small_blind)
        self.pot += sb_amount
        print(f"Player {sb_pos} posts small blind of {sb_amount}")
        
        # Post big blind
        bb_amount = self.players[bb_pos].bet(self.big_blind)
        self.pot += bb_amount
        print(f"Player {bb_pos} posts big blind of {bb_amount}")
        
        # Set the current bet to the big blind
        self.current_bet = self.big_blind
        
        return True
    
    def _print_player_status(self, stage):
        """Print the status of all players and the game state."""
        print(f"\n{stage} player status:")
        for i, p in enumerate(self.players):
            status = "FOLDED" if p.folded else "ALL-IN" if p.all_in else "ACTIVE"
            print(f"Player {i}: Chips={p.chips}, Bet={p.current_bet}, Status={status}")
        print(f"Pot: {self.pot}, Current bet: {self.current_bet}")
    
    def get_player_action(self, player, current_bet):
        """Get the action for a player. This will be overridden with neural network actions."""
        # For now, return a weighted action to reduce folding
        # Make calling/checking more likely than folding or raising
        
        # If player can check (no bet to call), prioritize check over fold
        can_check = (current_bet - player.current_bet) == 0
        
        if can_check:
            # When checking is possible, make it more likely
            action_weights = {"fold": 1, "call": 8, "raise": 3}  # "call" here means "check"
        else:
            # When facing a bet, adjust weights
            action_weights = {"fold": 2, "call": 6, "raise": 3}
        
        actions = list(action_weights.keys())
        weights = list(action_weights.values())
        action = random.choices(actions, weights=weights, k=1)[0]
        
        if action == "raise":
            # Ensure min_raise is at least the big blind or 2x current bet
            min_raise = max(self.big_blind, current_bet * 2)
            if min_raise <= player.current_bet:
                min_raise = player.current_bet + self.big_blind
            
            max_raise = player.chips + player.current_bet
            
            # Make sure max_raise is not less than min_raise
            if max_raise < min_raise:
                action = "call"  # Can't afford to raise, so call instead
                return {"action": action}
            
            # Calculate a reasonable raise amount
            raise_to = min_raise + random.randint(0, max(0, max_raise - min_raise))
            return {"action": action, "amount": raise_to}
        else:
            return {"action": action}
    
    def betting_round(self, starting_player=None):
        """Execute a betting round.
        
        Args:
            starting_player: Position of the player to start the betting round.
                             If None, determine based on the stage.
        """
        # Determine who starts betting
        if starting_player is None:
            if not self.community_cards:  # Pre-flop
                # UTG position (first player after big blind)
                sb_pos = self.get_next_active_player(self.dealer_pos)
                bb_pos = self.get_next_active_player(sb_pos)
                starting_player = self.get_next_active_player(bb_pos)
            else:  # Post-flop
                # First active player after dealer
                starting_player = self.get_next_active_player(self.dealer_pos)
        
        if starting_player is None:
            print("No active starting player found for betting round.")
            return
        
        print(f"Starting betting round with player {starting_player}, current bet: {self.current_bet}")
        
        players_acted = 0
        current_player = starting_player
        last_raiser = None
        timeout_thread = None
        self.timeout_occurred = False
        
        # Count active players before starting
        active_count = sum(1 for p in self.players if not p.folded and not p.all_in)
        
        # If only 0 or 1 active players, skip betting
        if active_count <= 1:
            print(f"Skipping betting round - only {active_count} active players.")
            return
        
        # Set up timeout handling
        if platform.system() != "Windows":
            # Unix-based systems can use signal.SIGALRM
            def unix_timeout_handler(signum, frame):
                self.timeout_occurred = True
                
            signal.signal(signal.SIGALRM, unix_timeout_handler)
            signal.alarm(TOURNAMENT_TIMEOUT)
        else:
            # Windows uses a thread-based timeout
            timeout_thread = self.TimeoutThread(TOURNAMENT_TIMEOUT, self.handle_timeout)
            timeout_thread.start()
        
        try:
            # Main betting loop
            round_complete = False
            max_iterations = len(self.players) * 4  # Safety to prevent infinite loops
            iteration_count = 0
            
            while not round_complete and iteration_count < max_iterations:
                iteration_count += 1
                
                # Check if timeout was triggered
                if self.timeout_occurred:
                    print("Timeout occurred in betting round.")
                    # Force fold remaining players
                    current = self.get_next_active_player(current_player)
                    while current is not None:
                        if not self.players[current].folded and not self.players[current].all_in:
                            self.players[current].folded = True
                            print(f"Player {current} forced to fold due to timeout")
                        current = self.get_next_active_player(current)
                    break
                
                # Check if we still have the current player (might have been removed)
                if current_player >= len(self.players):
                    current_player = current_player % len(self.players)
                
                player = self.players[current_player]
                
                # Skip folded or all-in players
                if player.folded or player.all_in:
                    # Get next active player
                    next_player = self.get_next_active_player(current_player)
                    if next_player is None:
                        print("No next active player, ending betting round.")
                        round_complete = True
                        break
                    current_player = next_player
                    continue
                
                # Check if betting round is complete
                to_call = self.current_bet - player.current_bet
                
                # If player has already matched the current bet and we've gone around the table
                # since the last raise, end the betting round
                if to_call == 0 and (last_raiser is None or players_acted >= len(self.players)):
                    print(f"Betting round complete - all active players have matched the current bet of {self.current_bet}.")
                    round_complete = True
                    break
                
                # Get player action
                print(f"Player {current_player} to act (chips: {player.chips}, current bet: {player.current_bet}, to call: {to_call})")
                action_info = self.get_player_action(player, self.current_bet)
                
                action = action_info["action"]
                amount = action_info.get("amount", 0)
                
                if action == "fold":
                    player.folded = True
                    print(f"Player {current_player} folds")
                elif action == "call":
                    if to_call == 0:
                        print(f"Player {current_player} checks")
                    else:
                        call_amount = min(to_call, player.chips)
                        self.pot += player.bet(call_amount)
                        print(f"Player {current_player} calls {call_amount}, pot now {self.pot}")
                elif action == "raise":
                    # Amount is the total bet, not the raise amount
                    raise_to = min(amount, player.chips + player.current_bet)
                    raise_amount = raise_to - player.current_bet
                    
                    # Ensure minimum raise
                    min_raise = max(self.big_blind, self.current_bet - player.current_bet + self.big_blind)
                    
                    if raise_amount < min_raise and raise_amount < player.chips:
                        # Raise too small, convert to call
                        print(f"Raise amount {raise_amount} too small, minimum is {min_raise}")
                        call_amount = min(to_call, player.chips)
                        self.pot += player.bet(call_amount)
                        print(f"Player {current_player} calls {call_amount} instead, pot now {self.pot}")
                    else:
                        # Valid raise
                        self.pot += player.bet(raise_amount)
                        self.current_bet = player.current_bet
                        last_raiser = current_player
                        players_acted = 0
                        print(f"Player {current_player} raises to {player.current_bet}, pot now {self.pot}")
                
                players_acted += 1
                
                # Get next active player
                next_player = self.get_next_active_player(current_player)
                if next_player is None:
                    print("No next active player, ending betting round.")
                    round_complete = True
                    break
                current_player = next_player
                
                # Check if we're the only active player left
                active_count = sum(1 for p in self.players if not p.folded and not p.all_in)
                if active_count <= 1:
                    print(f"Only {active_count} active player left, ending betting round.")
                    round_complete = True
                    break
                
                # If everyone has called or folded (or is all-in), we're done
                all_matched = True
                for p in self.players:
                    if not p.folded and not p.all_in and p.current_bet != self.current_bet:
                        all_matched = False
                        break
                
                if all_matched and players_acted > 0:
                    print("All active players have matched bets, ending betting round.")
                    round_complete = True
                    break
                    
            # Ensure we log if we hit the safety limit
            if iteration_count >= max_iterations:
                print(f"Warning: Betting round hit maximum iterations ({max_iterations})")
                
        finally:
            # Clean up timeout handlers
            if platform.system() != "Windows":
                # Disable the alarm on Unix systems
                signal.alarm(0)
            elif timeout_thread:
                # Stop the timeout thread on Windows
                timeout_thread.stop()
                
        # Debug info
        print(f"Betting round completed. Active players: {sum(1 for p in self.players if not p.folded)}, Pot: {self.pot}")
    
    def play_hand(self):
        """Play a single hand of poker from start to finish."""
        # Reset game state for a new hand
        self.pot = 0
        self.current_bet = 0
        self.community_cards = []
        self.side_pots = []
        self.timeout_occurred = False
        
        for player in self.players:
            player.reset_hand()
        
        # Check if there are enough players to play
        if len(self.players) < 2:
            print("Not enough players to play a hand.")
            return False  # Not enough players
        
        # Move dealer button
        self.dealer_pos = self.get_next_active_player(self.dealer_pos)
        if self.dealer_pos is None:
            print("No active player found for dealer position.")
            return False  # Game over, only one player left
        
        try:
            print("\n=== NEW HAND ===")
            print(f"Dealer position: Player {self.dealer_pos}")
            
            # Post blinds
            if not self.post_blinds():
                print("Failed to post blinds.")
                return False
            
            # Deal hole cards (2 cards to each player)
            self.deal_hands()
            print("Hole cards dealt to all players.")
            
            # Print all players' status before pre-flop
            self._print_player_status("Before pre-flop betting")
            
            # BETTING ROUND 1: Pre-flop
            print("\n--- PRE-FLOP BETTING ---")
            self.betting_round()  # Starting player is UTG (after BB)
            
            # Print all players' status after pre-flop
            self._print_player_status("After pre-flop betting")
            
            # Count active players after pre-flop
            active_players = [p for p in self.players if not p.folded]
            print(f"Active players after pre-flop: {len(active_players)}")
            
            # If only 0 or 1 players remain, end the hand
            if len(active_players) <= 1:
                print("Hand ends after pre-flop - not enough active players for flop.")
                if len(active_players) == 1:
                    active_players[0].chips += self.pot
                    print(f"Player {active_players[0].player_id} wins {self.pot} chips.")
                return True
            
            # DEAL FLOP (burn 1, deal 3 community cards)
            print("\n--- FLOP ---")
            flop_cards = self.deal_flop()
            self.community_cards = flop_cards
            print(f"Flop cards: {self.community_cards}")
            
            # BETTING ROUND 2: Flop
            # Reset current bet for new betting round
            self.current_bet = 0
            for player in self.players:
                player.current_bet = 0
                
            print("\n--- FLOP BETTING ---")
            self.betting_round()  # Starting player is first active after dealer
            
            # Print all players' status after flop
            self._print_player_status("After flop betting")
            
            # Count active players after flop
            active_players = [p for p in self.players if not p.folded]
            print(f"Active players after flop: {len(active_players)}")
            
            # If only 0 or 1 players remain, end the hand
            if len(active_players) <= 1:
                print("Hand ends after flop - not enough active players for turn.")
                if len(active_players) == 1:
                    active_players[0].chips += self.pot
                    print(f"Player {active_players[0].player_id} wins {self.pot} chips.")
                return True
            
            # DEAL TURN (burn 1, deal 1 community card)
            print("\n--- TURN ---")
            turn_card = self.deal_turn()
            self.community_cards.extend(turn_card)
            print(f"Turn card: {turn_card[0]}")
            print(f"Community cards: {self.community_cards}")
            
            # BETTING ROUND 3: Turn
            # Reset current bet for new betting round
            self.current_bet = 0
            for player in self.players:
                player.current_bet = 0
                
            print("\n--- TURN BETTING ---")
            self.betting_round()  # Starting player is first active after dealer
            
            # Print all players' status after turn
            self._print_player_status("After turn betting")
            
            # Count active players after turn
            active_players = [p for p in self.players if not p.folded]
            print(f"Active players after turn: {len(active_players)}")
            
            # If only 0 or 1 players remain, end the hand
            if len(active_players) <= 1:
                print("Hand ends after turn - not enough active players for river.")
                if len(active_players) == 1:
                    active_players[0].chips += self.pot
                    print(f"Player {active_players[0].player_id} wins {self.pot} chips.")
                return True
            
            # DEAL RIVER (burn 1, deal 1 community card)
            print("\n--- RIVER ---")
            river_card = self.deal_river()
            self.community_cards.extend(river_card)
            print(f"River card: {river_card[0]}")
            print(f"Community cards: {self.community_cards}")
            
            # BETTING ROUND 4: River
            # Reset current bet for new betting round
            self.current_bet = 0
            for player in self.players:
                player.current_bet = 0
                
            print("\n--- RIVER BETTING ---")
            self.betting_round()  # Starting player is first active after dealer
            
            # Print all players' status after river
            self._print_player_status("After river betting")
            
            # Count active players after river for showdown
            active_players = [p for p in self.players if not p.folded]
            print(f"Active players for showdown: {len(active_players)}")
            
            # If only 0 or 1 players remain, end the hand
            if len(active_players) <= 1:
                print("Hand ends after river - not enough active players for showdown.")
                if len(active_players) == 1:
                    active_players[0].chips += self.pot
                    print(f"Player {active_players[0].player_id} wins {self.pot} chips.")
                return True
            
            # SHOWDOWN (compare hands if multiple players remain)
            if len(active_players) > 1:
                print("\n--- SHOWDOWN ---")
                winners = self.determine_winners()
                print(f"Final community cards: {self.community_cards}")
                for winner in winners:
                    print(f"Winner: Player {winner.player_id} with hand: {winner.hand}")
            
        except Exception as e:
            print(f"Error during hand: {e}")
            import traceback
            traceback.print_exc()
            # Emergency recovery - refund the pot to active players
            active_players = [p for p in self.players if not p.folded]
            if active_players:
                split = self.pot // len(active_players)
                for p in active_players:
                    p.chips += split
        
        return True
    
    def calculate_hand_strength(self, hand):
        """Calculate the strength of a hand combined with community cards."""
        def card_value(card):
            values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            return values[card.value]
        
        def is_straight(cards):
            values = sorted([card_value(card) for card in cards])
            # Check for A-5 straight
            if set(values) == {2, 3, 4, 5, 14}:
                return 5
            # Check for regular straight
            return values[-1] if values == list(range(values[0], values[0] + 5)) else 0
            
        def is_flush(cards):
            suits = [card.suit for card in cards]
            return cards[0].suit if len(set(suits)) == 1 else None
        
        # Combine hole cards and community cards
        all_cards = hand + self.community_cards
        
        # Generate all possible 5-card combinations
        best_hand_value = 0
        best_hand = []
        
        for i in range(len(all_cards)):
            for j in range(i+1, len(all_cards)):
                for k in range(j+1, len(all_cards)):
                    for l in range(k+1, len(all_cards)):
                        for m in range(l+1, len(all_cards)):
                            five_cards = [all_cards[i], all_cards[j], all_cards[k], all_cards[l], all_cards[m]]
                            hand_value = self.evaluate_five_card_hand(five_cards)
                            if hand_value > best_hand_value:
                                best_hand_value = hand_value
                                best_hand = five_cards
        
        return best_hand_value, best_hand
    
    def evaluate_five_card_hand(self, cards):
        """Evaluate the strength of a five-card poker hand."""
        def card_value(card):
            values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
            return values[card.value]
        
        values = [card_value(card) for card in cards]
        suits = [card.suit for card in cards]
        
        # Count occurrences of each value
        value_counts = Counter(values)
        suit_counts = Counter(suits)
        
        # Check for flush
        is_flush = max(suit_counts.values()) == 5
        
        # Check for straight
        sorted_values = sorted(values)
        is_straight = False
        
        # Regular straight
        if sorted_values == list(range(min(sorted_values), min(sorted_values) + 5)):
            is_straight = True
        
        # A-5 straight (Ace counts as 1)
        if set(sorted_values) == {2, 3, 4, 5, 14}:
            is_straight = True
            sorted_values = [1, 2, 3, 4, 5]  # Ace counts as low
        
        # Royal flush: A, K, Q, J, 10 of same suit
        if is_flush and set(values) == {10, 11, 12, 13, 14}:
            return 9 * 10**10  # Royal flush
        
        # Straight flush
        if is_straight and is_flush:
            return 8 * 10**10 + max(sorted_values)  # Straight flush
        
        # Four of a kind
        if 4 in value_counts.values():
            four_val = next(val for val, count in value_counts.items() if count == 4)
            kicker = next(val for val, count in value_counts.items() if count == 1)
            return 7 * 10**10 + four_val * 10**2 + kicker  # Four of a kind
        
        # Full house
        if 3 in value_counts.values() and 2 in value_counts.values():
            three_val = next(val for val, count in value_counts.items() if count == 3)
            pair_val = next(val for val, count in value_counts.items() if count == 2)
            return 6 * 10**10 + three_val * 10**2 + pair_val  # Full house
        
        # Flush
        if is_flush:
            return 5 * 10**10 + sum(val * 10**(2*i) for i, val in enumerate(sorted(values, reverse=True)))  # Flush
        
        # Straight
        if is_straight:
            return 4 * 10**10 + max(sorted_values)  # Straight
        
        # Three of a kind
        if 3 in value_counts.values():
            three_val = next(val for val, count in value_counts.items() if count == 3)
            kickers = sorted([val for val, count in value_counts.items() if count == 1], reverse=True)
            return 3 * 10**10 + three_val * 10**4 + kickers[0] * 10**2 + kickers[1]  # Three of a kind
        
        # Two pair
        if list(value_counts.values()).count(2) == 2:
            pairs = sorted([val for val, count in value_counts.items() if count == 2], reverse=True)
            kicker = next(val for val, count in value_counts.items() if count == 1)
            return 2 * 10**10 + pairs[0] * 10**4 + pairs[1] * 10**2 + kicker  # Two pair
        
        # One pair
        if 2 in value_counts.values():
            pair_val = next(val for val, count in value_counts.items() if count == 2)
            kickers = sorted([val for val, count in value_counts.items() if count == 1], reverse=True)
            return 1 * 10**10 + pair_val * 10**6 + kickers[0] * 10**4 + kickers[1] * 10**2 + kickers[2]  # One pair
        
        # High card
        return sum(val * 10**(2*i) for i, val in enumerate(sorted(values, reverse=True)))  # High card
    
    def determine_winners(self):
        """Determine the winners of a showdown and distribute the pot."""
        active_players = [p for p in self.players if not p.folded]
        
        if len(active_players) == 0:
            print("Warning: No active players found in determine_winners()")
            return []
        
        if len(active_players) == 1:
            winner = active_players[0]
            winner.chips += self.pot
            print(f"Player {winner.player_id} wins {self.pot} chips as the only remaining player.")
            return [winner]
        
        # Calculate hand strengths for active players
        hand_strengths = []
        for player in active_players:
            try:
                strength = self.calculate_hand_strength(player.hand)
                hand_strengths.append((player, strength))
                print(f"Player {player.player_id}'s hand: {player.hand}, strength: {strength[0]}")
            except Exception as e:
                print(f"Error calculating hand strength for player {player.player_id}: {e}")
                # Give this player lowest possible hand strength
                hand_strengths.append((player, (0, [])))
        
        if not hand_strengths:
            print("Warning: No valid hand strengths calculated")
            # Just distribute pot equally among active players
            split_amount = self.pot // len(active_players)
            for player in active_players:
                player.chips += split_amount
            return active_players
            
        best_value = max(strength[0] for _, strength in hand_strengths)
        print(f"Best hand value: {best_value}")
        
        winners = [player for player, strength in hand_strengths if strength[0] == best_value]
        
        # Split pot among winners
        split_amount = self.pot // len(winners)
        remainder = self.pot % len(winners)
        
        for winner in winners:
            winner.chips += split_amount
            print(f"Player {winner.player_id} wins {split_amount} chips")
        
        # Give remainder to a random winner
        if remainder > 0 and winners:
            random_winner = random.choice(winners)
            random_winner.chips += remainder
            print(f"Player {random_winner.player_id} gets {remainder} remainder chips")
        
        return winners
        
    def play_tournament(self, max_hands=1000):
        """Play a tournament until only one player remains or max_hands is reached."""
        self.setup_game()
        
        hand_num = 0
        try:
            while hand_num < max_hands:
                # Remove players with no chips
                self.players = [p for p in self.players if p.chips > 0]
                
                if len(self.players) <= 1:
                    print("Tournament ended - only one player remains.")
                    break
                
                print(f"\n\n === HAND #{hand_num+1} === ")
                hand_success = self.play_hand()
                if not hand_success:
                    print("Hand could not be played. Ending tournament.")
                    break
                
                hand_num += 1
                
                # Print player standings after each hand
                print("\n === CURRENT STANDINGS === ")
                sorted_players = sorted(self.players, key=lambda p: p.chips, reverse=True)
                for i, player in enumerate(sorted_players):
                    print(f"#{i+1}: Player {player.player_id} - {player.chips} chips")
                
        except Exception as e:
            print(f"Tournament error: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n === TOURNAMENT COMPLETE === ")
        print(f"Total hands played: {hand_num}")
        
        # Return the winner(s)
        winners = sorted(self.players, key=lambda p: p.chips, reverse=True)
        return winners