#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Hand Evaluator

This module evaluates poker hands and calculates hand strength, 
equity, and potential for decision making.
"""

import logging
import random
import itertools
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from collections import Counter

from .constants import HandRank
from vision.card_detector import Card, CardValue, CardSuit

logger = logging.getLogger("PokerVision.HandEvaluator")


@dataclass
class HandEvaluation:
    """Class representing the evaluation of a poker hand."""
    hand_rank: HandRank
    hand_value: int
    kickers: List[CardValue]
    description: str
    equity: float = 0.0
    potential: float = 0.0
    
    def __str__(self) -> str:
        """String representation of the hand evaluation."""
        return f"{self.description} ({self.hand_rank.name}, Equity: {self.equity:.2%})"


class HandEvaluator:
    """Class for evaluating poker hands."""
    
    def __init__(self):
        """Initialize the HandEvaluator."""
        logger.info("Initializing HandEvaluator")
        self.simulation_iterations = 1000  # Number of Monte Carlo iterations for equity calculations
    
    def evaluate_hand(self, player_cards: List[Card], community_cards: List[Card]) -> HandEvaluation:
        """
        Evaluate a poker hand.
        
        Args:
            player_cards: List of player's hole cards
            community_cards: List of community cards
            
        Returns:
            HandEvaluation object with hand information
        """
        logger.debug("Evaluating hand")
        
        # All available cards
        all_cards = player_cards + community_cards
        
        # Get the best 5-card hand
        best_hand_result = self._get_best_five_card_hand(all_cards)
        best_hand, hand_rank, hand_value, kickers = best_hand_result
        
        # Generate a description of the hand
        description = self._get_hand_description(best_hand, hand_rank)
        
        # Calculate equity if we have hole cards and some community cards
        equity = 0.0
        potential = 0.0
        if player_cards and community_cards:
            if len(community_cards) < 5:
                equity, potential = self._calculate_equity_and_potential(
                    player_cards, community_cards
                )
            else:
                # If all community cards are out, equity is either 1 (win) or 0 (lose)
                # This would require knowledge of other players' cards
                # For now, just set to 0 as a placeholder
                equity = 0.0
                potential = 0.0
            
        # Create and return the evaluation
        evaluation = HandEvaluation(
            hand_rank=hand_rank,
            hand_value=hand_value,
            kickers=kickers,
            description=description,
            equity=equity,
            potential=potential
        )
        
        logger.info(f"Hand evaluation: {evaluation}")
        return evaluation
    
    def _get_best_five_card_hand(self, cards: List[Card]) -> Tuple[List[Card], HandRank, int, List[CardValue]]:
        """
        Get the best 5-card hand from the available cards.
        
        Args:
            cards: List of available cards
            
        Returns:
            Tuple of (best 5-card hand, hand rank, hand value, kickers)
        """
        if len(cards) < 5:
            # Not enough cards for a 5-card hand
            return cards, HandRank.HIGH_CARD, 0, []
        
        # Generate all possible 5-card combinations
        best_hand = None
        best_rank = HandRank.HIGH_CARD
        best_value = 0
        best_kickers = []
        
        for hand in itertools.combinations(cards, 5):
            hand_list = list(hand)
            rank, value, kickers = self._evaluate_five_card_hand(hand_list)
            
            # Check if this hand is better than the current best
            if rank.value > best_rank.value or (rank == best_rank and value > best_value):
                best_hand = hand_list
                best_rank = rank
                best_value = value
                best_kickers = kickers
        
        return best_hand, best_rank, best_value, best_kickers
    
    def _evaluate_five_card_hand(self, hand: List[Card]) -> Tuple[HandRank, int, List[CardValue]]:
        """
        Evaluate a 5-card poker hand.
        
        Args:
            hand: List of 5 cards
            
        Returns:
            Tuple of (hand rank, hand value, kickers)
        """
        if len(hand) != 5:
            raise ValueError("Hand must contain exactly 5 cards")
        
        # Sort cards by value (descending)
        sorted_hand = sorted(hand, key=lambda card: card.value.value, reverse=True)
        
        # Extract values and suits
        values = [card.value for card in sorted_hand]
        suits = [card.suit for card in sorted_hand]
        
        # Count occurrences of each value and suit
        value_counts = Counter(values)
        suit_counts = Counter(suits)
        
        # Check for flush
        is_flush = max(suit_counts.values()) == 5
        
        # Check for straight
        value_list = [v.value for v in values]
        is_straight = False
        
        # Regular straight
        if len(set(value_list)) == 5 and max(value_list) - min(value_list) == 4:
            is_straight = True
        
        # A-5-4-3-2 straight (Ace-low or "wheel")
        if set(value_list) == {14, 5, 4, 3, 2}:
            is_straight = True
            # For A-5-4-3-2, we need to move Ace to the end for proper ranking
            sorted_hand = sorted_hand[1:] + [sorted_hand[0]]
            values = [card.value for card in sorted_hand]
            value_list = [v.value for v in values]
        
        # Determine hand rank and value
        if is_straight and is_flush:
            # Check for royal flush (A-K-Q-J-10 of same suit)
            if set(value_list) == {14, 13, 12, 11, 10}:
                return HandRank.ROYAL_FLUSH, 0, []
            else:
                return HandRank.STRAIGHT_FLUSH, max(value_list), []
        
        # Check for four of a kind
        if 4 in value_counts.values():
            quads_value = [v.value for v, count in value_counts.items() if count == 4][0]
            kicker = [v for v in values if v.value != quads_value]
            return HandRank.FOUR_OF_A_KIND, quads_value, kicker
        
        # Check for full house
        if 3 in value_counts.values() and 2 in value_counts.values():
            trips_value = [v.value for v, count in value_counts.items() if count == 3][0]
            pair_value = [v.value for v, count in value_counts.items() if count == 2][0]
            return HandRank.FULL_HOUSE, trips_value * 100 + pair_value, []
        
        # Check for flush
        if is_flush:
            return HandRank.FLUSH, sum(v.value * (100 ** i) for i, v in enumerate(values)), []
        
        # Check for straight
        if is_straight:
            return HandRank.STRAIGHT, max(value_list), []
        
        # Check for three of a kind
        if 3 in value_counts.values():
            trips_value = [v.value for v, count in value_counts.items() if count == 3][0]
            kickers = [v for v in values if v.value != trips_value]
            return HandRank.THREE_OF_A_KIND, trips_value, kickers
        
        # Check for two pair
        if list(value_counts.values()).count(2) == 2:
            pairs = [v.value for v, count in value_counts.items() if count == 2]
            pairs.sort(reverse=True)
            kicker = [v for v in values if v.value not in pairs]
            return HandRank.TWO_PAIR, pairs[0] * 100 + pairs[1], kicker
        
        # Check for one pair
        if 2 in value_counts.values():
            pair_value = [v.value for v, count in value_counts.items() if count == 2][0]
            kickers = [v for v in values if v.value != pair_value]
            return HandRank.ONE_PAIR, pair_value, kickers
        
        # High card
        return HandRank.HIGH_CARD, values[0].value, values[1:]
    
    def _get_hand_description(self, hand: List[Card], hand_rank: HandRank) -> str:
        """
        Generate a human-readable description of the hand.
        
        Args:
            hand: The 5-card hand
            hand_rank: The rank of the hand
            
        Returns:
            Description string
        """
        if not hand:
            return "No hand"
        
        # Sort cards by value (descending)
        sorted_hand = sorted(hand, key=lambda card: card.value.value, reverse=True)
        
        # Extract values
        values = [card.value for card in sorted_hand]
        value_counts = Counter(values)
        
        # Map card values to names
        value_names = {
            CardValue.ACE: "Ace",
            CardValue.KING: "King",
            CardValue.QUEEN: "Queen",
            CardValue.JACK: "Jack",
            CardValue.TEN: "Ten",
            CardValue.NINE: "Nine",
            CardValue.EIGHT: "Eight",
            CardValue.SEVEN: "Seven",
            CardValue.SIX: "Six",
            CardValue.FIVE: "Five",
            CardValue.FOUR: "Four",
            CardValue.THREE: "Three",
            CardValue.TWO: "Two"
        }
        
        # Generate description based on hand rank
        if hand_rank == HandRank.ROYAL_FLUSH:
            return "Royal Flush"
        
        elif hand_rank == HandRank.STRAIGHT_FLUSH:
            high_card = max(values, key=lambda v: v.value)
            return f"{value_names[high_card]}-high Straight Flush"
        
        elif hand_rank == HandRank.FOUR_OF_A_KIND:
            quads_value = [v for v, count in value_counts.items() if count == 4][0]
            return f"Four of a Kind, {value_names[quads_value]}s"
        
        elif hand_rank == HandRank.FULL_HOUSE:
            trips_value = [v for v, count in value_counts.items() if count == 3][0]
            pair_value = [v for v, count in value_counts.items() if count == 2][0]
            return f"Full House, {value_names[trips_value]}s over {value_names[pair_value]}s"
        
        elif hand_rank == HandRank.FLUSH:
            high_card = values[0]
            return f"{value_names[high_card]}-high Flush"
        
        elif hand_rank == HandRank.STRAIGHT:
            high_card = max(values, key=lambda v: v.value)
            # Special case for A-5-4-3-2 straight
            if set(v.value for v in values) == {14, 5, 4, 3, 2}:
                return "Five-high Straight (Wheel)"
            return f"{value_names[high_card]}-high Straight"
        
        elif hand_rank == HandRank.THREE_OF_A_KIND:
            trips_value = [v for v, count in value_counts.items() if count == 3][0]
            return f"Three of a Kind, {value_names[trips_value]}s"
        
        elif hand_rank == HandRank.TWO_PAIR:
            pairs = [v for v, count in value_counts.items() if count == 2]
            pairs.sort(key=lambda v: v.value, reverse=True)
            return f"Two Pair, {value_names[pairs[0]]}s and {value_names[pairs[1]]}s"
        
        elif hand_rank == HandRank.ONE_PAIR:
            pair_value = [v for v, count in value_counts.items() if count == 2][0]
            return f"Pair of {value_names[pair_value]}s"
        
        else:  # HandRank.HIGH_CARD
            high_card = values[0]
            return f"{value_names[high_card]} High"
    
    def _calculate_equity_and_potential(self, player_cards: List[Card], 
                                      community_cards: List[Card]) -> Tuple[float, float]:
        """
        Calculate equity and potential using Monte Carlo simulation.
        
        Args:
            player_cards: List of player's hole cards
            community_cards: List of community cards
            
        Returns:
            Tuple of (equity, potential)
        """
        if not player_cards:
            return 0.0, 0.0
        
        # Create a deck of remaining cards
        all_cards = set()
        
        # Add all possible cards to the deck
        for suit in CardSuit:
            if suit != CardSuit.UNKNOWN:
                for value in CardValue:
                    if value != CardValue.UNKNOWN:
                        all_cards.add((value, suit))
        
        # Remove player cards and community cards from the deck
        for card in player_cards + community_cards:
            if (card.value, card.suit) in all_cards:
                all_cards.remove((card.value, card.suit))
        
        # Convert set back to list for random.sample
        remaining_deck = [Card(value, suit) for value, suit in all_cards]
        
        # Number of community cards to deal
        cards_to_deal = 5 - len(community_cards)
        
        # Simulate hands
        win_count = 0
        improvement_count = 0
        
        current_hand_rank = self.evaluate_hand(player_cards, community_cards).hand_rank
        
        # Run Monte Carlo simulation
        for _ in range(self.simulation_iterations):
            # Deal remaining community cards
            if cards_to_deal > 0:
                additional_cards = random.sample(remaining_deck, cards_to_deal)
                complete_community = community_cards + additional_cards
            else:
                complete_community = community_cards
            
            # Evaluate player's hand
            player_evaluation = self.evaluate_hand(player_cards, complete_community)
            
            # Check if hand improved
            if player_evaluation.hand_rank.value > current_hand_rank.value:
                improvement_count += 1
            
            # Simulate opponent hands (assuming one opponent for simplicity)
            # In a real implementation, you'd want to simulate multiple opponents
            # and consider the probability of winning against all of them
            opponent_wins = False
            
            # Try 5 random opponent hands
            for _ in range(5):
                # Deal cards to opponent
                opponent_cards = random.sample([card for card in remaining_deck 
                                               if card not in additional_cards], 2)
                
                # Evaluate opponent's hand
                opponent_evaluation = self.evaluate_hand(opponent_cards, complete_community)
                
                # Compare hands
                if (opponent_evaluation.hand_rank.value > player_evaluation.hand_rank.value or
                    (opponent_evaluation.hand_rank == player_evaluation.hand_rank and 
                     opponent_evaluation.hand_value > player_evaluation.hand_value)):
                    opponent_wins = True
                    break
            
            if not opponent_wins:
                win_count += 1
        
        # Calculate equity (probability of winning)
        equity = win_count / self.simulation_iterations
        
        # Calculate potential (probability of improvement)
        potential = improvement_count / self.simulation_iterations
        
        return equity, potential
    
    def get_hand_odds(self, player_cards: List[Card], 
                    community_cards: List[Card], 
                    target_rank: HandRank) -> float:
        """
        Calculate the odds of making a specific hand.
        
        Args:
            player_cards: List of player's hole cards
            community_cards: List of community cards
            target_rank: Target hand rank
            
        Returns:
            Probability of achieving the target hand rank
        """
        if not player_cards:
            return 0.0
        
        # Create a deck of remaining cards
        all_cards = set()
        
        # Add all possible cards to the deck
        for suit in CardSuit:
            if suit != CardSuit.UNKNOWN:
                for value in CardValue:
                    if value != CardValue.UNKNOWN:
                        all_cards.add((value, suit))
        
        # Remove player cards and community cards from the deck
        for card in player_cards + community_cards:
            if (card.value, card.suit) in all_cards:
                all_cards.remove((card.value, card.suit))
        
        # Convert set back to list for random.sample
        remaining_deck = [Card(value, suit) for value, suit in all_cards]
        
        # Number of community cards to deal
        cards_to_deal = 5 - len(community_cards)
        
        # Current hand evaluation
        current_evaluation = self.evaluate_hand(player_cards, community_cards)
        
        # If we already have the target hand or better, return 1.0
        if current_evaluation.hand_rank.value >= target_rank.value:
            return 1.0
        
        # Simulate hands
        success_count = 0
        
        # Run Monte Carlo simulation
        for _ in range(self.simulation_iterations):
            # Deal remaining community cards
            if cards_to_deal > 0:
                additional_cards = random.sample(remaining_deck, cards_to_deal)
                complete_community = community_cards + additional_cards
            else:
                complete_community = community_cards
            
            # Evaluate player's hand
            player_evaluation = self.evaluate_hand(player_cards, complete_community)
            
            # Check if hand reaches or exceeds target rank
            if player_evaluation.hand_rank.value >= target_rank.value:
                success_count += 1
        
        # Calculate probability
        probability = success_count / self.simulation_iterations
        
        return probability
    
    def analyze_outs(self, player_cards: List[Card], 
                    community_cards: List[Card]) -> Dict[str, int]:
        """
        Analyze and count the number of outs (cards that can improve the hand).
        
        Args:
            player_cards: List of player's hole cards
            community_cards: List of community cards
            
        Returns:
            Dictionary of out types and counts
        """
        if len(player_cards) != 2 or not community_cards:
            return {}
        
        # Current hand evaluation
        current_evaluation = self.evaluate_hand(player_cards, community_cards)
        
        # Create a deck of remaining cards
        all_cards = set()
        
        # Add all possible cards to the deck
        for suit in CardSuit:
            if suit != CardSuit.UNKNOWN:
                for value in CardValue:
                    if value != CardValue.UNKNOWN:
                        all_cards.add((value, suit))
        
        # Remove player cards and community cards from the deck
        for card in player_cards + community_cards:
            if (card.value, card.suit) in all_cards:
                all_cards.remove((card.value, card.suit))
        
        # Convert set back to list
        remaining_deck = [Card(value, suit) for value, suit in all_cards]
        
        # Initialize out counters
        outs = {
            "pair": 0,
            "two_pair": 0,
            "three_of_a_kind": 0,
            "straight": 0,
            "flush": 0,
            "full_house": 0,
            "four_of_a_kind": 0,
            "straight_flush": 0,
            "royal_flush": 0,
            "total": 0
        }
        
        # Check each potential out
        for card in remaining_deck:
            # Add the card to the community cards
            new_community = community_cards + [card]
            
            # Evaluate the new hand
            new_evaluation = self.evaluate_hand(player_cards, new_community)
            
            # If the hand improved, count it as an out
            if new_evaluation.hand_rank.value > current_evaluation.hand_rank.value:
                outs["total"] += 1
                
                # Categorize the out
                if new_evaluation.hand_rank == HandRank.ONE_PAIR:
                    outs["pair"] += 1
                elif new_evaluation.hand_rank == HandRank.TWO_PAIR:
                    outs["two_pair"] += 1
                elif new_evaluation.hand_rank == HandRank.THREE_OF_A_KIND:
                    outs["three_of_a_kind"] += 1
                elif new_evaluation.hand_rank == HandRank.STRAIGHT:
                    outs["straight"] += 1
                elif new_evaluation.hand_rank == HandRank.FLUSH:
                    outs["flush"] += 1
                elif new_evaluation.hand_rank == HandRank.FULL_HOUSE:
                    outs["full_house"] += 1
                elif new_evaluation.hand_rank == HandRank.FOUR_OF_A_KIND:
                    outs["four_of_a_kind"] += 1
                elif new_evaluation.hand_rank == HandRank.STRAIGHT_FLUSH:
                    outs["straight_flush"] += 1
                elif new_evaluation.hand_rank == HandRank.ROYAL_FLUSH:
                    outs["royal_flush"] += 1
            
            # Also count as an out if the hand is the same rank but higher value
            elif (new_evaluation.hand_rank == current_evaluation.hand_rank and 
                  new_evaluation.hand_value > current_evaluation.hand_value):
                outs["total"] += 1
                
                # Categorize the out based on current hand
                if current_evaluation.hand_rank == HandRank.HIGH_CARD:
                    outs["pair"] += 1
                elif current_evaluation.hand_rank == HandRank.ONE_PAIR:
                    outs["two_pair"] += 1
                elif current_evaluation.hand_rank == HandRank.TWO_PAIR:
                    outs["two_pair"] += 1
                elif current_evaluation.hand_rank == HandRank.THREE_OF_A_KIND:
                    outs["three_of_a_kind"] += 1
                elif current_evaluation.hand_rank == HandRank.STRAIGHT:
                    outs["straight"] += 1
                elif current_evaluation.hand_rank == HandRank.FLUSH:
                    outs["flush"] += 1
                elif current_evaluation.hand_rank == HandRank.FULL_HOUSE:
                    outs["full_house"] += 1
                elif current_evaluation.hand_rank == HandRank.FOUR_OF_A_KIND:
                    outs["four_of_a_kind"] += 1
                elif current_evaluation.hand_rank == HandRank.STRAIGHT_FLUSH:
                    outs["straight_flush"] += 1
        
        return outs
    
    def get_starting_hand_rank(self, player_cards: List[Card]) -> int:
        """
        Get a numerical rank for a starting hand (1-169).
        
        Args:
            player_cards: List of player's hole cards (must be 2 cards)
            
        Returns:
            Starting hand rank (1 = best, 169 = worst)
        """
        if len(player_cards) != 2:
            return 169  # Worst rank
        
        # Extract values and sort them (higher value first)
        values = sorted([card.value.value for card in player_cards], reverse=True)
        
        # Check if suited
        is_suited = player_cards[0].suit == player_cards[1].suit
        
        # Pre-defined rankings for all 169 starting hands
        # This is a simplified version, you might want to use a more detailed ranking
        if values[0] == values[1]:  # Pocket pair
            # Rank pairs from AA (1) to 22 (13)
            return 14 - values[0]
        
        elif is_suited:  # Suited cards
            # Calculate rank: (high_card_rank - 2) * 12 + (low_card_rank - 2) + 13
            # This gives ranks from 14 (AKs) to 89 (32s)
            rank = (14 - values[0]) * 12 + (14 - values[1]) + 13
            return min(rank, 169)
        
        else:  # Offsuit cards
            # Calculate rank: (high_card_rank - 2) * 12 + (low_card_rank - 2) + 90
            # This gives ranks from 90 (AKo) to 169 (32o)
            rank = (14 - values[0]) * 12 + (14 - values[1]) + 90
            return min(rank, 169)
    
    def get_starting_hand_group(self, player_cards: List[Card]) -> str:
        """
        Get a descriptive group for a starting hand.
        
        Args:
            player_cards: List of player's hole cards (must be 2 cards)
            
        Returns:
            Starting hand group description
        """
        if len(player_cards) != 2:
            return "Unknown"
        
        rank = self.get_starting_hand_rank(player_cards)
        
        # Group hands into categories
        if rank <= 5:       # AA, KK, QQ, JJ, AKs
            return "Premium"
        elif rank <= 20:    # Including TT, AQs, AJs, AKo, etc.
            return "Strong"
        elif rank <= 50:    # Including 99, 88, ATs, KQs, etc.
            return "Playable"
        elif rank <= 100:   # Middle suited connectors, small pairs, etc.
            return "Marginal"
        else:               # Weak hands
            return "Weak"


# Test function
def test_hand_evaluator():
    """Test the hand evaluator functionality."""
    from vision.card_detector import Card, CardValue, CardSuit
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize hand evaluator
    evaluator = HandEvaluator()
    
    # Test hands
    hands = [
        # Royal flush
        [
            Card(CardValue.ACE, CardSuit.HEARTS),
            Card(CardValue.KING, CardSuit.HEARTS),
            Card(CardValue.QUEEN, CardSuit.HEARTS),
            Card(CardValue.JACK, CardSuit.HEARTS),
            Card(CardValue.TEN, CardSuit.HEARTS)
        ],
        # Straight flush
        [
            Card(CardValue.NINE, CardSuit.CLUBS),
            Card(CardValue.EIGHT, CardSuit.CLUBS),
            Card(CardValue.SEVEN, CardSuit.CLUBS),
            Card(CardValue.SIX, CardSuit.CLUBS),
            Card(CardValue.FIVE, CardSuit.CLUBS)
        ],
        # Four of a kind
        [
            Card(CardValue.ACE, CardSuit.HEARTS),
            Card(CardValue.ACE, CardSuit.DIAMONDS),
            Card(CardValue.ACE, CardSuit.CLUBS),
            Card(CardValue.ACE, CardSuit.SPADES),
            Card(CardValue.KING, CardSuit.HEARTS)
        ],
        # Full house
        [
            Card(CardValue.KING, CardSuit.HEARTS),
            Card(CardValue.KING, CardSuit.DIAMONDS),
            Card(CardValue.KING, CardSuit.CLUBS),
            Card(CardValue.QUEEN, CardSuit.HEARTS),
            Card(CardValue.QUEEN, CardSuit.DIAMONDS)
        ],
        # Flush
        [
            Card(CardValue.ACE, CardSuit.DIAMONDS),
            Card(CardValue.JACK, CardSuit.DIAMONDS),
            Card(CardValue.NINE, CardSuit.DIAMONDS),
            Card(CardValue.SEVEN, CardSuit.DIAMONDS),
            Card(CardValue.FIVE, CardSuit.DIAMONDS)
        ],
        # Straight
        [
            Card(CardValue.NINE, CardSuit.HEARTS),
            Card(CardValue.EIGHT, CardSuit.DIAMONDS),
            Card(CardValue.SEVEN, CardSuit.CLUBS),
            Card(CardValue.SIX, CardSuit.SPADES),
            Card(CardValue.FIVE, CardSuit.HEARTS)
        ],
        # Three of a kind
        [
            Card(CardValue.QUEEN, CardSuit.HEARTS),
            Card(CardValue.QUEEN, CardSuit.DIAMONDS),
            Card(CardValue.QUEEN, CardSuit.CLUBS),
            Card(CardValue.NINE, CardSuit.HEARTS),
            Card(CardValue.SEVEN, CardSuit.DIAMONDS)
        ],
        # Two pair
        [
            Card(CardValue.JACK, CardSuit.HEARTS),
            Card(CardValue.JACK, CardSuit.DIAMONDS),
            Card(CardValue.NINE, CardSuit.CLUBS),
            Card(CardValue.NINE, CardSuit.SPADES),
            Card(CardValue.ACE, CardSuit.HEARTS)
        ],
        # One pair
        [
            Card(CardValue.TEN, CardSuit.HEARTS),
            Card(CardValue.TEN, CardSuit.DIAMONDS),
            Card(CardValue.KING, CardSuit.CLUBS),
            Card(CardValue.SEVEN, CardSuit.SPADES),
            Card(CardValue.THREE, CardSuit.HEARTS)
        ],
        # High card
        [
            Card(CardValue.ACE, CardSuit.HEARTS),
            Card(CardValue.KING, CardSuit.DIAMONDS),
            Card(CardValue.JACK, CardSuit.CLUBS),
            Card(CardValue.EIGHT, CardSuit.SPADES),
            Card(CardValue.THREE, CardSuit.HEARTS)
        ]
    ]
    
    # Evaluate each hand
    for i, hand in enumerate(hands):
        evaluation = evaluator._evaluate_five_card_hand(hand)
        print(f"Hand {i+1}: {[str(card) for card in hand]}")
        print(f"Rank: {evaluation[0].name}, Value: {evaluation[1]}")
        print(f"Description: {evaluator._get_hand_description(hand, evaluation[0])}")
        print()
    
    # Test hand equity calculation
    player_cards = [
        Card(CardValue.ACE, CardSuit.HEARTS),
        Card(CardValue.KING, CardSuit.HEARTS)
    ]
    
    community_cards = [
        Card(CardValue.QUEEN, CardSuit.HEARTS),
        Card(CardValue.JACK, CardSuit.DIAMONDS),
        Card(CardValue.TWO, CardSuit.CLUBS)
    ]
    
    evaluation = evaluator.evaluate_hand(player_cards, community_cards)
    print("\nHand evaluation with equity:")
    print(f"Player cards: {[str(card) for card in player_cards]}")
    print(f"Community cards: {[str(card) for card in community_cards]}")
    print(f"Evaluation: {evaluation}")
    print(f"Equity: {evaluation.equity:.2%}")
    print(f"Potential: {evaluation.potential:.2%}")
    
    # Test outs analysis
    outs = evaluator.analyze_outs(player_cards, community_cards)
    print("\nOuts analysis:")
    for out_type, count in outs.items():
        print(f"{out_type}: {count}")
    
    # Test starting hand ranking
    starting_hands = [
        # Premium hands
        [Card(CardValue.ACE, CardSuit.HEARTS), Card(CardValue.ACE, CardSuit.SPADES)],
        [Card(CardValue.ACE, CardSuit.HEARTS), Card(CardValue.KING, CardSuit.HEARTS)],
        
        # Strong hands
        [Card(CardValue.TEN, CardSuit.HEARTS), Card(CardValue.TEN, CardSuit.DIAMONDS)],
        [Card(CardValue.ACE, CardSuit.HEARTS), Card(CardValue.QUEEN, CardSuit.DIAMONDS)],
        
        # Playable hands
        [Card(CardValue.EIGHT, CardSuit.HEARTS), Card(CardValue.EIGHT, CardSuit.DIAMONDS)],
        [Card(CardValue.KING, CardSuit.HEARTS), Card(CardValue.QUEEN, CardSuit.HEARTS)],
        
        # Marginal hands
        [Card(CardValue.SEVEN, CardSuit.HEARTS), Card(CardValue.SIX, CardSuit.HEARTS)],
        
        # Weak hands
        [Card(CardValue.SEVEN, CardSuit.HEARTS), Card(CardValue.TWO, CardSuit.DIAMONDS)],
        [Card(CardValue.THREE, CardSuit.HEARTS), Card(CardValue.TWO, CardSuit.DIAMONDS)]
    ]
    
    print("\nStarting hand rankings:")
    for hand in starting_hands:
        rank = evaluator.get_starting_hand_rank(hand)
        group = evaluator.get_starting_hand_group(hand)
        print(f"{hand[0]} {hand[1]}: Rank {rank}, Group: {group}")


if __name__ == "__main__":
    # Run test
    test_hand_evaluator()