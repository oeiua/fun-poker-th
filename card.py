"""
Card classes for the poker game implementation.
"""
import random
from enum import Enum, auto
from typing import List, Optional

class Suit(Enum):
    """Enum for card suits."""
    HEARTS = 0
    DIAMONDS = 1
    CLUBS = 2
    SPADES = 3
    
    def __str__(self):
        return self.name.lower()

class Rank(Enum):
    """Enum for card ranks."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    
    def __str__(self):
        if self.value <= 10:
            return str(self.value)
        else:
            return self.name[0]  # First letter of face cards (J, Q, K, A)

class Card:
    """
    Represents a playing card with a rank and suit.
    """
    def __init__(self, rank: Rank, suit: Suit):
        """
        Initialize a card with a rank and suit.
        
        Args:
            rank (Rank): The card's rank (2-A)
            suit (Suit): The card's suit
        """
        self.rank = rank
        self.suit = suit
    
    def __eq__(self, other):
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    def __str__(self):
        from config import PokerConfig
        suit_symbol = PokerConfig.CARD_SYMBOLS[str(self.suit)]
        return f"{str(self.rank)}{suit_symbol}"
    
    def __repr__(self):
        return self.__str__()
    
    def to_feature_vector(self) -> List[float]:
        """
        Convert the card to a feature vector representation.
        
        Returns:
            List[float]: [rank_normalized, is_hearts, is_diamonds, is_clubs, is_spades]
        """
        # Normalize rank to [0, 1]
        rank_normalized = (self.rank.value - 2) / 12  # 2 -> 0, Ace -> 1
        
        # One-hot encode suit
        is_hearts = 1.0 if self.suit == Suit.HEARTS else 0.0
        is_diamonds = 1.0 if self.suit == Suit.DIAMONDS else 0.0
        is_clubs = 1.0 if self.suit == Suit.CLUBS else 0.0
        is_spades = 1.0 if self.suit == Suit.SPADES else 0.0
        
        return [rank_normalized, is_hearts, is_diamonds, is_clubs, is_spades]

class Deck:
    """
    Represents a deck of 52 playing cards.
    """
    def __init__(self):
        """Initialize a standard 52-card deck."""
        self.cards = []
        self.reset()
    
    def reset(self):
        """Reset and recreate a full deck of cards."""
        self.cards = []
        for suit in Suit:
            for rank in Rank:
                self.cards.append(Card(rank, suit))
    
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def deal(self) -> Optional[Card]:
        """
        Deal a card from the top of the deck.
        
        Returns:
            Card: The dealt card, or None if the deck is empty
        """
        if not self.cards:
            return None
        return self.cards.pop()
    
    def deal_multiple(self, count: int) -> List[Card]:
        """
        Deal multiple cards from the deck.
        
        Args:
            count (int): Number of cards to deal
            
        Returns:
            List[Card]: List of dealt cards
        """
        return [self.deal() for _ in range(min(count, len(self.cards)))]
    
    def __len__(self):
        return len(self.cards)
