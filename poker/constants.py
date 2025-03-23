#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Constants

This module defines constants used in poker game analysis.
"""

from enum import IntEnum, auto

class HandRank(IntEnum):
    """Enumeration of poker hand ranks."""
    HIGH_CARD = 1
    ONE_PAIR = 2
    TWO_PAIR = 3
    THREE_OF_A_KIND = 4
    STRAIGHT = 5
    FLUSH = 6
    FULL_HOUSE = 7
    FOUR_OF_A_KIND = 8
    STRAIGHT_FLUSH = 9
    ROYAL_FLUSH = 10


class Position(IntEnum):
    """Enumeration of poker table positions."""
    EARLY = 1    # UTG, UTG+1, UTG+2
    MIDDLE = 2   # MP, MP+1, MP+2
    LATE = 3     # CO, BTN
    BLINDS = 4   # SB, BB


class GameStage(IntEnum):
    """Enumeration of poker game stages."""
    PREFLOP = 1
    FLOP = 2
    TURN = 3
    RIVER = 4


class ActionType(IntEnum):
    """Enumeration of possible poker actions."""
    FOLD = 1
    CHECK = 2
    CALL = 3
    BET = 4
    RAISE = 5
    ALL_IN = 6


# Starting hand groups
PREMIUM_HANDS = [
    'AA', 'KK', 'QQ', 'JJ', 'AKs'
]

STRONG_HANDS = [
    'TT', '99', 'AQs', 'AJs', 'ATs', 'AKo', 'KQs', 'KJs', 'QJs'
]

PLAYABLE_HANDS = [
    '88', '77', '66', 'A9s', 'A8s', 'A7s', 'A6s', 'A5s', 'A4s', 'A3s', 'A2s',
    'AQo', 'AJo', 'ATo', 'A9o', 'KQo', 'KJo', 'KTo', 'QJo', 'QTo', 'JTo',
    'K9s', 'K8s', 'K7s', 'Q9s', 'Q8s', 'J9s', 'T9s', '98s', '87s', '76s', '65s', '54s'
]

MARGINAL_HANDS = [
    '55', '44', '33', '22', 'A8o', 'A7o', 'A6o', 'A5o', 'A4o', 'A3o', 'A2o',
    'K9o', 'K8o', 'K7o', 'K6o', 'K5o', 'K4o', 'K3o', 'K2o', 'Q9o', 'Q8o', 'Q7o',
    'J8o', 'J7o', 'T8o', 'T7o', '97o', '96o', '86o', '75o', '64o', '53o', '43o',
    'J8s', 'T8s', '97s', '86s', '75s', '64s', '53s', '43s', '32s'
]

# All other hands are considered weak


# Pot odds and implied odds thresholds
MIN_POT_ODDS_TO_CALL = 0.25  # Minimum pot odds to call
MIN_IMPLIED_ODDS_TO_CALL = 0.2  # Minimum implied odds to call
MAX_RISK_PERCENTAGE = 0.15  # Maximum percentage of stack to risk without strong hand


# Position-based aggression factors
POSITION_AGGRESSION = {
    Position.EARLY: 0.7,    # Less aggressive in early position
    Position.MIDDLE: 0.85,  # Moderately aggressive in middle position
    Position.LATE: 1.2,     # More aggressive in late position
    Position.BLINDS: 0.8    # Moderately aggressive in blinds
}


# Bet sizing guidelines
class BetSizing:
    """Class with constants for bet sizing."""
    # Standard bet sizes as percentage of pot
    STANDARD_BET = 0.75       # 75% of pot
    STANDARD_RAISE = 2.5      # 2.5x the previous bet
    
    # Preflop raise sizes
    PREFLOP_OPEN_RAISE = 3.0  # 3x the big blind
    PREFLOP_3BET = 3.0        # 3x the previous raise
    PREFLOP_4BET = 2.5        # 2.5x the 3-bet
    
    # Continuation bet sizes
    CBET_DRY_BOARD = 0.5      # 50% of pot on dry boards
    CBET_WET_BOARD = 0.75     # 75% of pot on wet boards
    
    # Turn and river bet sizes
    TURN_BARREL = 0.75        # 75% of pot on turn
    RIVER_VALUE_BET = 0.6     # 60% of pot for value on river
    RIVER_BLUFF = 0.8         # 80% of pot for bluffs on river


# Player type profiles
class PlayerType:
    """Player type profiles with adjustment factors."""
    TIGHT_PASSIVE = {
        'hand_range_adjustment': 0.7,   # Plays fewer hands
        'aggression_adjustment': 0.6,   # Less aggressive
        'bluff_frequency': 0.2,         # Bluffs less often
        'fold_to_3bet': 0.7,            # Folds to 3-bets more often
        'fold_to_cbet': 0.6,            # Folds to c-bets more often
        'value_bet_threshold': 0.8      # Only value bets strong hands
    }
    
    TIGHT_AGGRESSIVE = {
        'hand_range_adjustment': 0.75,  # Plays fewer hands
        'aggression_adjustment': 1.2,   # More aggressive
        'bluff_frequency': 0.4,         # Bluffs sometimes
        'fold_to_3bet': 0.5,            # Folds to 3-bets sometimes
        'fold_to_cbet': 0.4,            # Folds to c-bets sometimes
        'value_bet_threshold': 0.7      # Value bets strong to medium hands
    }
    
    LOOSE_PASSIVE = {
        'hand_range_adjustment': 1.3,   # Plays more hands
        'aggression_adjustment': 0.6,   # Less aggressive
        'bluff_frequency': 0.3,         # Bluffs sometimes
        'fold_to_3bet': 0.4,            # Rarely folds to 3-bets
        'fold_to_cbet': 0.3,            # Rarely folds to c-bets
        'value_bet_threshold': 0.6      # Value bets weaker hands
    }
    
    LOOSE_AGGRESSIVE = {
        'hand_range_adjustment': 1.5,   # Plays many hands
        'aggression_adjustment': 1.5,   # Very aggressive
        'bluff_frequency': 0.6,         # Bluffs often
        'fold_to_3bet': 0.3,            # Rarely folds to 3-bets
        'fold_to_cbet': 0.25,           # Rarely folds to c-bets
        'value_bet_threshold': 0.5      # Value bets wide range
    }
    
    MANIAC = {
        'hand_range_adjustment': 2.0,   # Plays almost any hand
        'aggression_adjustment': 2.0,   # Extremely aggressive
        'bluff_frequency': 0.8,         # Bluffs very often
        'fold_to_3bet': 0.2,            # Almost never folds to 3-bets
        'fold_to_cbet': 0.2,            # Almost never folds to c-bets
        'value_bet_threshold': 0.3      # Value bets very wide range
    }
    
    ROCK = {
        'hand_range_adjustment': 0.5,   # Plays very few hands
        'aggression_adjustment': 0.5,   # Very passive
        'bluff_frequency': 0.1,         # Almost never bluffs
        'fold_to_3bet': 0.8,            # Almost always folds to 3-bets
        'fold_to_cbet': 0.7,            # Almost always folds to c-bets
        'value_bet_threshold': 0.9      # Only value bets very strong hands
    }
    
    CALLING_STATION = {
        'hand_range_adjustment': 1.3,   # Plays many hands
        'aggression_adjustment': 0.4,   # Very passive
        'bluff_frequency': 0.1,         # Almost never bluffs
        'fold_to_3bet': 0.2,            # Almost never folds to 3-bets
        'fold_to_cbet': 0.1,            # Almost never folds to c-bets
        'value_bet_threshold': 0.6      # Value bets weaker hands
    }


# Board texture classifications
class BoardTexture:
    """Board texture classifications with adjustment factors."""
    # Wetness (draw potential)
    VERY_DRY = 0.2    # Few draws possible
    DRY = 0.4         # Some draws possible
    NEUTRAL = 0.6     # Average number of draws
    WET = 0.8         # Many draws possible
    VERY_WET = 0.95   # Almost all draws possible
    
    # Connectedness
    DISCONNECTED = 0.1     # No connected cards
    SLIGHTLY_CONNECTED = 0.3  # One pair of connected cards
    CONNECTED = 0.6        # Multiple connected cards
    HIGHLY_CONNECTED = 0.9  # All cards connected
    
    # Paired
    UNPAIRED = 0.0      # No paired cards
    PAIRED = 0.5        # One pair on board
    TRIPS = 0.9         # Three of a kind on board
    
    # Suitedness
    RAINBOW = 0.1       # Three different suits
    TWO_SUIT = 0.5      # Two cards of the same suit
    MONOTONE = 0.9      # All cards of the same suit


# Stack size classifications
class StackSize:
    """Stack size classifications."""
    SHORT = 25      # 25 big blinds or less
    MEDIUM = 50     # 26-50 big blinds
    DEEP = 100      # 51-100 big blinds
    VERY_DEEP = 200  # Over 100 big blinds