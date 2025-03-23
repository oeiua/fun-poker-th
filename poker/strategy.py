#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Strategy Engine

This module provides decision-making algorithms for poker strategy
based on game state, hand evaluation, and poker fundamentals.
"""

import logging
import random
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass

from .constants import HandRank, Position, GameStage, ActionType
from .constants import PREMIUM_HANDS, STRONG_HANDS, PLAYABLE_HANDS, MARGINAL_HANDS
from .constants import BetSizing, PlayerType, BoardTexture, StackSize
from .constants import POSITION_AGGRESSION, MIN_POT_ODDS_TO_CALL, MIN_IMPLIED_ODDS_TO_CALL
from .hand_evaluator import HandEvaluator, HandEvaluation
from vision.card_detector import Card
from vision.game_state_extractor import GameState, PlayerState, PlayerAction

logger = logging.getLogger("PokerVision.StrategyEngine")


@dataclass
class DecisionContext:
    """Class representing the context for a poker decision."""
    game_state: GameState
    hand_evaluation: HandEvaluation
    position: Position
    game_stage: GameStage
    pot_odds: float = 0.0
    implied_odds: float = 0.0
    fold_equity: float = 0.0
    required_equity: float = 0.0
    aggressive_factor: float = 1.0
    board_texture: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.board_texture is None:
            self.board_texture = {
                'wetness': 0.5,
                'connectedness': 0.5,
                'paired': 0.0,
                'suitedness': 0.0
            }


@dataclass
class Decision:
    """Class representing a poker decision."""
    action: ActionType
    amount: float = 0.0
    confidence: float = 0.0
    reasoning: str = ""
    alternative_actions: List[Tuple[ActionType, float]] = None
    
    def __post_init__(self):
        """Initialize derived fields."""
        if self.alternative_actions is None:
            self.alternative_actions = []
    
    def __str__(self) -> str:
        """String representation of the decision."""
        action_str = {
            ActionType.FOLD: "Fold",
            ActionType.CHECK: "Check",
            ActionType.CALL: "Call",
            ActionType.BET: "Bet",
            ActionType.RAISE: "Raise",
            ActionType.ALL_IN: "All-In"
        }.get(self.action, "Unknown")
        
        if self.action in [ActionType.BET, ActionType.RAISE, ActionType.CALL, ActionType.ALL_IN]:
            return f"{action_str} ${self.amount:.2f} ({self.confidence:.0%} confidence)"
        else:
            return f"{action_str} ({self.confidence:.0%} confidence)"


class StrategyEngine:
    """Class for making poker decisions."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the StrategyEngine.
        
        Args:
            config: Configuration dictionary
        """
        logger.info("Initializing StrategyEngine")
        
        # Set default config if not provided
        if config is None:
            config = {}
        
        self.config = config
        
        # Initialize hand evaluator
        self.hand_evaluator = HandEvaluator()
        
        # Set aggression level (0.0 to 2.0, where 1.0 is balanced)
        self.aggression = config.get('aggression', 1.0)
        
        # Set bluff frequency (0.0 to 1.0)
        self.bluff_frequency = config.get('bluff_frequency', 0.3)
        
        # Set value betting threshold (0.0 to 1.0)
        self.value_bet_threshold = config.get('value_bet_threshold', 0.6)
        
        # Set continuation bet frequency (0.0 to 1.0)
        self.cbet_frequency = config.get('cbet_frequency', 0.7)
    
    def get_decision(self, game_state: GameState, hand_evaluation: HandEvaluation) -> Decision:
        """
        Get a decision for the current game state.
        
        Args:
            game_state: Current game state
            hand_evaluation: Evaluation of the player's hand
            
        Returns:
            Decision object with the recommended action
        """
        logger.info("Making decision based on game state and hand evaluation")
        
        # Create decision context
        context = self._create_decision_context(game_state, hand_evaluation)
        
        # Get appropriate decision based on game stage
        if context.game_stage == GameStage.PREFLOP:
            return self._preflop_decision(context)
        elif context.game_stage == GameStage.FLOP:
            return self._flop_decision(context)
        elif context.game_stage == GameStage.TURN:
            return self._turn_decision(context)
        else:  # RIVER
            return self._river_decision(context)
    
    def _create_decision_context(self, game_state: GameState, 
                               hand_evaluation: HandEvaluation) -> DecisionContext:
        """
        Create a decision context from the game state and hand evaluation.
        
        Args:
            game_state: Current game state
            hand_evaluation: Evaluation of the player's hand
            
        Returns:
            DecisionContext object
        """
        # Determine game stage
        game_stage = self._determine_game_stage(game_state.community_cards)
        
        # Determine position
        position = self._determine_position(game_state)
        
        # Calculate pot odds
        pot_odds = self._calculate_pot_odds(game_state)
        
        # Calculate implied odds
        implied_odds = self._calculate_implied_odds(game_state, hand_evaluation)
        
        # Calculate fold equity
        fold_equity = self._calculate_fold_equity(game_state)
        
        # Calculate required equity to call
        required_equity = self._calculate_required_equity(game_state)
        
        # Calculate aggressive factor
        aggressive_factor = self._calculate_aggressive_factor(position, game_state)
        
        # Analyze board texture
        board_texture = self._analyze_board_texture(game_state.community_cards)
        
        # Create and return context
        context = DecisionContext(
            game_state=game_state,
            hand_evaluation=hand_evaluation,
            position=position,
            game_stage=game_stage,
            pot_odds=pot_odds,
            implied_odds=implied_odds,
            fold_equity=fold_equity,
            required_equity=required_equity,
            aggressive_factor=aggressive_factor,
            board_texture=board_texture
        )
        
        return context
    
    def _determine_game_stage(self, community_cards: List[Card]) -> GameStage:
        """
        Determine the current game stage.
        
        Args:
            community_cards: List of community cards
            
        Returns:
            GameStage enum value
        """
        num_cards = len(community_cards)
        
        if num_cards == 0:
            return GameStage.PREFLOP
        elif num_cards == 3:
            return GameStage.FLOP
        elif num_cards == 4:
            return GameStage.TURN
        elif num_cards == 5:
            return GameStage.RIVER
        else:
            # Shouldn't happen, but default to preflop
            logger.warning(f"Unexpected number of community cards: {num_cards}")
            return GameStage.PREFLOP
    
    def _determine_position(self, game_state: GameState) -> Position:
        """
        Determine the player's position.
        
        Args:
            game_state: Current game state
            
        Returns:
            Position enum value
        """
        # Get the number of players
        num_players = len(game_state.players)
        
        if num_players <= 2:
            # Heads up
            if game_state.dealer_position == 0:  # Player is dealer/button
                return Position.LATE
            else:
                return Position.BLINDS
        
        # Find the player and dealer indices
        player_index = 0  # Assuming player is always at index 0
        dealer_index = game_state.dealer_position
        
        # Calculate relative position
        if dealer_index == -1:
            # Dealer position unknown, assume middle position
            return Position.MIDDLE
        
        # Calculate positions relative to the dealer
        positions_from_dealer = (player_index - dealer_index) % num_players
        
        # Categorize position
        if positions_from_dealer <= 2:  # UTG, UTG+1, UTG+2
            return Position.EARLY
        elif positions_from_dealer <= 5:  # MP, MP+1, MP+2
            return Position.MIDDLE
        elif positions_from_dealer <= 7:  # CO, BTN
            return Position.LATE
        else:  # SB, BB
            return Position.BLINDS
    
    def _calculate_pot_odds(self, game_state: GameState) -> float:
        """
        Calculate the pot odds.
        
        Args:
            game_state: Current game state
            
        Returns:
            Pot odds as a float (0.0 to 1.0)
        """
        # Get the current player
        player = game_state.get_current_player()
        if player is None:
            # Assume it's our turn
            player = game_state.get_player(0)
        
        if player is None:
            return 0.0
        
        # Find the amount to call
        call_amount = 0.0
        for p in game_state.players:
            if p.is_active and p.bet > player.bet:
                call_amount = max(call_amount, p.bet - player.bet)
        
        # Calculate pot odds
        total_pot = game_state.pot + sum(p.bet for p in game_state.players if p.is_active)
        
        if call_amount <= 0:
            return 0.0  # No need to call
        
        pot_odds = call_amount / (total_pot + call_amount)
        
        return pot_odds
    
    def _calculate_implied_odds(self, game_state: GameState, 
                              hand_evaluation: HandEvaluation) -> float:
        """
        Calculate the implied odds.
        
        Args:
            game_state: Current game state
            hand_evaluation: Evaluation of the player's hand
            
        Returns:
            Implied odds as a float (0.0 to 1.0)
        """
        # Base implied odds on hand potential
        implied_odds = hand_evaluation.potential
        
        # Adjust based on stack sizes
        avg_stack = sum(p.stack for p in game_state.players if p.is_active) / len(game_state.players)
        
        # More chips behind = better implied odds
        if avg_stack > StackSize.DEEP:
            implied_odds *= 1.2
        elif avg_stack > StackSize.MEDIUM:
            implied_odds *= 1.1
        elif avg_stack < StackSize.SHORT:
            implied_odds *= 0.8
        
        # Adjust based on number of opponents
        num_opponents = len([p for p in game_state.players if p.is_active and p.player_id != 0])
        
        # More opponents = better implied odds
        implied_odds *= min(1.5, 1.0 + (num_opponents * 0.1))
        
        return min(implied_odds, 1.0)
    
    def _calculate_fold_equity(self, game_state: GameState) -> float:
        """
        Calculate the fold equity.
        
        Args:
            game_state: Current game state
            
        Returns:
            Fold equity as a float (0.0 to 1.0)
        """
        # Count number of active opponents
        num_opponents = len([p for p in game_state.players if p.is_active and p.player_id != 0])
        
        if num_opponents == 0:
            return 0.0
        
        # Assume a default fold equity of 0.3 (30% chance opponents will fold)
        fold_equity = 0.3
        
        # Adjust based on opponent actions
        fold_prone_opponents = 0
        for player in game_state.players:
            if player.is_active and player.player_id != 0:
                if player.last_action in [PlayerAction.CHECK, PlayerAction.CALL]:
                    # Players who call or check are less likely to fold
                    fold_prone_opponents += 0.5
                elif player.last_action == PlayerAction.FOLD:
                    # Players who have already folded don't count
                    num_opponents -= 1
                else:
                    # Other players (especially limpers or unknowns) might fold
                    fold_prone_opponents += 1
        
        if num_opponents == 0:
            return 0.0
        
        # Calculate fold equity
        fold_equity = fold_prone_opponents / num_opponents
        
        return min(fold_equity, 1.0)
    
    def _calculate_required_equity(self, game_state: GameState) -> float:
        """
        Calculate the required equity to call.
        
        Args:
            game_state: Current game state
            
        Returns:
            Required equity as a float (0.0 to 1.0)
        """
        # Start with pot odds
        pot_odds = self._calculate_pot_odds(game_state)
        
        if pot_odds <= 0:
            return 0.0  # No call needed
        
        # Required equity is the pot odds
        required_equity = pot_odds
        
        # Adjust based on game state
        player = game_state.get_current_player()
        if player is None:
            player = game_state.get_player(0)
        
        if player is None:
            return required_equity
        
        # Adjust based on stack size
        if player.stack < StackSize.SHORT:
            # Short stack requires higher equity
            required_equity *= 1.1
        elif player.stack > StackSize.DEEP:
            # Deep stack can take more chances
            required_equity *= 0.9
        
        return min(required_equity, 1.0)
    
    def _calculate_aggressive_factor(self, position: Position, game_state: GameState) -> float:
        """
        Calculate the aggressive factor based on position and game state.
        
        Args:
            position: Player's position
            game_state: Current game state
            
        Returns:
            Aggressive factor as a float
        """
        # Base aggression on position
        aggressive_factor = POSITION_AGGRESSION.get(position, 1.0)
        
        # Adjust based on global aggression setting
        aggressive_factor *= self.aggression
        
        # Adjust based on stack size
        player = game_state.get_player(0)
        if player is not None:
            if player.stack < StackSize.SHORT:
                # Short stack should be more selective
                aggressive_factor *= 0.8
            elif player.stack > StackSize.DEEP:
                # Deep stack can be more aggressive
                aggressive_factor *= 1.2
        
        # Adjust based on number of opponents
        num_opponents = len([p for p in game_state.players if p.is_active and p.player_id != 0])
        
        # More opponents = less aggression
        if num_opponents > 3:
            aggressive_factor *= 0.8
        elif num_opponents <= 1:
            aggressive_factor *= 1.2
        
        return aggressive_factor
    
    def _analyze_board_texture(self, community_cards: List[Card]) -> Dict[str, float]:
        """
        Analyze the board texture.
        
        Args:
            community_cards: List of community cards
            
        Returns:
            Dictionary with board texture metrics
        """
        if not community_cards:
            return {
                'wetness': 0.0,
                'connectedness': 0.0,
                'paired': 0.0,
                'suitedness': 0.0
            }
        
        # Extract values and suits
        values = [card.value.value for card in community_cards]
        suits = [card.suit for card in community_cards]
        
        # Analyze wetness (draw potential)
        # Check for straight draws
        values_sorted = sorted(values)
        straight_draw_potential = 0.0
        if len(values_sorted) >= 3:
            # Check for open-ended straight draws
            for i in range(len(values_sorted) - 2):
                if values_sorted[i+2] - values_sorted[i] <= 4:
                    straight_draw_potential = max(straight_draw_potential, 0.7)
                elif values_sorted[i+2] - values_sorted[i] <= 5:
                    straight_draw_potential = max(straight_draw_potential, 0.5)
        
        # Check for flush draws
        suit_counts = {}
        for suit in suits:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        flush_draw_potential = 0.0
        for count in suit_counts.values():
            if count >= 3:
                flush_draw_potential = 0.7
            elif count == 2:
                flush_draw_potential = 0.3
        
        wetness = max(straight_draw_potential, flush_draw_potential)
        
        # Analyze connectedness
        connectedness = 0.0
        if len(values_sorted) >= 2:
            gaps = [values_sorted[i+1] - values_sorted[i] for i in range(len(values_sorted) - 1)]
            avg_gap = sum(gaps) / len(gaps)
            connectedness = 1.0 - min(1.0, avg_gap / 4.0)
        
        # Analyze paired
        paired = 0.0
        value_counts = {}
        for value in values:
            value_counts[value] = value_counts.get(value, 0) + 1
        
        max_count = max(value_counts.values()) if value_counts else 0
        if max_count == 2:
            paired = 0.5
        elif max_count >= 3:
            paired = 0.9
        
        # Analyze suitedness
        suitedness = 0.0
        if suit_counts:
            max_suit_count = max(suit_counts.values())
            if max_suit_count == len(community_cards):
                suitedness = 0.9  # Monotone
            elif max_suit_count >= 2:
                suitedness = 0.5  # Two-tone
            else:
                suitedness = 0.1  # Rainbow
        
        return {
            'wetness': wetness,
            'connectedness': connectedness,
            'paired': paired,
            'suitedness': suitedness
        }
    
    def _preflop_decision(self, context: DecisionContext) -> Decision:
        """
        Make a preflop decision.
        
        Args:
            context: Decision context
            
        Returns:
            Decision object
        """
        logger.debug("Making preflop decision")
        
        # Get player cards
        player = context.game_state.get_player(0)
        if player is None or not player.cards or len(player.cards) != 2:
            # Not enough information for a decision
            return Decision(
                action=ActionType.CHECK,
                confidence=0.3,
                reasoning="Not enough information for a solid preflop decision"
            )
        
        # Get starting hand rank and group
        hand_rank = self.hand_evaluator.get_starting_hand_rank(player.cards)
        hand_group = self.hand_evaluator.get_starting_hand_group(player.cards)
        
        # Get pot odds and required equity
        pot_odds = context.pot_odds
        required_equity = context.required_equity
        
        # Check if anyone has bet
        active_bet = False
        max_bet = 0.0
        for p in context.game_state.players:
            if p.is_active and p.bet > 0:
                active_bet = True
                max_bet = max(max_bet, p.bet)
        
        # If no one has bet, we can open
        if not active_bet:
            # Decision based on hand strength and position
            if hand_group == "Premium":
                # Always raise with premium hands
                raise_amount = context.game_state.big_blind * BetSizing.PREFLOP_OPEN_RAISE
                return Decision(
                    action=ActionType.RAISE,
                    amount=raise_amount,
                    confidence=0.9,
                    reasoning=f"Premium hand ({hand_rank}/169) in {context.position.name} position - standard open raise"
                )
            
            elif hand_group == "Strong":
                # Raise with strong hands
                raise_amount = context.game_state.big_blind * BetSizing.PREFLOP_OPEN_RAISE
                confidence = 0.8
                
                if context.position in [Position.EARLY]:
                    confidence = 0.7  # Slightly less confident in early position
                
                return Decision(
                    action=ActionType.RAISE,
                    amount=raise_amount,
                    confidence=confidence,
                    reasoning=f"Strong hand ({hand_rank}/169) in {context.position.name} position - standard open raise"
                )
            
            elif hand_group == "Playable":
                # Raise with playable hands in good position
                if context.position in [Position.LATE, Position.BLINDS]:
                    raise_amount = context.game_state.big_blind * BetSizing.PREFLOP_OPEN_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.7,
                        reasoning=f"Playable hand ({hand_rank}/169) in {context.position.name} position - position-based open raise"
                    )
                elif context.position == Position.MIDDLE and random.random() < context.aggressive_factor:
                    raise_amount = context.game_state.big_blind * BetSizing.PREFLOP_OPEN_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.6,
                        reasoning=f"Playable hand ({hand_rank}/169) in {context.position.name} position - mixing it up with a raise"
                    )
                else:
                    return Decision(
                        action=ActionType.CHECK,
                        confidence=0.6,
                        reasoning=f"Playable hand ({hand_rank}/169) in {context.position.name} position - checking to see the flop"
                    )
            
            elif hand_group == "Marginal":
                # Only play marginal hands in late position
                if context.position in [Position.LATE]:
                    if random.random() < context.aggressive_factor * 0.7:
                        raise_amount = context.game_state.big_blind * BetSizing.PREFLOP_OPEN_RAISE
                        return Decision(
                            action=ActionType.RAISE,
                            amount=raise_amount,
                            confidence=0.5,
                            reasoning=f"Marginal hand ({hand_rank}/169) in {context.position.name} position - steal attempt"
                        )
                    else:
                        return Decision(
                            action=ActionType.CHECK,
                            confidence=0.5,
                            reasoning=f"Marginal hand ({hand_rank}/169) in {context.position.name} position - checking to see the flop"
                        )
                elif context.position == Position.BLINDS:
                    return Decision(
                        action=ActionType.CHECK,
                        confidence=0.6,
                        reasoning=f"Marginal hand ({hand_rank}/169) in {context.position.name} position - checking option"
                    )
                else:
                    # Fold in early/middle position
                    return Decision(
                        action=ActionType.FOLD,
                        confidence=0.7,
                        reasoning=f"Marginal hand ({hand_rank}/169) in {context.position.name} position - folding to avoid trouble"
                    )
            
            else:  # Weak hand
                # Only consider playing weak hands in late position
                if context.position == Position.LATE and random.random() < context.aggressive_factor * self.bluff_frequency:
                    raise_amount = context.game_state.big_blind * BetSizing.PREFLOP_OPEN_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.4,
                        reasoning=f"Weak hand ({hand_rank}/169) in {context.position.name} position - steal attempt"
                    )
                elif context.position == Position.BLINDS:
                    return Decision(
                        action=ActionType.CHECK,
                        confidence=0.7,
                        reasoning=f"Weak hand ({hand_rank}/169) in {context.position.name} position - checking option"
                    )
                else:
                    # Fold in all other positions
                    return Decision(
                        action=ActionType.FOLD,
                        confidence=0.8,
                        reasoning=f"Weak hand ({hand_rank}/169) in {context.position.name} position - easy fold"
                    )
        
        else:
            # Someone has bet, we need to decide whether to call, raise, or fold
            
            # Estimate our equity
            equity = 0.0
            if hand_group == "Premium":
                equity = 0.7
            elif hand_group == "Strong":
                equity = 0.6
            elif hand_group == "Playable":
                equity = 0.5
            elif hand_group == "Marginal":
                equity = 0.4
            else:  # Weak
                equity = 0.3
            
            # Factor in position
            if context.position in [Position.LATE, Position.BLINDS]:
                equity += 0.05
            elif context.position == Position.EARLY:
                equity -= 0.05
            
            # Check if we have enough equity to call
            if equity >= required_equity:
                # We have enough equity, now decide between calling and raising
                
                if hand_group in ["Premium", "Strong"]:
                    # Usually 3-bet with premium hands
                    raise_amount = max_bet * BetSizing.PREFLOP_3BET
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.8,
                        reasoning=f"Strong hand ({hand_rank}/169) with good equity ({equity:.1%}) - 3-betting for value"
                    )
                
                elif hand_group == "Playable" and random.random() < context.aggressive_factor * 0.6:
                    # Sometimes 3-bet with playable hands
                    raise_amount = max_bet * BetSizing.PREFLOP_3BET
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.6,
                        reasoning=f"Playable hand ({hand_rank}/169) with decent equity ({equity:.1%}) - semi-bluff 3-bet"
                    )
                
                elif (hand_group == "Marginal" and context.position in [Position.LATE] and 
                     random.random() < context.aggressive_factor * self.bluff_frequency):
                    # Occasionally 3-bet bluff with marginal hands in position
                    raise_amount = max_bet * BetSizing.PREFLOP_3BET
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.4,
                        reasoning=f"Marginal hand ({hand_rank}/169) in position - bluff 3-bet"
                    )
                
                else:
                    # Call with the remaining hands that have enough equity
                    return Decision(
                        action=ActionType.CALL,
                        amount=max_bet - player.bet,
                        confidence=0.6,
                        reasoning=f"{hand_group} hand ({hand_rank}/169) with sufficient equity ({equity:.1%}) vs. required ({required_equity:.1%}) - calling"
                    )
            
            else:
                # Not enough equity, but we might still bluff
                if (context.fold_equity > 0.5 and 
                    context.position in [Position.LATE] and 
                    random.random() < context.aggressive_factor * self.bluff_frequency):
                    # Bluff with high fold equity
                    raise_amount = max_bet * BetSizing.PREFLOP_3BET
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.3,
                        reasoning=f"Weak hand but good fold equity ({context.fold_equity:.1%}) - bluff 3-bet"
                    )
                else:
                    # Fold with insufficient equity
                    return Decision(
                        action=ActionType.FOLD,
                        confidence=0.7,
                        reasoning=f"Insufficient equity ({equity:.1%}) vs. required ({required_equity:.1%}) - folding"
                    )
    
    def _flop_decision(self, context: DecisionContext) -> Decision:
        """
        Make a flop decision.
        
        Args:
            context: Decision context
            
        Returns:
            Decision object
        """
        logger.debug("Making flop decision")
        
        # Get player
        player = context.game_state.get_player(0)
        if player is None:
            return Decision(
                action=ActionType.CHECK,
                confidence=0.3,
                reasoning="Not enough information for a solid flop decision"
            )
        
        # Get hand evaluation
        hand_eval = context.hand_evaluation
        
        # Get pot odds and required equity
        pot_odds = context.pot_odds
        required_equity = context.required_equity
        
        # Get board texture
        board_texture = context.board_texture
        
        # Check if anyone has bet
        active_bet = False
        max_bet = 0.0
        for p in context.game_state.players:
            if p.is_active and p.bet > 0:
                active_bet = True
                max_bet = max(max_bet, p.bet)
        
        # If no one has bet, we can bet first
        if not active_bet:
            # Calculate pot size
            pot_size = context.game_state.pot
            
            # Decision based on hand strength and board texture
            if hand_eval.hand_rank.value >= HandRank.TWO_PAIR.value:
                # Strong made hand, bet for value
                bet_amount = pot_size * BetSizing.STANDARD_BET
                return Decision(
                    action=ActionType.BET,
                    amount=bet_amount,
                    confidence=0.8,
                    reasoning=f"Strong made hand ({hand_eval.description}) - value betting"
                )
            
            elif hand_eval.hand_rank.value >= HandRank.ONE_PAIR.value:
                # One pair, bet depending on kicker and board texture
                if board_texture['wetness'] < 0.5 and board_texture['paired'] < 0.5:
                    # Dry board, more likely our pair is good
                    bet_amount = pot_size * BetSizing.CBET_DRY_BOARD
                    return Decision(
                        action=ActionType.BET,
                        amount=bet_amount,
                        confidence=0.7,
                        reasoning=f"One pair ({hand_eval.description}) on dry board - value betting"
                    )
                else:
                    # Wet board, more cautious
                    if random.random() < self.cbet_frequency * context.aggressive_factor:
                        bet_amount = pot_size * BetSizing.CBET_WET_BOARD
                        return Decision(
                            action=ActionType.BET,
                            amount=bet_amount,
                            confidence=0.6,
                            reasoning=f"One pair ({hand_eval.description}) on wet board - thin value bet / protection"
                        )
                    else:
                        return Decision(
                            action=ActionType.CHECK,
                            confidence=0.6,
                            reasoning=f"One pair ({hand_eval.description}) on wet board - checking for pot control"
                        )
            
            elif hand_eval.equity > 0.5:
                # Drawing hand with good equity
                bet_amount = pot_size * BetSizing.CBET_WET_BOARD
                return Decision(
                    action=ActionType.BET,
                    amount=bet_amount,
                    confidence=0.6,
                    reasoning=f"Drawing hand with good equity ({hand_eval.equity:.1%}) - semi-bluff betting"
                )
            
            elif hand_eval.potential > 0.3:
                # Drawing hand with decent potential
                if random.random() < self.cbet_frequency * context.aggressive_factor:
                    bet_amount = pot_size * BetSizing.CBET_WET_BOARD
                    return Decision(
                        action=ActionType.BET,
                        amount=bet_amount,
                        confidence=0.5,
                        reasoning=f"Drawing hand with potential ({hand_eval.potential:.1%}) - semi-bluff betting"
                    )
                else:
                    return Decision(
                        action=ActionType.CHECK,
                        confidence=0.6,
                        reasoning=f"Drawing hand with potential ({hand_eval.potential:.1%}) - checking to realize equity"
                    )
            
            else:
                # Weak hand
                if (context.position in [Position.LATE] and 
                    random.random() < self.bluff_frequency * context.aggressive_factor):
                    # Sometimes bluff in position
                    bet_amount = pot_size * BetSizing.CBET_DRY_BOARD
                    return Decision(
                        action=ActionType.BET,
                        amount=bet_amount,
                        confidence=0.4,
                        reasoning=f"Weak hand in position - bluffing"
                    )
                else:
                    # Otherwise check
                    return Decision(
                        action=ActionType.CHECK,
                        confidence=0.7,
                        reasoning=f"Weak hand ({hand_eval.description}) - checking"
                    )
        
        else:
            # Someone has bet, decide whether to call, raise, or fold
            
            # Check if we have enough equity to call
            if hand_eval.equity >= required_equity:
                # We have enough equity, now decide between calling and raising
                
                if hand_eval.hand_rank.value >= HandRank.TWO_PAIR.value:
                    # Strong made hand, raise for value
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.8,
                        reasoning=f"Strong made hand ({hand_eval.description}) with good equity ({hand_eval.equity:.1%}) - raising for value"
                    )
                
                elif (hand_eval.hand_rank == HandRank.ONE_PAIR and 
                     board_texture['wetness'] < 0.5 and 
                     hand_eval.equity > 0.6):
                    # Strong one pair on dry board
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.7,
                        reasoning=f"Strong one pair ({hand_eval.description}) on dry board with good equity ({hand_eval.equity:.1%}) - raising for value"
                    )
                
                elif hand_eval.equity > 0.6 and hand_eval.potential > 0.5:
                    # Strong drawing hand, semi-bluff raise
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.6,
                        reasoning=f"Strong drawing hand with good equity ({hand_eval.equity:.1%}) and potential ({hand_eval.potential:.1%}) - semi-bluff raising"
                    )
                
                else:
                    # Call with the remaining hands that have enough equity
                    return Decision(
                        action=ActionType.CALL,
                        amount=max_bet - player.bet,
                        confidence=0.7,
                        reasoning=f"{hand_eval.description} with sufficient equity ({hand_eval.equity:.1%}) vs. required ({required_equity:.1%}) - calling"
                    )
            
            elif (hand_eval.potential > 0.5 and 
                 context.implied_odds > MIN_IMPLIED_ODDS_TO_CALL):
                # Drawing hand with good potential and implied odds
                return Decision(
                    action=ActionType.CALL,
                    amount=max_bet - player.bet,
                    confidence=0.6,
                    reasoning=f"Drawing hand with good potential ({hand_eval.potential:.1%}) and implied odds ({context.implied_odds:.1%}) - calling"
                )
            
            else:
                # Not enough equity, but we might still bluff raise
                if (context.fold_equity > 0.6 and 
                    context.position in [Position.LATE] and 
                    random.random() < self.bluff_frequency * context.aggressive_factor):
                    # Bluff raise with high fold equity
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.4,
                        reasoning=f"Weak hand but good fold equity ({context.fold_equity:.1%}) - bluff raising"
                    )
                else:
                    # Fold with insufficient equity
                    return Decision(
                        action=ActionType.FOLD,
                        confidence=0.7,
                        reasoning=f"Insufficient equity ({hand_eval.equity:.1%}) vs. required ({required_equity:.1%}) - folding"
                    )
    
    def _turn_decision(self, context: DecisionContext) -> Decision:
        """
        Make a turn decision.
        
        Args:
            context: Decision context
            
        Returns:
            Decision object
        """
        logger.debug("Making turn decision")
        
        # Similar structure to flop decision, but with adjusted values
        # Get player
        player = context.game_state.get_player(0)
        if player is None:
            return Decision(
                action=ActionType.CHECK,
                confidence=0.3,
                reasoning="Not enough information for a solid turn decision"
            )
        
        # Get hand evaluation
        hand_eval = context.hand_evaluation
        
        # Get pot odds and required equity
        pot_odds = context.pot_odds
        required_equity = context.required_equity
        
        # Check if anyone has bet
        active_bet = False
        max_bet = 0.0
        for p in context.game_state.players:
            if p.is_active and p.bet > 0:
                active_bet = True
                max_bet = max(max_bet, p.bet)
        
        # If no one has bet, we can bet first
        if not active_bet:
            # Calculate pot size
            pot_size = context.game_state.pot
            
            # Decision based on hand strength and board texture
            if hand_eval.hand_rank.value >= HandRank.TWO_PAIR.value:
                # Strong made hand, bet for value
                bet_amount = pot_size * BetSizing.TURN_BARREL
                return Decision(
                    action=ActionType.BET,
                    amount=bet_amount,
                    confidence=0.8,
                    reasoning=f"Strong made hand ({hand_eval.description}) - value betting"
                )
            
            elif (hand_eval.hand_rank == HandRank.ONE_PAIR and 
                 hand_eval.equity > 0.6):
                # Strong one pair
                bet_amount = pot_size * BetSizing.TURN_BARREL
                return Decision(
                    action=ActionType.BET,
                    amount=bet_amount,
                    confidence=0.7,
                    reasoning=f"Strong one pair ({hand_eval.description}) with good equity ({hand_eval.equity:.1%}) - value betting"
                )
            
            elif hand_eval.equity > 0.5 and hand_eval.potential > 0.3:
                # Strong drawing hand
                bet_amount = pot_size * BetSizing.TURN_BARREL
                return Decision(
                    action=ActionType.BET,
                    amount=bet_amount,
                    confidence=0.6,
                    reasoning=f"Strong drawing hand with good equity ({hand_eval.equity:.1%}) - semi-bluff betting"
                )
            
            else:
                # Weak hand, usually check
                if (context.position in [Position.LATE] and 
                    random.random() < self.bluff_frequency * context.aggressive_factor * 0.8):
                    # Occasionally bluff in position
                    bet_amount = pot_size * BetSizing.TURN_BARREL
                    return Decision(
                        action=ActionType.BET,
                        amount=bet_amount,
                        confidence=0.4,
                        reasoning=f"Weak hand in position - bluffing"
                    )
                else:
                    # Otherwise check
                    return Decision(
                        action=ActionType.CHECK,
                        confidence=0.7,
                        reasoning=f"Weak hand ({hand_eval.description}) - checking"
                    )
        
        else:
            # Someone has bet, decide whether to call, raise, or fold
            
            # Check if we have enough equity to call
            if hand_eval.equity >= required_equity:
                # We have enough equity, now decide between calling and raising
                
                if hand_eval.hand_rank.value >= HandRank.TWO_PAIR.value:
                    # Strong made hand, raise for value
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.8,
                        reasoning=f"Strong made hand ({hand_eval.description}) with good equity ({hand_eval.equity:.1%}) - raising for value"
                    )
                
                elif (hand_eval.hand_rank == HandRank.ONE_PAIR and 
                     hand_eval.equity > 0.65):
                    # Very strong one pair
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.6,
                        reasoning=f"Strong one pair ({hand_eval.description}) with very good equity ({hand_eval.equity:.1%}) - raising for value"
                    )
                
                elif hand_eval.equity > 0.7 and hand_eval.potential > 0.5:
                    # Very strong drawing hand
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.6,
                        reasoning=f"Very strong drawing hand with excellent equity ({hand_eval.equity:.1%}) - semi-bluff raising"
                    )
                
                else:
                    # Call with the remaining hands that have enough equity
                    return Decision(
                        action=ActionType.CALL,
                        amount=max_bet - player.bet,
                        confidence=0.7,
                        reasoning=f"{hand_eval.description} with sufficient equity ({hand_eval.equity:.1%}) vs. required ({required_equity:.1%}) - calling"
                    )
            
            elif (hand_eval.potential > 0.6 and 
                 context.implied_odds > MIN_IMPLIED_ODDS_TO_CALL * 1.2):
                # Very strong draw with good implied odds
                return Decision(
                    action=ActionType.CALL,
                    amount=max_bet - player.bet,
                    confidence=0.6,
                    reasoning=f"Strong drawing hand with excellent potential ({hand_eval.potential:.1%}) and implied odds ({context.implied_odds:.1%}) - calling"
                )
            
            else:
                # Not enough equity, but we might still bluff raise
                if (context.fold_equity > 0.7 and 
                    context.position in [Position.LATE] and 
                    random.random() < self.bluff_frequency * context.aggressive_factor * 0.7):
                    # Bluff raise with very high fold equity
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.4,
                        reasoning=f"Weak hand but excellent fold equity ({context.fold_equity:.1%}) - bluff raising"
                    )
                else:
                    # Fold with insufficient equity
                    return Decision(
                        action=ActionType.FOLD,
                        confidence=0.8,
                        reasoning=f"Insufficient equity ({hand_eval.equity:.1%}) vs. required ({required_equity:.1%}) - folding"
                    )
    
    def _river_decision(self, context: DecisionContext) -> Decision:
        """
        Make a river decision.
        
        Args:
            context: Decision context
            
        Returns:
            Decision object
        """
        logger.debug("Making river decision")
        
        # Get player
        player = context.game_state.get_player(0)
        if player is None:
            return Decision(
                action=ActionType.CHECK,
                confidence=0.3,
                reasoning="Not enough information for a solid river decision"
            )
        
        # Get hand evaluation
        hand_eval = context.hand_evaluation
        
        # Get pot odds and required equity
        pot_odds = context.pot_odds
        required_equity = context.required_equity
        
        # Check if anyone has bet
        active_bet = False
        max_bet = 0.0
        for p in context.game_state.players:
            if p.is_active and p.bet > 0:
                active_bet = True
                max_bet = max(max_bet, p.bet)
        
        # If no one has bet, we can bet first
        if not active_bet:
            # Calculate pot size
            pot_size = context.game_state.pot
            
            # On the river, we either value bet or bluff
            if hand_eval.equity > self.value_bet_threshold:
                # Strong enough for a value bet
                bet_amount = pot_size * BetSizing.RIVER_VALUE_BET
                return Decision(
                    action=ActionType.BET,
                    amount=bet_amount,
                    confidence=0.8,
                    reasoning=f"Strong hand ({hand_eval.description}) with good equity ({hand_eval.equity:.1%}) - value betting"
                )
            
            else:
                # Not strong enough for a value bet, consider bluffing
                if (context.position in [Position.LATE] and 
                    random.random() < self.bluff_frequency * context.aggressive_factor * 0.6):
                    # Bluff in position sometimes
                    bet_amount = pot_size * BetSizing.RIVER_BLUFF
                    return Decision(
                        action=ActionType.BET,
                        amount=bet_amount,
                        confidence=0.4,
                        reasoning=f"Weak hand ({hand_eval.description}) in position - bluffing"
                    )
                else:
                    # Otherwise check
                    return Decision(
                        action=ActionType.CHECK,
                        confidence=0.7,
                        reasoning=f"Weak hand ({hand_eval.description}) - checking"
                    )
        
        else:
            # Someone has bet, decide whether to call, raise, or fold
            
            # On the river, all draws are complete, so our equity is our showdown value
            if hand_eval.equity >= required_equity:
                # We have enough equity to call
                
                if hand_eval.equity > 0.8:
                    # Very strong hand, raise for value
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.8,
                        reasoning=f"Very strong hand ({hand_eval.description}) with excellent equity ({hand_eval.equity:.1%}) - raising for value"
                    )
                
                else:
                    # Call with hands that have enough equity but aren't strong enough to raise
                    return Decision(
                        action=ActionType.CALL,
                        amount=max_bet - player.bet,
                        confidence=0.7,
                        reasoning=f"{hand_eval.description} with sufficient equity ({hand_eval.equity:.1%}) vs. required ({required_equity:.1%}) - calling"
                    )
            
            else:
                # Not enough equity, but we might still bluff raise
                if (context.fold_equity > 0.8 and 
                    context.position in [Position.LATE] and 
                    random.random() < self.bluff_frequency * context.aggressive_factor * 0.5):
                    # Bluff raise with extremely high fold equity
                    raise_amount = max_bet * BetSizing.STANDARD_RAISE
                    return Decision(
                        action=ActionType.RAISE,
                        amount=raise_amount,
                        confidence=0.3,
                        reasoning=f"Bluff raising with excellent fold equity ({context.fold_equity:.1%})"
                    )
                else:
                    # Fold with insufficient equity
                    return Decision(
                        action=ActionType.FOLD,
                        confidence=0.8,
                        reasoning=f"Insufficient equity ({hand_eval.equity:.1%}) vs. required ({required_equity:.1%}) - folding"
                    )


# Test function
def test_strategy_engine():
    """Test the strategy engine functionality."""
    from vision.card_detector import Card, CardValue, CardSuit
    from vision.game_state_extractor import GameState, PlayerState, PlayerAction
    
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Initialize strategy engine
    strategy = StrategyEngine()
    
    # Initialize hand evaluator
    hand_evaluator = HandEvaluator()
    
    # Create a sample game state
    game_state = GameState()
    game_state.pot = 10.0
    game_state.big_blind = 1.0
    game_state.small_blind = 0.5
    game_state.dealer_position = 5
    
    # Community cards
    game_state.community_cards = [
        Card(CardValue.ACE, CardSuit.HEARTS),
        Card(CardValue.KING, CardSuit.HEARTS),
        Card(CardValue.SEVEN, CardSuit.DIAMONDS)
    ]
    
    # Players
    player1 = PlayerState(player_id=0, name="Hero", stack=100.0, is_active=True, is_current=True)
    player1.cards = [
        Card(CardValue.QUEEN, CardSuit.HEARTS),
        Card(CardValue.JACK, CardSuit.HEARTS)
    ]
    
    player2 = PlayerState(player_id=1, name="Villain1", stack=120.0, bet=2.0, is_active=True)
    player3 = PlayerState(player_id=2, name="Villain2", stack=80.0, is_active=True)
    player4 = PlayerState(player_id=3, name="Villain3", stack=150.0, is_active=True)
    player5 = PlayerState(player_id=4, name="Villain4", stack=200.0, is_active=False)
    player6 = PlayerState(player_id=5, name="Villain5", stack=50.0, is_active=True, is_dealer=True)
    
    game_state.players = [player1, player2, player3, player4, player5, player6]
    game_state.current_player_id = 0
    
    # Evaluate the hand
    hand_evaluation = hand_evaluator.evaluate_hand(player1.cards, game_state.community_cards)
    
    # Get a decision
    decision = strategy.get_decision(game_state, hand_evaluation)
    
    # Print the results
    print("Game State:")
    print(f"  Pot: ${game_state.pot}")
    print(f"  Community Cards: {[str(card) for card in game_state.community_cards]}")
    print(f"  Player Cards: {[str(card) for card in player1.cards]}")
    print(f"  Current Player: {game_state.current_player_id}")
    print()
    
    print("Hand Evaluation:")
    print(f"  Hand: {hand_evaluation.description}")
    print(f"  Equity: {hand_evaluation.equity:.2%}")
    print(f"  Potential: {hand_evaluation.potential:.2%}")
    print()
    
    print("Decision:")
    print(f"  Action: {decision.action.name}")
    if decision.amount > 0:
        print(f"  Amount: ${decision.amount:.2f}")
    print(f"  Confidence: {decision.confidence:.2%}")
    print(f"  Reasoning: {decision.reasoning}")
    print()
    
    # Test different scenarios
    print("Testing different scenarios:")
    
    # Scenario 1: Premium preflop hand
    player1.cards = [
        Card(CardValue.ACE, CardSuit.SPADES),
        Card(CardValue.ACE, CardSuit.HEARTS)
    ]
    game_state.community_cards = []
    
    hand_evaluation = hand_evaluator.evaluate_hand(player1.cards, game_state.community_cards)
    decision = strategy.get_decision(game_state, hand_evaluation)
    
    print("Scenario 1: Premium preflop hand (AA)")
    print(f"  Decision: {decision}")
    print(f"  Reasoning: {decision.reasoning}")
    print()
    
    # Scenario 2: Marginal hand facing a bet
    player1.cards = [
        Card(CardValue.EIGHT, CardSuit.CLUBS),
        Card(CardValue.SEVEN, CardSuit.HEARTS)
    ]
    player2.bet = 5.0
    
    hand_evaluation = hand_evaluator.evaluate_hand(player1.cards, game_state.community_cards)
    decision = strategy.get_decision(game_state, hand_evaluation)
    
    print("Scenario 2: Marginal preflop hand (87o) facing a bet")
    print(f"  Decision: {decision}")
    print(f"  Reasoning: {decision.reasoning}")
    print()
    
    # Scenario 3: Strong draw on flop
    player1.cards = [
        Card(CardValue.QUEEN, CardSuit.HEARTS),
        Card(CardValue.JACK, CardSuit.HEARTS)
    ]
    game_state.community_cards = [
        Card(CardValue.ACE, CardSuit.HEARTS),
        Card(CardValue.KING, CardSuit.DIAMONDS),
        Card(CardValue.TEN, CardSuit.CLUBS)
    ]
    player2.bet = 0.0
    
    hand_evaluation = hand_evaluator.evaluate_hand(player1.cards, game_state.community_cards)
    decision = strategy.get_decision(game_state, hand_evaluation)
    
    print("Scenario 3: Strong draw on flop (OESD + flush draw)")
    print(f"  Decision: {decision}")
    print(f"  Reasoning: {decision.reasoning}")
    print()


if __name__ == "__main__":
    # Run test
    test_strategy_engine()