"""
Human player interface for poker AI.
"""

import logging
from typing import Dict, List, Tuple, Any

from src.agent.base_agent import BaseAgent
from src.environment.state import GameState


class HumanAgent(BaseAgent):
    """
    Human player interface for poker games.
    """
    
    def __init__(self, player_id: int):
        """
        Initialize the human agent.
        
        Args:
            player_id: Unique identifier for this agent
        """
        super().__init__(player_id)
    
    def get_action(self, state: GameState, legal_actions: Dict[str, List[int]]) -> Tuple[str, int]:
        """
        Get the human player's action through console input.
        
        Args:
            state: Current game state
            legal_actions: Dictionary of legal actions with their parameters
            
        Returns:
            Tuple of (action_type, amount)
        """
        self._display_game_state(state)
        self._display_legal_actions(legal_actions)
        
        while True:
            try:
                action = input("Enter your action (format: 'action_type amount'): ").strip()
                
                # Parse input
                if action.lower() == 'fold':
                    if 'fold' in legal_actions:
                        return 'fold', 0
                    else:
                        print("Folding is not a legal action right now.")
                        continue
                        
                if action.lower() in ['check', 'call']:
                    if 'check_call' in legal_actions:
                        return 'check_call', legal_actions['check_call'][0]
                    else:
                        print("Checking/calling is not a legal action right now.")
                        continue
                
                # Handle bet/raise
                parts = action.split()
                if len(parts) == 2 and parts[0].lower() in ['bet', 'raise']:
                    if 'bet_raise' not in legal_actions:
                        print("Betting/raising is not a legal action right now.")
                        continue
                    
                    try:
                        amount = int(parts[1])
                        
                        # Check if amount is legal
                        min_amount = min(legal_actions['bet_raise'])
                        max_amount = max(legal_actions['bet_raise'])
                        
                        if amount < min_amount:
                            print(f"Minimum bet/raise is {min_amount}.")
                            continue
                        
                        if amount > max_amount:
                            print(f"Maximum bet/raise is {max_amount}.")
                            continue
                        
                        return 'bet_raise', amount
                        
                    except ValueError:
                        print("Invalid amount. Please enter a number.")
                        continue
                
                print("Invalid action. Try again.")
                
            except KeyboardInterrupt:
                print("\nAuto-folding due to interrupt.")
                if 'fold' in legal_actions:
                    return 'fold', 0
                else:
                    return 'check_call', legal_actions.get('check_call', [0])[0]
    
    def _display_game_state(self, state: GameState) -> None:
        """
        Display the current game state to the console.
        
        Args:
            state: Current game state
        """
        print("\n" + "="*50)
        print("YOUR TURN")
        print("="*50)
        
        # Display hole cards
        card_indices = [i for i, v in enumerate(state.hand_cards) if v == 1]
        hole_cards = self._indices_to_cards(card_indices)
        print(f"Your hole cards: {' '.join(hole_cards)}")
        
        # Display board cards
        board_indices = [i for i, v in enumerate(state.board_cards) if v == 1]
        board = self._indices_to_cards(board_indices)
        print(f"Board: {' '.join(board) if board else 'No cards yet'}")
        
        # Display pot and position
        print(f"Pot: {state.pot}")
        print(f"Position: {state.player_index}")
        
        # Display player stacks and bets
        print("\nPlayer information:")
        for i in range(len(state.player_stacks)):
            active = "Active" if state.active_players[i] else "Folded"
            you = " (YOU)" if i == state.player_index else ""
            print(f"Player {i}{you}: Stack: {state.player_stacks[i]}, Bet: {state.player_bets[i]}, {active}")
        
        print("="*50)
    
    def _display_legal_actions(self, legal_actions: Dict[str, List[int]]) -> None:
        """
        Display the legal actions to the console.
        
        Args:
            legal_actions: Dictionary of legal actions with their parameters
        """
        print("\nLegal actions:")
        
        if 'fold' in legal_actions:
            print("- fold")
        
        if 'check_call' in legal_actions:
            amount = legal_actions['check_call'][0]
            action = "check" if amount == 0 else f"call {amount}"
            print(f"- {action}")
        
        if 'bet_raise' in legal_actions:
            min_amount = min(legal_actions['bet_raise'])
            max_amount = max(legal_actions['bet_raise'])
            print(f"- bet/raise [amount between {min_amount} and {max_amount}]")
            
            # Suggest some standard bet sizes
            suggestions = [
                min_amount,
                min(max_amount, min_amount * 2),  # 2x min bet
                min(max_amount, min_amount * 3),  # 3x min bet
                max_amount  # all-in
            ]
            
            print(f"  Suggested amounts: {', '.join(map(str, suggestions))}")
        
        print("="*50)
    
    @staticmethod
    def _indices_to_cards(indices: List[int]) -> List[str]:
        """
        Convert card indices to human-readable card representations.
        
        Args:
            indices: List of card indices (0-51)
            
        Returns:
            List of card strings (e.g., ["As", "Kd", "Qh"])
        """
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        suits = ['c', 'd', 'h', 's']
        
        cards = []
        for idx in indices:
            rank_idx = idx // 4
            suit_idx = idx % 4
            cards.append(f"{ranks[rank_idx]}{suits[suit_idx]}")
        
        return cards