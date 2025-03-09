"""
Poker hand evaluation functionality.
"""
from typing import List, Tuple, Dict, Any
from collections import Counter

from card import Card, Rank, Suit

class HandRank:
    """Enumeration of poker hand ranks."""
    HIGH_CARD = 0
    ONE_PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9

class HandEvaluator:
    """
    Class for evaluating poker hands.
    """
    @staticmethod
    def evaluate_hand(hole_cards: List[Card], community_cards: List[Card]) -> Tuple[int, List[Card]]:
        """
        Evaluate the best 5-card poker hand from the given hole and community cards.
        
        Args:
            hole_cards (List[Card]): The player's two hole cards
            community_cards (List[Card]): The community cards on the table
            
        Returns:
            Tuple[int, List[Card]]: A tuple containing the hand rank value and the 5 cards making the best hand
        """
        all_cards = hole_cards + community_cards
        best_hand_rank = -1
        best_hand_cards = []
        
        # Generate all possible 5-card combinations
        for i1 in range(len(all_cards)):
            for i2 in range(i1 + 1, len(all_cards)):
                for i3 in range(i2 + 1, len(all_cards)):
                    for i4 in range(i3 + 1, len(all_cards)):
                        for i5 in range(i4 + 1, len(all_cards)):
                            hand = [
                                all_cards[i1],
                                all_cards[i2],
                                all_cards[i3],
                                all_cards[i4],
                                all_cards[i5]
                            ]
                            
                            hand_rank, hand_value = HandEvaluator._get_hand_rank_and_value(hand)
                            
                            # Replace best hand if this hand is better
                            if hand_rank > best_hand_rank or (hand_rank == best_hand_rank and hand_value > best_hand_rank):
                                best_hand_rank = hand_rank
                                best_hand_cards = hand
        
        return best_hand_rank, best_hand_cards
    
    @staticmethod
    def _get_hand_rank_and_value(cards: List[Card]) -> Tuple[int, int]:
        """
        Get the rank and value of a 5-card poker hand.
        
        Args:
            cards (List[Card]): The 5 cards to evaluate
            
        Returns:
            Tuple[int, int]: A tuple containing the hand rank and a value for comparing hands of the same rank
        """
        if len(cards) != 5:
            raise ValueError("Hand must contain exactly 5 cards")
        
        # Check for flush
        is_flush = all(card.suit == cards[0].suit for card in cards)
        
        # Check for straight
        ranks = sorted([card.rank.value for card in cards])
        # Handle Ace-low straight (A-2-3-4-5)
        if set(ranks) == {2, 3, 4, 5, 14}:
            is_straight = True
            # Adjust Ace value to 1 for comparison
            ranks = [1, 2, 3, 4, 5]
        else:
            is_straight = (ranks == list(range(min(ranks), max(ranks) + 1)))
        
        # Get rank counts for pairs, trips, etc.
        rank_counts = Counter([card.rank.value for card in cards])
        count_values = sorted(rank_counts.values(), reverse=True)
        
        # Determine hand rank
        if is_straight and is_flush:
            if max(ranks) == 14:  # Ace high
                return HandRank.ROYAL_FLUSH, 0
            else:
                return HandRank.STRAIGHT_FLUSH, max(ranks)
        
        if count_values == [4, 1]:
            # Get the rank of the four of a kind
            four_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            # Get the rank of the kicker
            kicker = [rank for rank, count in rank_counts.items() if count == 1][0]
            return HandRank.FOUR_OF_A_KIND, four_rank * 100 + kicker
        
        if count_values == [3, 2]:
            # Get the rank of the three of a kind and the pair
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            return HandRank.FULL_HOUSE, three_rank * 100 + pair_rank
        
        if is_flush:
            return HandRank.FLUSH, sum(rank * (100 ** i) for i, rank in enumerate(sorted(ranks, reverse=True)))
        
        if is_straight:
            return HandRank.STRAIGHT, max(ranks)
        
        if count_values == [3, 1, 1]:
            # Get the rank of the three of a kind
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            # Get the ranks of the kickers
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            return HandRank.THREE_OF_A_KIND, three_rank * 10000 + kickers[0] * 100 + kickers[1]
        
        if count_values == [2, 2, 1]:
            # Get the ranks of the two pairs
            pairs = sorted([rank for rank, count in rank_counts.items() if count == 2], reverse=True)
            # Get the rank of the kicker
            kicker = [rank for rank, count in rank_counts.items() if count == 1][0]
            return HandRank.TWO_PAIR, pairs[0] * 10000 + pairs[1] * 100 + kicker
        
        if count_values == [2, 1, 1, 1]:
            # Get the rank of the pair
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            # Get the ranks of the kickers
            kickers = sorted([rank for rank, count in rank_counts.items() if count == 1], reverse=True)
            return HandRank.ONE_PAIR, pair_rank * 1000000 + kickers[0] * 10000 + kickers[1] * 100 + kickers[2]
        
        # High card
        return HandRank.HIGH_CARD, sum(rank * (100 ** i) for i, rank in enumerate(sorted(ranks, reverse=True)))
    
    @staticmethod
    def get_hand_description(hand_rank: int, cards: List[Card]) -> str:
        """
        Get a human-readable description of the poker hand.
        
        Args:
            hand_rank (int): The rank of the hand
            cards (List[Card]): The 5 cards in the hand
            
        Returns:
            str: A description of the hand
        """
        rank_names = {
            HandRank.ROYAL_FLUSH: "Royal Flush",
            HandRank.STRAIGHT_FLUSH: "Straight Flush",
            HandRank.FOUR_OF_A_KIND: "Four of a Kind",
            HandRank.FULL_HOUSE: "Full House",
            HandRank.FLUSH: "Flush",
            HandRank.STRAIGHT: "Straight",
            HandRank.THREE_OF_A_KIND: "Three of a Kind",
            HandRank.TWO_PAIR: "Two Pair",
            HandRank.ONE_PAIR: "One Pair",
            HandRank.HIGH_CARD: "High Card"
        }
        
        # Get basic description
        description = rank_names.get(hand_rank, "Unknown Hand")
        
        # Add details based on hand type
        if hand_rank == HandRank.FOUR_OF_A_KIND:
            # Find the rank of the four of a kind
            rank_counts = Counter([card.rank for card in cards])
            four_rank = [rank for rank, count in rank_counts.items() if count == 4][0]
            description += f" - {four_rank.name}s"
        
        elif hand_rank == HandRank.FULL_HOUSE:
            # Find the ranks of the three of a kind and the pair
            rank_counts = Counter([card.rank for card in cards])
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            description += f" - {three_rank.name}s full of {pair_rank.name}s"
        
        elif hand_rank == HandRank.FLUSH:
            highest_card = max(cards, key=lambda card: card.rank.value)
            description += f" - {highest_card.rank.name} high"
        
        elif hand_rank == HandRank.STRAIGHT:
            highest_card = max(cards, key=lambda card: card.rank.value)
            description += f" - {highest_card.rank.name} high"
        
        elif hand_rank == HandRank.THREE_OF_A_KIND:
            # Find the rank of the three of a kind
            rank_counts = Counter([card.rank for card in cards])
            three_rank = [rank for rank, count in rank_counts.items() if count == 3][0]
            description += f" - {three_rank.name}s"
        
        elif hand_rank == HandRank.TWO_PAIR:
            # Find the ranks of the two pairs
            rank_counts = Counter([card.rank for card in cards])
            pairs = [rank for rank, count in rank_counts.items() if count == 2]
            pairs.sort(key=lambda r: r.value, reverse=True)
            description += f" - {pairs[0].name}s and {pairs[1].name}s"
        
        elif hand_rank == HandRank.ONE_PAIR:
            # Find the rank of the pair
            rank_counts = Counter([card.rank for card in cards])
            pair_rank = [rank for rank, count in rank_counts.items() if count == 2][0]
            description += f" - {pair_rank.name}s"
        
        elif hand_rank == HandRank.HIGH_CARD:
            highest_card = max(cards, key=lambda card: card.rank.value)
            description += f" - {highest_card.rank.name} high"
        
        return description
    
    @staticmethod
    def compare_hands(player_hands: Dict[Any, Tuple[int, List[Card]]]) -> List[Any]:
        """
        Compare multiple player hands to determine the winner(s).
        
        Args:
            player_hands: Dictionary mapping players to their best hand (rank, cards)
            
        Returns:
            List[Any]: List of winners (can be multiple in case of a tie)
        """
        if not player_hands:
            return []
        
        # Find the highest hand rank
        max_rank = max(rank for rank, _ in player_hands.values())
        
        # Filter players with the highest hand rank
        players_with_max_rank = {
            player: (rank, cards) 
            for player, (rank, cards) in player_hands.items() 
            if rank == max_rank
        }
        
        # If there's only one player with the highest rank, they're the winner
        if len(players_with_max_rank) == 1:
            return list(players_with_max_rank.keys())
        
        # Otherwise, we need to compare their hand values
        # Re-evaluate each hand to get its value for comparison
        player_values = {}
        for player, (rank, cards) in players_with_max_rank.items():
            _, value = HandEvaluator._get_hand_rank_and_value(cards)
            player_values[player] = value
        
        # Find the highest value
        max_value = max(player_values.values())
        
        # Return all players with the highest value (ties are possible)
        return [player for player, value in player_values.items() if value == max_value]
