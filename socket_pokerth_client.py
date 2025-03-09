#!/usr/bin/env python3
"""
Socket-Based PokerTH Client

A direct socket implementation of the PokerTH protocol, designed to work with your AI model.
This avoids the compatibility issues with the pokerthproto library.
"""
import os
import sys
import socket
import struct
import time
import logging
import argparse
import getpass
import random
from typing import Dict, List, Tuple, Any, Optional, Union
from enum import IntEnum

# Import your AI components
try:
    from config import PokerConfig, Action, GamePhase
    from model import PokerModel
    from player import AIPlayer
    from card import Card, Rank, Suit
    from utils import calculate_hand_strength
except ImportError:
    print("Error: Could not import AI components.")
    print("Make sure your AI model files are in the Python path.")
    sys.exit(1)

logging.getLogger().setLevel(logging.DEBUG)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pokerth_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PokerTH.SocketClient")

class MessageType(IntEnum):
    """PokerTH protocol message types."""
    MSG_AUTH_CLIENT_REQUEST = 1
    MSG_AUTH_SERVER_CHALLENGE = 2
    MSG_AUTH_CLIENT_RESPONSE = 3
    MSG_AUTH_SERVER_VERIFICATION = 4
    MSG_LOBBY = 5
    MSG_GAME = 6
    MSG_ERROR = 8  # Error message
    MSG_INIT_DONE = 13
    MSG_GAME_LIST = 15
    MSG_JOIN_GAME_REQUEST = 16
    MSG_JOIN_GAME_REPLY = 17
    MSG_CREATE_GAME_REQUEST = 18
    MSG_CREATE_GAME_REPLY = 19
class GameMessageType(IntEnum):
    """Game message types."""
    MANAGEMENT = 1
    ENGINE = 2
    PLAYER = 3

class GameManagementMessageType(IntEnum):
    """Game management message types."""
    GAME_MANAGEMENT_PLAYER_JOINED = 5
    GAME_MANAGEMENT_PLAYER_LEFT = 6
    GAME_MANAGEMENT_GAME_STARTED = 9

class GameEngineMessageType(IntEnum):
    """Game engine message types."""
    GAME_ENGINE_PLAYER_ID_LIST = 1
    GAME_ENGINE_NEW_GAME = 2
    GAME_ENGINE_NEW_HAND_CARDS = 4
    GAME_ENGINE_DEAL_FLOP_CARDS = 13
    GAME_ENGINE_DEAL_TURN_CARD = 14
    GAME_ENGINE_DEAL_RIVER_CARD = 15
    GAME_ENGINE_NEXT_PLAYER_TO_ACT = 7
    GAME_ENGINE_HAND_FINISHED = 8

class GamePlayerMessageType(IntEnum):
    """Game player message types."""
    GAME_PLAYER_ACTION = 3

class PlayerActionType(IntEnum):
    """Player action types."""
    FOLD = 0
    CHECK = 1
    CALL = 2
    BET = 3
    RAISE = 4
    ALL_IN = 5

class AuthServerVerificationResult(IntEnum):
    """Authentication verification results."""
    SUCCESS = 0
    FAILED = 1

class AuthServerVerificationError(IntEnum):
    """Authentication verification error reasons."""
    INVALID_PASSWORD = 1
    INVALID_USERNAME = 2
    INVALID_VERSION = 3
    ALREADY_LOGGED_IN = 4

class JoinGameResult(IntEnum):
    """Join game result codes."""
    SUCCESS = 0
    INVALID_GAME_ID = 1
    GAME_FULL = 2
    ALREADY_STARTED = 3
    INVALID_PASSWORD = 4

class GameType(IntEnum):
    """Game types."""
    NORMAL = 0
    REGISTERED_ONLY = 1
    INVITE_ONLY = 2
    RANKING = 3

class SocketPokerTHClient:
    """
    Socket-based client for connecting to PokerTH server and playing with a trained model.
    """
    # PokerTH packet header format: 4 bytes message size, 1 byte message type
    HEADER_FORMAT = "!IB"
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
    
    def __init__(
        self, 
        server: str, 
        port: int, 
        username: str, 
        password: str = "", 
        model_path: str = None,
        game_name: str = None,
        use_guest: bool = False,
        protocol_version: int = 5
    ):
        """
        Initialize the PokerTH client.
        
        Args:
            server (str): Server address
            port (int): Server port
            username (str): Username for authentication
            password (str): Password for authentication
            model_path (str): Path to the AI model file
            game_name (str): Name of the game to join/create (optional)
            use_guest (bool): Whether to use guest login instead of credentials
            protocol_version (int): PokerTH protocol version to use
        """
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.model_path = model_path
        self.game_name = game_name
        self.is_guest = use_guest
        self.protocol_version = protocol_version
        
        # Client state
        self.sock = None
        self.buffer = bytearray()
        self.ai_model = None
        self.connected = False
        self.authenticated = False
        self.player_id = None
        self.session_id = None
        
        # Game state
        self.game_id = None
        self.joined_game = False
        self.my_seat = None
        self.hand_cards = []
        self.community_cards = []
        self.pot_size = 0
        self.current_bet = 0
        self.my_chips = 1000  # Default starting chips
        self.phase = GamePhase.PREFLOP
        self.dealer_position = 0
        self.active_players = 0
        self.player_states = {}
        self.my_turn = False
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    def _handle_game_management(self, data: bytes):
        """
        Handle game management message.
        
        Args:
            data (bytes): Message data
        """
        if len(data) < 1:
            return
        
        # Management message type (1 byte)
        mgmt_type = data[0]
        
        if mgmt_type == GameManagementMessageType.GAME_MANAGEMENT_PLAYER_JOINED:
            self._handle_player_joined(data[1:])
        
        elif mgmt_type == GameManagementMessageType.GAME_MANAGEMENT_PLAYER_LEFT:
            self._handle_player_left(data[1:])
        
        elif mgmt_type == GameManagementMessageType.GAME_MANAGEMENT_GAME_STARTED:
            self._handle_game_started(data[1:])
        
        else:
            logger.debug(f"Unhandled game management message type: {mgmt_type}")

    def _handle_player_joined(self, data: bytes):
        """
        Handle player joined message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Player ID (4 bytes)
            player_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Player name (length-prefixed string)
            name_len = struct.unpack("!H", data[index:index+2])[0]
            index += 2
            player_name = data[index:index+name_len].decode('utf-8')
            index += name_len
            
            # Update player count
            self.active_players += 1
            
            logger.info(f"Player joined: {player_name} (ID: {player_id})")
            
            # Add player to our state
            self.player_states[player_id] = {
                'name': player_name,
                'active': True,
                'chips': 0,
                'bet': 0
            }
        
        except Exception as e:
            logger.error(f"Error handling player joined: {e}")

    def _handle_player_left(self, data: bytes):
        """
        Handle player left message.
        
        Args:
            data (bytes): Message data
        """
        try:
            # Player ID (4 bytes)
            player_id = struct.unpack("!I", data[0:4])[0]
            
            # Update player count
            self.active_players -= 1
            
            # Get player name if available
            player_name = self.player_states.get(player_id, {}).get('name', f"Player {player_id}")
            
            logger.info(f"Player left: {player_name} (ID: {player_id})")
            
            # Remove player from our state
            if player_id in self.player_states:
                del self.player_states[player_id]
        
        except Exception as e:
            logger.error(f"Error handling player left: {e}")

    def _handle_game_started(self, data: bytes):
        """
        Handle game started message.
        
        Args:
            data (bytes): Message data
        """
        logger.info("Game has started")
        
        # Reset game state
        self.hand_cards = []
        self.community_cards = []
        self.pot_size = 0
        self.current_bet = 0
        self.phase = GamePhase.PREFLOP

    def _handle_game_engine(self, data: bytes):
        """
        Handle game engine message.
        
        Args:
            data (bytes): Message data
        """
        if len(data) < 1:
            return
        
        # Engine message type (1 byte)
        engine_type = data[0]
        
        if engine_type == GameEngineMessageType.GAME_ENGINE_PLAYER_ID_LIST:
            self._handle_player_id_list(data[1:])
        
        elif engine_type == GameEngineMessageType.GAME_ENGINE_NEW_GAME:
            self._handle_new_game(data[1:])
        
        elif engine_type == GameEngineMessageType.GAME_ENGINE_NEW_HAND_CARDS:
            self._handle_new_hand_cards(data[1:])
        
        elif engine_type == GameEngineMessageType.GAME_ENGINE_DEAL_FLOP_CARDS:
            self._handle_flop_cards(data[1:])
        
        elif engine_type == GameEngineMessageType.GAME_ENGINE_DEAL_TURN_CARD:
            self._handle_turn_card(data[1:])
        
        elif engine_type == GameEngineMessageType.GAME_ENGINE_DEAL_RIVER_CARD:
            self._handle_river_card(data[1:])
        
        elif engine_type == GameEngineMessageType.GAME_ENGINE_NEXT_PLAYER_TO_ACT:
            self._handle_next_player_to_act(data[1:])
        
        elif engine_type == GameEngineMessageType.GAME_ENGINE_HAND_FINISHED:
            self._handle_hand_finished(data[1:])
        
        else:
            logger.debug(f"Unhandled game engine message type: {engine_type}")

    def _handle_player_id_list(self, data: bytes):
        """
        Handle player ID list message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Game ID (4 bytes)
            game_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Number of players (4 bytes)
            num_players = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            logger.info(f"Player ID list for game {game_id}: {num_players} players")
            
            # Parse player IDs
            for _ in range(num_players):
                # Player ID (4 bytes)
                player_id = struct.unpack("!I", data[index:index+4])[0]
                index += 4
                
                # Check if this is us
                if player_id == self.player_id:
                    logger.info("Found ourselves in the player list")
            
            # Update active player count
            self.active_players = num_players
        
        except Exception as e:
            logger.error(f"Error handling player ID list: {e}")

    def _handle_new_game(self, data: bytes):
        """
        Handle new game message.
        
        Args:
            data (bytes): Message data
        """
        logger.info("New game starting")

    def _handle_new_hand_cards(self, data: bytes):
        """
        Handle new hand cards message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Game ID (4 bytes)
            game_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Number of cards (1 byte)
            num_cards = struct.unpack("!B", data[index:index+1])[0]
            index += 1
            
            logger.info(f"Received {num_cards} new hand cards")
            
            # Reset hand state
            self.hand_cards = []
            self.community_cards = []
            self.phase = GamePhase.PREFLOP
            
            # Parse cards
            for _ in range(num_cards):
                # Card value (1 byte) and suit (1 byte)
                card_value = struct.unpack("!B", data[index:index+1])[0]
                index += 1
                card_suit = struct.unpack("!B", data[index:index+1])[0]
                index += 1
                
                # Convert to our Card format
                rank, suit = self._convert_card(card_value, card_suit)
                card = Card(rank, suit)
                
                # Add to hand
                self.hand_cards.append(card)
                
                logger.info(f"Card: {card}")
        
        except Exception as e:
            logger.error(f"Error handling new hand cards: {e}")

    def _handle_flop_cards(self, data: bytes):
        """
        Handle flop cards message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Game ID (4 bytes)
            game_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Reset community cards
            self.community_cards = []
            
            # Parse three flop cards
            for _ in range(3):
                # Card value (1 byte) and suit (1 byte)
                card_value = struct.unpack("!B", data[index:index+1])[0]
                index += 1
                card_suit = struct.unpack("!B", data[index:index+1])[0]
                index += 1
                
                # Convert to our Card format
                rank, suit = self._convert_card(card_value, card_suit)
                card = Card(rank, suit)
                
                # Add to community cards
                self.community_cards.append(card)
            
            # Update phase
            self.phase = GamePhase.FLOP
            
            logger.info(f"Flop cards: {[str(card) for card in self.community_cards]}")
        
        except Exception as e:
            logger.error(f"Error handling flop cards: {e}")

    def _handle_turn_card(self, data: bytes):
        """
        Handle turn card message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Game ID (4 bytes)
            game_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Card value (1 byte) and suit (1 byte)
            card_value = struct.unpack("!B", data[index:index+1])[0]
            index += 1
            card_suit = struct.unpack("!B", data[index:index+1])[0]
            
            # Convert to our Card format
            rank, suit = self._convert_card(card_value, card_suit)
            card = Card(rank, suit)
            
            # Add to community cards
            self.community_cards.append(card)
            
            # Update phase
            self.phase = GamePhase.TURN
            
            logger.info(f"Turn card: {card}")
        
        except Exception as e:
            logger.error(f"Error handling turn card: {e}")

    def _handle_river_card(self, data: bytes):
        """
        Handle river card message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Game ID (4 bytes)
            game_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Card value (1 byte) and suit (1 byte)
            card_value = struct.unpack("!B", data[index:index+1])[0]
            index += 1
            card_suit = struct.unpack("!B", data[index:index+1])[0]
            
            # Convert to our Card format
            rank, suit = self._convert_card(card_value, card_suit)
            card = Card(rank, suit)
            
            # Add to community cards
            self.community_cards.append(card)
            
            # Update phase
            self.phase = GamePhase.RIVER
            
            logger.info(f"River card: {card}")
        
        except Exception as e:
            logger.error(f"Error handling river card: {e}")

    def _handle_next_player_to_act(self, data: bytes):
        """
        Handle next player to act message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Game ID (4 bytes)
            game_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Player ID (4 bytes)
            player_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Action timeout (4 bytes in 1/10 seconds)
            timeout_10ms = struct.unpack("!I", data[index:index+4])[0]
            timeout_sec = timeout_10ms / 10.0
            
            # Check if it's our turn
            if player_id == self.player_id:
                logger.info(f"It's our turn to act (timeout: {timeout_sec:.1f}s)")
                self.my_turn = True
                
                # In a full implementation, we would also parse:
                # - current highest bet
                # - amount to call
                # - minimum raise
                # - maximum raise
                
                # For now, let's assume some defaults
                self.current_bet = 100  # Example value
                to_call = 100
                min_raise = 200
                max_raise = self.my_chips
                
                # Make a decision
                self._make_decision(to_call, min_raise, max_raise)
            else:
                # Get player name if available
                player_name = self.player_states.get(player_id, {}).get('name', f"Player {player_id}")
                logger.info(f"Player {player_name} to act next")
        
        except Exception as e:
            logger.error(f"Error handling next player to act: {e}")

    def _handle_hand_finished(self, data: bytes):
        """
        Handle hand finished message.
        
        Args:
            data (bytes): Message data
        """
        logger.info("Hand finished")
        
        # Reset hand state
        self.hand_cards = []
        self.community_cards = []
        self.my_turn = False

    def _handle_game_player(self, data: bytes):
        """
        Handle game player message.
        
        Args:
            data (bytes): Message data
        """
        if len(data) < 1:
            return
        
        # Player message type (1 byte)
        player_type = data[0]
        
        if player_type == GamePlayerMessageType.GAME_PLAYER_ACTION:
            self._handle_player_action(data[1:])
        
        else:
            logger.debug(f"Unhandled game player message type: {player_type}")

    def _handle_player_action(self, data: bytes):
        """
        Handle player action message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Game ID (4 bytes)
            game_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Player ID (4 bytes)
            player_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Action type (1 byte)
            action_type = struct.unpack("!B", data[index:index+1])[0]
            index += 1
            
            # Get player name if available
            player_name = self.player_states.get(player_id, {}).get('name', f"Player {player_id}")
            
            # Parse based on action type
            if action_type == PlayerActionType.FOLD:
                logger.info(f"Player {player_name} folds")
                
                # Update player state
                if player_id in self.player_states:
                    self.player_states[player_id]['active'] = False
            
            elif action_type == PlayerActionType.CHECK:
                logger.info(f"Player {player_name} checks")
            
            elif action_type == PlayerActionType.CALL:
                logger.info(f"Player {player_name} calls")
            
            elif action_type == PlayerActionType.BET:
                # Bet amount (4 bytes)
                bet_amount = struct.unpack("!I", data[index:index+4])[0]
                logger.info(f"Player {player_name} bets {bet_amount}")
                
                # Update game state
                self.current_bet = bet_amount
            
            elif action_type == PlayerActionType.RAISE:
                # Raise amount (4 bytes)
                raise_amount = struct.unpack("!I", data[index:index+4])[0]
                logger.info(f"Player {player_name} raises to {raise_amount}")
                
                # Update game state
                self.current_bet = raise_amount
            
            elif action_type == PlayerActionType.ALL_IN:
                logger.info(f"Player {player_name} goes all-in")
        
        except Exception as e:
            logger.error(f"Error handling player action: {e}")

    def _make_decision(self, to_call: int, min_raise: int, max_raise: int):
        """
        Make a decision when it's our turn to act.
        
        Args:
            to_call (int): Amount to call
            min_raise (int): Minimum raise amount
            max_raise (int): Maximum raise amount
        """
        try:
            if self.ai_model:
                # Use AI model for decision
                action, amount = self._make_ai_decision(to_call, min_raise, max_raise)
            else:
                # Use simple logic
                action, amount = self._make_simple_decision(to_call, min_raise, max_raise)
            
            logger.info(f"Decision: {action.name}, amount: {amount}")
            
            # Send action to server
            self._send_player_action(action, amount)
            
            # Reset turn flag
            self.my_turn = False
        
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            
            # Default to safest action (fold)
            self._send_player_action(Action.FOLD, None)
            self.my_turn = False

    def _make_simple_decision(self, to_call: int, min_raise: int, max_raise: int) -> Tuple[Action, Optional[int]]:
        """
        Make a simple poker decision without using the AI model.
        
        Args:
            to_call (int): Amount to call
            min_raise (int): Minimum raise amount
            max_raise (int): Maximum raise amount
            
        Returns:
            Tuple[Action, Optional[int]]: Action and bet amount
        """
        # Calculate hand strength
        hand_strength = calculate_hand_strength(
            self.hand_cards,
            self.community_cards,
            self.active_players
        )
        
        logger.info(f"Hand strength: {hand_strength:.2f}")
        
        # Simple decision based on hand strength
        if hand_strength < 0.3:
            # Weak hand - fold or check
            if to_call > 0:
                return Action.FOLD, None
            else:
                return Action.CHECK, None
        
        elif hand_strength < 0.6:
            # Medium hand - call or small raise
            if to_call > self.my_chips * 0.25:
                return Action.FOLD, None
            elif to_call > 0:
                return Action.CALL, None
            else:
                bet_amount = min(min_raise, self.my_chips // 10)
                return Action.BET, bet_amount
        
        else:
            # Strong hand - raise or all-in
            if hand_strength > 0.8:
                return Action.ALL_IN, self.my_chips
            elif to_call > 0:
                raise_amount = min(min_raise * 3, self.my_chips)
                return Action.RAISE, raise_amount
            else:
                bet_amount = min(min_raise * 2, self.my_chips // 4)
                return Action.BET, bet_amount

    def _make_ai_decision(self, to_call: int, min_raise: int, max_raise: int) -> Tuple[Action, Optional[int]]:
        """
        Make a poker decision using the AI model.
        
        Args:
            to_call (int): Amount to call
            min_raise (int): Minimum raise amount
            max_raise (int): Maximum raise amount
            
        Returns:
            Tuple[Action, Optional[int]]: Action and bet amount
        """
        # Create a temporary AIPlayer to use the model
        ai_player = AIPlayer("temp", self.ai_model, self.my_chips)
        ai_player.hole_cards = self.hand_cards
        ai_player.position = self.my_seat if self.my_seat is not None else 0
        
        # Determine valid actions
        valid_actions = []
        
        # You can always fold
        valid_actions.append(Action.FOLD)
        
        # Check is valid if no bet to call
        if to_call == 0:
            valid_actions.append(Action.CHECK)
        else:
            # Call is valid if player has chips to call
            if self.my_chips > 0:
                valid_actions.append(Action.CALL)
        
        # Bet/Raise is valid if player has enough chips
        if to_call == 0:
            valid_actions.append(Action.BET)
        else:
            valid_actions.append(Action.RAISE)
        
        # All-in is always valid if player has chips
        if self.my_chips > 0:
            valid_actions.append(Action.ALL_IN)
        
        # Create game state for the AI
        game_state = {
            'community_cards': self.community_cards,
            'to_call': to_call,
            'pot_size': self.pot_size,
            'phase': self.phase,
            'active_players': self.active_players,
            'dealer_position': self.dealer_position,
            'hand_strength': calculate_hand_strength(
                self.hand_cards,
                self.community_cards,
                self.active_players
            )
        }
        
        # Get AI's action
        action, bet_amount = ai_player.get_action(game_state, valid_actions)
        
        return action, bet_amount

    def _send_player_action(self, action: Action, amount: Optional[int]):
        """
        Send a player action to the server.
        
        Args:
            action (Action): The action to take
            amount (Optional[int]): Bet/raise amount if applicable
        """
        try:
            payload = bytearray()
            
            # Game ID (4 bytes)
            payload.extend(struct.pack("!I", self.game_id))
            
            # Convert our action to PokerTH action type
            action_type = self._convert_action_type(action)
            
            # Action type (1 byte)
            payload.extend(struct.pack("!B", action_type))
            
            # For bet/raise actions, include amount
            if action == Action.BET or action == Action.RAISE:
                if amount is not None:
                    payload.extend(struct.pack("!I", amount))
                else:
                    # Use a minimum amount if none specified
                    payload.extend(struct.pack("!I", 100))  # Default 100 chips
            
            # Create game player message
            game_player_msg = bytearray()
            game_player_msg.append(GamePlayerMessageType.GAME_PLAYER_ACTION)
            game_player_msg.extend(payload)
            
            # Create game message
            game_msg = bytearray()
            game_msg.append(GameMessageType.PLAYER)
            game_msg.extend(game_player_msg)
            
            # Send packet
            self._send_packet(MessageType.MSG_GAME, game_msg)
            
            logger.info(f"Sent player action: {action.name}")
        
        except Exception as e:
            logger.error(f"Error sending player action: {e}")

    def _convert_action_type(self, action: Action) -> int:
        """
        Convert our Action enum to PokerTH action type.
        
        Args:
            action (Action): Our action
            
        Returns:
            int: PokerTH action type
        """
        action_map = {
            Action.FOLD: PlayerActionType.FOLD,
            Action.CHECK: PlayerActionType.CHECK,
            Action.CALL: PlayerActionType.CALL,
            Action.BET: PlayerActionType.BET,
            Action.RAISE: PlayerActionType.RAISE,
            Action.ALL_IN: PlayerActionType.ALL_IN
        }
        
        return action_map.get(action, PlayerActionType.FOLD)

    def _convert_card(self, card_value: int, card_suit: int) -> Tuple[Rank, Suit]:
        """
        Convert PokerTH card value and suit to our format.
        
        Args:
            card_value (int): PokerTH card value (0-12)
            card_suit (int): PokerTH card suit (0-3)
            
        Returns:
            Tuple[Rank, Suit]: Card rank and suit in our format
        """
        # PokerTH uses 0-12 for card values (0=2, 12=Ace)
        # Our Rank enum uses 2-14 (2=Two, 14=Ace)
        rank_value = card_value + 2
        rank = Rank(rank_value)
        
        # PokerTH uses 0-3 for suits
        # Map to our Suit enum
        suit_map = {
            0: Suit.CLUBS,
            1: Suit.DIAMONDS,
            2: Suit.HEARTS,
            3: Suit.SPADES
        }
        suit = suit_map.get(card_suit, Suit.SPADES)
        
        return rank, suit
    
    def _load_model(self, model_path: str):
        """
        Load the AI model from the given path.
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            logger.info(f"Loading AI model from {model_path}")
            self.ai_model = PokerModel(PokerConfig.STATE_SIZE, PokerConfig.ACTION_SIZE)
            self.ai_model.load(model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.ai_model = None
    
    def connect(self) -> bool:
        """
        Connect to the PokerTH server.
        
        Returns:
            bool: True if connected successfully, False otherwise
        """
        try:
            logger.info(f"Connecting to {self.server}:{self.port}")
            
            # Create socket
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(10)  # 10 second timeout for initial connection
            
            # Connect to server
            self.sock.connect((self.server, self.port))
            self.sock.settimeout(None)  # Remove timeout for normal operation
            self.connected = True
            
            # Send authentication request based on mode
            if self.is_guest:
                self._send_guest_auth_request()
            else:
                self._send_auth_request()
            
            logger.info("Connected to server and sent auth request")
            return True
        
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the server."""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass
        
        self.connected = False
        self.authenticated = False
        logger.info("Disconnected from server")
    
    def _send_auth_request(self):
        """Send authentication request to the server using exact PokerTH protocol format."""
        try:
            # Authentication request message
            payload = bytearray()
            
            # Protocol version (1 byte) - Current PokerTH version uses 5
            payload.extend(struct.pack("!B", 5))
            
            # Authentication type (1 byte)
            # 0 = init session
            # 1 = authenticate with credentials
            # 2 = authenticate as guest
            payload.extend(struct.pack("!B", 1))  # 1 = authenticate with credentials
            
            # Username (length-prefixed string)
            username_bytes = self.username.encode('utf-8')
            payload.extend(struct.pack("!H", len(username_bytes)))
            payload.extend(username_bytes)
            
            # Password (length-prefixed string)
            password_bytes = self.password.encode('utf-8')
            payload.extend(struct.pack("!H", len(password_bytes)))
            payload.extend(password_bytes)
            
            # ClientUserData structure
            # App version string (length-prefixed string)
            version_str = "PokerTH 1.1.2".encode('utf-8')  # Use official PokerTH version
            payload.extend(struct.pack("!H", len(version_str)))
            payload.extend(version_str)
            
            # Build number (4 bytes) - Use a recent build number
            payload.extend(struct.pack("!I", 20200101))  # Example build number
            
            # My unique player ID (length-prefixed string) - Just use a random string if we don't have one
            player_id = "".encode('utf-8')  # Empty for initial login
            payload.extend(struct.pack("!H", len(player_id)))
            payload.extend(player_id)
            
            # My avatar hash (length-prefixed string) - Empty for no avatar
            avatar_hash = "".encode('utf-8')  # Empty for no avatar
            payload.extend(struct.pack("!H", len(avatar_hash)))
            payload.extend(avatar_hash)
            
            # Send the authentication request
            self._send_packet(MessageType.MSG_AUTH_CLIENT_REQUEST, payload)
            
            # Log packet details for debugging
            logger.debug(f"Sent auth packet: {payload.hex()}")
            logger.info(f"Sent authentication request for user: {self.username}")
        
        except Exception as e:
            logger.error(f"Error sending auth request: {e}")
            self.disconnect()

    def _send_guest_auth_request(self):
        """Send guest authentication request to the server."""
        try:
            # Authentication request message
            payload = bytearray()
            
            # Protocol version (1 byte)
            payload.extend(struct.pack("!B", 5))
            
            # Authentication type (1 byte): 2 = guest login
            payload.extend(struct.pack("!B", 2))
            
            # Guest name (length-prefixed string)
            # Use a random number to avoid name collisions
            import random
            guest_name = f"Guest_{random.randint(1000, 9999)}".encode('utf-8')
            payload.extend(struct.pack("!H", len(guest_name)))
            payload.extend(guest_name)
            
            # ClientUserData structure
            # App version string (length-prefixed string)
            version_str = "PokerTH 1.1.2".encode('utf-8')
            payload.extend(struct.pack("!H", len(version_str)))
            payload.extend(version_str)
            
            # Build number (4 bytes)
            payload.extend(struct.pack("!I", 20200101))
            
            # My unique player ID (length-prefixed string)
            player_id = "".encode('utf-8')  # Empty for initial login
            payload.extend(struct.pack("!H", len(player_id)))
            payload.extend(player_id)
            
            # My avatar hash (length-prefixed string)
            avatar_hash = "".encode('utf-8')
            payload.extend(struct.pack("!H", len(avatar_hash)))
            payload.extend(avatar_hash)
            
            # Send the authentication request
            self._send_packet(MessageType.MSG_AUTH_CLIENT_REQUEST, payload)
            
            # Log packet details for debugging
            logger.debug(f"Sent guest auth packet: {payload.hex()}")
            logger.info(f"Sent guest authentication request as: {guest_name.decode('utf-8')}")
            
            # Store the guest name we used
            self.username = guest_name.decode('utf-8')
        
        except Exception as e:
            logger.error(f"Error sending guest auth request: {e}")
            self.disconnect()

    def _send_packet(self, msg_type: int, payload: bytearray):
        """
        Send a packet to the server with enhanced debugging.
        
        Args:
            msg_type (int): Message type
            payload (bytearray): Message payload
        """
        if not self.connected:
            logger.error("Cannot send packet: not connected")
            return
        
        try:
            # Calculate message size (payload + message type byte)
            msg_size = len(payload) + 1
            
            # Create header
            header = struct.pack(self.HEADER_FORMAT, msg_size, msg_type)
            
            # Create full packet
            packet = header + payload
            
            # Debug logging
            logger.debug(f">>> SEND Packet {msg_type}, size {msg_size}")
            logger.debug(f">>> Header: {header.hex()}")
            if len(payload) > 0:
                logger.debug(f">>> First 32 bytes: {payload[:min(32, len(payload))].hex()}")
            
            # Send packet
            self.sock.sendall(packet)
        
        except Exception as e:
            logger.error(f"Error sending packet: {e}")
            self.disconnect()
    
    def _receive_data(self) -> bool:
        """
        Receive data from the server with enhanced error handling.
        
        Returns:
            bool: True if data was received, False on error or disconnect
        """
        if not self.connected:
            return False
        
        try:
            # Receive data with a small timeout to avoid blocking
            self.sock.settimeout(0.1)
            data = self.sock.recv(4096)
            self.sock.settimeout(None)
            
            if not data:
                logger.warning("Server closed connection")
                self.connected = False
                return False
            
            # Debug logging
            logger.debug(f"<<< RECV {len(data)} bytes: {data[:min(32, len(data))].hex()}")
            
            # Add data to buffer
            self.buffer.extend(data)
            return True
        
        except socket.timeout:
            # Timeout is normal, just no data available
            return True
        
        except socket.error as e:
            # More detailed socket error handling
            if e.errno == 104:  # Connection reset by peer
                logger.error("Connection reset by peer")
            elif e.errno == 32:  # Broken pipe
                logger.error("Broken pipe")
            elif e.errno == 111:  # Connection refused
                logger.error("Connection refused")
            else:
                logger.error(f"Socket error {e.errno}: {e}")
            
            self.connected = False
            return False
        
        except Exception as e:
            logger.error(f"Error receiving data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.connected = False
            return False
    
    def _process_buffer(self) -> bool:
        """
        Process buffered data and extract complete packets with enhanced debugging.
        
        Returns:
            bool: True if processing should continue, False otherwise
        """
        while len(self.buffer) >= self.HEADER_SIZE:
            # Parse header
            header_data = self.buffer[:self.HEADER_SIZE]
            msg_size, msg_type = struct.unpack(self.HEADER_FORMAT, header_data)
            
            # Calculate total packet size
            total_size = self.HEADER_SIZE + msg_size - 1  # -1 because msg_size includes the type byte
            
            # Debug
            logger.debug(f"<<< Processing header: size={msg_size}, type={msg_type}, total_size={total_size}")
            
            # If we don't have the complete packet yet, wait for more data
            if len(self.buffer) < total_size:
                logger.debug(f"Incomplete packet: have {len(self.buffer)} bytes, need {total_size}")
                break
            
            # Extract the complete packet
            packet_data = self.buffer[self.HEADER_SIZE:total_size]
            
            # Debug packet content
            logger.debug(f"<<< Complete packet type {msg_type}, payload size {len(packet_data)}")
            if len(packet_data) > 0:
                logger.debug(f"<<< Payload (hex): {packet_data.hex()[:64]}")
            
            # Remove processed data from buffer
            self.buffer = self.buffer[total_size:]
            
            # Process the packet
            self._process_packet(msg_type, packet_data)
        
        return True
    
    def _process_packet(self, msg_type: int, data: bytes):
        """
        Process a complete packet.
        
        Args:
            msg_type (int): Message type
            data (bytes): Message data
        """
        logger.debug(f"Processing packet type {msg_type}, size {len(data)}")
        
        try:
            if msg_type == MessageType.MSG_AUTH_SERVER_CHALLENGE:
                self._handle_auth_challenge(data)
            
            elif msg_type == MessageType.MSG_AUTH_SERVER_VERIFICATION:
                self._handle_auth_verification(data)
            
            elif msg_type == MessageType.MSG_INIT_DONE:
                self._handle_init_done()
            
            elif msg_type == MessageType.MSG_GAME_LIST:
                self._handle_game_list(data)
            
            elif msg_type == MessageType.MSG_JOIN_GAME_REPLY:
                self._handle_join_game_reply(data)
            
            elif msg_type == MessageType.MSG_CREATE_GAME_REPLY:
                self._handle_create_game_reply(data)
            
            elif msg_type == MessageType.MSG_GAME:
                self._handle_game_message(data)
            
            elif msg_type == MessageType.MSG_LOBBY:
                self._handle_lobby_message(data)
            
            elif msg_type == MessageType.MSG_ERROR:
                self._handle_error_message(data)
            
            else:
                logger.debug(f"Unhandled message type: {msg_type}")
        
        except Exception as e:
            logger.error(f"Error processing packet type {msg_type}: {e}")
    
    def _handle_auth_challenge(self, data: bytes):
        """
        Handle authentication challenge message.
        
        Args:
            data (bytes): Message data
        """
        try:
            # Parse challenge
            index = 0
            
            # Session ID (4 bytes)
            self.session_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Server ID (length-prefixed string)
            server_id_len = struct.unpack("!H", data[index:index+2])[0]
            index += 2
            server_id = data[index:index+server_id_len].decode('utf-8')
            
            logger.info(f"Received auth challenge: session ID {self.session_id}")
            
            # In a full implementation, we would compute a proper response
            # For now, we're just sending a minimal response
            self._send_auth_response()
        
        except Exception as e:
            logger.error(f"Error handling auth challenge: {e}")
    
    def _send_auth_response(self):
        """Send authentication response."""
        try:
            # Simple auth response
            payload = bytearray()
            
            # Session ID
            payload.extend(struct.pack("!I", self.session_id))
            
            # Client user data (placeholder)
            # Player Name (empty string)
            payload.extend(struct.pack("!H", 0))
            
            # Avatar MD5 (empty string)
            payload.extend(struct.pack("!H", 0))
            
            # Send the response
            self._send_packet(MessageType.MSG_AUTH_CLIENT_RESPONSE, payload)
            logger.info("Sent authentication response")
        
        except Exception as e:
            logger.error(f"Error sending auth response: {e}")
    
    def _handle_auth_verification(self, data: bytes):
        """
        Handle authentication verification message.
        
        Args:
            data (bytes): Message data
        """
        try:
            # Parse verification result
            index = 0
            
            # Verification result (1 byte)
            result = struct.unpack("!B", data[index:index+1])[0]
            index += 1
            
            if result == AuthServerVerificationResult.SUCCESS:
                # Authentication successful
                # Player ID (4 bytes)
                self.player_id = struct.unpack("!I", data[index:index+4])[0]
                self.authenticated = True
                logger.info(f"Authentication successful, player ID: {self.player_id}")
            else:
                # Authentication failed
                # Error code (1 byte)
                error_code = struct.unpack("!B", data[index:index+1])[0]
                error_msg = self._get_auth_error_message(error_code)
                logger.error(f"Authentication failed: {error_msg}")
                self.disconnect()
        
        except Exception as e:
            logger.error(f"Error handling auth verification: {e}")
    
    def _handle_init_done(self):
        """Handle initialization done message."""
        logger.info("Connection initialization completed")
        
        # Request game list or join/create a game
        if self.game_name:
            # We have a specific game to join/create
            self._request_game_list()
        else:
            # Just request the game list
            self._request_game_list()
    
    def _request_game_list(self):
        """Request the list of available games."""
        try:
            # Empty payload for game list request
            payload = bytearray()
            
            # Send the request
            self._send_packet(MessageType.MSG_GAME_LIST, payload)
            logger.info("Sent game list request")
        
        except Exception as e:
            logger.error(f"Error requesting game list: {e}")
    
    def _handle_game_list(self, data: bytes):
        """
        Handle game list message.
        
        Args:
            data (bytes): Message data
        """
        try:
            games = []
            index = 0
            
            # Number of games (4 bytes)
            num_games = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Parse each game
            for _ in range(num_games):
                game = {}
                
                # Game ID (4 bytes)
                game['id'] = struct.unpack("!I", data[index:index+4])[0]
                index += 4
                
                # Game name (length-prefixed string)
                name_len = struct.unpack("!H", data[index:index+2])[0]
                index += 2
                game['name'] = data[index:index+name_len].decode('utf-8')
                index += name_len
                
                # Game creator name (length-prefixed string)
                creator_len = struct.unpack("!H", data[index:index+2])[0]
                index += 2
                game['creator'] = data[index:index+creator_len].decode('utf-8')
                index += creator_len
                
                # Game type (1 byte)
                game['type'] = struct.unpack("!B", data[index:index+1])[0]
                index += 1
                
                # Game mode (1 byte)
                game['mode'] = struct.unpack("!B", data[index:index+1])[0]
                index += 1
                
                # Current players (1 byte)
                game['players'] = struct.unpack("!B", data[index:index+1])[0]
                index += 1
                
                # Maximum players (1 byte)
                game['max_players'] = struct.unpack("!B", data[index:index+1])[0]
                index += 1
                
                games.append(game)
            
            logger.info(f"Received {len(games)} games:")
            for i, game in enumerate(games):
                logger.info(f"{i+1}. {game['name']} ({game['players']}/{game['max_players']}) by {game['creator']}")
            
            # If we have a specific game name, try to find and join it
            if self.game_name:
                # Look for a game with matching name
                for game in games:
                    if game['name'] == self.game_name:
                        logger.info(f"Found game '{self.game_name}', joining...")
                        self._join_game(game['id'])
                        return
                
                # Game not found, create it
                logger.info(f"Game '{self.game_name}' not found, creating...")
                self._create_game(self.game_name)
            elif games:
                # Join the first game
                logger.info(f"Joining game '{games[0]['name']}'")
                self._join_game(games[0]['id'])
            else:
                # No games available, create one
                default_name = f"{self.username}'s Game"
                logger.info(f"No games available, creating '{default_name}'")
                self._create_game(default_name)
        
        except Exception as e:
            logger.error(f"Error handling game list: {e}")
    
    def _join_game(self, game_id: int, password: str = ""):
        """
        Send a request to join a game.
        
        Args:
            game_id (int): Game ID to join
            password (str): Password if required
        """
        try:
            payload = bytearray()
            
            # Game ID (4 bytes)
            payload.extend(struct.pack("!I", game_id))
            
            # Password (length-prefixed string)
            password_bytes = password.encode('utf-8')
            payload.extend(struct.pack("!H", len(password_bytes)))
            payload.extend(password_bytes)
            
            # Game State (1 byte, 0 = active)
            payload.extend(struct.pack("!B", 0))
            
            # Send the request
            self._send_packet(MessageType.MSG_JOIN_GAME_REQUEST, payload)
            logger.info(f"Sent request to join game {game_id}")
        
        except Exception as e:
            logger.error(f"Error joining game: {e}")
    
    def _handle_join_game_reply(self, data: bytes):
        """
        Handle join game reply message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Game ID (4 bytes)
            game_id = struct.unpack("!I", data[index:index+4])[0]
            index += 4
            
            # Join result (1 byte)
            result = struct.unpack("!B", data[index:index+1])[0]
            
            if result == JoinGameResult.SUCCESS:
                self.game_id = game_id
                self.joined_game = True
                logger.info(f"Successfully joined game {game_id}")
            else:
                error_msg = self._get_join_game_error_message(result)
                logger.error(f"Failed to join game: {error_msg}")
                
                # If we failed to join, create our own game
                if self.game_name:
                    logger.info(f"Creating game '{self.game_name}' instead")
                    self._create_game(self.game_name)
                else:
                    default_name = f"{self.username}'s Game"
                    logger.info(f"Creating game '{default_name}' instead")
                    self._create_game(default_name)
        
        except Exception as e:
            logger.error(f"Error handling join game reply: {e}")
    
    def _create_game(self, game_name: str, password: str = ""):
        """
        Send a request to create a new game.
        
        Args:
            game_name (str): Name for the new game
            password (str): Password protection (optional)
        """
        try:
            payload = bytearray()
            
            # Game name (length-prefixed string)
            game_name_bytes = game_name.encode('utf-8')
            payload.extend(struct.pack("!H", len(game_name_bytes)))
            payload.extend(game_name_bytes)
            
            # Password (length-prefixed string)
            password_bytes = password.encode('utf-8')
            payload.extend(struct.pack("!H", len(password_bytes)))
            payload.extend(password_bytes)
            
            # Game type (1 byte, 0 = normal)
            payload.extend(struct.pack("!B", 0))
            
            # Maximum number of players (1 byte)
            payload.extend(struct.pack("!B", 10))
            
            # Starting blinds (4 bytes)
            payload.extend(struct.pack("!I", 50))  # Small blind
            payload.extend(struct.pack("!I", 100))  # Big blind
            
            # Starting chip amount (4 bytes)
            payload.extend(struct.pack("!I", 10000))
            
            # Blind raise mode (1 byte, 0 = no raise)
            payload.extend(struct.pack("!B", 1))  # 1 = Raise on hand number
            
            # Blind raise every (4 bytes)
            payload.extend(struct.pack("!I", 10))  # Raise every 10 hands
            
            # Send the request
            self._send_packet(MessageType.MSG_CREATE_GAME_REQUEST, payload)
            logger.info(f"Sent request to create game '{game_name}'")
        
        except Exception as e:
            logger.error(f"Error creating game: {e}")
    
    def _handle_create_game_reply(self, data: bytes):
        """
        Handle create game reply message.
        
        Args:
            data (bytes): Message data
        """
        try:
            index = 0
            
            # Result (1 byte, 0 = success)
            result = struct.unpack("!B", data[index:index+1])[0]
            index += 1
            
            if result == 0:
                # Game ID (4 bytes)
                game_id = struct.unpack("!I", data[index:index+4])[0]
                self.game_id = game_id
                self.joined_game = True
                logger.info(f"Successfully created and joined game {game_id}")
            else:
                logger.error("Failed to create game")
                self.disconnect()
        
        except Exception as e:
            logger.error(f"Error handling create game reply: {e}")
    
    def _handle_game_message(self, data: bytes):
        """
        Handle game message.
        
        Args:
            data (bytes): Message data
        """
        try:
            if len(data) < 1:
                return
            
            # Game message type (1 byte)
            game_msg_type = data[0]
            
            # Handle different types of game messages
            if game_msg_type == 1:  # Game management
                self._handle_game_management(data[1:])
            elif game_msg_type == 2:  # Game engine
                self._handle_game_engine(data[1:])
            elif game_msg_type == 3:  # Game player
                self._handle_game_player(data[1:])
            else:
                logger.debug(f"Unhandled game message type: {game_msg_type}")
        
        except Exception as e:
            logger.error(f"Error handling game message: {e}")
    
    def _handle_game_management(self, data: bytes):
        """
        Handle game management message.
        
        Args:
            data (bytes): Message data
        """
        # In a full implementation, we would parse and handle various management messages
        # Such as player joining/leaving, game starting, etc.
        logger.debug("Received game management message")
    
    def _handle_game_engine(self, data: bytes):
        """
        Handle game engine message.
        
        Args:
            data (bytes): Message data
        """
        # In a full implementation, we would parse and handle various engine messages
        # Such as dealing cards, betting rounds, etc.
        logger.debug("Received game engine message")
    
    def _handle_game_player(self, data: bytes):
        """
        Handle game player message.
        
        Args:
            data (bytes): Message data
        """
        # In a full implementation, we would parse and handle various player messages
        # Such as player actions, etc.
        logger.debug("Received game player message")
    
    def _handle_lobby_message(self, data: bytes):
        """
        Handle lobby message.
        
        Args:
            data (bytes): Message data
        """
        # In a full implementation, we would parse and handle various lobby messages
        logger.debug("Received lobby message")
    
    def _get_auth_error_message(self, error_code: int) -> str:
        """
        Get a human-readable message for an authentication error code.
        
        Args:
            error_code (int): Error code
            
        Returns:
            str: Error message
        """
        error_messages = {
            AuthServerVerificationError.INVALID_PASSWORD: "Invalid password",
            AuthServerVerificationError.INVALID_USERNAME: "Invalid username",
            AuthServerVerificationError.INVALID_VERSION: "Invalid client version",
            AuthServerVerificationError.ALREADY_LOGGED_IN: "Already logged in"
        }
        
        return error_messages.get(error_code, f"Unknown error ({error_code})")
    
    def _get_join_game_error_message(self, error_code: int) -> str:
        """
        Get a human-readable message for a join game error code.
        
        Args:
            error_code (int): Error code
            
        Returns:
            str: Error message
        """
        error_messages = {
            JoinGameResult.INVALID_GAME_ID: "Invalid game ID",
            JoinGameResult.GAME_FULL: "Game is full",
            JoinGameResult.ALREADY_STARTED: "Game already started",
            JoinGameResult.INVALID_PASSWORD: "Invalid password"
        }
        
        return error_messages.get(error_code, f"Unknown error ({error_code})")
    
    def _handle_error_message(self, data: bytes):
        """
        Handle error message from the server with enhanced debugging.
        
        Args:
            data (bytes): Message data
        """
        try:
            if len(data) < 1:
                logger.error("Received empty error message")
                return
            
            # Error code (1 byte)
            error_code = data[0]
            
            # Dump full error message in hex for debugging
            logger.debug(f"Error packet hex dump: {data.hex()}")
            
            # Try to extract more info
            error_msg = f"Error code {error_code}"
            error_details = ""
            
            # Try to extract a string message if available
            if len(data) > 3:  # At least 1 byte error code + 2 bytes length + some content
                try:
                    msg_len = struct.unpack("!H", data[1:3])[0]
                    if len(data) >= 3 + msg_len and msg_len > 0:
                        error_details = data[3:3+msg_len].decode('utf-8', errors='replace')
                        error_msg += f" - {error_details}"
                except:
                    pass
            
            logger.error(f"Server error: {error_msg}")
            
            # Handle specific errors - update with any known PokerTH error codes
            if error_code == 0:
                logger.error("Generic error")
            elif error_code == 1:
                logger.error("Invalid request or protocol error")
                # Maybe try a different protocol version or format
                logger.info("Trying guest authentication instead...")
                self._send_guest_auth_request()
            elif error_code == 2:
                logger.error("Authentication failed")
                # Mark as not authenticated to retry
                self.authenticated = False
            elif error_code == 3:
                logger.error("Not authorized for this action")
            elif error_code == 4:
                logger.error("Server is full")
            elif error_code == 5:
                logger.error("Game server version mismatch")
            elif error_code == 73:
                logger.error("Session timeout - server is closing the connection")
                # This is normal after inactivity
            else:
                logger.error(f"Unknown error code: {error_code}")
            
            # For protocol errors, retry with guest login
            if error_code == 1 and not self.is_guest:
                logger.info("Trying guest login instead...")
                self.is_guest = True
                self._send_guest_auth_request()
        
        except Exception as e:
            logger.error(f"Error handling error message: {e}")

    def retry_as_guest(self):
        """Switch to guest mode and retry authentication."""
        if self.is_guest:
            # Already in guest mode, don't retry
            return
        
        logger.info("Switching to guest authentication mode")
        self.is_guest = True
        self._send_guest_auth_request()

    def run(self):
        """Run the client, connecting and playing in games."""
        if not self.connect():
            logger.error("Failed to connect to server")
            return
        
        logger.info("Starting main loop")
        try:
            while self.connected:
                # Receive data from server
                if not self._receive_data():
                    break
                
                # Process received data
                if not self._process_buffer():
                    break
                
                # Check if it's our turn
                if self.my_turn:
                    # Default values if we don't have actual data
                    self._make_decision(100, 200, self.my_chips)
                
                # Small sleep to avoid busy-waiting
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Client interrupted by user")
        finally:
            self.disconnect()


def main():
    """Main entry point for the PokerTH client."""
    parser = argparse.ArgumentParser(description='Socket-Based PokerTH Client with AI Model')
    parser.add_argument('--server', type=str, default='pokerth.net',
                        help='Server address (default: pokerth.net)')
    parser.add_argument('--port', type=int, default=7234,
                        help='Server port (default: 7234)')
    parser.add_argument('--username', type=str, default='Guest',
                        help='Username for authentication')
    parser.add_argument('--password', type=str, default='',
                        help='Password for authentication')
    parser.add_argument('--guest', action='store_true',
                        help='Use guest login instead of credentials')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the AI model to use')
    parser.add_argument('--game', type=str, default=None,
                        help='Name of the game to join or create')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--protocol-version', type=int, default=5,
                        help='PokerTH protocol version to use (default: 5)')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # If password not provided via args, prompt for it (unless using guest mode)
    password = args.password
    if not password and not args.guest and args.username != 'Guest':
        password = getpass.getpass(f"Enter password for {args.username}: ")
    
    # Create and run the client
    client = SocketPokerTHClient(
        server=args.server,
        port=args.port,
        username=args.username,
        password=password,
        model_path=args.model,
        game_name=args.game,
        use_guest=args.guest,
        protocol_version=args.protocol_version
    )
    
    client.run()


if __name__ == "__main__":
    main()