"""
PokerTH Client using Protocol Buffers

This implementation uses the Protocol Buffers approach required by PokerTH.
Before using this, you'll need to:

1. Install Protocol Buffers: pip install protobuf
2. Get the pokerth.proto file from the PokerTH source repository
3. Generate Python code: protoc --python_out=. pokerth.proto

This will generate pokerth_pb2.py which is imported by this client.
"""
import socket
import struct
import logging
import argparse
import getpass
import time
import os
import random
from typing import Optional, Dict, List, Any, Tuple

try:
    import pokerth_pb2
except ImportError:
    print("Error: pokerth_pb2 module not found.")
    print("You need to generate it from the pokerth.proto file:")
    print("1. Get pokerth.proto from the PokerTH source repository")
    print("2. Run: protoc --python_out=. pokerth.proto")
    print("This will generate the pokerth_pb2.py file needed by this client.")
    exit(1)

# Import your AI components
try:
    from config import PokerConfig, Action, GamePhase
    from model import PokerModel
    from player import AIPlayer
    from card import Card, Rank, Suit
    from utils import calculate_hand_strength
except ImportError:
    print("Warning: AI components not found. Using basic decision logic.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pokerth_client.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PokerTH.ProtobufClient")

class PokerTHProtobufClient:
    """
    PokerTH client using Protocol Buffers for communication.
    """
    # Header format: 4 bytes size, 1 byte type
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
        """Send authentication request using Protocol Buffers."""
        try:
            # Create auth request message
            auth_request = pokerth_pb2.AuthClientRequestMessage()
            auth_request.requested_version = self.protocol_version
            auth_request.login_type = pokerth_pb2.AuthClientRequestMessage.Type.AUTHENTIFICATION
            auth_request.login_name = self.username
            auth_request.password = self.password
            
            # Set client user data
            auth_request.client_user_data.application_version = "PokerTH 1.1.2"
            auth_request.client_user_data.build_id = 20200101  # Example build number
            
            # Create envelope message
            envelope = pokerth_pb2.PokerTHMessage()
            envelope.type = pokerth_pb2.PokerTHMessage.Type.Type_AuthClientRequestMessage
            envelope.auth_client_request_message.CopyFrom(auth_request)
            
            # Serialize and send
            self._send_protobuf_message(envelope)
            logger.info(f"Sent authentication request for user: {self.username}")
        
        except Exception as e:
            logger.error(f"Error sending auth request: {e}")
            self.disconnect()
    
    def _send_guest_auth_request(self):
        """Send guest authentication request using Protocol Buffers."""
        try:
            # Create guest name with random number to avoid collisions
            guest_name = f"Guest_{random.randint(1000, 9999)}"
            
            # Create auth request message
            auth_request = pokerth_pb2.AuthClientRequestMessage()
            auth_request.requested_version = self.protocol_version
            auth_request.login_type = pokerth_pb2.AuthClientRequestMessage.Type.GUEST_LOGIN
            auth_request.guest_name = guest_name
            
            # Set client user data
            auth_request.client_user_data.application_version = "PokerTH 1.1.2"
            auth_request.client_user_data.build_id = 20200101  # Example build number
            
            # Create envelope message
            envelope = pokerth_pb2.PokerTHMessage()
            envelope.type = pokerth_pb2.PokerTHMessage.Type.Type_AuthClientRequestMessage
            envelope.auth_client_request_message.CopyFrom(auth_request)
            
            # Serialize and send
            self._send_protobuf_message(envelope)
            
            # Store the guest name
            self.username = guest_name
            logger.info(f"Sent guest authentication request as: {guest_name}")
        
        except Exception as e:
            logger.error(f"Error sending guest auth request: {e}")
            self.disconnect()
    
    def _send_auth_challenge_response(self, session_id: int):
        """
        Send response to authentication challenge.
        
        Args:
            session_id (int): Session ID from challenge
        """
        try:
            # Create response message
            response = pokerth_pb2.AuthClientResponseMessage()
            response.session_id = session_id
            
            # Create envelope message
            envelope = pokerth_pb2.PokerTHMessage()
            envelope.type = pokerth_pb2.PokerTHMessage.Type.Type_AuthClientResponseMessage
            envelope.auth_client_response_message.CopyFrom(response)
            
            # Serialize and send
            self._send_protobuf_message(envelope)
            logger.info(f"Sent authentication challenge response for session: {session_id}")
        
        except Exception as e:
            logger.error(f"Error sending auth challenge response: {e}")
            self.disconnect()
    
    def _send_game_list_request(self):
        """Send game list request using Protocol Buffers."""
        try:
            # Create game list request message
            request = pokerth_pb2.GameListMessage()
            request.type = pokerth_pb2.GameListMessage.Type.GAME_LIST_REQUEST
            
            # Create envelope message
            envelope = pokerth_pb2.PokerTHMessage()
            envelope.type = pokerth_pb2.PokerTHMessage.Type.Type_GameListMessage
            envelope.game_list_message.CopyFrom(request)
            
            # Serialize and send
            self._send_protobuf_message(envelope)
            logger.info("Sent game list request")
        
        except Exception as e:
            logger.error(f"Error sending game list request: {e}")
    
    def _send_join_game_request(self, game_id: int, password: str = ""):
        """
        Send request to join a game.
        
        Args:
            game_id (int): ID of the game to join
            password (str): Password if required
        """
        try:
            # Create join game request message
            request = pokerth_pb2.JoinGameRequestMessage()
            request.game_id = game_id
            if password:
                request.password = password
            
            # Create envelope message
            envelope = pokerth_pb2.PokerTHMessage()
            envelope.type = pokerth_pb2.PokerTHMessage.Type.Type_JoinGameRequestMessage
            envelope.join_game_request_message.CopyFrom(request)
            
            # Serialize and send
            self._send_protobuf_message(envelope)
            logger.info(f"Sent request to join game {game_id}")
        
        except Exception as e:
            logger.error(f"Error sending join game request: {e}")
    
    def _send_create_game_request(self, game_name: str, password: str = ""):
        """
        Send request to create a new game.
        
        Args:
            game_name (str): Name for the new game
            password (str): Password protection (optional)
        """
        try:
            # Create game creation request message
            request = pokerth_pb2.CreateGameRequestMessage()
            request.game_name = game_name
            if password:
                request.password = password
            
            # Set game rules
            request.max_num_players = 10
            request.start_money = 10000
            request.small_blind = 50
            request.big_blind = 100
            request.raise_interval_mode = pokerth_pb2.CreateGameRequestMessage.RaiseIntervalMode.RAISE_ON_HANDNUMBER
            request.raise_every_hands = 10
            
            # Create envelope message
            envelope = pokerth_pb2.PokerTHMessage()
            envelope.type = pokerth_pb2.PokerTHMessage.Type.Type_CreateGameRequestMessage
            envelope.create_game_request_message.CopyFrom(request)
            
            # Serialize and send
            self._send_protobuf_message(envelope)
            logger.info(f"Sent request to create game '{game_name}'")
        
        except Exception as e:
            logger.error(f"Error sending create game request: {e}")
    
    def _send_player_action(self, action_type: int, amount: Optional[int] = None):
        """
        Send a player action to the server.
        
        Args:
            action_type (int): Type of action
            amount (Optional[int]): Bet/raise amount if applicable
        """
        try:
            # Create game engine message
            game_engine = pokerth_pb2.GameEngineMessage()
            game_engine.type = pokerth_pb2.GameEngineMessage.Type.GAME_ENGINE_PLAYER_ACTION
            
            # Create player action message
            action = game_engine.player_action
            action.game_id = self.game_id
            action.player_id = self.player_id
            action.action_type = action_type
            
            # Set amount if applicable
            if amount is not None:
                if action_type == pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_BET:
                    action.bet_amount = amount
                elif action_type == pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_RAISE:
                    action.raise_amount = amount
            
            # Create game message
            game_msg = pokerth_pb2.GameMessage()
            game_msg.type = pokerth_pb2.GameMessage.Type.GAME_ENGINE
            game_msg.game_engine_message.CopyFrom(game_engine)
            
            # Create envelope message
            envelope = pokerth_pb2.PokerTHMessage()
            envelope.type = pokerth_pb2.PokerTHMessage.Type.Type_GameMessage
            envelope.game_message.CopyFrom(game_msg)
            
            # Serialize and send
            self._send_protobuf_message(envelope)
            logger.info(f"Sent player action: {action_type}")
        
        except Exception as e:
            logger.error(f"Error sending player action: {e}")
    
    def _send_protobuf_message(self, message):
        """
        Serialize and send a Protocol Buffers message.
        
        Args:
            message: Protocol Buffers message to send
        """
        if not self.connected:
            logger.error("Cannot send message: not connected")
            return
        
        try:
            # Serialize the message
            serialized_data = message.SerializeToString()
            
            # Send the message
            self.sock.sendall(serialized_data)
            
            # Debug logging
            logger.debug(f">>> SEND Message type: {message.type}")
        
        except Exception as e:
            logger.error(f"Error sending protobuf message: {e}")
            self.disconnect()
    
    def _receive_data(self) -> bool:
        """
        Receive data from the server.
        
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
            logger.debug(f"<<< RECV {len(data)} bytes")
            
            # Process received data
            self._process_message(data)
            
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
    
    def _process_message(self, data: bytes):
        """
        Process a received Protocol Buffers message.
        
        Args:
            data (bytes): Received data
        """
        try:
            # Parse the message
            message = pokerth_pb2.PokerTHMessage()
            message.ParseFromString(data)
            
            # Handle based on message type
            if message.type == pokerth_pb2.PokerTHMessage.Type.Type_AuthServerChallengeMessage:
                self._handle_auth_challenge(message.auth_server_challenge_message)
            
            elif message.type == pokerth_pb2.PokerTHMessage.Type.Type_AuthServerVerificationMessage:
                self._handle_auth_verification(message.auth_server_verification_message)
            
            elif message.type == pokerth_pb2.PokerTHMessage.Type.Type_InitDoneMessage:
                self._handle_init_done()
            
            elif message.type == pokerth_pb2.PokerTHMessage.Type.Type_GameListMessage:
                self._handle_game_list(message.game_list_message)
            
            elif message.type == pokerth_pb2.PokerTHMessage.Type.Type_JoinGameReplyMessage:
                self._handle_join_game_reply(message.join_game_reply_message)
            
            elif message.type == pokerth_pb2.PokerTHMessage.Type.Type_CreateGameReplyMessage:
                self._handle_create_game_reply(message.create_game_reply_message)
            
            elif message.type == pokerth_pb2.PokerTHMessage.Type.Type_GameMessage:
                self._handle_game_message(message.game_message)
            
            elif message.type == pokerth_pb2.PokerTHMessage.Type.Type_ErrorMessage:
                self._handle_error(message.error_message)
            
            else:
                logger.debug(f"Received unhandled message type: {message.type}")
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _handle_auth_challenge(self, challenge):
        """
        Handle authentication challenge.
        
        Args:
            challenge: Auth challenge message
        """
        try:
            # Store session ID
            self.session_id = challenge.session_id
            logger.info(f"Received auth challenge, session ID: {self.session_id}")
            
            # Send response
            self._send_auth_challenge_response(self.session_id)
        
        except Exception as e:
            logger.error(f"Error handling auth challenge: {e}")
    
    def _handle_auth_verification(self, verification):
        """
        Handle authentication verification.
        
        Args:
            verification: Auth verification message
        """
        try:
            if verification.verification_result == pokerth_pb2.AuthServerVerificationMessage.VerificationResult.SUCCESS:
                self.authenticated = True
                self.player_id = verification.player_id
                logger.info(f"Authentication successful, player ID: {self.player_id}")
            else:
                # Handle failure
                error_code = verification.verification_error
                error_msg = self._get_auth_error_message(error_code)
                logger.error(f"Authentication failed: {error_msg}")
                
                # If regular auth failed, try guest login
                if not self.is_guest:
                    logger.info("Trying guest login instead...")
                    self.is_guest = True
                    self._send_guest_auth_request()
                else:
                    # Both regular and guest login failed
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
    

    def _handle_game_list(self, game_list):
        """
        Handle game list message.
        
        Args:
            game_list: Game list message
        """
        try:
            if game_list.type == pokerth_pb2.GameListMessage.Type.GAME_LIST_EVENT:
                games = game_list.game_info
                
                logger.info(f"Received {len(games)} games:")
                for i, game in enumerate(games):
                    logger.info(f"{i+1}. {game.game_name} ({len(game.player_ids)}/{game.max_num_players})")
                
                # If we have a specific game name, try to find and join it
                if self.game_name:
                    # Look for a game with matching name
                    for game in games:
                        if game.game_name == self.game_name:
                            logger.info(f"Found game '{self.game_name}', joining...")
                            self._send_join_game_request(game.game_id)
                            return
                    
                    # Game not found, create it
                    logger.info(f"Game '{self.game_name}' not found, creating...")
                    self._send_create_game_request(self.game_name)
                elif games:
                    # Join the first game
                    logger.info(f"Joining game '{games[0].game_name}'")
                    self._send_join_game_request(games[0].game_id)
                else:
                    # No games available, create one
                    default_name = f"{self.username}'s Game"
                    logger.info(f"No games available, creating '{default_name}'")
                    self._send_create_game_request(default_name)
        
        except Exception as e:
            logger.error(f"Error handling game list: {e}")
    
    def _handle_join_game_reply(self, reply):
        """
        Handle join game reply message.
        
        Args:
            reply: Join game reply message
        """
        try:
            if reply.result == pokerth_pb2.JoinGameReplyMessage.JoinGameResult.JOIN_GAME_RESULT_SUCCESS:
                self.game_id = reply.game_id
                self.joined_game = True
                logger.info(f"Successfully joined game {reply.game_id}")
            else:
                error_msg = self._get_join_game_error_message(reply.result)
                logger.error(f"Failed to join game: {error_msg}")
                
                # If we failed to join, create our own game
                if self.game_name:
                    logger.info(f"Creating game '{self.game_name}' instead")
                    self._send_create_game_request(self.game_name)
                else:
                    default_name = f"{self.username}'s Game"
                    logger.info(f"Creating game '{default_name}' instead")
                    self._send_create_game_request(default_name)
        
        except Exception as e:
            logger.error(f"Error handling join game reply: {e}")
    
    def _handle_create_game_reply(self, reply):
        """
        Handle create game reply message.
        
        Args:
            reply: Create game reply message
        """
        try:
            if reply.result == pokerth_pb2.CreateGameReplyMessage.CreateGameResult.CREATE_GAME_RESULT_SUCCESS:
                self.game_id = reply.game_id
                self.joined_game = True
                logger.info(f"Successfully created and joined game {reply.game_id}")
            else:
                logger.error("Failed to create game")
                self.disconnect()
        
        except Exception as e:
            logger.error(f"Error handling create game reply: {e}")
    
    def _handle_game_message(self, game_message):
        """
        Handle game message.
        
        Args:
            game_message: Game message
        """
        try:
            if game_message.type == pokerth_pb2.GameMessage.Type.GAME_MANAGEMENT:
                self._handle_game_management(game_message.game_management_message)
            
            elif game_message.type == pokerth_pb2.GameMessage.Type.GAME_ENGINE:
                self._handle_game_engine(game_message.game_engine_message)
            
            elif game_message.type == pokerth_pb2.GameMessage.Type.GAME_PLAYER:
                self._handle_game_player(game_message.game_player_message)
            
            else:
                logger.debug(f"Unhandled game message type: {game_message.type}")
        
        except Exception as e:
            logger.error(f"Error handling game message: {e}")
    
    def _handle_game_management(self, management_message):
        """
        Handle game management message.
        
        Args:
            management_message: Game management message
        """
        try:
            if management_message.type == pokerth_pb2.GameManagementMessage.Type.GAME_MANAGEMENT_PLAYER_JOINED:
                player = management_message.player_joined
                logger.info(f"Player joined: {player.player_name} (ID: {player.player_id})")
                
                # Update player count
                self.active_players += 1
                
                # Add player to our state
                self.player_states[player.player_id] = {
                    'name': player.player_name,
                    'active': True,
                    'chips': 0,
                    'bet': 0
                }
            
            elif management_message.type == pokerth_pb2.GameManagementMessage.Type.GAME_MANAGEMENT_PLAYER_LEFT:
                player_id = management_message.player_left.player_id
                
                # Update player count
                self.active_players -= 1
                
                # Get player name if available
                player_name = self.player_states.get(player_id, {}).get('name', f"Player {player_id}")
                logger.info(f"Player left: {player_name} (ID: {player_id})")
                
                # Remove player from our state
                if player_id in self.player_states:
                    del self.player_states[player_id]
            
            elif management_message.type == pokerth_pb2.GameManagementMessage.Type.GAME_MANAGEMENT_GAME_STARTED:
                logger.info("Game has started")
        
        except Exception as e:
            logger.error(f"Error handling game management message: {e}")
    
    def _handle_game_engine(self, engine_message):
        """
        Handle game engine message.
        
        Args:
            engine_message: Game engine message
        """
        try:
            if engine_message.type == pokerth_pb2.GameEngineMessage.Type.GAME_ENGINE_PLAYER_ID_LIST:
                # Handle player ID list
                game_id = engine_message.player_id_list.game_id
                player_ids = engine_message.player_id_list.player_ids
                
                logger.info(f"Player ID list for game {game_id}: {len(player_ids)} players")
                
                # Update active player count
                self.active_players = len(player_ids)
            
            elif engine_message.type == pokerth_pb2.GameEngineMessage.Type.GAME_ENGINE_NEW_HAND_CARDS:
                # Handle new hand cards
                game_id = engine_message.new_hand_cards.game_id
                cards = engine_message.new_hand_cards.cards
                
                # Reset hand state
                self.hand_cards = []
                self.community_cards = []
                self.phase = GamePhase.PREFLOP
                
                # Convert cards to our format
                for card in cards:
                    rank, suit = self._convert_card(card)
                    self.hand_cards.append(Card(rank, suit))
                
                logger.info(f"Received {len(cards)} new hand cards: {[str(card) for card in self.hand_cards]}")
            
            elif engine_message.type == pokerth_pb2.GameEngineMessage.Type.GAME_ENGINE_DEAL_FLOP_CARDS:
                # Handle flop cards
                game_id = engine_message.deal_flop_cards.game_id
                cards = engine_message.deal_flop_cards.cards
                
                # Reset community cards
                self.community_cards = []
                
                # Convert cards to our format
                for card in cards:
                    rank, suit = self._convert_card(card)
                    self.community_cards.append(Card(rank, suit))
                
                # Update phase
                self.phase = GamePhase.FLOP
                
                logger.info(f"Flop cards: {[str(card) for card in self.community_cards]}")
            
            elif engine_message.type == pokerth_pb2.GameEngineMessage.Type.GAME_ENGINE_DEAL_TURN_CARD:
                # Handle turn card
                game_id = engine_message.deal_turn_card.game_id
                card = engine_message.deal_turn_card.card
                
                # Convert card to our format
                rank, suit = self._convert_card(card)
                turn_card = Card(rank, suit)
                
                # Add to community cards
                self.community_cards.append(turn_card)
                
                # Update phase
                self.phase = GamePhase.TURN
                
                logger.info(f"Turn card: {turn_card}")
            
            elif engine_message.type == pokerth_pb2.GameEngineMessage.Type.GAME_ENGINE_DEAL_RIVER_CARD:
                # Handle river card
                game_id = engine_message.deal_river_card.game_id
                card = engine_message.deal_river_card.card
                
                # Convert card to our format
                rank, suit = self._convert_card(card)
                river_card = Card(rank, suit)
                
                # Add to community cards
                self.community_cards.append(river_card)
                
                # Update phase
                self.phase = GamePhase.RIVER
                
                logger.info(f"River card: {river_card}")
            
            elif engine_message.type == pokerth_pb2.GameEngineMessage.Type.GAME_ENGINE_NEXT_PLAYER_TO_ACT:
                # Handle next player to act
                player_id = engine_message.next_player_to_act.player_id
                
                # Check if it's our turn
                if player_id == self.player_id:
                    logger.info("It's our turn to act")
                    
                    # Update game state
                    self.my_turn = True
                    self.current_bet = engine_message.next_player_to_act.highest_set
                    
                    # Make a decision
                    self._make_decision(
                        engine_message.next_player_to_act.highest_set,
                        engine_message.next_player_to_act.min_raise,
                        self.my_chips
                    )
                else:
                    # Get player name if available
                    player_name = self.player_states.get(player_id, {}).get('name', f"Player {player_id}")
                    logger.info(f"Player {player_name} to act next")
            
            elif engine_message.type == pokerth_pb2.GameEngineMessage.Type.GAME_ENGINE_HAND_FINISHED:
                # Handle hand finished
                logger.info("Hand finished")
                
                # Reset hand state
                self.hand_cards = []
                self.community_cards = []
                self.my_turn = False
        
        except Exception as e:
            logger.error(f"Error handling game engine message: {e}")
    
    def _handle_game_player(self, player_message):
        """
        Handle game player message.
        
        Args:
            player_message: Game player message
        """
        try:
            if player_message.type == pokerth_pb2.GamePlayerMessage.Type.GAME_PLAYER_ACTION:
                # Handle player action
                player_id = player_message.player_action.player_id
                action_type = player_message.player_action.action_type
                
                # Get player name if available
                player_name = self.player_states.get(player_id, {}).get('name', f"Player {player_id}")
                
                # Log action
                if action_type == pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_FOLD:
                    logger.info(f"Player {player_name} folds")
                    
                    # Update player state
                    if player_id in self.player_states:
                        self.player_states[player_id]['active'] = False
                
                elif action_type == pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_CHECK:
                    logger.info(f"Player {player_name} checks")
                
                elif action_type == pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_CALL:
                    logger.info(f"Player {player_name} calls")
                
                elif action_type == pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_BET:
                    bet_amount = player_message.player_action.bet_amount
                    logger.info(f"Player {player_name} bets {bet_amount}")
                    
                    # Update game state
                    self.current_bet = bet_amount
                
                elif action_type == pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_RAISE:
                    raise_amount = player_message.player_action.raise_amount
                    logger.info(f"Player {player_name} raises to {raise_amount}")
                    
                    # Update game state
                    self.current_bet = raise_amount
                
                elif action_type == pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_ALL_IN:
                    logger.info(f"Player {player_name} goes all-in")
        
        except Exception as e:
            logger.error(f"Error handling game player message: {e}")
    
    def _handle_error(self, error_message):
        """
        Handle error message.
        
        Args:
            error_message: Error message
        """
        try:
            error_code = error_message.error_code
            
            # Log error
            logger.error(f"Server error: code {error_code}")
            
            # Handle specific errors
            if error_code == pokerth_pb2.ErrorMessage.ErrorCode.ERROR_INVALID_STATE:
                logger.error("Invalid state")
            elif error_code == pokerth_pb2.ErrorMessage.ErrorCode.ERROR_INVALID_PACKET:
                logger.error("Invalid packet")
            
            # For protocol errors, retry with guest login
            if error_code == pokerth_pb2.ErrorMessage.ErrorCode.ERROR_INVALID_PACKET and not self.is_guest:
                logger.info("Trying guest login instead...")
                self.is_guest = True
                self._send_guest_auth_request()
        
        except Exception as e:
            logger.error(f"Error handling error message: {e}")
    
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
            
            # Convert our action to PokerTH action type
            pokerth_action = self._convert_action_type(action)
            
            # Send action to server
            self._send_player_action(pokerth_action, amount)
            
            # Reset turn flag
            self.my_turn = False
        
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            
            # Default to safest action (fold)
            self._send_player_action(pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_FOLD)
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
    
    def _convert_action_type(self, action: Action) -> int:
        """
        Convert our Action enum to PokerTH action type.
        
        Args:
            action (Action): Our action
            
        Returns:
            int: PokerTH action type
        """
        action_map = {
            Action.FOLD: pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_FOLD,
            Action.CHECK: pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_CHECK,
            Action.CALL: pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_CALL,
            Action.BET: pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_BET,
            Action.RAISE: pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_RAISE,
            Action.ALL_IN: pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_ALL_IN
        }
        
        return action_map.get(action, pokerth_pb2.PlayerAction.ActionType.PLAYER_ACTION_FOLD)
    
    def _convert_card(self, card) -> Tuple[Rank, Suit]:
        """
        Convert PokerTH card to our format.
        
        Args:
            card: PokerTH card
            
        Returns:
            Tuple[Rank, Suit]: Card rank and suit in our format
        """
        # PokerTH uses 0-12 for card values (0=2, 12=Ace)
        # Our Rank enum uses 2-14 (2=Two, 14=Ace)
        rank_value = card.card_value + 2
        rank = Rank(rank_value)
        
        # PokerTH uses 0-3 for suits
        # Map to our Suit enum
        suit_map = {
            0: Suit.CLUBS,
            1: Suit.DIAMONDS,
            2: Suit.HEARTS,
            3: Suit.SPADES
        }
        suit = suit_map.get(card.card_suit, Suit.SPADES)
        
        return rank, suit
    
    def _get_auth_error_message(self, error_code: int) -> str:
        """
        Get a human-readable message for an authentication error code.
        
        Args:
            error_code (int): Error code
            
        Returns:
            str: Error message
        """
        error_messages = {
            pokerth_pb2.AuthServerVerificationMessage.VerificationErrorReason.INVALID_PASSWORD: "Invalid password",
            pokerth_pb2.AuthServerVerificationMessage.VerificationErrorReason.INVALID_USERNAME: "Invalid username",
            pokerth_pb2.AuthServerVerificationMessage.VerificationErrorReason.INVALID_VERSION: "Invalid client version",
            pokerth_pb2.AuthServerVerificationMessage.VerificationErrorReason.ALREADY_LOGGED_IN: "Already logged in"
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
            pokerth_pb2.JoinGameReplyMessage.JoinGameResult.JOIN_GAME_RESULT_INVALID_GAME_ID: "Invalid game ID",
            pokerth_pb2.JoinGameReplyMessage.JoinGameResult.JOIN_GAME_RESULT_GAME_FULL: "Game is full",
            pokerth_pb2.JoinGameReplyMessage.JoinGameResult.JOIN_GAME_RESULT_ALREADY_STARTED: "Game already started",
            pokerth_pb2.JoinGameReplyMessage.JoinGameResult.JOIN_GAME_RESULT_INVALID_PASSWORD: "Invalid password"
        }
        
        return error_messages.get(error_code, f"Unknown error ({error_code})")
    
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
                
                # Sleep a bit to avoid busy-waiting
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            logger.info("Client interrupted by user")
        finally:
            self.disconnect()


def main():
    """Main entry point for the PokerTH client."""
    parser = argparse.ArgumentParser(description='PokerTH Client with Protocol Buffers')
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
    client = PokerTHProtobufClient(
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