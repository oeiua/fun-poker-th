import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
from config import INPUT_SIZE, HIDDEN_LAYERS, OUTPUT_SIZE, LEARNING_RATE, DEVICE

class PokerNet(nn.Module):
    def __init__(self):
        super(PokerNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(INPUT_SIZE, HIDDEN_LAYERS[0]))
        
        # Hidden layers
        for i in range(len(HIDDEN_LAYERS) - 1):
            self.layers.append(nn.Linear(HIDDEN_LAYERS[i], HIDDEN_LAYERS[i+1]))
        
        # Output layer
        self.layers.append(nn.Linear(HIDDEN_LAYERS[-1], OUTPUT_SIZE))
        
        # Move to device
        self.to(DEVICE)
    
    def forward(self, x):
        x = x.to(DEVICE)
        
        for i in range(len(self.layers) - 1):
            x = F.relu(self.layers[i](x))
        
        # Output layer without activation (for fold)
        # and with sigmoid (for call, raise percentage)
        logits = self.layers[-1](x)
        fold_prob = torch.sigmoid(logits[:, 0]).unsqueeze(1)
        call_raise = torch.softmax(logits[:, 1:], dim=1)
        
        return torch.cat((fold_prob, call_raise), dim=1)
    
    def decide_action(self, state_tensor, deterministic=False):
        with torch.no_grad():
            output = self.forward(state_tensor)
            
            if deterministic:
                # Deterministic decision (for evaluation)
                fold_prob = output[0, 0].item()
                if fold_prob > 0.5:
                    return {"action": "fold"}
                
                call_prob = output[0, 1].item()
                raise_prob = output[0, 2].item()
                
                if call_prob > raise_prob:
                    return {"action": "call"}
                else:
                    raise_amount = raise_prob * state_tensor[0, -1].item()  # Last input is max possible raise
                    return {"action": "raise", "amount": max(2, int(raise_amount))}
            else:
                # Stochastic decision (for training)
                fold_prob = output[0, 0].item()
                if random.random() < fold_prob:
                    return {"action": "fold"}
                
                call_prob = output[0, 1].item() / (output[0, 1].item() + output[0, 2].item())
                
                if random.random() < call_prob:
                    return {"action": "call"}
                else:
                    raise_amount = output[0, 2].item() * state_tensor[0, -1].item()
                    return {"action": "raise", "amount": max(2, int(raise_amount))}

def create_state_tensor(player, game):
    """Convert game state to a tensor for neural network input"""
    state = []
    
    # Card encoding: 52 positions for cards, 1 is present, 0 is not
    card_encoding = [0] * 52
    
    # Helper to convert a card to an index
    def card_to_index(card):
        suits = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}
        values = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6, 
                  '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
        return values[card.value] * 4 + suits[card.suit]
    
    # Encode hole cards
    for card in player.hand:
        card_encoding[card_to_index(card)] = 1
    
    # Encode community cards
    for card in game.community_cards:
        card_encoding[card_to_index(card)] = 1
    
    state.extend(card_encoding)
    
    # Player position relative to dealer
    player_pos = (player.player_id - game.dealer_pos) % len(game.players)
    position_encoding = [0] * len(game.players)
    position_encoding[player_pos] = 1
    state.extend(position_encoding)
    
    # Pot and current bet
    state.append(game.pot / 1000.0)  # Normalize
    state.append(game.current_bet / 1000.0)
    
    # Player's chips
    state.append(player.chips / 1000.0)
    
    # Opponent chip stacks (anonymized)
    opponent_chips = sorted([p.chips for p in game.players if p.player_id != player.player_id])
    while len(opponent_chips) < len(game.players) - 1:
        opponent_chips.append(0)  # Pad if some players are out
    state.extend([c / 1000.0 for c in opponent_chips])
    
    # Player's current bet
    state.append(player.current_bet / 1000.0)
    
    # Amount to call
    amount_to_call = game.current_bet - player.current_bet
    state.append(amount_to_call / 1000.0)
    
    # Number of active players
    active_players = sum(1 for p in game.players if not p.folded)
    state.append(active_players / len(game.players))
    
    # Max possible raise (used for scaling the raise output)
    max_raise = player.chips
    state.append(max_raise / 1000.0)
    
    # Ensure the state vector has exactly INPUT_SIZE elements
    # Pad or truncate if necessary to match INPUT_SIZE
    if len(state) < INPUT_SIZE:
        state.extend([0] * (INPUT_SIZE - len(state)))
    elif len(state) > INPUT_SIZE:
        state = state[:INPUT_SIZE]
        
    return torch.tensor([state], dtype=torch.float32)

class EvolutionTrainer:
    def __init__(self, population_size=10, mutation_rate=0.1, crossover_rate=0.7):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = [PokerNet() for _ in range(population_size)]
        self.fitness_history = []
    
    def evaluate_fitness(self, game, tournaments=5):
        fitness_scores = [0] * self.population_size
        
        for _ in range(tournaments):
            # Play a tournament with all models
            for i in range(self.population_size):
                # Set up a game where one model plays against random copies of other models
                game.setup_game(ai_players=game.num_players)
                
                # Store the original method
                original_method = game.get_player_action
                
                # Define a new method that doesn't call the original one
                def get_action_wrapper(player, current_bet):
                    if player.player_id == 0:  # This is our targeted player
                        # Use the neural network
                        state_tensor = create_state_tensor(player, game)
                        return self.population[i].decide_action(state_tensor, deterministic=True)
                    else:
                        # Other players use a simple random strategy directly
                        actions = ["fold", "call", "raise"]
                        action = random.choice(actions)
                        
                        if action == "raise":
                            min_raise = current_bet * 2
                            max_raise = player.chips
                            amount = random.randint(min_raise, max(min_raise, max_raise))
                            return {"action": action, "amount": amount}
                        else:
                            return {"action": action}
                
                # Replace the method
                game.get_player_action = get_action_wrapper
                
                try:
                    # Play the tournament
                    winners = game.play_tournament(max_hands=100)
                    
                    # Score is based on final position and remaining chips
                    if winners and winners[0].player_id == 0:
                        fitness_scores[i] += 10 + (winners[0].chips / 1000)
                    elif len(winners) > 1:
                        # Find our player's rank
                        for j, winner in enumerate(winners):
                            if winner.player_id == 0:
                                fitness_scores[i] += (len(winners) - j) / len(winners) * 5
                                break
                finally:
                    # Restore the original method to avoid memory leaks
                    game.get_player_action = original_method
        
        # Normalize fitness scores
        if sum(fitness_scores) > 0:
            fitness_scores = [score / sum(fitness_scores) for score in fitness_scores]
        else:
            fitness_scores = [1.0 / self.population_size] * self.population_size
        
        self.fitness_history.append(fitness_scores)
        return fitness_scores
    
    def select_parent(self, fitness_scores):
        # Tournament selection
        indices = random.sample(range(self.population_size), 3)
        tournament = [(i, fitness_scores[i]) for i in indices]
        tournament.sort(key=lambda x: x[1], reverse=True)
        return tournament[0][0]
    
    def crossover(self, parent1, parent2):
        child = PokerNet()
        
        # Crossover parameters
        for child_param, parent1_param, parent2_param in zip(
            child.parameters(), parent1.parameters(), parent2.parameters()
        ):
            if random.random() < self.crossover_rate:
                # Perform crossover at parameter level
                crossover_point = random.randint(0, child_param.data.numel() - 1)
                flat_p1 = parent1_param.data.flatten()
                flat_p2 = parent2_param.data.flatten()
                flat_child = child_param.data.flatten()
                
                flat_child[:crossover_point] = flat_p1[:crossover_point]
                flat_child[crossover_point:] = flat_p2[crossover_point:]
                
                child_param.data = flat_child.reshape(child_param.data.shape)
            else:
                # No crossover, inherit from first parent
                child_param.data = parent1_param.data.clone()
        
        return child
    
    def mutate(self, model):
        for param in model.parameters():
            if random.random() < self.mutation_rate:
                # Add Gaussian noise to parameters
                noise = torch.randn_like(param.data) * 0.1
                param.data += noise
    
    def evolve(self, game):
        # Evaluate fitness of current population
        fitness_scores = self.evaluate_fitness(game)
        
        # Sort population by fitness
        sorted_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)
        sorted_population = [self.population[i] for i in sorted_indices]
        
        # Elitism: keep top performers
        new_population = [sorted_population[0]]  # Keep the best model
        
        # Create rest of new population
        while len(new_population) < self.population_size:
            # Select parents
            parent1_idx = self.select_parent(fitness_scores)
            parent2_idx = self.select_parent(fitness_scores)
            
            # Create child through crossover
            child = self.crossover(self.population[parent1_idx], self.population[parent2_idx])
            
            # Mutate child
            self.mutate(child)
            
            # Add to new population
            new_population.append(child)
        
        self.population = new_population
        return sorted_population[0], fitness_scores[sorted_indices[0]]  # Return best model and its fitness