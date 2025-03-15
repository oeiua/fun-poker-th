import os
import torch
import matplotlib
# Force matplotlib to use a non-interactive backend to avoid Tkinter issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
from poker_game import PokerGame
from neural_network import PokerNet, EvolutionTrainer, create_state_tensor
from config import (
    POPULATION_SIZE, MUTATION_RATE, CROSSOVER_RATE, NUM_GENERATIONS,
    CHECKPOINT_INTERVAL, CHECKPOINT_DIR, RESULTS_DIR, NUM_PLAYERS, STARTING_CHIPS,
    DEVICE
)

def save_model(model, generation, fitness):
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_gen_{generation}.pt")
    torch.save({
        'generation': generation,
        'model_state_dict': model.state_dict(),
        'fitness': fitness
    }, checkpoint_path)
    
    # Also save as best model if it's the latest
    best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")
    torch.save({
        'generation': generation,
        'model_state_dict': model.state_dict(),
        'fitness': fitness
    }, best_model_path)

def load_latest_model():
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("model_gen_")]
    
    if not checkpoints:
        return None, 0, 0
    
    # Find the latest generation
    latest_gen = max([int(f.split("_")[-1].split(".")[0]) for f in checkpoints])
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_gen_{latest_gen}.pt")
    
    checkpoint = torch.load(checkpoint_path)
    model = PokerNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, checkpoint['generation'], checkpoint['fitness']

def plot_fitness_history(fitness_history, generation):
    plt.figure(figsize=(12, 6))
    
    # Plot best fitness per generation
    best_fitness = [max(gen_fitness) for gen_fitness in fitness_history]
    plt.plot(range(1, len(best_fitness) + 1), best_fitness, 'b-', label='Best Fitness')
    
    # Plot average fitness per generation
    avg_fitness = [sum(gen_fitness) / len(gen_fitness) for gen_fitness in fitness_history]
    plt.plot(range(1, len(avg_fitness) + 1), avg_fitness, 'r-', label='Average Fitness')
    
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.title(f'Fitness History (Generation 1-{generation})')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(RESULTS_DIR, f'fitness_history_{generation}.png'))
    plt.close()

def train_poker_ai():
    print("Initializing training environment...")
    game = PokerGame(num_players=NUM_PLAYERS, starting_chips=STARTING_CHIPS)
    
    # Try to load existing model
    best_model, start_generation, best_fitness = load_latest_model()
    
    if best_model:
        print(f"Loaded model from generation {start_generation} with fitness {best_fitness}")
        # Initialize trainer with loaded model
        trainer = EvolutionTrainer(
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE
        )
        trainer.population[0] = best_model  # Replace first model with loaded model
    else:
        print("Starting training from scratch")
        trainer = EvolutionTrainer(
            population_size=POPULATION_SIZE,
            mutation_rate=MUTATION_RATE,
            crossover_rate=CROSSOVER_RATE
        )
        start_generation = 0
        best_fitness = 0
    
    print(f"Training with population size: {POPULATION_SIZE}, generations: {NUM_GENERATIONS}")
    print(f"Using device: {DEVICE}")
    
    # Training loop
    for gen in range(start_generation + 1, start_generation + NUM_GENERATIONS + 1):
        print(f"\nStarting generation {gen}/{start_generation + NUM_GENERATIONS}...")
        start_time = time.time()
        
        try:
            # Evolve population
            best_model, fitness = trainer.evolve(game)
            
            elapsed_time = time.time() - start_time
            
            print(f"Generation {gen} | Best Fitness: {fitness:.4f} | Time: {elapsed_time:.2f}s")
            
            # Update best fitness
            if fitness > best_fitness:
                best_fitness = fitness
                print(f"New best fitness achieved: {best_fitness:.4f}")
            
            # Save checkpoint
            if gen % CHECKPOINT_INTERVAL == 0:
                save_model(best_model, gen, fitness)
                plot_fitness_history(trainer.fitness_history, gen)
                print(f"Checkpoint saved at generation {gen}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current progress...")
            save_model(best_model, gen, best_fitness)
            plot_fitness_history(trainer.fitness_history, gen)
            print(f"Progress saved at generation {gen}")
            return best_model
        except Exception as e:
            print(f"Error during generation {gen}: {e}")
            import traceback
            traceback.print_exc()
            print("Attempting to continue with next generation...")
    
    # Save final model
    print("\nTraining complete! Saving final model...")
    save_model(best_model, start_generation + NUM_GENERATIONS, best_fitness)
    plot_fitness_history(trainer.fitness_history, start_generation + NUM_GENERATIONS)
    
    return best_model

if __name__ == "__main__":
    best_model = train_poker_ai()
    print("Training complete!")