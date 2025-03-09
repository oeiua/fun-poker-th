# Evolutionary Poker AI

A comprehensive poker AI system using Python, TensorFlow, and evolutionary algorithms to simulate Texas Hold'em games with neural networks. The AI models compete against each other and improve strategies over time through evolutionary learning.

## Features

- **Neural Network AI**: Uses TensorFlow to make betting decisions based on game state
- **Evolutionary Training**: Models improve over generations through selection, crossover, and mutation
- **Complete Game Logic**: Implements full Texas Hold'em rules with proper betting rounds
- **Interactive Play**: Human players can compete against trained AI models
- **Visualization Tools**: Monitor training progress and game results with visual analytics

## Project Structure

```
.
├── main.py                  # Entry point for the application
├── config.py                # Configuration settings and game constants
├── card.py                  # Card and deck classes
├── evaluator.py             # Poker hand evaluation functionality
├── model.py                 # Neural network model for the AI
├── player.py                # Player classes (human and AI)
├── game_engine.py           # Core game management and rules
├── evolutionary_trainer.py  # Evolutionary algorithm implementation
├── visualizer.py            # Training and game visualization tools
├── utils.py                 # Utility functions
├── models/                  # Directory for saved models
├── logs/                    # Training logs and checkpoints
└── README.md                # This documentation file
```

## Technical Details

### Neural Network Architecture

The AI uses a neural network with the following input features:
- Player position and chip state
- Hole cards and community cards
- Pot size and current bet
- Game phase (pre-flop, flop, turn, river)
- Number of active players

The output layer produces action probabilities for:
- Fold
- Check
- Call
- Bet
- Raise
- All-in

### Evolutionary Algorithm

The training process uses:
- Tournament selection for parent selection
- Elitism to preserve the best models
- Crossover between successful models
- Mutation to explore new strategies
- Fitness evaluation based on tournament performance

## Usage

### Installation

1. Clone the repository
2. Install dependencies:
```bash
python3 -m venv poker_env
pip install -r requirements.txt
```

### Training the AI

```bash
python3 main.py --mode train --generations 100 --population 50
```

Training parameters:
- `--generations`: Number of generations for evolutionary training
- `--population`: Population size
- `--games`: Number of games per evaluation
- `--verbose`: Enable detailed output
- `--checkpoint`: Continue training from checkpoint

### Playing Against the AI

```bash
python3 main.py --mode play --num_players 10
```

Play parameters:
- `--num_players`: Total number of players (including human)
- `--initial_chips`: Starting chips for each player
- `--model_path`: Path to load trained models

### Evaluating Models

```bash
python3 main.py --mode evaluate --games 100
```

Evaluation parameters:
- `--games`: Number of games to evaluate
- `--verbose`: Show detailed results

## Game Rules

The implementation follows standard Texas Hold'em rules:

1. **Pre-flop**: Each player receives two hole cards, betting starts left of big blind
2. **Flop**: Three community cards are dealt, betting starts left of dealer
3. **Turn**: Fourth community card is dealt, another betting round
4. **River**: Final community card is dealt, final betting round
5. **Showdown**: Best five-card hand wins the pot

## Future Improvements

- Implement multi-table tournaments
- Add support for different poker variants
- Incorporate reinforcement learning techniques
- Create a web-based interface for human players
- Implement player profiling for adaptive strategies

## License

This project is available under the MIT License. See LICENSE file for details.
