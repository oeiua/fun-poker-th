# Evolutionary Neural Network Poker AI

This project implements a comprehensive poker AI system using evolutionary algorithms and neural networks to learn optimal Texas Hold'em strategies. The system uses the `pokerkit` library for game mechanics and implements an evolutionary training approach where neural networks compete against each other and evolve over generations.

## Features

- **Neural Network Decision Making**: Modern neural networks make strategic poker decisions
- **Evolutionary Training**: Populations of agents evolve through selection, crossover, and mutation
- **Multi-Agent Competition**: Agents compete against each other to improve strategies
- **Resource Optimization**: Efficient CPU/GPU utilization for faster training
- **Progress Monitoring**: Detailed training visualizations and metrics
- **Checkpointing**: Save and resume training from checkpoints
- **Human vs AI Play**: Play against the trained AI
- **Configurable**: All important parameters easily adjustable through YAML configuration

## Requirements

- python3 3.8+
- TensorFlow 2.x
- pokerkit 0.5.0+
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/poker-ai.git
cd poker-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Configuration

Edit the `config/config.yaml` file to adjust parameters for:
- Game mechanics (blinds, stacks, etc.)
- Neural network architecture
- Training parameters
- Resource usage

### Training

Train the AI using the following command:

```bash
python3 main.py --mode train --config config/config.yaml
```

To continue training from a checkpoint:

```bash
python3 main.py --mode train --config config/config.yaml --checkpoint checkpoints/generation_50
```

### Playing Against the AI

Play against the trained AI:

```bash
python3 main.py --mode play --config config/config.yaml --checkpoint checkpoints/final
```

### Evaluation

Evaluate the performance of a trained model:

```bash
python3 main.py --mode evaluate --config config/config.yaml --checkpoint checkpoints/final
```

## Project Structure

```
poker_ai/
├── config/
│   └── config.yaml         # Configuration file
├── src/
│   ├── agent/              # Agent implementations
│   ├── environment/        # Game environment
│   ├── models/             # Neural network models
│   ├── training/           # Training components
│   ├── utils/              # Utility functions
│   └── main.py             # Entry point
├── checkpoints/            # Saved models
├── logs/                   # Training logs
└── README.md               # This file
```

## Training Methodology

The AI is trained using a combination of evolutionary algorithms and reinforcement learning:

1. **Initialization**: Create a population of randomly initialized neural networks
2. **Evaluation**: Agents play poker against each other to determine fitness
3. **Selection**: Higher-performing agents are more likely to be selected for reproduction
4. **Crossover**: Combine weights from two parent networks to create offspring
5. **Mutation**: Randomly modify weights to explore new strategies
6. **Iteration**: Repeat the process over many generations

This approach allows the AI to discover effective poker strategies without human guidance or labeled training data.

## Neural Network Architecture

The neural network takes the game state as input (including cards, pot size, player stacks, etc.) and outputs action values for folding, checking/calling, and betting/raising. The architecture includes:

- Input layer for game state features
- Multiple hidden layers with ReLU activation
- Output layer for action values
- Dropout for regularization
- Batch normalization for stable training

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [pokerkit](https://github.com/uoftcprg/pokerkit) for poker game mechanics
- TensorFlow team for the deep learning framework