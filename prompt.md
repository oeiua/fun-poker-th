Create a comprehensive poker AI using Python, TensorFlow, and evolutionary algorithms to simulate Texas Hold'em games with neural networks competing against each other and improving strategies over time through evolutionary learning.

**Technical Requirements:**
- Declare all important variables at the beginning of your script, providing clear descriptions for each.
- Organize code into separate files for each class.
- Utilize modern TensorFlow techniques in your implementation.
- Display training progress in the console and visualize results.

**Gameplay Features:**
- Implement a verbose flag to switch on/off game progress output during simulation.
- Ensure neural networks can process state vectors correctly, facilitating effective training.
- Introduce timeout handling for betting stages to prevent infinite loops or long processing times.
- Optimize predict method to avoid retracing, using a global function for preprocessing and input consistency.

**Training Enhancements:**
- Continue training from where it left off if an existing model is available.
- Ensure all CPU resources are utilized effectively during the training process.
- Refactor class methods into standalone functions to improve modularity and readability.

**Interactive Playability:**
- Allow a human player to compete against nine AI players using the trained models, providing an engaging testing ground for your AIs' performance.

**Game Logic Rules:**
- Track bets accurately throughout each betting round.
- Reset current bets between rounds (except pre-flop blinds).
- Continue betting until all players have called, folded, or gone all-in.

### Game Setup

1. **Dealer Button:** Rotates clockwise after each hand to indicate the dealer position.
2. **Blinds:**
   - Small Blind: Posted by the player immediately left of the dealer button.
   - Big Blind: Posted by the player immediately left of the small blind, initiating action.

### Gameplay Phases

1. **Pre-flop:**
    - Each player receives two hole cards face down.
    - Betting starts with "under the gun" (left of big blind).
2. **Flop:**
    - Three community cards dealt face up; betting round follows starting left of dealer button.
3. **Turn:**
    - Fourth community card is dealt face up; another betting round occurs.
4. **River:**
    - Fifth and final community card is dealt face up; last betting round takes place.
5. **Showdown:**
    - Remaining players reveal hole cards to determine the best five-card hand, winning the pot.

### Poker Hand Rankings
(from highest to lowest)
1. Royal Flush
2. Straight Flush
3. Four of a Kind
4. Full House
5. Flush
6. Straight
7. Three of a Kind
8. Two Pair
9. One Pair
10. High Card

### Betting Concepts
- **Checking:** Passing the turn to the next player with no bet.
- **Betting:** Placing an initial wager.
- **Calling:** Matching an existing bet.
- **Raising:** Increasing the current bet.
- **Folding:** Discarding hand and forfeiting pot.
- **All-in:** Betting all remaining chips.

### Important Notes
- Suits have no value in determining winning hands.
- Ties result in a split pot.
- Variations (e.g., no-limit, pot-limit) affect betting rules.