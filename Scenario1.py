#  Agent must collect 1 package located somewhere in the environment
#  Agent starts at a random position in the environment
#  Agent can move in 4 directions: UP, DOWN, LEFT, RIGHT

# Maphuti Shilabje

from FourRooms import FourRooms
import numpy as np
import random

GRID_COLS = 11
GRID_ROWS = 11


ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
NUM_ACTIONS = len(ACTIONS)


# The learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
NUM_EPOCHS = 1000
STEPS_PER_EPOCH = 250

# The exploration parameters
# Epsilon-greedy exploration
INITIAL_EPSILON = 0.9 
MIN_EPSILON = 0.1
EPSILON_DECAY_VALUE = (MIN_EPSILON / INITIAL_EPSILON) ** (1.0 / (NUM_EPOCHS* 0.8)) # Decay rate for epsilon

q_table = np.zeros((GRID_ROWS, GRID_COLS, NUM_ACTIONS, 2, NUM_ACTIONS)) # Initialize Q-table

def get_state_index(pos, packages_remaining):
    """
    Convert the agent's position and packages remaining to a state index.
    """
    x = pos[0] - 1    
    y = pos[1] - 1

    # Ensure the position is within the grid bounds

    x = np.clip(x, 0, GRID_ROWS - 1) # Ensure x is within bounds
    y = np.clip(y, 0, GRID_COLS - 1) # Ensure y is within bounds

    packages = packages_remaining 

    return (pos[0], pos[1], packages)


def choose_action(state, current_epoch):
    """
    Selects an action using an epsilon-greedy policy.
    With probability epsilon, a random action is chosen (exploration).
    """
    if random.random() < current_epoch:
        # Explore: choose a random action index
        return random.randrange(NUM_ACTIONS)
    else:
        # Exploit: choose the action with the highest Q-value
        # for the current state
        return np.argmax(q_table[state])

def get_reward(prev_package_count, curr_package_count, is_terminal_state):
    """
    Calculates the reward for the agent's last action.
    - prev_package_count: Number of packages before the action.
    - curr_package_count: Number of packages after the action.
    - is_terminal_state: Boolean indicating if the new state is terminal.
    """
    reward = 0
    # Penalty for each step to encourage efficiency
    reward -= 1 

    if curr_package_count < prev_package_count:
        # Agent picked up a package
        reward += 100  # Large positive reward for picking up the package
    
    if is_terminal_state:
        if curr_package_count == 0:
            reward += 200 # Large positive reward for reaching the terminal state with all packages collected
        else:
            reward -= 50 # Penalty if terminal for other reasons (e.g. error)
            
    return reward


def train_agent():

    print("Training agent...")
    fourRoomsObj = FourRooms('simple') # Create FourRooms Object

    current_epsilon = INITIAL_EPSILON

    epoch_rewards = []
    epoch_steps = []

    print(f"Q-table shape: {q_table.shape}")
    print(f"Training agent with {NUM_EPOCHS} epochs and {STEPS_PER_EPOCH} steps per epoch...")

    for epoch in range(NUM_EPOCHS):
        fourRoomsObj.newEpoch()  # Reset environment to a new random start state

        # Loop through the steps in the epoch
        if current_epsilon > MIN_EPSILON:
            current_epsilon *= EPSILON_DECAY_VALUE
            current_epsilon = max(MIN_EPSILON, current_epsilon) # Ensure it doesn't go below min

        if (epoch + 1) % 100 == 0: # Log progress for every 100 epochs
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} completed. Epsilon: {current_epsilon:.4f}")
        

    print("Training complete.")
    print("displaying path from last epoch")

    fourRoomsObj.showPath(-1,"scenario1_final_path.png")  # Show the path taken in the last epoch
    print("Path image saved to scenario1_final_path_placeholder.png")


if __name__ == "__main__":
    train_agent()
