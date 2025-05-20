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

def get_reward(prev_package_count, curr_package_count, is_terminal_state, grid_cell_type):
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

        is_terminal = False
        steps_this_epoch = 0
        total_reward_this_epoch = 0

        current_pos = fourRoomsObj.getPosition()
        current_packages_left_val = fourRoomsObj.getPackagesRemaining()
        current_state_idx = get_state_index(current_pos, current_packages_left_val)

        while not is_terminal and steps_this_epoch < STEPS_PER_EPOCH:
            # 1. Choose action
            action_idx = choose_action(current_state_idx, current_epsilon)
            action_to_take = ACTIONS[action_idx] 
            
            old_packages_left_val = current_state_idx[2] # Packages before action, from current Q-state index

            # 2. Take action in environment
            # takeAction returns: (gridCellType, newPos, packagescount, isTerminal)
            grid_cell_type, new_pos, new_packages_left_val, terminal_from_env = fourRoomsObj.takeAction(action_to_take)
            
            # 3. Calculate Reward
            reward = get_reward(old_packages_left_val, new_packages_left_val, terminal_from_env, grid_cell_type)
            total_reward_this_epoch += reward

            # 4. Observe new state
            next_state_idx = get_state_index(new_pos, new_packages_left_val)

            # 5. Update Q-Table (Q-learning formula)
            old_q_value = q_table[current_state_idx + (action_idx,)] # Q(s,a)
            
            # Best Q-value for the next state: Q_max(s',a')
            # If the next state is terminal, there are no future actions, so next_max_q is 0.
            if terminal_from_env:
                next_max_q = 0.0
            else:
                next_max_q = np.max(q_table[next_state_idx])
            
            # Q-update rule: Q(s,a) <- Q(s,a) + alpha * (R + gamma * max_a' Q(s',a') - Q(s,a))
            new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q_value)
            q_table[current_state_idx + (action_idx,)] = new_q_value # Update Q-table
            
            # 6. Update current state and step count
            current_state_idx = next_state_idx
            is_terminal = terminal_from_env
            steps_this_epoch += 1
        
        epoch_rewards.append(total_reward_this_epoch)
        epoch_steps.append(steps_this_epoch)

        

        # Loop through the steps in the epoch
        if current_epsilon > MIN_EPSILON:
            current_epsilon *= EPSILON_DECAY_VALUE
            current_epsilon = max(MIN_EPSILON, current_epsilon) # Ensure it doesn't go below min

        if (epoch + 1) % 100 == 0: # Log progress for every 100 epochs
            avg_reward_last_100 = np.mean(epoch_rewards[-100:])
            avg_steps_last_100 = np.mean(epoch_steps[-100:])
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Steps: {steps_this_epoch} | Epsilon: {current_epsilon:.4f} | Last Reward: {total_reward_this_epoch} | Avg Reward (100): {avg_reward_last_100:.2f} | Avg Steps (100): {avg_steps_last_100:.2f}")
        
            if is_terminal and new_packages_left_val == 0:
                 print(f"  SUCCESS: Package collected in {steps_this_epoch} steps.")
            elif steps_this_epoch >= STEPS_PER_EPOCH:
                 print(f"  TIMEOUT: Reached max steps.")

    print("Training complete.")
    print("displaying path from last epoch")

    fourRoomsObj.showPath(-1,"scenario1_final_path.png")  # Show the path taken in the last epoch
    print("Path image saved to scenario1_final_path_placeholder.png")


if __name__ == "__main__":
    train_agent()
