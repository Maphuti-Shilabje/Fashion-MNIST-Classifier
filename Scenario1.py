# Scenario1.py
# Agent must collect 1 package located somewhere in the environment
# Agent starts at a random position in the environment
# Agent can move in 4 directions: UP, DOWN, LEFT, RIGHT

# Maphuti Shilabje

from FourRooms import FourRooms
import numpy as np
import random

GRID_COLS = 11  # Effective traversable columns (indices 0-10)
GRID_ROWS = 11  # Effective traversable rows (indices 0-10)

ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
NUM_ACTIONS = len(ACTIONS)

# The learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
NUM_EPOCHS = 2000  # Increased for potentially better convergence
STEPS_PER_EPOCH = 250

EXPLORATION_STRATEGY = 'linear' # or 'linear'

# The exploration parameters
# Epsilon-greedy exploration
INITIAL_EPSILON = 1.0 # Start with full exploration
MIN_EPSILON = 0.01    # Minimum exploration rate
# Adjusted decay to aim for MIN_EPSILON over ~80% of epochs

if EXPLORATION_STRATEGY == 'multiplicative':
    EPSILON_DECAY_VALUE = (MIN_EPSILON / INITIAL_EPSILON) ** (1.0 / (NUM_EPOCHS * 0.8))
elif EXPLORATION_STRATEGY == 'linear':
    EPSILON_DECAY_STEPS = NUM_EPOCHS * 0.8
    EPSILON_DECREMENT = (INITIAL_EPSILON - MIN_EPSILON) / (NUM_EPOCHS * 0.8)
else:
    raise ValueError("Invalid exploration strategy. Choose 'multiplicative' or 'linear'.")

# Initialize Q-table
# State: (x_idx, y_idx, package_remaining_idx)
# package_remaining_idx: 0 (no packages left), 1 (1 package left)
# Shape: (GRID_ROWS, GRID_COLS, num_package_states, NUM_ACTIONS)
q_table = np.zeros((GRID_ROWS, GRID_COLS, 2, NUM_ACTIONS)) # Corrected Q-table shape

def get_state_index(pos, packages_remaining):
    """
    Convert the agent's 1-indexed position and packages remaining
    to a 0-indexed state tuple suitable for Q-table indexing.
    """
    x_idx = pos[0] - 1    # Convert from 1-based (env) to 0-based (q-table)
    y_idx = pos[1] - 1    # Convert from 1-based (env) to 0-based (q-table)

    # Ensure the 0-indexed position is within the grid bounds
    x_idx = np.clip(x_idx, 0, GRID_ROWS - 1)
    y_idx = np.clip(y_idx, 0, GRID_COLS - 1)

    # packages_remaining is already 0 or 1, which is suitable for indexing
    package_idx = packages_remaining 

    return (x_idx, y_idx, package_idx) # Return 0-indexed state tuple

def choose_action(state, current_epsilon): # Corrected parameter name
    """
    Selects an action using an epsilon-greedy policy.
    'state' is the 0-indexed tuple (x_idx, y_idx, package_idx).
    """
    if random.random() < current_epsilon:
        # Explore: choose a random action index
        return random.randrange(NUM_ACTIONS)
    else:
        # Exploit: choose the action with the highest Q-value for the current state
        # q_table[state] retrieves the 1D array of Q-values for all actions in that state
        return np.argmax(q_table[state])

def get_reward(prev_package_count, curr_package_count, is_terminal_state, grid_cell_type):
    """
    Calculates the reward for the agent's last action.
    """
    reward = 0
    # Penalty for each step to encourage efficiency
    reward -= 1 

    if curr_package_count < prev_package_count:
        # Agent picked up a package
        reward += 100  # Large positive reward for picking up the package
    
    if is_terminal_state:
        if curr_package_count == 0: # Successfully collected the package
            reward += 200 # Bonus for completing the task
        # else: # Terminal but package not collected (shouldn't happen in 'simple' if logic is correct)
            # reward -= 50 # Penalty if terminal for other reasons (e.g. error or wrong order in other scenarios)
            
    return reward

def train_agent():
    print(f"Training agent with {EXPLORATION_STRATEGY} epsilon decay...")
    # If "from FourRooms import FourRooms" is used, then FourRooms() is correct.
    # Otherwise, it would be module_name.FourRooms()
    fourRoomsObj = FourRooms(scenario='simple') # Create FourRooms Object for 'simple' scenario

    current_epsilon = INITIAL_EPSILON

    epoch_rewards = []
    epoch_steps = []

    print(f"Q-table shape: {q_table.shape}") # Should now be (11, 11, 2, 4)
    print(f"Training agent with {NUM_EPOCHS} epochs and up to {STEPS_PER_EPOCH} steps per epoch...")

    for epoch in range(NUM_EPOCHS):
        fourRoomsObj.newEpoch()  # Reset environment

        is_terminal = False
        steps_this_epoch = 0
        total_reward_this_epoch = 0

        current_pos_env = fourRoomsObj.getPosition() # e.g., (1,5) - 1-indexed from environment
        current_packages_left_val = fourRoomsObj.getPackagesRemaining() # 1 or 0
        current_state_idx = get_state_index(current_pos_env, current_packages_left_val) # 0-indexed for Q-table

        while not is_terminal and steps_this_epoch < STEPS_PER_EPOCH:
            # 1. Choose action
            action_idx = choose_action(current_state_idx, current_epsilon) # action_idx is 0,1,2 or 3
            action_to_take = ACTIONS[action_idx] # Map to FourRooms.UP, etc.
            
            # For reward calculation, what was the package count *before* this action?
            # This comes from the 'package_idx' part of the current_state_idx
            old_packages_left_val_for_reward = current_state_idx[2] 

            # 2. Take action in environment
            grid_cell_type, new_pos_env, new_packages_left_val, terminal_from_env = fourRoomsObj.takeAction(action_to_take)
            
            # 3. Calculate Reward
            reward = get_reward(old_packages_left_val_for_reward, new_packages_left_val, terminal_from_env, grid_cell_type)
            total_reward_this_epoch += reward

            # 4. Observe new state (0-indexed for Q-table)
            next_state_idx = get_state_index(new_pos_env, new_packages_left_val)

            # 5. Update Q-Table (Q-learning formula)
            # Q(s,a) is q_table[current_state_idx_tuple + (action_idx,)]
            # current_state_idx is (x,y,pkg_idx), action_idx is int.
            # (x,y,pkg_idx) + (act_idx,) gives (x,y,pkg_idx,act_idx)
            q_table_access_index_for_sa = current_state_idx + (action_idx,)
            old_q_value = q_table[q_table_access_index_for_sa]
            
            if terminal_from_env:
                next_max_q = 0.0
            else:
                # q_table[next_state_idx] gives the array of Q-values for actions in the next state
                next_max_q = np.max(q_table[next_state_idx]) 
            
            new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q_value)
            q_table[q_table_access_index_for_sa] = new_q_value
            
            # 6. Update current state and step count
            current_state_idx = next_state_idx
            is_terminal = terminal_from_env
            steps_this_epoch += 1
        
        epoch_rewards.append(total_reward_this_epoch)
        epoch_steps.append(steps_this_epoch)
        
        # Decay epsilon
        if EXPLORATION_STRATEGY == 'multiplicative':
            # Multiplicative decay
            if current_epsilon > MIN_EPSILON:
                current_epsilon *= EPSILON_DECAY_VALUE
                current_epsilon = max(MIN_EPSILON, current_epsilon) # Ensure it doesn't go below min
        elif EXPLORATION_STRATEGY == 'linear':
            if current_epsilon > MIN_EPSILON:
                # current_epsilon = INITIAL_EPSILON - (epoch / EPSILON_DECAY_STEPS) * (INITIAL_EPSILON - MIN_EPSILON)
                current_epsilon -= EPSILON_DECREMENT
                current_epsilon = max(MIN_EPSILON, current_epsilon)

        if (epoch + 1) % 100 == 0: 
            avg_reward_last_100 = np.mean(epoch_rewards[-100:]) if len(epoch_rewards) >= 100 else np.mean(epoch_rewards)
            avg_steps_last_100 = np.mean(epoch_steps[-100:]) if len(epoch_steps) >= 100 else np.mean(epoch_steps)
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Steps: {steps_this_epoch} | Epsilon: {current_epsilon:.4f} | Last Reward: {total_reward_this_epoch} | Avg Reward (100): {avg_reward_last_100:.2f} | Avg Steps (100): {avg_steps_last_100:.2f}")
        
            if is_terminal and new_packages_left_val == 0: # Use new_packages_left_val from takeAction
                 print(f"  SUCCESS: Package collected in {steps_this_epoch} steps.")
            elif steps_this_epoch >= STEPS_PER_EPOCH:
                 print(f"  TIMEOUT: Reached max steps for epoch.")

    print("Training complete.")
    print("Displaying path from the agent's behavior in the final epoch...")

    # Use savefig to save the image to a file
    fourRoomsObj.showPath(index=-1, savefig="scenario1_final_path.png")
    print("Path image saved to scenario1_final_path.png")

    # --- Data Saving for Plotting (New) ---
    # After the loop, save the results for later plotting
    results_filename_suffix = f"_{EXPLORATION_STRATEGY}"
    np.save(f"scenario1_epoch_rewards{results_filename_suffix}.npy", np.array(epoch_rewards))
    np.save(f"scenario1_epoch_steps{results_filename_suffix}.npy", np.array(epoch_steps))
    print(f"Saved epoch rewards to scenario1_epoch_rewards{results_filename_suffix}.npy")
    print(f"Saved epoch steps to scenario1_epoch_steps{results_filename_suffix}.npy")

if __name__ == "__main__":
    train_agent()