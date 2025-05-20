# Scenario1.py
# Agent must collect 1 package located somewhere in the environment
# Agent starts at a random position in the environment
# Agent can move in 4 directions: UP, DOWN, LEFT, RIGHT

# Maphuti Shilabje

from FourRooms import FourRooms
import numpy as np
import random
import argparse
import os

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
# q_table = np.zeros((GRID_ROWS, GRID_COLS, 2, NUM_ACTIONS)) # Corrected Q-table shape

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

def choose_action(q_table_ref, state, current_epsilon): # Corrected parameter name
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
        return np.argmax(q_table_ref[state])

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

def train_single_run(run_id, num_epochs, exploration_strategy, use_stochastic_env, results_dir="results/scenario1"):
    print(f"\n--- Starting Run {run_id} for Scenario 1 ---")
    print(f"Strategy: {exploration_strategy}, Stochastic Env: {use_stochastic_env}, Epochs: {num_epochs}")

    # Initialize Q-table for this run
    q_table = np.zeros((GRID_ROWS, GRID_COLS, 2, NUM_ACTIONS))

    # Epsilon parameters based on strategy
    current_epsilon = INITIAL_EPSILON
    epsilon_decay_value = 0
    epsilon_decrement = 0

    if exploration_strategy == 'multiplicative':
        # Ensure num_epochs > 0 to avoid division by zero if used in decay calc
        if num_epochs * 0.8 > 0:
            epsilon_decay_value = (MIN_EPSILON / INITIAL_EPSILON) ** (1.0 / (num_epochs * 0.8))
        else: # Handle case of very few epochs, e.g. direct to min epsilon or no decay
            epsilon_decay_value = 0 # effectively no decay if it leads to MIN_EPSILON quickly
    elif exploration_strategy == 'linear':
        decay_steps = num_epochs * 0.8
        if decay_steps > 0 :
            epsilon_decrement = (INITIAL_EPSILON - MIN_EPSILON) / decay_steps
        else:
            epsilon_decrement = INITIAL_EPSILON # Go to MIN_EPSILON in one step
    else:
        raise ValueError("Invalid exploration_strategy.")

    fourRoomsObj = FourRooms(scenario='simple', stochastic=use_stochastic_env)

    epoch_rewards = []
    epoch_steps = []

    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch()
        is_terminal = False
        steps_this_epoch = 0
        total_reward_this_epoch = 0

        current_pos_env = fourRoomsObj.getPosition()
        current_packages_left_val = fourRoomsObj.getPackagesRemaining()
        current_state_idx = get_state_index(current_pos_env, current_packages_left_val)

        while not is_terminal and steps_this_epoch < STEPS_PER_EPOCH:
            action_idx = choose_action(q_table, current_state_idx, current_epsilon) # Pass q_table
            action_to_take = ACTIONS[action_idx]
            
            old_packages_left_val_for_reward = current_state_idx[2]
            grid_cell_type, new_pos_env, new_packages_left_val, terminal_from_env = fourRoomsObj.takeAction(action_to_take)
            reward = get_reward(old_packages_left_val_for_reward, new_packages_left_val, terminal_from_env, grid_cell_type)
            total_reward_this_epoch += reward
            next_state_idx = get_state_index(new_pos_env, new_packages_left_val)

            q_table_access_index_for_sa = current_state_idx + (action_idx,)
            old_q_value = q_table[q_table_access_index_for_sa]
            
            next_max_q = 0.0 if terminal_from_env else np.max(q_table[next_state_idx])
            
            new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q_value)
            q_table[q_table_access_index_for_sa] = new_q_value
            
            current_state_idx = next_state_idx
            is_terminal = terminal_from_env
            steps_this_epoch += 1
        
        epoch_rewards.append(total_reward_this_epoch)
        epoch_steps.append(steps_this_epoch)
        
        if exploration_strategy == 'multiplicative':
            if current_epsilon > MIN_EPSILON and epsilon_decay_value > 0 : # check decay_value to prevent issues with few epochs
                current_epsilon *= epsilon_decay_value
        elif exploration_strategy == 'linear':
            if current_epsilon > MIN_EPSILON:
                current_epsilon -= epsilon_decrement
        current_epsilon = max(MIN_EPSILON, current_epsilon)

        if (epoch + 1) % 100 == 0 or epoch == num_epochs -1 :
            avg_r = np.mean(epoch_rewards[-100:]) if len(epoch_rewards) >= 100 else np.mean(epoch_rewards)
            avg_s = np.mean(epoch_steps[-100:]) if len(epoch_steps) >= 100 else np.mean(epoch_steps)
            print(f"Run {run_id} Epoch {epoch + 1}/{num_epochs} | Steps: {steps_this_epoch} | Eps: {current_epsilon:.3f} | AvgR: {avg_r:.2f} | AvgS: {avg_s:.2f}")
            if is_terminal and new_packages_left_val == 0: print(f"  SUCCESS Run {run_id}: Package collected in {steps_this_epoch} steps.")
            elif steps_this_epoch >= STEPS_PER_EPOCH: print(f"  TIMEOUT Run {run_id}: Max steps.")

    print(f"Run {run_id} training complete.")
    
    # Save Q-table and path for the last epoch of this run
    stochastic_suffix = "_stochastic" if use_stochastic_env else ""
    final_path_filename = f"{results_dir}/{exploration_strategy}_run{run_id}{stochastic_suffix}_final_path.png"
    q_table_filename = f"{results_dir}/{exploration_strategy}_run{run_id}{stochastic_suffix}_q_table.npy"
    
    os.makedirs(results_dir, exist_ok=True) # Ensure directory exists
    fourRoomsObj.showPath(index=-1, savefig=final_path_filename)
    np.save(q_table_filename, q_table)
    print(f"Saved final path to {final_path_filename}")
    print(f"Saved Q-table to {q_table_filename}")

    # Return collected data for this run
    return np.array(epoch_rewards), np.array(epoch_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-Learning for CSC3022F Assignment Scenario 1.")
    parser.add_argument("--stochastic", action="store_true", help="Enable stochastic actions in the environment.")
    parser.add_argument("--strategy", type=str, default="multiplicative", choices=["multiplicative", "linear"],
                        help="Exploration strategy for epsilon decay.")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent training runs.")
    parser.add_argument("--epochs", type=int, default=2000, help="Number of epochs per run.")
    parser.add_argument("--results_dir", type=str, default="results/scenario1", help="Directory to save results.")

    args = parser.parse_args()

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    all_runs_rewards = []
    all_runs_steps = []

    for i in range(1, args.runs + 1):
        # For true independence between runs, ensure Python's random and NumPy's random are seeded differently
        # or not seeded within the loop if you want different random sequences per run.
        # If you want reproducible sets of runs, seed *before* this loop.
        # random.seed(i) # Optional: for reproducible sequences of runs
        # np.random.seed(i) # Optional

        run_rewards, run_steps = train_single_run(
            run_id=i,
            num_epochs=args.epochs,
            exploration_strategy=args.strategy,
            use_stochastic_env=args.stochastic,
            results_dir=args.results_dir
        )
        all_runs_rewards.append(run_rewards)
        all_runs_steps.append(run_steps)

    # Save aggregated results (average over runs)
    # Ensure all runs had the same number of epochs for straightforward averaging
    if all_runs_rewards: # Check if list is not empty
        avg_rewards_over_runs = np.mean(np.array(all_runs_rewards), axis=0)
        avg_steps_over_runs = np.mean(np.array(all_runs_steps), axis=0)
        std_rewards_over_runs = np.std(np.array(all_runs_rewards), axis=0) # For stability analysis
        std_steps_over_runs = np.std(np.array(all_runs_steps), axis=0)     # For stability analysis

        stochastic_suffix = "_stochastic" if args.stochastic else ""
        avg_rewards_filename = f"{args.results_dir}/{args.strategy}{stochastic_suffix}_avg_rewards_over_{args.runs}_runs.npy"
        avg_steps_filename = f"{args.results_dir}/{args.strategy}{stochastic_suffix}_avg_steps_over_{args.runs}_runs.npy"
        std_rewards_filename = f"{args.results_dir}/{args.strategy}{stochastic_suffix}_std_rewards_over_{args.runs}_runs.npy"
        std_steps_filename = f"{args.results_dir}/{args.strategy}{stochastic_suffix}_std_steps_over_{args.runs}_runs.npy"

        np.save(avg_rewards_filename, avg_rewards_over_runs)
        np.save(avg_steps_filename, avg_steps_over_runs)
        np.save(std_rewards_filename, std_rewards_over_runs)
        np.save(std_steps_filename, std_steps_over_runs)

        print(f"\nSaved averaged rewards to {avg_rewards_filename}")
        print(f"Saved averaged steps to {avg_steps_filename}")
        print(f"Saved std dev rewards to {std_rewards_filename}")
        print(f"Saved std dev steps to {std_steps_filename}")

        # You would then modify your plotting script to load these averaged files
        # and potentially the std deviation files to show error bands.
    else:
        print("No runs were completed, so no aggregate results saved.")

    print("\nAll runs complete for Scenario 1.")