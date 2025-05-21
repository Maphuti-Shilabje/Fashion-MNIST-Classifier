from FourRooms import FourRooms
import numpy as np
import random
import argparse
import os


# --- Environment & Q-Learning Constants ---
GRID_COLS = 11  # Effective traversable columns (indices 0-10)
GRID_ROWS = 11  # Effective traversable rows (indices 0-10)

ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
NUM_ACTIONS = len(ACTIONS)

NUM_PACKAGE_STATES = 5

# The learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
NUM_EPOCHS = 2000  # Increased for potentially better convergence
STEPS_PER_EPOCH = 300

EXPLORATION_STRATEGY = 'linear' # or 'linear'

# The exploration parameters
# Epsilon-greedy exploration
INITIAL_EPSILON = 1.0 # Start with full exploration
MIN_EPSILON = 0.01    # Minimum exploration rate
# Adjusted decay to aim for MIN_EPSILON over ~80% of epochs


def get_state_index(pos, packages_remaining):
    """
    Converts the agent's 1-indexed position and packages remaining
    to a 0-indexed state tuple suitable for Q-table indexing.
    For Scenario 2, packages_remaining directly maps to an index (0-4).
    """
    x_idx = pos[0] - 1
    y_idx = pos[1] - 1
    x_idx = np.clip(x_idx, 0, GRID_ROWS - 1)
    y_idx = np.clip(y_idx, 0, GRID_COLS - 1)

    # packages_remaining will be 4, 3, 2, 1, or 0.
    # This value can be directly used as the index for the package dimension.
    package_idx = packages_remaining
    if not (0 <= package_idx < NUM_PACKAGE_STATES):
        # This should ideally not happen if FourRooms.getPackagesRemaining() is consistent
        print(f"Warning: packages_remaining ({packages_remaining}) out of expected range for Q-table.")
        package_idx = np.clip(package_idx, 0, NUM_PACKAGE_STATES - 1)


    return (x_idx, y_idx, package_idx)


def choose_action(q_table_ref, state, current_epsilon):
    """
    Selects an action using an epsilon-greedy policy.
    'state' is the 0-indexed tuple (x_idx, y_idx, package_idx).
    """
    if random.random() < current_epsilon:
        return random.randrange(NUM_ACTIONS)
    else:
        return np.argmax(q_table_ref[state])
    

def get_reward_s2(prev_package_count, curr_package_count, is_terminal_state, grid_cell_type):
    """
    Calculates the reward for Scenario 2.
    """
    reward = 0
    # Penalty for each step to encourage efficiency
    reward -= 1

    if curr_package_count < prev_package_count:
        # Agent picked up a package
        reward += 100  # Reward for picking up any package

    if is_terminal_state:
        if curr_package_count == 0:
            # All 4 packages collected!
            reward += 500 # Large bonus for completing the entire task

    return reward

# --- Main Training Function (per run) ---
def train_single_run_s2(run_id, num_epochs, exploration_strategy, use_stochastic_env, results_dir="results/scenario2"):
    print(f"\n--- Scenario 2: Starting Run {run_id} ---")
    print(f"Strategy: {exploration_strategy}, Stochastic Env: {use_stochastic_env}, Epochs: {num_epochs}")

    # Initialize Q-table with dimensions (GRID_ROWS, GRID_COLS, NUM_PACKAGE_STATES, NUM_ACTIONS)
    q_table = np.zeros((GRID_ROWS, GRID_COLS, NUM_PACKAGE_STATES, NUM_ACTIONS)) # Correct dimensions for S2

    current_epsilon = INITIAL_EPSILON
    epsilon_decay_value = 0
    epsilon_decrement = 0

    if exploration_strategy == 'multiplicative':
        if num_epochs * 0.8 > 0:
            epsilon_decay_value = (MIN_EPSILON / INITIAL_EPSILON) ** (1.0 / (num_epochs * 0.8))
        else:
            epsilon_decay_value = 0
    elif exploration_strategy == 'linear':
        decay_steps = num_epochs * 0.8
        if decay_steps > 0:
            epsilon_decrement = (INITIAL_EPSILON - MIN_EPSILON) / decay_steps
        else:
            epsilon_decrement = INITIAL_EPSILON

    # Initialize FourRooms for SCENARIO 'multi'
    fourRoomsObj = FourRooms(scenario='multi', stochastic=use_stochastic_env) # Key change for S2

    epoch_rewards = []
    epoch_steps = []

    print(f"Q-Table shape for S2: {q_table.shape}") # Verify shape

    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch()
        is_terminal = False
        steps_this_epoch = 0
        total_reward_this_epoch = 0

        current_pos_env = fourRoomsObj.getPosition()
        current_packages_left_val = fourRoomsObj.getPackagesRemaining() # Will be 4, 3, 2, 1, or 0
        current_state_idx = get_state_index(current_pos_env, current_packages_left_val)

        while not is_terminal and steps_this_epoch < STEPS_PER_EPOCH:
            action_idx = choose_action(q_table, current_state_idx, current_epsilon)
            action_to_take = ACTIONS[action_idx]
            
            old_packages_left_val_for_reward = current_state_idx[2] # This is the package index (0-4)
                                                                   # For reward, we need the actual count.
            
            # Get current package count before action for reward calc
            prev_pkg_count_from_env = fourRoomsObj.getPackagesRemaining()

            grid_cell_type, new_pos_env, new_packages_left_val, terminal_from_env = fourRoomsObj.takeAction(action_to_take)
            
            # Use the S2 specific reward function
            reward = get_reward_s2(prev_pkg_count_from_env, new_packages_left_val, terminal_from_env, grid_cell_type)
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
        
        # Epsilon decay logic
        if exploration_strategy == 'multiplicative':
            if current_epsilon > MIN_EPSILON and epsilon_decay_value > 0:
                current_epsilon *= epsilon_decay_value
        elif exploration_strategy == 'linear':
            if current_epsilon > MIN_EPSILON:
                current_epsilon -= epsilon_decrement
        current_epsilon = max(MIN_EPSILON, current_epsilon)

        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1 : # Log every 100 epochs or last epoch
            avg_r = np.mean(epoch_rewards[-100:]) if len(epoch_rewards) >= 100 else np.mean(epoch_rewards) if epoch_rewards else 0
            avg_s = np.mean(epoch_steps[-100:]) if len(epoch_steps) >= 100 else np.mean(epoch_steps) if epoch_steps else 0
            print(f"S2 Run {run_id} Ep {epoch + 1}/{num_epochs} | Steps: {steps_this_epoch} | Eps: {current_epsilon:.3f} | AvgR: {avg_r:.2f} | AvgS: {avg_s:.2f}")
            if is_terminal and new_packages_left_val == 0: print(f"  SUCCESS S2 Run {run_id}: All packages collected in {steps_this_epoch} steps.")
            elif steps_this_epoch >= STEPS_PER_EPOCH: print(f"  TIMEOUT S2 Run {run_id}: Max steps.")

    print(f"Run {run_id} (Scenario 2) training complete.")
    
    stochastic_suffix = "_stochastic" if use_stochastic_env else ""
    # Using more descriptive filenames for S2
    base_results_filename = f"s2_{exploration_strategy}_run{run_id}{stochastic_suffix}"

    final_path_filename = os.path.join(results_dir, f"{base_results_filename}_final_path.png")
    q_table_filename = os.path.join(results_dir, f"{base_results_filename}_q_table.npy")
    
    os.makedirs(results_dir, exist_ok=True)
    fourRoomsObj.showPath(index=-1, savefig=final_path_filename)
    # np.save(q_table_filename, q_table) 
    print(f"Saved final path for S2 Run {run_id} to {final_path_filename}")
    # print(f"Saved Q-table for S2 Run {run_id} to {q_table_filename}")

    return np.array(epoch_rewards), np.array(epoch_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-Learning for CSC3022F Assignment Scenario 2: Multiple Packages.")
    parser.add_argument("--stochastic", action="store_true", help="Enable stochastic actions in the environment.")
    parser.add_argument("--strategy", type=str, default="multiplicative", choices=["multiplicative", "linear"],
                        help="Exploration strategy for epsilon decay.")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent training runs (start with 1 for S2).") # Default to 1 for faster initial testing
    parser.add_argument("--epochs", type=int, default=3000, help="Number of epochs per run.") # Default more epochs for S2
    parser.add_argument("--results_dir", type=str, default="results/scenario2", help="Directory to save results.")


    args = parser.parse_args()


    os.makedirs(args.results_dir, exist_ok=True)

    all_runs_rewards_s2 = []
    all_runs_steps_s2 = []

    for i in range(1, args.runs + 1):
        run_rewards, run_steps = train_single_run_s2( # Call the S2 specific function
            run_id=i,
            num_epochs=args.epochs,
            exploration_strategy=args.strategy,
            use_stochastic_env=args.stochastic,
            results_dir=args.results_dir
        )
        if run_rewards.size > 0 and run_steps.size > 0 : 
            all_runs_rewards_s2.append(run_rewards)
            all_runs_steps_s2.append(run_steps)

    if all_runs_rewards_s2:
        avg_rewards_over_runs = np.mean(np.array(all_runs_rewards_s2), axis=0)
        avg_steps_over_runs = np.mean(np.array(all_runs_steps_s2), axis=0)

        stochastic_suffix = "_stochastic" if args.stochastic else ""
        # Using more descriptive filenames for S2
        base_filename = f"s2_{args.strategy}{stochastic_suffix}_over_{args.runs}_runs"
        
        np.save(os.path.join(args.results_dir, f"{base_filename}_avg_rewards.npy"), avg_rewards_over_runs)
        np.save(os.path.join(args.results_dir, f"{base_filename}_avg_steps.npy"), avg_steps_over_runs)


        print(f"\nSaved Scenario 2 aggregated rewards to {os.path.join(args.results_dir, f'{base_filename}_avg_rewards.npy')}")
        print(f"Saved Scenario 2 aggregated steps to {os.path.join(args.results_dir, f'{base_filename}_avg_steps.npy')}")
    else:
        print("No Scenario 2 runs completed or dummy function used, so no aggregate results saved.")

    print("\nScenario 2 processing complete (or placeholder executed).")