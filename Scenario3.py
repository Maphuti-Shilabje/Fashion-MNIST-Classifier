# Scenario3.py
from FourRooms import FourRooms
import numpy as np
import random
import argparse
import os

# --- Environment & Q-Learning Constants ---
GRID_ROWS = 11
GRID_COLS = 11
ACTIONS = [FourRooms.UP, FourRooms.DOWN, FourRooms.LEFT, FourRooms.RIGHT]
NUM_ACTIONS = len(ACTIONS)
NUM_PACKAGE_STATES = 3 # 0: Target Red, 1: Target Green, 2: Target Blue

# --- Hyperparameters ---
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
# Default STEPS_PER_EPOCH, can be overridden by CLI arg
DEFAULT_STEPS_PER_EPOCH = 350

# --- Exploration Parameters (Initial Epsilon, Min Epsilon is CLI arg) ---
INITIAL_EPSILON = 1.0
# MIN_EPSILON is now a command-line argument, default 0.01

def get_state_index_s3(pos, packages_remaining_from_env):
    x_idx = pos[0] - 1
    y_idx = pos[1] - 1
    x_idx = np.clip(x_idx, 0, GRID_ROWS - 1)
    y_idx = np.clip(y_idx, 0, GRID_COLS - 1)

    target_package_idx = -1 
    if packages_remaining_from_env == 3: target_package_idx = 0
    elif packages_remaining_from_env == 2: target_package_idx = 1
    elif packages_remaining_from_env == 1: target_package_idx = 2
    
    if target_package_idx == -1 and packages_remaining_from_env > 0:
        # Fallback for safety, though ideally not reached with correct env behavior for S3
        # print(f"S3 Warning: Invalid target_package_idx for packages_remaining: {packages_remaining_from_env}")
        target_package_idx = 0 

    return (x_idx, y_idx, target_package_idx)

def choose_action(q_table_ref, state, current_epsilon):
    if random.random() < current_epsilon:
        return random.randrange(NUM_ACTIONS)
    else:
        return np.argmax(q_table_ref[state])

def get_reward_s3(prev_pkg_count_env, new_pkg_count_env, is_terminal_env, grid_cell_type_env):
    reward = 0.0
    reward -= 1.0  # Step penalty

    package_was_picked_up = new_pkg_count_env < prev_pkg_count_env

    if package_was_picked_up:
        if is_terminal_env and new_pkg_count_env > 0: # Picked up, but terminated with packages left -> wrong order
            reward -= 250.0 
        else: # Correct package pickup (or last one for success)
            reward += 150.0
    
    if is_terminal_env:
        if new_pkg_count_env == 0: # All packages collected in order
            reward += 1000.0
        # else if not package_was_picked_up and new_pkg_count_env > 0:
            # Terminated (e.g. timeout) without picking a package this step, step penalty applies.
            # If it was a wrong order pickup, the 'package_was_picked_up' block handles it.

    return reward

def train_single_run_s3(run_id, num_epochs, exploration_strategy, use_stochastic_env, 
                        min_epsilon_runtime, results_dir="results/scenario3", 
                        steps_per_epoch_override=DEFAULT_STEPS_PER_EPOCH):
    print(f"\n--- Scenario 3: Starting Run {run_id} ---")
    print(f"Strategy: {exploration_strategy}, Stochastic: {use_stochastic_env}, Epochs: {num_epochs}, Steps/Ep: {steps_per_epoch_override}, MinEps: {min_epsilon_runtime}")

    q_table = np.zeros((GRID_ROWS, GRID_COLS, NUM_PACKAGE_STATES, NUM_ACTIONS))

    current_epsilon = INITIAL_EPSILON
    epsilon_decay_value = 0.0
    epsilon_decrement = 0.0

    # Calculate decay parameters using min_epsilon_runtime
    if exploration_strategy == 'multiplicative':
        if num_epochs * 0.8 > 0: 
            epsilon_decay_value = (min_epsilon_runtime / INITIAL_EPSILON) ** (1.0 / (num_epochs * 0.8))
    elif exploration_strategy == 'linear':
        decay_steps = num_epochs * 0.8
        if decay_steps > 0: 
            epsilon_decrement = (INITIAL_EPSILON - min_epsilon_runtime) / decay_steps
        else: 
            epsilon_decrement = INITIAL_EPSILON 

    fourRoomsObj = FourRooms(scenario='rgb', stochastic=use_stochastic_env)
    epoch_rewards = []
    epoch_steps = []
    print(f"Q-Table shape for S3: {q_table.shape}")

    for epoch in range(num_epochs):
        fourRoomsObj.newEpoch()
        is_terminal_current_epoch = False # Renamed to avoid conflict with loop variable 'is_terminal'
        steps_this_epoch = 0
        total_reward_this_epoch = 0
        # Initialize new_packages_left_env for logging if epoch ends before first step
        new_packages_left_env = fourRoomsObj.getPackagesRemaining() 


        current_pos_env = fourRoomsObj.getPosition()
        current_packages_left_env = fourRoomsObj.getPackagesRemaining()
        current_state_idx = get_state_index_s3(current_pos_env, current_packages_left_env)

        while not is_terminal_current_epoch and steps_this_epoch < steps_per_epoch_override:
            if current_state_idx[2] == -1: # Safety break
                # print(f"S3 Warning Ep{epoch+1} Step{steps_this_epoch}: Invalid current_state_idx {current_state_idx}. PkgsLeft: {current_packages_left_env}")
                break 

            action_idx = choose_action(q_table, current_state_idx, current_epsilon)
            action_to_take = ACTIONS[action_idx]
            
            prev_pkg_count_env = fourRoomsObj.getPackagesRemaining()

            grid_cell_type, new_pos_env, new_packages_left_env, terminal_from_env = fourRoomsObj.takeAction(action_to_take)
            reward = get_reward_s3(prev_pkg_count_env, new_packages_left_env, terminal_from_env, grid_cell_type)
            total_reward_this_epoch += reward
            
            next_s_idx_for_q_update = None
            effective_terminal_for_q = terminal_from_env 

            if not terminal_from_env:
                next_s_idx_for_q_update = get_state_index_s3(new_pos_env, new_packages_left_env)
                if next_s_idx_for_q_update[2] == -1: 
                    # print(f"S3 Warning Ep{epoch+1} Step{steps_this_epoch}: Invalid next_state_idx {next_s_idx_for_q_update}. PkgsLeft: {new_packages_left_env}. Terminal: {terminal_from_env}")
                    effective_terminal_for_q = True 

            q_table_access_index_for_sa = current_state_idx + (action_idx,)
            old_q_value = q_table[q_table_access_index_for_sa]
            
            next_max_q = 0.0
            if not effective_terminal_for_q and next_s_idx_for_q_update is not None: # Implicitly next_s_idx_for_q_update[2] != -1 due to effective_terminal_for_q
                next_max_q = np.max(q_table[next_s_idx_for_q_update])
            
            new_q_value = old_q_value + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q - old_q_value)
            q_table[q_table_access_index_for_sa] = new_q_value
            
            if not effective_terminal_for_q and next_s_idx_for_q_update is not None:
                current_state_idx = next_s_idx_for_q_update
            
            is_terminal_current_epoch = terminal_from_env 
            steps_this_epoch += 1
        
        epoch_rewards.append(total_reward_this_epoch)
        epoch_steps.append(steps_this_epoch)
        
        # Epsilon decay using min_epsilon_runtime
        if exploration_strategy == 'multiplicative':
            if current_epsilon > min_epsilon_runtime and epsilon_decay_value > 0: current_epsilon *= epsilon_decay_value
        elif exploration_strategy == 'linear':
            if current_epsilon > min_epsilon_runtime: current_epsilon -= epsilon_decrement
        current_epsilon = max(min_epsilon_runtime, current_epsilon)

        if (epoch + 1) % 100 == 0 or epoch == num_epochs - 1:
            avg_r = np.mean(epoch_rewards[-100:]) if len(epoch_rewards) >= 100 else np.mean(epoch_rewards) if epoch_rewards else 0
            avg_s = np.mean(epoch_steps[-100:]) if len(epoch_steps) >= 100 else np.mean(epoch_steps) if epoch_steps else 0
            log_msg = (f"S3 Run {run_id} Ep {epoch + 1}/{num_epochs} | Steps: {steps_this_epoch} | "
                       f"Eps: {current_epsilon:.3f} | AvgR: {avg_r:.2f} | AvgS: {avg_s:.2f}")
            print(log_msg)
            
            # For logging final status, use the state at the end of the 'while' loop
            final_packages_at_epoch_end = new_packages_left_env 
            
            if is_terminal_current_epoch:
                if final_packages_at_epoch_end == 0:
                    print(f"  SUCCESS S3 Run {run_id}: All packages collected in order in {steps_this_epoch} steps.")
                else:
                    print(f"  FAILURE S3 Run {run_id}: Terminated with {final_packages_at_epoch_end} package(s) left.")
            elif steps_this_epoch >= steps_per_epoch_override:
                print(f"  TIMEOUT S3 Run {run_id}: Max steps with {final_packages_at_epoch_end} package(s) left.")

    print(f"Run {run_id} (Scenario 3) training complete.")
    stochastic_suffix = "_stochastic" if use_stochastic_env else ""
    base_results_filename = f"s3_{exploration_strategy}_run{run_id}{stochastic_suffix}"
    final_path_filename = os.path.join(results_dir, f"{base_results_filename}_final_path.png")
    os.makedirs(results_dir, exist_ok=True)
    fourRoomsObj.showPath(index=-1, savefig=final_path_filename)
    print(f"Saved final path for S3 Run {run_id} to {final_path_filename}")
    return np.array(epoch_rewards), np.array(epoch_steps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Q-Learning for Scenario 3: Ordered Collection.")
    parser.add_argument("--stochastic", action="store_true", help="Enable stochastic actions.")
    parser.add_argument("--strategy", type=str, default="multiplicative", choices=["multiplicative", "linear"], help="Epsilon decay strategy.")
    parser.add_argument("--runs", type=int, default=1, help="Number of training runs.")
    parser.add_argument("--epochs", type=int, default=10000, help="Number of epochs per run for S3.") # Increased default
    parser.add_argument("--steps_epoch", type=int, default=DEFAULT_STEPS_PER_EPOCH, help="Max steps per epoch for S3.")
    parser.add_argument("--min_epsilon", type=float, default=0.05, help="Minimum epsilon value for exploration.") # Added min_epsilon arg with new default
    parser.add_argument("--results_dir", type=str, default="results/scenario3", help="Directory for results.")

    args = parser.parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    all_runs_rewards_s3 = []
    all_runs_steps_s3 = []
    for i in range(1, args.runs + 1):
        run_rewards, run_steps = train_single_run_s3(
            run_id=i,
            num_epochs=args.epochs,
            exploration_strategy=args.strategy,
            use_stochastic_env=args.stochastic,
            min_epsilon_runtime=args.min_epsilon, # Pass parsed min_epsilon
            results_dir=args.results_dir,
            steps_per_epoch_override=args.steps_epoch
        )
        if run_rewards.size > 0 and run_steps.size > 0: # Check if actual data was returned
            all_runs_rewards_s3.append(run_rewards)
            all_runs_steps_s3.append(run_steps)

    if all_runs_rewards_s3:
        avg_rewards = np.mean(np.array(all_runs_rewards_s3), axis=0)
        avg_steps = np.mean(np.array(all_runs_steps_s3), axis=0)
        stochastic_suffix = "_stochastic" if args.stochastic else ""
        base_fn = f"s3_{args.strategy}{stochastic_suffix}_over_{args.runs}_runs_eps{args.min_epsilon}" # Add min_eps to filename for clarity
        
        np.save(os.path.join(args.results_dir, f"{base_fn}_avg_rewards.npy"), avg_rewards)
        np.save(os.path.join(args.results_dir, f"{base_fn}_avg_steps.npy"), avg_steps)
        print(f"\nS3: Saved aggregated rewards to {os.path.join(args.results_dir, f'{base_fn}_avg_rewards.npy')}")
        print(f"S3: Saved aggregated steps to {os.path.join(args.results_dir, f'{base_fn}_avg_steps.npy')}")
    else:
        print("S3: No valid runs completed or data generated; no aggregate results saved.")
    print("\nScenario 3 processing complete.")