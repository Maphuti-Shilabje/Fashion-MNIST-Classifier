# In plot_results.py
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_learning_curves(strategies_data, title_prefix="", output_dir="results/scenario1", base_filename="learning_curves"):
    plt.figure(figsize=(14, 6))
    colors = {'Multiplicative Decay': 'blue', 'Linear Decay': 'green'} # Add more if needed

    # Plot Rewards
    plt.subplot(1, 2, 1)
    for strategy_name, data in strategies_data.items():
        if 'avg_rewards' in data:
            epochs_axis = np.arange(len(data['avg_rewards']))
            plt.plot(epochs_axis, data['avg_rewards'], label=f"{strategy_name} Rewards", color=colors.get(strategy_name))
            if 'std_rewards' in data:
                plt.fill_between(epochs_axis,
                                 data['avg_rewards'] - data['std_rewards'],
                                 data['avg_rewards'] + data['std_rewards'],
                                 alpha=0.2, color=colors.get(strategy_name))
    plt.xlabel("Epochs")
    plt.ylabel("Average Reward")
    plt.title(f"{title_prefix}Learning Curve (Avg Rewards)")
    plt.legend()
    plt.grid(True)

    # Plot Steps
    plt.subplot(1, 2, 2)
    for strategy_name, data in strategies_data.items():
        if 'avg_steps' in data:
            epochs_axis = np.arange(len(data['avg_steps']))
            plt.plot(epochs_axis, data['avg_steps'], label=f"{strategy_name} Steps", color=colors.get(strategy_name))
            if 'std_steps' in data:
                plt.fill_between(epochs_axis,
                                 data['avg_steps'] - data['std_steps'],
                                 data['avg_steps'] + data['std_steps'],
                                 alpha=0.2, color=colors.get(strategy_name))
    plt.xlabel("Epochs")
    plt.ylabel("Average Steps per Epoch")
    plt.title(f"{title_prefix}Learning Curve (Avg Steps)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # Ensure output_dir exists if it's part of the path
    os.makedirs(os.path.dirname(os.path.join(output_dir, base_filename)), exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{base_filename}.png"))
    print(f"Plot saved to {os.path.join(output_dir, f'{base_filename}.png')}")
    plt.show()

if __name__ == "__main__":
    results_dir = "results/scenario1"
    num_runs = 3 # Should match the --runs parameter used for Scenario1.py

    scenario1_data_deterministic = {}
    scenario1_data_stochastic = {}

    strategies = ["multiplicative", "linear"]
    stochastic_modes = {"deterministic": False, "stochastic": True}

    for strategy in strategies:
        for mode_name, is_stochastic in stochastic_modes.items():
            stochastic_suffix = "_stochastic" if is_stochastic else ""
            plot_filename_suffix = f"_{mode_name}"
            
            try:
                avg_rewards_file = os.path.join(results_dir, f"{strategy}{stochastic_suffix}_avg_rewards_over_{num_runs}_runs.npy")
                avg_steps_file = os.path.join(results_dir, f"{strategy}{stochastic_suffix}_avg_steps_over_{num_runs}_runs.npy")
                std_rewards_file = os.path.join(results_dir, f"{strategy}{stochastic_suffix}_std_rewards_over_{num_runs}_runs.npy")
                std_steps_file = os.path.join(results_dir, f"{strategy}{stochastic_suffix}_std_steps_over_{num_runs}_runs.npy")

                if os.path.exists(avg_rewards_file) and os.path.exists(avg_steps_file):
                    current_data_dict = scenario1_data_stochastic if is_stochastic else scenario1_data_deterministic
                    
                    current_data_dict[f"{strategy.capitalize()} Decay"] = {
                        "avg_rewards": np.load(avg_rewards_file),
                        "avg_steps": np.load(avg_steps_file),
                        "std_rewards": np.load(std_rewards_file) if os.path.exists(std_rewards_file) else None,
                        "std_steps": np.load(std_steps_file) if os.path.exists(std_steps_file) else None
                    }
                    print(f"Loaded data for {strategy} ({mode_name})")
                else:
                    print(f"Data files not found for {strategy} ({mode_name}). Run Scenario1.py first.")
            except Exception as e:
                print(f"Error loading data for {strategy} ({mode_name}): {e}")
    
    if scenario1_data_deterministic:
        plot_learning_curves(scenario1_data_deterministic, title_prefix="Scenario 1 (Deterministic): ", output_dir=results_dir, base_filename="s1_deterministic_comparison")
    if scenario1_data_stochastic:
        plot_learning_curves(scenario1_data_stochastic, title_prefix="Scenario 1 (Stochastic): ", output_dir=results_dir, base_filename="s1_stochastic_comparison")

    print("Plotting complete (if data was found).")