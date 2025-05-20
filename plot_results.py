# plot_results.py
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curves(strategies_data, title_prefix=""):
    plt.figure(figsize=(12, 5))

    # Plot Rewards
    plt.subplot(1, 2, 1)
    for strategy_name, data in strategies_data.items():
        if 'rewards' in data:
            # Apply a simple moving average for smoother curves
            rewards = data['rewards']
            window_size = 50 # Adjust as needed
            if len(rewards) >= window_size:
                smoothed_rewards = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_rewards, label=f"{strategy_name} Rewards")
            else:
                 plt.plot(rewards, label=f"{strategy_name} Rewards (raw)")
    plt.xlabel("Epochs (smoothed)")
    plt.ylabel("Average Reward")
    plt.title(f"{title_prefix}Smoothed Learning Curve (Rewards)")
    plt.legend()
    plt.grid(True)

    # Plot Steps
    plt.subplot(1, 2, 2)
    for strategy_name, data in strategies_data.items():
        if 'steps' in data:
            steps = data['steps']
            window_size = 50 # Adjust as needed
            if len(steps) >= window_size:
                smoothed_steps = np.convolve(steps, np.ones(window_size)/window_size, mode='valid')
                plt.plot(smoothed_steps, label=f"{strategy_name} Steps")
            else:
                plt.plot(steps, label=f"{strategy_name} Steps (raw)")
    plt.xlabel("Epochs (smoothed)")
    plt.ylabel("Average Steps per Epoch")
    plt.title(f"{title_prefix}Smoothed Learning Curve (Steps)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{title_prefix.lower().replace(' ', '_')}learning_curves.png")
    plt.show()

if __name__ == "__main__":
    # Load data for Scenario 1
    try:
        rewards_mult = np.load("scenario1_epoch_rewards_multiplicative.npy")
        steps_mult = np.load("scenario1_epoch_steps_multiplicative.npy")
        rewards_lin = np.load("scenario1_epoch_rewards_linear.npy")
        steps_lin = np.load("scenario1_epoch_steps_linear.npy")

        scenario1_data = {
            "Multiplicative Decay": {"rewards": rewards_mult, "steps": steps_mult},
            "Linear Decay": {"rewards": rewards_lin, "steps": steps_lin}
        }
        plot_learning_curves(scenario1_data, title_prefix="Scenario 1: ")
    except FileNotFoundError:
        print("Make sure to run Scenario1.py for both 'multiplicative' and 'linear' strategies first to generate .npy files.")