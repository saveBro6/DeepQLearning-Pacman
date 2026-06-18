import matplotlib.pyplot as plt
import numpy as np

def plot_training_metrics(log_steps, log_rewards, save_path):
    moving_avg = np.zeros_like(log_rewards, dtype=np.float32)
    for i in range(len(log_rewards)):
        start_idx = max(0, i - 100 + 1)
        moving_avg[i] = np.mean(log_rewards[start_idx:i+1])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(log_steps, log_rewards, alpha=0.25, color='royalblue', label='Episode Reward')
    ax.plot(log_steps, moving_avg, color='crimson', linewidth=2.5, label='100-Episode Moving Avg')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward / Score')
    ax.set_title('DCQN Training Progress on ALE/Pacman-v5')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()