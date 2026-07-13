import torch

CONFIG = {
    "ENV_NAME": "ALE/Pacman-v5",
    "FRAME_STACK": 4,
    "BATCH_SIZE": 32,
    "GAMMA": 0.99,
    "LR": 0.00025,
    "BUFFER_SIZE": 100000,
    "MIN_REPLAY_SIZE": 10000,
    "EPS_START": 1.0,
    "EPS_END": 0.1,
    "EPS_DECAY_STEPS": 500000,
    "TARGET_UPDATE_FREQ": 10000,
    "LOG_INTERVAL_EPISODES": 10,
    "MAX_STEPS": 200000,
    "MODEL_SAVE_PATH": "weights2/dqn_pacman_latest.pth",
    "BEST_MODEL_SAVE_PATH": "weights/dqn_pacman_best.pth", #dqn_pacman_best.pth",
    "PLOT_SAVE_PATH": "logs/training_metrics.png",
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# import random
# from collections import deque

# # Hyperparameters Configuration
# CONFIG = {
#     "ENV_NAME": "ALE/Pacman-v5",
#     "FRAME_STACK": 4,
#     "BATCH_SIZE": 32,
#     "GAMMA": 0.99,                 # Discount factor
#     "LR": 0.00025,                 # Learning rate
#     "BUFFER_SIZE": 100000,         # Max capacity of replay buffer
#     "MIN_REPLAY_SIZE": 10000,      # Steps to take before training starts
#     "EPS_START": 1.0,              # Initial exploration probability
#     "EPS_END": 0.1,                # Final exploration probability
#     "EPS_DECAY_STEPS": 500000,     # Over how many steps to decay epsilon
#     "TARGET_UPDATE_FREQ": 10000,   # How often to sync target network (in steps)
#     "LOG_INTERVAL_EPISODES": 10,   # Print logs every X episodes
#     "MAX_STEPS": 2000000,          # Total environment steps to train for
# }
