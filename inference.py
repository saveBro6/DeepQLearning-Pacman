import os
import numpy as np
import torch
import time
from config import CONFIG
from src.preprocessing_wrappers import make_env
from models.DeepQ_CNN import DQNCNN

def live_inference(model_path):
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at '{model_path}'.")
        return
    # Instantiate with explicit 'human' mode rendering
    env = make_env(CONFIG["ENV_NAME"], render_mode="human")
    model = DQNCNN(CONFIG["FRAME_STACK"], env.action_space.n).to(CONFIG["DEVICE"])
    model.load_state_dict(torch.load(model_path, map_location=CONFIG["DEVICE"]))
    model.eval()
    
    state, _ = env.reset()
    done = False
    while not done:
        time.sleep(0.02) # Keeps game speed viewable
        with torch.no_grad():
            st = torch.tensor(np.array(state), device=CONFIG["DEVICE"]).unsqueeze(0)
            action = model(st).argmax(dim=1).item()
        next_state, reward, term, trunc, _ = env.step(action)
        if reward != 0.0 and reward != 1.0:
            print("Reward received:", reward)
        state = np.array(next_state)
        done = term or trunc
    env.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference using DQN agent on Pacman")
    parser.add_argument("model_path", type=str, help="Path to the model weights file")
    args = parser.parse_args()
    live_inference(args.model_path)