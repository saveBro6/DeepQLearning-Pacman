import os
import numpy as np
import torch
import time
from config import CONFIG
from src.preprocessing_wrappers import make_env
from models.DeepQ_CNN import DQNCNN

def live_inference():
    if not os.path.exists(CONFIG["MODEL_SAVE_PATH"]):
        print(f"Error: Model checkpoint not found at '{CONFIG['MODEL_SAVE_PATH']}'. Please run train.py first.")
        return
    # Instantiate with explicit 'human' mode rendering
    env = make_env(CONFIG["ENV_NAME"], render_mode="human")
    model = DQNCNN(CONFIG["FRAME_STACK"], env.action_space.n).to(CONFIG["DEVICE"])
    model.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"], map_location=CONFIG["DEVICE"]))
    model.eval()
    
    state, _ = env.reset()
    done = False
    while not done:
        time.sleep(0.02) # Keeps game speed viewable
        with torch.no_grad():
            st = torch.tensor(np.array(state), device=CONFIG["DEVICE"]).unsqueeze(0)
            action = model(st).argmax(dim=1).item()
        next_state, _, term, trunc, _ = env.step(action)
        state = np.array(next_state)
        done = term or trunc
    env.close()

if __name__ == "__main__":
    live_inference()