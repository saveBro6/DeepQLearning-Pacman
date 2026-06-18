import os
import numpy as np
import torch
from config import CONFIG
from src.preprocessing_wrappers import make_env
from models.DeepQ_CNN import DQNCNN

def evaluate(num_episodes=100):
    if not os.path.exists(CONFIG["MODEL_SAVE_PATH"]):
        print(f"Error: Model checkpoint not found at '{CONFIG['MODEL_SAVE_PATH']}'. Please run train.py first.")
        return
    env = make_env(CONFIG["ENV_NAME"])
    model = DQNCNN(CONFIG["FRAME_STACK"], env.action_space.n).to(CONFIG["DEVICE"])
    model.load_state_dict(torch.load(CONFIG["MODEL_SAVE_PATH"], map_location=CONFIG["DEVICE"]))
    model.eval()
    
    scores = []
    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        done, total_reward = False, 0
        while not done:
            with torch.no_grad():
                st = torch.tensor(np.array(state), device=CONFIG["DEVICE"]).unsqueeze(0)
                action = model(st).argmax(dim=1).item()
            next_state, reward, term, trunc, _ = env.step(action)
            total_reward += reward
            state = np.array(next_state)
            done = term or trunc
        scores.append(total_reward)
        
    print(f"\nEvaluation Complete over {num_episodes} games.")
    print(f"Mean Score: {np.mean(scores):.2f} | Max: {np.max(scores)} | Min: {np.min(scores)}")
    env.close()

if __name__ == "__main__":
    evaluate()