import os
import numpy as np
import torch
from config import CONFIG
from src.preprocessing_wrappers import make_env
from models.DeepQ_CNN import DQNCNN

def evaluate(model_path, num_episodes=100):
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at '{model_path}'.")
        return
    env = make_env(CONFIG["ENV_NAME"])
    model = DQNCNN(CONFIG["FRAME_STACK"], env.action_space.n).to(CONFIG["DEVICE"])
    model.load_state_dict(torch.load(model_path, map_location=CONFIG["DEVICE"]))
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
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DQN agent on Pacman")
    parser.add_argument("model_path", type=str, help="Path to the model weights file")
    parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to evaluate")
    args = parser.parse_args()
    evaluate(args.model_path, num_episodes=args.num_episodes)