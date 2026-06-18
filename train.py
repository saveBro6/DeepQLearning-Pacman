import numpy as np
import torch
from config import CONFIG
from src.preprocessing_wrappers import make_env
from src.agent import DCQNAgent
from src.utils import plot_training_metrics

def main():
    print(f"Using device: {CONFIG['DEVICE']}")
    env = make_env(CONFIG["ENV_NAME"])
    agent = DCQNAgent(env.action_space.n, CONFIG["DEVICE"])
    
    steps_log, rewards_log = [], []
    episode_rewards = []
    episode_count, current_ep_reward = 0, 0
    state, _ = env.reset()
    state = np.array(state)
    
    print("Populating replay buffer... Training will begin shortly.")

    for global_step in range(1, CONFIG["MAX_STEPS"] + 1):
        # Calculate linear epsilon decay
        epsilon = max(CONFIG["EPS_END"], CONFIG["EPS_START"] - (global_step / CONFIG["EPS_DECAY_STEPS"]) * (CONFIG["EPS_START"] - CONFIG["EPS_END"]))
        
        # Action Step
        action = agent.select_action(state, epsilon)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Save to replay memory
        agent.memory.push(state, action, reward, np.array(next_state), done)
        state = np.array(next_state)
        current_ep_reward += reward
        
        # Step training optimization
        agent.train_step()
        
        # Periodically update Target Network
        if global_step % CONFIG["TARGET_UPDATE_FREQ"] == 0:
            agent.update_target_network()

        # Handle Episode Ending    
        if done:
            episode_count += 1
            episode_rewards.append(current_ep_reward)
            steps_log.append(global_step)
            rewards_log.append(current_ep_reward)
            running_reward = np.mean(episode_rewards[-100:]) # 100 episode moving average

            # Print logs every X episodes
            if episode_count % CONFIG["LOG_INTERVAL_EPISODES"] == 0:
                print(f"Episode: {episode_count} | "
                      f"Step: {global_step}/{CONFIG["MAX_STEPS"]} | "
                      f"Reward: {current_ep_reward:.1f} | "
                      f"Avg Reward (100 ep): {running_reward:.2f} | "
                      f"Epsilon: {epsilon:.3f} | "
                      f"Buffer Size: {len(agent.memory)}")
            
            # Reset for next episode
            state, _ = env.reset()
            state = np.array(state)
            current_ep_reward = 0

    torch.save(agent.policy_net.state_dict(), CONFIG["MODEL_SAVE_PATH"])
    plot_training_metrics(steps_log, rewards_log, CONFIG["PLOT_SAVE_PATH"])
    env.close()

if __name__ == "__main__":
    main()