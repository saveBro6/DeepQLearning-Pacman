import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from models.DeepQ_CNN import DQNCNN
from config import CONFIG

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (np.array(state), np.array(action, dtype=np.int64), 
                np.array(reward, dtype=np.float32), np.array(next_state), 
                np.array(done, dtype=np.uint8))
                
    def __len__(self):
        return len(self.buffer)

class DCQNAgent:
    def __init__(self, num_actions, device):
        self.num_actions = num_actions
        self.device = device
        
        # Initialize Policy and Target Networks
        self.policy_net = DQNCNN(CONFIG["FRAME_STACK"], num_actions).to(device)
        self.target_net = DQNCNN(CONFIG["FRAME_STACK"], num_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=CONFIG["LR"])
        self.memory = ReplayBuffer(CONFIG["BUFFER_SIZE"])

    def select_action(self, state, epsilon):
        # Epsilon-greedy selection
        if random.random() < epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            with torch.no_grad():
                # State shape: (4, 84, 84) -> add batch dim: (1, 4, 84, 84)
                state_tensor = torch.tensor(np.array(state), device=self.device).unsqueeze(0)
                return self.policy_net(state_tensor).argmax(dim=1).item()

    def train_step(self):
        if len(self.memory) < CONFIG["MIN_REPLAY_SIZE"]:
            return None # Not enough data yet
            
        states, actions, rewards, next_states, dones = self.memory.sample(CONFIG["BATCH_SIZE"])
        
        states_t = torch.tensor(states, device=self.device)
        actions_t = torch.tensor(actions, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, device=self.device)
        next_states_t = torch.tensor(next_states, device=self.device)
        dones_t = torch.tensor(dones, device=self.device)
        
        # Compute current Q values: Q(s_t, a_t)
        current_q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(-1)
        
        # Compute Target Q values: max_a Q_target(s_{t+1}, a)
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1)[0]
            expected_q_values = rewards_t + (CONFIG["GAMMA"] * next_q_values * (1 - dones_t))
        
        # Compute Loss (Huber Loss is preferred for Atari)
        loss = nn.SmoothL1Loss()(current_q_values, expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())