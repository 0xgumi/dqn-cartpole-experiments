import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd
import os
from collections import deque

# Get the directory of the currently running file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set paths for saving results
CSV_FILE = os.path.join(BASE_DIR, "dqn_cartpole_v1_results.csv")
TXT_FILE = os.path.join(BASE_DIR, "dqn_cartpole_v1_summary.txt")

# Environment setup
def create_env():
    return gym.make("CartPole-v1")

# DQN Model Definition
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995  # Exploration decay rate
learning_rate = 0.001
batch_size = 32
memory_size = 5000  # Increased memory size
train_start = 1000
N = 5  # Target network update interval

def train_dqn(q_network, target_network, memory, optimizer, loss_fn, device):
    if len(memory) < train_start:
        return
    
    batch = random.sample(memory, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = q_network(states).gather(1, actions)
    next_q_values = target_network(next_states).max(1)[0].detach()
    target_q_values = rewards + (gamma * next_q_values * (1 - dones))

    loss = loss_fn(q_values.squeeze(), target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Run DQN Experiment
def run_experiment(episodes=300, trials=50):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    results = []

    for trial in range(trials):
        env = create_env()
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n

        q_network = DQN(state_size, action_size).to(device)
        target_network = DQN(state_size, action_size).to(device)
        target_network.load_state_dict(q_network.state_dict())
        target_network.eval()
        optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
        loss_fn = nn.MSELoss()
        memory = deque(maxlen=memory_size)

        last_50_scores = []
        epsilon = 1.0
        
        for episode in range(episodes):
            state, _ = env.reset()
            state = np.array(state)
            total_reward = 0
            
            for t in range(500):
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                        action = torch.argmax(q_network(state_tensor)).item()
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = np.array(next_state)
                done = terminated or truncated
                total_reward += reward
                
                memory.append((state, action, reward, next_state, done))
                state = next_state
                train_dqn(q_network, target_network, memory, optimizer, loss_fn, device)

                if done:
                    break
            
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay

            if episode % N == 0:
                target_network.load_state_dict(q_network.state_dict())

            if episode >= 250:
                last_50_scores.append(total_reward)
        
        min_score = min(last_50_scores)
        max_score = max(last_50_scores)
        avg_score = sum(last_50_scores) / len(last_50_scores)
        std_dev = np.std(last_50_scores)

        results.append([trial+1, min_score, max_score, avg_score, std_dev])
        env.close()
        print(f"Trial {trial+1}/{trials} completed: Avg {avg_score:.2f}, Std Dev {std_dev:.2f}")
    
    # Save results to CSV
    df = pd.DataFrame(results, columns=["Trial", "Min Score", "Max Score", "Average Score", "Std Dev"])
    df.to_csv(CSV_FILE, index=False)
    
    # Save summary to TXT
    with open(TXT_FILE, "w") as f:
        f.write("🔥 DQN Experiment Results (50 Trial Runs)\n")
        f.write("------------------------------------\n")
        for row in results:
            f.write(f"[Trial {row[0]}] Min: {row[1]}, Max: {row[2]}, Avg: {row[3]:.2f}, Std Dev: {row[4]:.2f}\n")
        f.write("------------------------------------\n")
        f.write(f"📊 Overall Averages:\n")
        f.write(f"- Min Score Avg: {np.mean([r[1] for r in results]):.2f}\n")
        f.write(f"- Max Score Avg: {np.mean([r[2] for r in results]):.2f}\n")
        f.write(f"- Average Score: {np.mean([r[3] for r in results]):.2f}\n")
        f.write(f"- Std Dev Avg: {np.mean([r[4] for r in results]):.2f}\n")

if __name__ == "__main__":
    run_experiment()
