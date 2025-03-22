import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# 1️⃣ Environment Setup
env = gym.make("CartPole-v1")  
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# 2️⃣ Define the DQN model
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

# 3️⃣ Hyperparameters
gamma = 0.99  
epsilon = 1.0  
epsilon_min = 0.01  
epsilon_decay = 0.995
learning_rate = 0.001  
batch_size = 32  
memory_size = 5000  
train_start = 1000  
N = 5  # Target network update interval

# 4️⃣ Initialize networks and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  
q_network = DQN(state_size, action_size).to(device)
target_network = DQN(state_size, action_size).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
memory = deque(maxlen=memory_size)

# 5️⃣ DQN Training function
def train_dqn():
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

# 6️⃣ Run a single experiment
episodes = 300
last_50_scores = []

for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state)
    total_reward = 0

    for t in range(500):
        # Action selection (exploration vs. exploitation)
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

        # Store experience
        memory.append((state, action, reward, next_state, done))
        state = next_state
        train_dqn()

        if done:
            break

    # Decay exploration rate
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Update target network every N episodes
    if episode % N == 0:
        target_network.load_state_dict(q_network.state_dict())

    # Record scores of last 50 episodes
    if episode >= 250:
        last_50_scores.append(total_reward)

    print(f"Episode {episode + 1}/{episodes}, Score: {total_reward}, Epsilon: {epsilon:.4f}")

# 7️⃣ Evaluate last 50 episodes
if len(last_50_scores) > 0:
    min_score = min(last_50_scores)
    max_score = max(last_50_scores)
    avg_score = sum(last_50_scores) / len(last_50_scores)
    std_dev = np.std(last_50_scores)

    print("\n🔥 Performance of Last 50 Episodes:")
    print(f"✅ Min Score: {min_score}")
    print(f"✅ Max Score: {max_score}")
    print(f"✅ Average Score: {avg_score:.2f}")
    print(f"✅ Standard Deviation: {std_dev:.2f}")

print("\nTraining Complete!")
env.close()  # Close environment
