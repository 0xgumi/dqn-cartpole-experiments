import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple
import heapq

# Environment Setup
env = gym.make("CartPole-v1")
env._max_episode_steps = 1000
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Define the DQN model
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)

        self.value_stream = nn.Linear(24, 1)
        self.advantage_stream = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean(1, keepdim=True))

# Hyperparameters
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
memory_size = 5000
train_start = 1000
N = 5  # Target network update interval
alpha = 0.6  # Controls prioritization strength in PER sampling
beta = 0.4   # Initial value for importance-sampling correction
beta_increment = 0.001  # Increment rate for beta

# Define Prioritized Experience Replay (PER)
class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        transition = self.Transition(state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)

        batch = self.Transition(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-5  # Prevent zero priority by adding small constant

# Initialize networks and optimizer
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
q_network = DuelingDQN(state_size, action_size).to(device)
target_network = DuelingDQN(state_size, action_size).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss(reduction="none")
memory = PrioritizedReplayBuffer(memory_size)

# Training function
def train_dqn():
    global beta
    if len(memory.buffer) < train_start:
        return

    beta = min(1.0, beta + beta_increment)

    batch, indices, weights = memory.sample(batch_size, beta)
    states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(batch.reward, dtype=torch.float32).to(device)
    next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
    dones = torch.tensor(batch.done, dtype=torch.float32).to(device)
    weights = weights.to(device)

    next_actions = torch.argmax(q_network(next_states), dim=1, keepdim=True)
    next_q_values = target_network(next_states).gather(1, next_actions).squeeze()

    target_q_values = rewards + (gamma * next_q_values * (1 - dones))
    q_values = q_network(states).gather(1, actions).squeeze()

    td_errors = q_values - target_q_values.detach()
    loss = (loss_fn(q_values, target_q_values.detach()) * weights).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    memory.update_priorities(indices, td_errors.detach().cpu().numpy())

# Run a single experiment
episodes = 300
last_50_scores = []

for episode in range(episodes):
    state, _ = env.reset()
    state = np.array(state)
    total_reward = 0

    for t in range(1000):
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

        memory.add(state, action, reward, next_state, done)
        state = next_state
        train_dqn()

        if done:
            break

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    if episode % N == 0:
        target_network.load_state_dict(q_network.state_dict())

    if episode >= 250:
        last_50_scores.append(total_reward)

    print(f"Episode {episode + 1}/{episodes}, Score: {total_reward}, Epsilon: {epsilon:.4f}")

# Evaluate last 50 episodes
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
env.close()
