# 🧠 Version 1000v1 – Basic DQN

This folder contains the implementation of the **basic Deep Q-Network (DQN)** algorithm, applied to the extended CartPole-v1 environment (max steps = 1000).

---

## 📌 Key Features

- Two-layer fully connected neural network
- Experience Replay using a fixed-size buffer
- Target Network updated every N episodes
- Epsilon-Greedy policy with decay

---

## ⚙️ Hyperparameters Used
```python
gamma = 0.99                # Discount factor  
epsilon = 1.0               # Initial exploration rate  
epsilon_min = 0.01          # Minimum exploration rate  
epsilon_decay = 0.995       # Exploration decay rate  
learning_rate = 0.001       # Adam optimizer learning rate  
batch_size = 32             # Batch size for training  
memory_size = 5000          # Experience replay buffer size  
train_start = 1000          # Start training after this many experiences  
N = 5                       # Target network update interval  
```

---

## 📁 Files and Purpose

- [`dqn_cartpole_1000v1.py`](./dqn_cartpole_1000v1.py): Main training script  
- [`dqn_cartpole_1000v1_test.py`](./dqn_cartpole_1000v1_test.py): Script to run 50 trials for statistical evaluation  
- [`../../results/1000v1/dqn_cartpole_1000v1_results.csv`](../../results/1000v1/dqn_cartpole_1000v1_results.csv): Result summary (CSV)
