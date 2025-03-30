# 🧠 Version 1 – Basic DQN

This folder contains the code implementation of the **basic Deep Q-Network (DQN)** agent applied to the CartPole-v1 environment.

---

## 📌 Key Features

- Fully connected network with 2 hidden layers (24 units each)
- Experience Replay using a fixed-size buffer
- Target Network updated every N episodes
- Epsilon-Greedy policy with exponential decay

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
## 📁 Files and Purpose

- [`dqn_cartpole_v1.py`](./dqn_cartpole_v1.py): Main training code for basic DQN  
- [`dqn_cartpole_v1_test.py`](./dqn_cartpole_v1_test.py): Repeated training script for statistical evaluation  
- [`results/v1/dqn_cartpole_v1_results.csv`](../../results/v1/dqn_cartpole_v1_results.csv): Output CSV of 50 trials
