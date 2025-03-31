# 🧠 Version 1000v3 – Dueling DQN

This folder contains the **Dueling DQN** implementation tested on the extended CartPole-v1 environment (1000 steps max).

---

## 📌 Key Features

- Separate estimation of state value (V) and advantage (A)
- Better generalization for states with similar value
- Slightly higher score ceiling in longer episodes

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

- [`dqn_cartpole_1000v3.py`](./dqn_cartpole_1000v3.py)  
- [`dqn_cartpole_1000v3_test.py`](./dqn_cartpole_1000v3_test.py)  
- [`../../results/1000v3/dqn_cartpole_1000v3_results.csv`](../../results/1000v3/dqn_cartpole_1000v3_results.csv)
