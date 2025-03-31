# 🧠 Version 2 – Double DQN

This folder contains the code implementation of the **Double Deep Q-Network (Double DQN)** agent applied to the CartPole-v1 environment.

---

## 📌 Key Features

- Uses separate networks for action selection and evaluation
- Mitigates Q-value overestimation common in vanilla DQN
- Same architecture as basic DQN: 2 hidden layers, 24 units each
- Target network updated every N episodes

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

- [`dqn_cartpole_v2.py`](./dqn_cartpole_v2.py): Main training code for Double DQN  
- [`dqn_cartpole_v2_test.py`](./dqn_cartpole_v2_test.py): Repeated training script for statistical evaluation  
- [`results/v2/dqn_cartpole_v2_results.csv`](../results/v2/dqn_cartpole_v2_results.csv): Output CSV of 50 trials
