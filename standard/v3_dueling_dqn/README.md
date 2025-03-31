# 🧠 Version 3 – Dueling DQN

This folder contains the code implementation of the **Dueling Deep Q-Network (DQN)** agent applied to the CartPole-v1 environment.

---

## 📌 Key Features

- Dueling architecture separates **state value (V)** and **advantage (A)** streams
- Better generalization in environments where action values are similar
- Maintains experience replay and target network from previous versions
- Uses same exploration and training settings as earlier versions

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

- [`dqn_cartpole_v3.py`](./dqn_cartpole_v3.py): Main training code with Dueling DQN  
- [`dqn_cartpole_v3_test.py`](./dqn_cartpole_v3_test.py): 50-trial evaluation for performance analysis  
- [`../../results/v3/dqn_cartpole_v3_results.csv`](../results/v3/dqn_cartpole_v3_results.csv): Result CSV summarizing 50 trials
