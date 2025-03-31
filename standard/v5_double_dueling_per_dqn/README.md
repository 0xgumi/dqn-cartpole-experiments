# 🧠 Version 5 – Double + Dueling + PER

This folder contains the most advanced version of the DQN agent, combining **Double Q-learning**, **Dueling architecture**, and **Prioritized Experience Replay (PER)**.

---

## 📌 Key Features

- Prioritized Experience Replay improves sample efficiency  
- Combines benefits of Double Q-learning and Dueling structure  
- More aggressive learning, but prone to instability in some trials  
- Aimed at maximizing long-term score potential

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
alpha = 0.6                # How much prioritization is used (0 = uniform)
beta = 0.4                 # Importance-sampling correction
beta_increment = 0.001     # How quickly beta moves toward 1.0

```

---

## 📁 Files and Purpose

- [`dqn_cartpole_v5.py`](./dqn_cartpole_v5.py): Advanced agent with PER integration  
- [`dqn_cartpole_v5_test.py`](./dqn_cartpole_v5_test.py): 50 repeated runs for evaluation  
- [`../../results/v5/dqn_cartpole_v5_results.csv`](../results/v5/dqn_cartpole_v5_results.csv): CSV summary of trial outcomes
