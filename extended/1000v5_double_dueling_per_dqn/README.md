# 🧠 Version 1000v5 – Double + Dueling + PER

This folder contains the **Double DQN + Dueling Network + Prioritized Experience Replay (PER)** implementation tested on the extended CartPole-v1 environment (1000 steps max).

---

## 📌 Key Features

- Adds prioritized sampling to emphasize valuable experiences
- Achieves highest peak scores, but with significant variance
- Best when long episodes allow full exploitation of PER

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

- [`dqn_cartpole_1000v5.py`](./dqn_cartpole_1000v5.py)  
- [`dqn_cartpole_1000v5_test.py`](./dqn_cartpole_1000v5_test.py)  
- [`../../results/1000v5/dqn_cartpole_1000v5_results.csv`](../../results/1000v5/dqn_cartpole_1000v5_results.csv)
