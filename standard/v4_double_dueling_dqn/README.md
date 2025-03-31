# 🧠 Version 4 – Double + Dueling DQN

This folder contains the implementation of a **combined Double DQN + Dueling DQN** agent for the CartPole-v1 environment.

---

## 📌 Key Features

- Combines Double Q-learning and Dueling architecture
- More robust value estimation with decoupled action selection and evaluation
- Enhanced separation between state value and action advantage
- Aims to combine stability (Double) and better representation (Dueling)

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

- [`dqn_cartpole_v4.py`](./dqn_cartpole_v4.py): Combined architecture with training logic  
- [`dqn_cartpole_v4_test.py`](./dqn_cartpole_v4_test.py): Repeated experiments for performance summary  
- [`../../results/v4/dqn_cartpole_v4_results.csv`](../../results/v4/dqn_cartpole_v4_results.csv): CSV result of 50 trials
