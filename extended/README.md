# 🧠 DQN CartPole - Extended Experiments

This directory contains the results of 5 Deep Q-Network (DQN) variants tested on the **extended CartPole environment** (max score = 1000).  
All experiments use the same architecture as the standard setting, but with longer episodes to better observe learning beyond the 500-point cap.

---

## 🎯 Objective

We extended the episode limit to 1000 to analyze long-term decision quality and model robustness, especially after early convergence in the standard setting.

---

## ⚙️ Shared Hyperparameters

```python
gamma = 0.99                # Discount factor  
epsilon = 1.0               # Initial exploration rate  
epsilon_min = 0.01          # Minimum exploration rate  
epsilon_decay = 0.995       # Exploration decay rate  
learning_rate = 0.001       # Adam optimizer learning rate  
batch_size = 32             # Batch size for training  
memory_size = 5000          # Size of experience replay buffer  
train_start = 1000          # Minimum experiences before training starts  
N = 5                       # Target network update interval (every N episodes)  
```

- Environment: Custom CartPole-v1 (max steps = 1000)
- Episodes per trial: 300
- Trials per version: 50

---

## 🧪 Version Summary

| Version | Description |
|---------|-------------|
| **1000v1** | Basic DQN |
| **1000v2** | Double DQN |
| **1000v3** | Dueling DQN |
| **1000v4** | Double + Dueling DQN |
| **1000v5** | Double + Dueling + PER |

These match the standard versions exactly in architecture and settings — only the episode length is different.

---

## 📊 Results Overview

| Version | Min Avg | Max Avg | Avg Score | Std Dev | Reached 1000 Ratio |
|---------|---------|---------|-----------|---------|---------------------|
| 1000v1 | 28.74 | 520.72 | 223.50 | 86.53 | 0.18 |
| 1000v2 | 31.18 | 560.50 | 251.47 | 96.84 | 0.22 |
| 1000v3 | 22.70 | 462.78 | 204.93 | 85.32 | 0.14 |
| 1000v4 | 20.02 | 542.04 | 243.88 | 91.10 | 0.24 |
| 1000v5 | 18.06 | 621.16 | 218.78 | 107.61 | 0.32 |

All values are averages over 50 trial runs.

---

## 🔍 Observations & Insights

- **v5** shows the highest max scores, reaching 1000 more often than others — but at the cost of high volatility.
- **v2 and v4** strike the best balance between performance and stability.
- **v3** underperforms in the extended setting, suggesting its weakness in long-term planning.
- Performance gaps became **more visible** than in the standard setting.

The longer episodes revealed score collapses that were previously hidden by the 500 cap — confirming the need for this extended evaluation.

---

## 💡 Summary

- Architectural improvements help reach higher scores, but also introduce instability.
- PER boosts max scores, but destabilizes training.
- Double DQN provides the most consistent improvement.
- The extended setup made it easier to tell these trade-offs apart.

---

## 📁 Files & References

- Each version contains:
  - `*.py`: Training and test code
  - `*_results.csv`: Performance logs
  - `README.md`: Version-specific explanation

- Visualization:
  - [Extended Results Graph](./results/extended_dual_axis.png)

> 📌 For background and comparison with the 500-point setting, see [`../standard/README.md`](../standard/README.md)

