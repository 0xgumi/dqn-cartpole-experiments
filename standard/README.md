# 🧠 DQN CartPole - Standard Experiments

This directory contains the results of 5 Deep Q-Network (DQN) variants tested on the standard CartPole-v1 environment (max score = 500).  
All experiments share identical training conditions, allowing us to isolate the effects of algorithmic improvements only.

---

## 🎯 Objective

The goal of this experiment series is to evaluate how specific upgrades to the DQN architecture affect performance.  
Each version modifies a distinct component (e.g., target estimation or network structure), and we observe how these changes impact learning dynamics.

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

- Environment: CartPole-v1 (max steps = 500)
- Episodes per trial: 300
- Number of trials per version: 50

---

## 🧪 Version Summary

| Version | Description |
|---------|-------------|
| **v1** | Basic DQN |
| **v2** | Double DQN (decouples action selection and evaluation) |
| **v3** | Dueling DQN (separates value and advantage streams) |
| **v4** | Double + Dueling DQN |
| **v5** | Double + Dueling + Prioritized Experience Replay (PER) |

Each version builds on the previous ones to test cumulative effects of DQN enhancements.

---

## 📊 Results Overview

| Version | Min Avg | Max Avg | Avg Score | Std Dev | Reached 500 Ratio |
|---------|---------|---------|-----------|---------|-------------------|
| v1 | 34.58 | 359.34 | 192.82 | 70.60 | 0.20 |
| v2 | 29.16 | 369.96 | 195.97 | 75.12 | 0.26 |
| v3 | 27.30 | 352.28 | 186.19 | 74.08 | 0.28 |
| v4 | 21.02 | 364.34 | 182.77 | 74.56 | 0.28 |
| v5 | 16.60 | 378.00 | 153.87 | 84.30 | 0.40 |

Each value represents the average across 50 training runs.

---

## 🔍 Observations & Analysis

- As versions progressed (v1 → v5), **minimum scores decreased**, indicating early-phase instability.
- **v2 and v4** achieved strong average scores with balanced learning.
- **v5 reached 500 points most frequently**, but also had the **highest variance** and lowest average — highlighting the instability introduced by PER.
- **v3**, while strong in theory, underperformed slightly in stability and average score.

Even when a model reaches the 500-point cap, its learning may collapse later — showing the limitations of a capped evaluation metric.

---

## 💡 Insights

- Advanced techniques like PER boost peak performance, but often at the cost of stability.
- Double DQN (v2) provided a good balance between performance and reliability.
- Every architectural improvement introduces **new trade-offs** rather than guaranteed upgrades.

---

## 🧠 Why We Moved to Extended Setting

The 500-score cap limited our ability to judge true performance differences.  
Many models (v3~v5) frequently maxed out early, making it hard to tell which ones were actually better.

So, we extended the episode limit to 1000 steps in the `extended/` experiments.  
This allowed us to better observe long-term stability, decision-making quality, and robustness.

> For continuation of this study, see [`extended/README.md`](../extended/README.md)

---

## 📁 Files & References

- Each version contains:
  - `*.py`: Training and testing code
  - `*_results.csv`: Performance logs
  - `README.md`: Explanation of the version

- Visualization:
  - [Overall Graphs in dqn_cartpole/](../../results/)

