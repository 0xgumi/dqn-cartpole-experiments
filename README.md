# 🧠 DQN CartPole Experiments

This repository contains Deep Q-Network (DQN) experiments on the CartPole-v1 environment using PyTorch.  
It is structured to support multiple experimental versions, with organized results and summaries across two tracks: **standard** and **extended**.

---

## 📁 Project Structure
```
dqn_cartpole/
├── standard/      ← CartPole-v1 default settings (max score = 500)
│   ├── v1_basic_dqn/
│   ├── v2_double_dqn/
│   ├── v3_dueling_dqn/
│   ├── v4_double_dueling_dqn/
│   └── v5_double_dueling_per_dqn/
│
├── extended/      ← Custom CartPole (max score = 1000)
│   ├── 1000v1_basic_dqn/
│   ├── 1000v2_double_dqn/
│   ├── 1000v3_dueling_dqn/
│   ├── 1000v4_double_dueling_dqn/
│   └── 1000v5_double_dueling_per_dqn/
│
├── results/
│   ├── v1/ ~ v5/
│   ├── 1000v1/ ~ 1000v5/
```

- Each version folder contains both training and test code.
- The `results/` directory contains CSV files summarizing 50-trial performance metrics for each version.

---

## 🧪 Experiment Overview

This project evaluates how various improvements to the Deep Q-Network algorithm affect learning performance on CartPole-v1.

**Implemented Variants:**

| Version | Algorithm                                |
|---------|-------------------------------------------|
| v1      | Basic DQN                                 |
| v2      | Double DQN                                |
| v3      | Dueling DQN                               |
| v4      | Double + Dueling DQN                      |
| v5      | Double + Dueling + Prioritized Replay (PER) |

Each version is tested under two different settings:

- **Standard** (CartPole-v1 max score = 500)
- **Extended** (custom max score = 1000)

This dual setup reveals performance differences more clearly, especially when models frequently reach the 500-point cap in the standard version.

---

## 📊 Key Findings

- **PER (v5)** shows the highest peak performance but also greater variability.
- **Dueling + Double (v4)** improves consistency but can be sensitive to exploration decay.
- The **extended setting** provides better separation for top-performing models.
- The **standard setting** underrepresents performance once the agent regularly scores 500.

> Detailed results and interpretation are provided in each version folder and summarized in `standard/README.md` and `extended/README.md`.

---

## 🔧 Requirements

- Python 3.10+
- `torch`
- `gymnasium`
- `numpy`
- `pandas`

Install all with:

```bash
pip install torch gymnasium numpy pandas
```
---

## 📌 Blog Version  
This project is also available on my blog:  
🔗 [https://0xgumi.netlify.app/projects/dqn-cartpole](https://0xgumi.netlify.app/projects/dqn-cartpole)
