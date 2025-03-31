# 📊 Results – Double DQN (v2)

This file summarizes the evaluation results of the **Double DQN agent** on CartPole-v1.  
We ran the experiment **50 times**, and computed the following:

---

## 🧪 Score Statistics

- **Minimum Score (Avg)**: 29.16  
- **Maximum Score (Avg)**: 369.96  
- **Average Score**: 195.97  
- **Standard Deviation**: 75.12  
- **Reached 500 Ratio**: 26%

---

## 📌 Observations

- Double DQN helps reduce Q-value overestimation compared to v1.
- Achieves slightly better average performance and stability.
- Still experiences variance, but with fewer catastrophic failures than basic DQN.
- Reaches the perfect score of 500 more often than v1.

---

## 🔍 Notes

- Hyperparameters are identical to all other versions.
- This version helps isolate the impact of more accurate value estimation.
