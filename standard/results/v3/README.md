# 📊 Results – Dueling DQN (v3)

This file summarizes the evaluation results of the **Dueling DQN agent** on CartPole-v1.  
We ran the experiment **50 times**, and computed the following:

---

## 🧪 Score Statistics

- **Minimum Score (Avg)**: 27.30  
- **Maximum Score (Avg)**: 352.28  
- **Average Score**: 186.19  
- **Standard Deviation**: 74.08  
- **Reached 500 Ratio**: 28%

---

## 📌 Observations

- The dueling architecture improves value estimation but adds sensitivity to noise.
- Slight improvement in score ceiling over v1, but not clearly superior to v2.
- Reaches 500 more often than v1, indicating better policy shaping.
- Slightly higher variance suggests instability in some episodes.

---

## 🔍 Notes

- Hyperparameters remained identical across versions.
- v3 shows benefits of architectural refinement but doesn’t yet outperform double Q-learning in consistency.
