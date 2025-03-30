# 📊 Results – Basic DQN (v1)

This file summarizes the evaluation results of the **basic DQN agent** on CartPole-v1.  
We ran the experiment **50 times**, and computed the following:

---

## 🧪 Score Statistics

- **Minimum Score (Avg)**: 34.58  
- **Maximum Score (Avg)**: 359.34  
- **Average Score**: 192.82  
- **Standard Deviation**: 70.60  
- **Reached 500 Ratio**: 20%

---

## 📌 Observations

- The basic DQN can reach moderate scores but is prone to early failure.
- Rarely reaches the perfect score of 500.
- Performance is unstable depending on initialization and early exploration.
- High variance suggests a need for better target estimation or value separation.

---

## 🔍 Notes

- Same hyperparameters were used for all versions.
- This serves as the baseline to evaluate the effect of algorithmic improvements in v2–v5.
