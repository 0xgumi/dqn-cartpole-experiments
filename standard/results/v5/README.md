# 📊 Results – Double + Dueling + PER (v5)

This file summarizes the evaluation results of the **most advanced DQN variant** on CartPole-v1.  
We ran the experiment **50 times**, and computed the following:

---

## 🧪 Score Statistics

- **Minimum Score (Avg)**: 16.60  
- **Maximum Score (Avg)**: 378.00  
- **Average Score**: 153.87  
- **Standard Deviation**: 84.30  
- **Reached 500 Ratio**: 40%

---

## 📌 Observations

- Highest rate of perfect score (500) among all standard models.
- Average performance dropped due to significant variance across trials.
- PER occasionally helps discover high-reward paths but introduces noise.
- Stability is reduced, possibly due to biased replay prioritization.

---

## 🔍 Notes

- This version trades off stability for potential high returns.
- PER hyperparameters (α, β) may require tuning for better consistency.
- Extended experiments with higher ceilings show this variant’s strength more clearly.
