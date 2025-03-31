# 📊 Results – Double + Dueling + PER (1000v5)

Summary of 50 trials with **Double + Dueling + Prioritized Replay (PER)** in the extended CartPole-v1 setup.

---

## 🧪 Score Statistics

- **Min Score (Avg)**: 18.76  
- **Max Score (Avg)**: 681.76  
- **Average Score**: 217.18  
- **Standard Deviation**: 122.15  
- **Reached 1000 Ratio**: 0.28

---

## 📌 Observations

- Highest potential among all variants but also the most unstable.
- PER leads to more exploratory and diverse behaviors.
- Volatility increases with longer episodes.

---

## 🔍 Notes

- Extended episodes emphasize PER’s trade-off between reward and consistency.
- Hyperparameters (α, β) could be tuned for improved stability.
