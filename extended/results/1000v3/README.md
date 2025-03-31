# 📊 Results – Dueling DQN (1000v3)

Summary of 50 trials with **Dueling DQN** in the extended CartPole-v1 setup.

---

## 🧪 Score Statistics

- **Min Score (Avg)**: 21.08  
- **Max Score (Avg)**: 581.34  
- **Average Score**: 222.11  
- **Standard Deviation**: 98.22  
- **Reached 1000 Ratio**: 0.18

---

## 📌 Observations

- The dueling network helps identify valuable states but suffers in long episodes.
- Increased variance compared to v2; instability more noticeable with 1000-step runs.
- May require tighter control of exploration or longer training for reliability.

---

## 🔍 Notes

- Same hyperparameters were used across all versions.
- Performance shows dueling’s strengths but also its limitations in extended settings.
