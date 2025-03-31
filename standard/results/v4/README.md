# 📊 Results – Double + Dueling DQN (v4)

This file summarizes the evaluation results of the **Double + Dueling DQN agent** on CartPole-v1.  
We ran the experiment **50 times**, and computed the following:

---

## 🧪 Score Statistics

- **Minimum Score (Avg)**: 21.02  
- **Maximum Score (Avg)**: 364.34  
- **Average Score**: 182.77  
- **Standard Deviation**: 74.56  
- **Reached 500 Ratio**: 28%

---

## 📌 Observations

- Combining Double and Dueling improves 500 reachability but not average score.
- Maintains decent max performance with similar variance to previous versions.
- Suggests architectural improvements do not always compound linearly.
- Best results often depend on interaction between exploration schedule and network complexity.

---

## 🔍 Notes

- Same epsilon strategy as other versions was used.
- Could benefit from fine-tuned PER or adaptive exploration scheduling.
