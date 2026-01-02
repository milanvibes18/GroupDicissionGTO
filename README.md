# Group Decision Optimization System (GTO-RL-XAI)

## 📌 Project Overview
This project simulates how large social groups reach consensus. It moves beyond traditional averaging by modeling **Leadership Influence**, **Social Susceptibility**, and **Conflict** using a novel combination of:
1.  **Gorilla Troops Optimizer (GTO):** A nature-inspired algorithm for searching the "best" group decision.
2.  **Reinforcement Learning (PPO):** An AI agent that dynamically adjusts social parameters (Weights) to resolve conflicts faster.
3.  **Explainable AI (SHAP):** An analytical layer that explains *why* a specific decision prevailed.

## 🛠️ Tech Stack
* **Simulation:** Python, NumPy (Vectorized Agent Logic)
* **Optimization:** Custom GTO Implementation
* **AI/ML:** Stable Baselines3 (PPO), Gymnasium
* **Analysis:** SHAP (Shapley Additive Explanations)

## 🚀 How to Run

### 1. Run the Static Simulation
To see the Gorilla Troops Optimizer in action with fixed weights:
```bash
python main.py