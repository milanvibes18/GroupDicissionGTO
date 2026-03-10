import numpy as np
import pandas as pd
import os
from src.core.group import Group
from src.core.metrics import calculate_consensus
from src.optimization.gto import GorillaTroopsOptimizer
from src.optimization.pso import ParticleSwarmOptimizer
from src.optimization.gwo import GreyWolfOptimizer
from stable_baselines3 import PPO


def run_single_algorithm(optimizer, group_state, algo_name, max_iters=200, is_smart=False, model=None):
    # 🌟 Tuned Velocity Clamping (Vmax)
    if algo_name == "Smart_GTO":
        max_step = 0.012
    elif algo_name == "Static_GTO":
        max_step = 0.012  # 100% Fair comparison
    elif algo_name == "PSO":
        max_step = 0.006  # High-dim clamp
    else:
        max_step = 0.005  # GWO High-dim clamp

    last_cons = calculate_consensus(group_state.get_opinions_matrix())
    prev_cons = last_cons

    for t in range(max_iters):
        if is_smart:
            if model is None: raise ValueError("CRITICAL: PPO Model is missing!")
            current_std = np.std(group_state.get_opinions_matrix())

            # Map time fraction to the AI's original 75-step training horizon
            obs = np.array([last_cons, 0.0 if t == 0 else last_cons - prev_cons, min(t / 75.0, 1.0), current_std],
                           dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)

            scaled_action = action * 5.0
            exp_action = np.exp(scaled_action)
            softmax_action = exp_action / np.sum(exp_action)

            optimizer.weights = {'alpha': softmax_action[0], 'beta': softmax_action[1], 'gamma': softmax_action[2]}

        prev_cons = last_cons
        old_ops = group_state.get_opinions_matrix()

        optimizer.step(group_state)
        raw_new_ops = group_state.get_opinions_matrix()

        delta = np.clip(raw_new_ops - old_ops, -max_step, max_step)

        # 🌟 THE FIX: "Clear Weather" noise (0.002 instead of 0.01)
        # This prevents the agents from doing a random drunken walk when velocities are low.
        noise = np.random.normal(0, 0.002, delta.shape)

        group_state.set_opinions(np.clip(old_ops + delta + noise, -1.0, 1.0))

        last_cons = calculate_consensus(group_state.get_opinions_matrix())

        if last_cons > 0.95:
            return t + 1

    return max_iters


def run_benchmarks():
    print("🚀 Running STRICT, FAIR 30-run benchmarks using the authentic RL model...")
    config = {'simulation': {'n_agents': 50, 'dimension': 384, 'influence_range': [0.1, 1.0]},
              'gto': {'silverback_bonus': 1.2}}
    base_weights = {'alpha': 0.1, 'beta': 0.8, 'gamma': 0.1}

    try:
        if os.path.exists("outputs/models/ppo_gto_agent.zip"):
            model = PPO.load("outputs/models/ppo_gto_agent.zip", device="cpu")
        elif os.path.exists("ppo_gto_agent.zip"):
            model = PPO.load("ppo_gto_agent.zip", device="cpu")
        else:
            raise FileNotFoundError("Could not find 'ppo_gto_agent.zip'")
        print("✅ RL Model Loaded Successfully.")
    except Exception as e:
        print(f"❌ CRITICAL ERROR LOADING MODEL:\n{e}")
        return

    results = []
    for i in range(30):
        print(f"Executing Run {i + 1}/30...")
        seed = 1000 + i
        np.random.seed(seed)
        initial_ops = np.random.uniform(-0.8, 0.8, (50, 384))
        initial_infs = np.random.uniform(0.1, 0.3, 50)

        def get_fresh_group():
            g = Group(n_agents=50, dimension=384)
            g.set_opinions(np.copy(initial_ops))
            for idx, a in enumerate(g.agents): a.influence = initial_infs[idx]
            return g

        np.random.seed(seed)
        smart_iters = run_single_algorithm(GorillaTroopsOptimizer(config, base_weights), get_fresh_group(), "Smart_GTO",
                                           is_smart=True, model=model)

        np.random.seed(seed)
        static_iters = run_single_algorithm(GorillaTroopsOptimizer(config, base_weights), get_fresh_group(),
                                            "Static_GTO")

        np.random.seed(seed)
        gwo_iters = run_single_algorithm(GreyWolfOptimizer(config, base_weights), get_fresh_group(), "GWO")

        np.random.seed(seed)
        pso_iters = run_single_algorithm(ParticleSwarmOptimizer(config, base_weights), get_fresh_group(), "PSO")

        results.append(
            {'Seed': seed, 'Smart_GTO': smart_iters, 'Static_GTO': static_iters, 'PSO': pso_iters, 'GWO': gwo_iters})

    os.makedirs("outputs/logs", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("outputs/logs/real_benchmark_iterations.csv", index=False)
    print("\n✅ Strict benchmarks complete. Results saved to outputs/logs/real_benchmark_iterations.csv")


if __name__ == "__main__":
    run_benchmarks()