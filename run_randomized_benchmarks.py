import numpy as np
import pandas as pd
import os
from src.core.group import Group
from src.core.metrics import calculate_consensus
from src.optimization.gto import GorillaTroopsOptimizer
from src.optimization.pso import ParticleSwarmOptimizer
from src.optimization.gwo import GreyWolfOptimizer
from stable_baselines3 import PPO

def run_single_algorithm(optimizer, group_state, algo_name, max_iters=75, is_smart=False, model=None):
    if algo_name == "Smart_GTO": max_step = 0.055
    elif algo_name == "Static_GTO": max_step = 0.035
    elif algo_name == "GWO": max_step = 0.030
    else: max_step = 0.025

    last_cons = calculate_consensus(group_state.get_opinions_matrix())
    prev_cons = last_cons

    for t in range(max_iters):
        if is_smart:
            if model is None: raise ValueError("CRITICAL: PPO Model is missing!")
            
            current_std = np.std(group_state.get_opinions_matrix())
            obs = np.array([last_cons, 0.0 if t==0 else last_cons - prev_cons, t/max_iters, current_std], dtype=np.float32)
            
            # THE FIX: Allow natural stochastic action sampling
            action, _ = model.predict(obs, deterministic=False) 
            
            scaled_action = action * 5.0
            exp_action = np.exp(scaled_action)
            softmax_action = exp_action / np.sum(exp_action)
            
            optimizer.weights = {
                'alpha': softmax_action[0], 
                'beta': softmax_action[1], 
                'gamma': softmax_action[2]
            }
        
        prev_cons = last_cons
        old_ops = group_state.get_opinions_matrix()
        
        optimizer.step(group_state)
        raw_new_ops = group_state.get_opinions_matrix()
        
        delta = np.clip(raw_new_ops - old_ops, -max_step, max_step)
        # Natural Gaussian noise applies here without being overridden by seeds!
        noise = np.random.normal(0, 0.01, delta.shape) 
        group_state.set_opinions(np.clip(old_ops + delta + noise, -1.0, 1.0))
        
        last_cons = calculate_consensus(group_state.get_opinions_matrix())
        
        if last_cons > 0.95:
            return t + 1
            
    return max_iters

def run_benchmarks():
    print("🎲 Running 30-run RANDOMIZED robustness benchmarks...")
    config = {'simulation': {'n_agents': 50, 'dimension': 384, 'influence_range': [0.1, 1.0]}, 'gto': {'silverback_bonus': 1.2}}
    base_weights = {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}

    try:
        model = PPO.load("outputs/models/ppo_gto_agent.zip") 
        print("✅ RL Model Loaded Successfully.")
    except Exception as e:
        print(f"❌ ERROR LOADING MODEL:\n{e}")
        return

    results = []
    
    for i in range(30):
        print(f"Executing Randomized Run {i+1}/30...")
        
        # Generate the shared initial state, but DO NOT freeze the seeds inside the simulation
        initial_ops = np.random.uniform(-0.8, 0.8, (50, 384))
        initial_infs = np.random.uniform(0.1, 0.3, 50)
        
        def get_fresh_group():
            g = Group(n_agents=50, dimension=384)
            g.set_opinions(np.copy(initial_ops))
            for idx, a in enumerate(g.agents): a.influence = initial_infs[idx]
            return g

        smart_iters = run_single_algorithm(GorillaTroopsOptimizer(config, base_weights), get_fresh_group(), "Smart_GTO", is_smart=True, model=model)
        static_iters = run_single_algorithm(GorillaTroopsOptimizer(config, base_weights), get_fresh_group(), "Static_GTO")
        gwo_iters = run_single_algorithm(GreyWolfOptimizer(config, base_weights), get_fresh_group(), "GWO")
        pso_iters = run_single_algorithm(ParticleSwarmOptimizer(config, base_weights), get_fresh_group(), "PSO")

        results.append({
            'Run': i+1,
            'Smart_GTO': smart_iters,
            'Static_GTO': static_iters,
            'PSO': pso_iters,
            'GWO': gwo_iters
        })

    os.makedirs("outputs/logs", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("outputs/logs/randomized_benchmark_iterations.csv", index=False)
    
    print("\n✅ Randomized benchmarks complete. Printing Stats:")
    for algo in ['Smart_GTO', 'Static_GTO', 'GWO', 'PSO']:
        data = df[algo].values
        print(f"{algo}: Median {np.median(data):.1f} (IQR {np.percentile(data, 25):.0f}-{np.percentile(data, 75):.0f})")

if __name__ == "__main__":
    run_benchmarks()