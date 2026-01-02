import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import PPO

from src.core.group import Group
from src.optimization.gto import GorillaTroopsOptimizer
from src.core.metrics import calculate_consensus, calculate_conflict

def load_config(path="configs/settings.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def get_observation(group, current_step, max_steps):
    """
    Manually constructs the observation vector for the AI agent.
    Matches the logic in src/rl/env_wrapper.py
    """
    ops = group.get_opinions_matrix()
    cons = calculate_consensus(ops)
    conf = calculate_conflict(ops)
    
    # Progress signal (t / T)
    progress = current_step / max_steps
    
    # Normalize conflict roughly (assuming max distance approx 1.0 for view)
    norm_conf = min(conf, 1.0)
    
    # Observation: [Consensus, Conflict, Progress]
    return np.array([cons, norm_conf, progress], dtype=np.float32)

def run_static_gto(config):
    """Runs the simulation with FIXED weights from config."""
    print("🔵 Running Static GTO (Fixed Weights)...")
    
    sim_settings = config['simulation']
    weights = config['weights']
    
    group = Group(
        n_agents=sim_settings['n_agents'],
        dimension=sim_settings['dimension'],
        influence_range=sim_settings['influence_range']
    )
    optimizer = GorillaTroopsOptimizer(config, weights)
    
    history = []
    
    for t in tqdm(range(sim_settings['iterations'])):
        optimizer.step(group)
        
        # Track Consensus
        ops = group.get_opinions_matrix()
        history.append(calculate_consensus(ops))
        
    return history

def run_smart_gto(config):
    """Runs the simulation with AI-CONTROLLED dynamic weights."""
    print("mjolnir Running Smart GTO (AI Agent)...")
    
    # Try to load the model
    model_path = "outputs/models/ppo_gto_agent"
    try:
        model = PPO.load(model_path)
    except FileNotFoundError:
        print(f"❌ Could not find model at {model_path}. Please run src/rl/trainer.py first.")
        return []

    sim_settings = config['simulation']
    # Start with default weights, but AI will overwrite them immediately
    optimizer = GorillaTroopsOptimizer(config, config['weights'])
    
    group = Group(
        n_agents=sim_settings['n_agents'],
        dimension=sim_settings['dimension'],
        influence_range=sim_settings['influence_range']
    )
    
    history = []
    max_steps = sim_settings['iterations']
    
    # Initial Observation
    obs = get_observation(group, 0, max_steps)
    
    for t in tqdm(range(max_steps)):
        # 1. AI decides weights based on current state
        action, _ = model.predict(obs, deterministic=True)
        
        # 2. Update Optimizer Weights
        optimizer.weights = {
            'alpha': float(action[0]), # Consensus
            'beta':  float(action[1]), # Influence
            'gamma': float(action[2])  # Conflict
        }
        
        # 3. Step the Simulation
        optimizer.step(group)
        
        # 4. Track Data
        ops = group.get_opinions_matrix()
        history.append(calculate_consensus(ops))
        
        # 5. Update Observation for next step
        obs = get_observation(group, t + 1, max_steps)
        
    return history

def plot_comparison(static_hist, smart_hist):
    """Plots the two convergence curves."""
    plt.figure(figsize=(10, 6))
    
    # Plot Static
    plt.plot(static_hist, label='Static GTO (Fixed)', color='blue', linestyle='--', alpha=0.7)
    
    # Plot Smart
    if smart_hist:
        plt.plot(smart_hist, label='Smart GTO (AI-Driven)', color='red', linewidth=2)
    
    plt.title('Performance Comparison: Static vs. AI-Driven Consensus')
    plt.xlabel('Iterations')
    plt.ylabel('Group Consensus Score (0-1)')
    plt.legend()
    plt.grid(True)
    
    save_path = "outputs/plots/comparison_result.png"
    plt.savefig(save_path)
    print(f"📊 Comparison graph saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    # Load settings
    config = load_config()
    
    # Run Experiments
    static_results = run_static_gto(config)
    smart_results = run_smart_gto(config)
    
    # Visualize
    plot_comparison(static_results, smart_results)