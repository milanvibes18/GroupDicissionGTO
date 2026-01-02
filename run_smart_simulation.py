import time
import numpy as np
import yaml
from stable_baselines3 import PPO
from src.core.group import Group
from src.optimization.gto import GorillaTroopsOptimizer
from src.core.metrics import calculate_consensus
from src.analysis.explainer import XAIExplainer

def run_smart():
    print("🤖 Loading the Trained Brain...")
    # Load Config
    with open("configs/settings.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    # Load Model (Ensure you ran trainer.py first!)
    try:
        model = PPO.load("outputs/models/ppo_gto_agent")
    except FileNotFoundError:
        print("❌ Model not found! Please run 'python -m src.rl.trainer' first.")
        return

    # Initialize Environment
    print("🦍 Initializing Smart Colony...")
    group = Group(
        n_agents=config['simulation']['n_agents'], 
        dimension=config['simulation']['dimension'],
        influence_range=config['simulation']['influence_range']
    )
    optimizer = GorillaTroopsOptimizer(config, config['weights'])
    
    # Simulation Loop
    obs = np.array([0.0, 1.0, 0.0]) # Initial dummy observation
    print("🚀 AI is taking control of the group...")
    
    for t in range(50):
        # 1. Ask AI for the best weights
        action, _ = model.predict(obs, deterministic=True)
        
        # 2. Update Optimizer with AI's decision
        optimizer.weights = {
            'alpha': float(action[0]), # Consensus Priority
            'beta':  float(action[1]), # Influence Priority
            'gamma': float(action[2])  # Conflict Penalty
        }
        
        # 3. Run Step
        fitness = optimizer.step(group)
        
        # 4. Update Observation (for next AI step)
        ops = group.get_opinions_matrix()
        cons = calculate_consensus(ops)
        
        if t % 10 == 0:
            print(f"   Step {t}: Consensus = {cons:.4f} | AI Weights: α={action[0]:.2f}, β={action[1]:.2f}")
            
    print("✅ Smart Simulation Complete.")
    
    # Explain the Result
    print("🔍 Asking SHAP to explain the final result...")
    explainer = XAIExplainer(optimizer.weights)
    explainer.explain_simulation(group)

if __name__ == "__main__":
    run_smart()