import yaml
import matplotlib.pyplot as plt
from src.core.group import Group
from src.optimization.gto import GorillaTroopsOptimizer
from src.analysis.explainer import XAIExplainer

def load_config():
    with open("configs/settings.yaml", 'r') as f:
        return yaml.safe_load(f)

def run_explanation():
    # 1. Setup
    config = load_config()
    group = Group(
        n_agents=50, 
        dimension=config['simulation']['dimension'],
        influence_range=config['simulation']['influence_range']
    )
    
    # 2. Run a short simulation to get a result
    optimizer = GorillaTroopsOptimizer(config, config['weights'])
    print("🦍 Running Short Simulation for Analysis...")
    for _ in range(10):
        optimizer.step(group)
        
    # 3. Explain the Result
    explainer = XAIExplainer(config['weights'])
    explainer.explain_simulation(group)

if __name__ == "__main__":
    run_explanation()