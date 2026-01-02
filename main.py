import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
import os

from src.core.group import Group
from src.optimization.gto import GorillaTroopsOptimizer
from src.core.metrics import calculate_consensus, calculate_conflict

def load_config(path="configs/settings.yaml"):
    """Load the simulation parameters."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_simulation():
    # 1. Setup
    print("🚀 Initializing Group Decision Simulation...")
    config = load_config()
    
    # Extract settings
    sim_settings = config['simulation']
    weights = config['weights']
    
    # Create the Social Group
    social_group = Group(
        n_agents=sim_settings['n_agents'],
        dimension=sim_settings['dimension'],
        influence_range=sim_settings['influence_range']
    )
    
    # Initialize the Optimizer
    optimizer = GorillaTroopsOptimizer(config, weights)
    
    # Tracking history for visualization
    history = {
        'fitness': [],
        'consensus': [],
        'conflict': []
    }
    
    # 2. The Optimization Loop
    iterations = sim_settings['iterations']
    print(f"🦍 Starting GTO for {iterations} iterations...")
    
    for t in tqdm(range(iterations)):
        # Run one step of GTO
        current_fitness = optimizer.step(social_group)
        
        # Calculate stats for this step
        current_ops = social_group.get_opinions_matrix()
        cons_score = calculate_consensus(current_ops)
        conf_score = calculate_conflict(current_ops)
        
        # Record history
        history['fitness'].append(current_fitness)
        history['consensus'].append(cons_score)
        history['conflict'].append(conf_score)

    # 3. Results & Visualization
    print("\n✅ Simulation Complete!")
    print(f"Final Consensus Score: {history['consensus'][-1]:.4f}")
    print(f"Final Conflict Score:  {history['conflict'][-1]:.4f}")
    
    plot_results(history)

def plot_results(history):
    """Generates a graph of the group's convergence."""
    plt.figure(figsize=(12, 5))
    
    # Plot Consensus (Should go UP)
    plt.subplot(1, 2, 1)
    plt.plot(history['consensus'], label='Consensus', color='blue')
    plt.title('Group Consensus Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Score (Higher is Better)')
    plt.grid(True)
    
    # Plot Conflict (Should go DOWN)
    plt.subplot(1, 2, 2)
    plt.plot(history['conflict'], label='Conflict', color='red')
    plt.title('Social Conflict Over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Distance (Lower is Better)')
    plt.grid(True)
    
    # Save the plot
    os.makedirs("outputs/plots", exist_ok=True)
    save_path = "outputs/plots/simulation_result.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Graph saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_simulation()