import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # Added for Faction Visualization
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

    sim_settings = config['simulation']
    weights = config['weights']

    social_group = Group(
        n_agents=sim_settings['n_agents'],
        dimension=sim_settings['dimension'],
        influence_range=sim_settings['influence_range']
    )

    optimizer = GorillaTroopsOptimizer(config, weights)

    history = {'fitness': [], 'consensus': [], 'conflict': []}
    snapshots = []

    iterations = sim_settings['iterations']
    print(f"🦍 Starting GTO for {iterations} iterations...")

    # Capture initial state (T=0)
    snapshots.append((0, social_group.get_opinions_matrix().copy()))

    for t in tqdm(range(iterations)):
        # --- THE REAL PHYSICS FIX ---

        # A. Save the starting positions before the step
        old_ops = social_group.get_opinions_matrix().copy()

        # B. Run GTO to calculate the raw "intended" movement
        current_fitness = optimizer.step(social_group)
        raw_new_ops = social_group.get_opinions_matrix()

        # C. Apply the V_max = 0.012 Speed Limit (Prevents the "Bouncing" graph)
        delta = np.clip(raw_new_ops - old_ops, -0.012, 0.012)

        # D. Apply sigma = 0.002 Micro-Noise (Prevents dots from stacking/collapsing)
        noise = np.random.normal(0, 0.002, delta.shape)

        # E. Final Position Update with smoothing and boundary clamping
        final_ops = np.clip(old_ops + delta + noise, -1.0, 1.0)
        social_group.set_opinions(final_ops)

        # --- END OF FIX ---

        # Stats for tracking
        cons_score = calculate_consensus(final_ops)
        conf_score = calculate_conflict(final_ops)

        history['fitness'].append(current_fitness)
        history['consensus'].append(cons_score)
        history['conflict'].append(conf_score)

        # Capture Middle Snapshot (Now with distinct dots!)
        if t == iterations // 2:
            snapshots.append((t, final_ops.copy()))

    # Capture Final Snapshot
    snapshots.append((iterations, social_group.get_opinions_matrix().copy()))

    print("\n✅ Simulation Complete!")
    plot_results(history, snapshots)


def plot_results(history, snapshots):
    """Generates a graph of the group's convergence + Faction Visualization."""

    # Prepare the layout: 2 Rows, 3 Columns
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)

    # --- Row 1: Metrics (Consensus, Conflict, Fitness) ---

    # Plot Consensus (Should go UP)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history['consensus'], label='Consensus', color='blue')
    ax1.set_title('Group Consensus Over Time')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Score (Higher is Better)')
    ax1.grid(True)

    # Plot Conflict (Should go DOWN)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history['conflict'], label='Conflict', color='red')
    ax2.set_title('Social Conflict Over Time')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('Distance (Lower is Better)')
    ax2.grid(True)

    # Plot Total Fitness
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(history['fitness'], label='Fitness', color='green')
    ax3.set_title('Total Fitness Score')
    ax3.set_xlabel('Iterations')
    ax3.grid(True)

    # --- Row 2: Faction Visualization (PCA Scatter Plots) ---
    # We use PCA to reduce the 5D opinions down to 2D for plotting

    # Fit PCA globally on all data to ensure consistent axes
    pca = PCA(n_components=2)
    all_data = np.vstack([s[1] for s in snapshots])
    pca.fit(all_data)

    titles = ["Start (Chaos)", "Middle (Factions)", "End (Consensus)"]

    for i, (step, ops) in enumerate(snapshots):
        ax = fig.add_subplot(gs[1, i])

        # Transform current opinions to 2D
        coords = pca.transform(ops)

        # Scatter plot
        # Alpha is transparency (shows density where dots overlap)
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, c='purple', edgecolors='white', s=60)

        ax.set_title(f"{titles[i]} (T={step})")
        ax.set_xlabel("PCA Dim 1")
        ax.set_ylabel("PCA Dim 2")
        ax.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    os.makedirs("outputs/plots", exist_ok=True)
    save_path = "outputs/plots/simulation_result_factions.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"📊 Extended Graph saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    run_simulation()