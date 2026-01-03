import streamlit as st
import yaml
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # 🆕 Added for Faction Visualization
from src.core.group import Group
from src.optimization.gto import GorillaTroopsOptimizer
from src.utils.nlp_handler import NLPHandler
from src.core.metrics import calculate_consensus

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="GTO Consensus AI", page_icon="🦍", layout="wide")

@st.cache_resource
def load_nlp():
    """Load the NLP model only once (caching speeds up the app)."""
    return NLPHandler()

def load_config():
    with open("configs/settings.yaml", 'r') as f:
        return yaml.safe_load(f)

# Load resources
nlp = load_nlp()
config = load_config()

# --- 2. SESSION STATE (Memory) ---
if 'group' not in st.session_state:
    st.session_state.group = Group(
        n_agents=0, 
        dimension=config['simulation']['dimension'],
        influence_range=config['simulation']['influence_range']
    )

if 'optimizer' not in st.session_state:
    st.session_state.optimizer = GorillaTroopsOptimizer(config, config['weights'])

if 'simulation_done' not in st.session_state:
    st.session_state.simulation_done = False
    st.session_state.final_consensus_text = ""
    st.session_state.snapshots = [] # Store history for PCA plot

# --- 3. SIDEBAR: ADD USERS ---
st.sidebar.title("👥 Join the Discussion")
st.sidebar.markdown("Add opinions to the group.")

user_input = st.sidebar.text_area("Your Opinion:", placeholder="E.g., We should invest in solar energy because...", height=100)
st.sidebar.caption("Press Ctrl+Enter to apply")

if st.sidebar.button("➕ Add Agent"):
    if user_input.strip():
        vector = nlp.text_to_vector(user_input)
        st.session_state.group.add_user_agent(user_input, vector)
        st.session_state.simulation_done = False # Reset if new agent joins
        st.success("Agent added!")
    else:
        st.sidebar.warning("Please write an opinion first.")

# Show current participants
st.sidebar.divider()
st.sidebar.subheader(f"Current Agents: {len(st.session_state.group)}")
for i, agent in enumerate(st.session_state.group.agents):
    with st.sidebar.expander(f"Agent {i+1}"):
        st.write(f"💬 *{agent.text_content}*")
        st.caption(f"Influence: {agent.influence:.2f}")

# --- 4. MAIN DASHBOARD ---
st.title("🦍 GTO: AI Consensus Engine")
st.markdown("This system uses **Gorilla Troops Optimization** to find the 'Mathematical Center' of diverse human opinions.")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Simulation")
    
    if st.button("🚀 Run Consensus Algorithm", type="primary"):
        if len(st.session_state.group) < 2:
            st.error("Need at least 2 agents to form a consensus!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Reset Optimizer & History
            st.session_state.optimizer = GorillaTroopsOptimizer(config, config['weights'])
            iterations = config['simulation']['iterations']
            history = []
            snapshots = [] # To store (step, positions)
            
            # Capture Start (T=0)
            snapshots.append((0, st.session_state.group.get_opinions_matrix().copy()))
            
            # --- THE LOOP ---
            for t in range(iterations):
                # GTO Step
                fitness = st.session_state.optimizer.step(st.session_state.group)
                
                # Track Metrics
                ops = st.session_state.group.get_opinions_matrix()
                cons = calculate_consensus(ops)
                history.append(cons)
                
                # Capture Middle (T=50%)
                if t == iterations // 2:
                    snapshots.append((t, ops.copy()))
                
                # Update UI
                progress_bar.progress((t + 1) / iterations)
                status_text.text(f"Iteration {t+1}/{iterations}: Consensus Score = {cons:.4f}")
                time.sleep(0.01) 
            
            # Capture End (T=100%)
            snapshots.append((iterations, st.session_state.group.get_opinions_matrix().copy()))
            
            # Save results to session state
            st.session_state.simulation_done = True
            st.session_state.history = history
            st.session_state.snapshots = snapshots

            # Find the Winner
            best_vector = st.session_state.optimizer.silverback_position
            if best_vector is not None:
                match_text, similarity = nlp.find_closest_opinion(best_vector, st.session_state.group.agents)
                st.session_state.final_consensus_text = match_text
                st.session_state.final_similarity = similarity
            
            st.success("Optimization Complete!")

with col2:
    st.subheader("📊 Metrics")
    if st.session_state.simulation_done:
        # 1. Line Chart (Consensus Score)
        chart_data = pd.DataFrame(st.session_state.history, columns=["Consensus Score"])
        st.line_chart(chart_data)
        st.metric("Final Consensus Score", f"{st.session_state.history[-1]:.4f}")

# --- 5. RESULTS & FACTIONS VISUALIZATION ---
st.divider()

if st.session_state.simulation_done:
    st.header("🏆 The AI Verdict")
    st.info("The algorithm has converged. Based on Influence and Centrality, the group consensus is closest to:")
    st.markdown(f"### ❝ {st.session_state.final_consensus_text} ❞")
    st.caption(f"Mathematical Similarity to Optimal Center: {st.session_state.final_similarity*100:.1f}%")
    st.write("---")

    # --- 🆕 NEW SECTION: FACTIONS GRAPH ---
    st.header("🌌 Faction Evolution (PCA)")
    st.markdown("See how the disparate opinions (dots) moved from chaos to unity.")
    
    # Check if we have enough agents for PCA (Need at least 2)
    if len(st.session_state.group) >= 2:
        snapshots = st.session_state.snapshots
        
        # Fit PCA globally on all data to ensure consistent axes
        pca = PCA(n_components=2)
        all_data = np.vstack([s[1] for s in snapshots])
        pca.fit(all_data)
        
        # Create Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ["Start (Chaos)", "Middle (Merging)", "End (Consensus)"]
        
        for i, (step, ops) in enumerate(snapshots):
            # Transform to 2D
            coords = pca.transform(ops)
            
            # Plot
            ax = axes[i]
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, c='purple', edgecolors='white', s=100)
            ax.set_title(f"{titles[i]} (T={step})")
            ax.grid(True, linestyle='--', alpha=0.5)
            
            # Remove ticks for cleaner look
            ax.set_xticks([])
            ax.set_yticks([])

        st.pyplot(fig)
    else:
        st.warning("Not enough agents to generate PCA plot.")
else:
    st.info("👈 Add at least 2 agents on the left, then click 'Run Consensus' to start.")