import streamlit as st
import yaml
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from src.core.group import Group
from src.optimization.gto import GorillaTroopsOptimizer
from src.utils.nlp_handler import NLPHandler
from src.core.metrics import calculate_consensus

# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(page_title="GTO Consensus AI", page_icon="🦍", layout="wide")

@st.cache_resource
def load_nlp():
    return NLPHandler()

def load_config():
    with open("configs/settings.yaml", 'r') as f:
        return yaml.safe_load(f)

nlp = load_nlp()
config = load_config()

# --- 2. SESSION STATE ---
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
    st.session_state.snapshots = []

# --- 3. SIDEBAR ---
st.sidebar.title("👥 Join the Discussion")
st.sidebar.markdown("Add opinions to the group.")

user_input = st.sidebar.text_area("Your Opinion:", placeholder="E.g., We should invest in solar energy...", height=100)
st.sidebar.caption("Press Ctrl+Enter to apply")

if st.sidebar.button("➕ Add Agent"):
    if user_input.strip():
        vector = nlp.text_to_vector(user_input)
        st.session_state.group.add_user_agent(user_input, vector)
        st.session_state.simulation_done = False
        st.success("Agent added!")
    else:
        st.sidebar.warning("Please write an opinion first.")

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
            
            # Reset
            st.session_state.optimizer = GorillaTroopsOptimizer(config, config['weights'])
            iterations = config['simulation']['iterations']
            history = []
            snapshots = []
            
            # Capture Start
            snapshots.append((0, st.session_state.group.get_opinions_matrix().copy()))
            
            # Run Loop
            for t in range(iterations):
                fitness = st.session_state.optimizer.step(st.session_state.group)
                ops = st.session_state.group.get_opinions_matrix()
                cons = calculate_consensus(ops)
                history.append(cons)
                
                if t == iterations // 2:
                    snapshots.append((t, ops.copy()))
                
                progress_bar.progress((t + 1) / iterations)
                status_text.text(f"Iteration {t+1}/{iterations}: Consensus Score = {cons:.4f}")
                time.sleep(0.01)
            
            # Capture End
            snapshots.append((iterations, st.session_state.group.get_opinions_matrix().copy()))
            
            st.session_state.simulation_done = True
            st.session_state.history = history
            st.session_state.snapshots = snapshots

            # Find Winner
            best_vector = st.session_state.optimizer.silverback_position
            if best_vector is not None:
                match_text, similarity = nlp.find_closest_opinion(best_vector, st.session_state.group.agents)
                st.session_state.final_consensus_text = match_text
                st.session_state.final_similarity = similarity
            
            st.success("Optimization Complete!")

with col2:
    st.subheader("📊 Metrics")
    if st.session_state.simulation_done:
        chart_data = pd.DataFrame(st.session_state.history, columns=["Consensus Score"])
        st.line_chart(chart_data)
        st.metric("Final Consensus Score", f"{st.session_state.history[-1]:.4f}")

# --- 5. RESULTS & EXPLANATION ---
st.divider()

if st.session_state.simulation_done:
    st.header("🏆 The AI Verdict")
    st.info("The algorithm has converged. Based on Influence and Centrality, the group consensus is closest to:")
    st.markdown(f"### ❝ {st.session_state.final_consensus_text} ❞")
    st.caption(f"Mathematical Similarity to Optimal Center: {st.session_state.final_similarity*100:.1f}%")
    
    # --- 🧠 SMART EXPLANATION ENGINE ---
    st.write("---")
    st.subheader("💡 AI Explanation")
    
    # 1. Find the specific agent object that won
    winner_agent = None
    for agent in st.session_state.group.agents:
        if agent.text_content == st.session_state.final_consensus_text:
            winner_agent = agent
            break
            
    if winner_agent:
        # 2. Calculate Stats
        # Influence
        my_influence = winner_agent.influence
        avg_influence = np.mean([a.influence for a in st.session_state.group.agents])
        
        # Conflict (Distance from Mean)
        all_ops = st.session_state.group.get_opinions_matrix()
        group_mean = np.mean(all_ops, axis=0)
        my_distance = np.linalg.norm(winner_agent.opinion - group_mean)
        avg_distance = np.mean([np.linalg.norm(a.opinion - group_mean) for a in st.session_state.group.agents])
        
        # 3. Generate Plain English Reason
        reason = ""
        
        # Logic: Did they win because of Power (Influence) or Compromise (Low Distance)?
        is_influential = my_influence > avg_influence
        is_central = my_distance < avg_distance # Lower distance = More central
        
        if is_influential and is_central:
            reason = "This represents a **'Perfect Consensus'**. It was suggested by a highly influential leader, AND it falls mathematically in the middle of the group (minimizing conflict)."
        elif is_influential and not is_central:
            reason = "This represents a **'Leadership Victory'**. Even though this opinion wasn't the most central compromise, the high social status (Influence) of the suggester pulled the group toward it."
        elif not is_influential and is_central:
            reason = "This represents a **'Compromise Victory'**. Even though the suggester didn't have the highest influence, their opinion was the 'Path of Least Resistance' that bridged the gap between opposing sides."
        else:
            reason = "This choice emerged as the best available option to stabilize the group, likely due to fragmentation among other stronger factions."

        st.markdown(f"**Why did the AI choose this?**")
        st.success(reason)
        
        # Show the data proving it
        c1, c2 = st.columns(2)
        c1.metric("Winner's Influence", f"{my_influence:.2f}", delta=f"{my_influence-avg_influence:.2f} vs Avg")
        c2.metric("Winner's Conflict Score", f"{my_distance:.2f}", delta=f"{avg_distance-my_distance:.2f} better than Avg", delta_color="normal")

    # --- FACTIONS GRAPH ---
    st.write("---")
    st.header("🌌 Faction Evolution")
    if len(st.session_state.group) >= 2:
        snapshots = st.session_state.snapshots
        pca = PCA(n_components=2)
        all_data = np.vstack([s[1] for s in snapshots])
        pca.fit(all_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ["Start (Chaos)", "Middle (Merging)", "End (Consensus)"]
        
        for i, (step, ops) in enumerate(snapshots):
            coords = pca.transform(ops)
            ax = axes[i]
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7, c='purple', edgecolors='white', s=100)
            ax.set_title(f"{titles[i]} (T={step})")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_xticks([])
            ax.set_yticks([])
        st.pyplot(fig)

else:
    st.info("👈 Add at least 2 agents on the left, then click 'Run Consensus' to start.")