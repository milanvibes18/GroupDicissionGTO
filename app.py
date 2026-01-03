import streamlit as st
import yaml
import numpy as np
import pandas as pd
import time
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
# Streamlit re-runs the script on every click. 
# We use 'session_state' to keep the Group alive between clicks.

if 'group' not in st.session_state:
    # Start with an empty group (0 random agents)
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

# --- 3. SIDEBAR: ADD USERS ---
st.sidebar.title("👥 Join the Discussion")
st.sidebar.markdown("Add opinions to the group.")

user_input = st.sidebar.text_area("Your Opinion:", placeholder="E.g., We should invest in solar energy because...")

if st.sidebar.button("➕ Add Agent"):
    if user_input.strip():
        # 1. Convert Text -> Vector
        vector = nlp.text_to_vector(user_input)
        
        # 2. Add to Group
        st.session_state.group.add_user_agent(user_input, vector)
        
        # 3. Reset Simulation State (New data needs new math)
        st.session_state.simulation_done = False
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
st.markdown("""
This system uses **Gorilla Troops Optimization** to find the 'Mathematical Center' of diverse human opinions.
""")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📡 Live Simulation")
    
    # Run Button
    if st.button("🚀 Run Consensus Algorithm", type="primary"):
        if len(st.session_state.group) < 2:
            st.error("Need at least 2 agents to form a consensus!")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Reset Optimizer for a fresh run
            st.session_state.optimizer = GorillaTroopsOptimizer(config, config['weights'])
            
            # Run the Loop
            iterations = config['simulation']['iterations']
            history = []
            
            for t in range(iterations):
                # GTO Step
                fitness = st.session_state.optimizer.step(st.session_state.group)
                
                # Track Metrics
                ops = st.session_state.group.get_opinions_matrix()
                cons = calculate_consensus(ops)
                history.append(cons)
                
                # Update UI
                progress_bar.progress((t + 1) / iterations)
                status_text.text(f"Iteration {t+1}/{iterations}: Consensus Score = {cons:.4f}")
                time.sleep(0.01) # Visual delay
            
            st.session_state.simulation_done = True
            st.session_state.history = history

            # --- DECODE THE RESULT ---
            # The "Silverback" is the best mathematical position found.
            # We compare this vector to all original inputs to see which one it matches best.
            best_vector = st.session_state.optimizer.silverback_position
            
            if best_vector is not None:
                match_text, similarity = nlp.find_closest_opinion(best_vector, st.session_state.group.agents)
                st.session_state.final_consensus_text = match_text
                st.session_state.final_similarity = similarity
            
            st.success("Optimization Complete!")

with col2:
    st.subheader("📊 Metrics")
    if st.session_state.simulation_done:
        # Plot Convergence
        chart_data = pd.DataFrame(st.session_state.history, columns=["Consensus Score"])
        st.line_chart(chart_data)
        
        st.metric("Final Consensus Score", f"{st.session_state.history[-1]:.4f}")

# --- 5. RESULTS AREA ---
st.divider()

if st.session_state.simulation_done:
    st.header("🏆 The AI Verdict")
    
    st.info("The algorithm has converged. Based on Influence and Centrality, the group consensus is closest to:")
    
    st.markdown(f"### ❝ {st.session_state.final_consensus_text} ❞")
    
    st.caption(f"Mathematical Similarity to Optimal Center: {st.session_state.final_similarity*100:.1f}%")
    
    # Explain why (Simple version)
    st.write("---")
    st.markdown("**Why this result?**")
    st.write("This opinion likely came from a high-influence agent or represented a 'middle ground' that minimized conflict with the majority of the group.")

else:
    st.info("👈 Add agents on the left, then click 'Run Consensus' to start.")