import numpy as np
import copy
from src.core.metrics import calculate_fitness

class GorillaTroopsOptimizer:
    def __init__(self, config, weights):
        """
        The Optimization Engine.
        
        Args:
            config: Dict containing 'simulation' and 'gto' settings from settings.yaml
            weights: Dict containing alpha, beta, gamma for fitness calculation.
        """
        self.config = config
        self.weights = weights
        
        # GTO Hyperparameters
        # p: Probability of demonstrating 'migration' behavior vs 'following' behavior
        self.p = 0.03  
        # beta: Controls the magnitude of the exploration step
        self.beta = 3.0
        
        # We store the best solution found so far (The Silverback)
        self.silverback_score = -np.inf
        self.silverback_position = None

    def _get_fitness(self, opinions, influences):
        """Helper to calculate fitness for a batch of agents."""
        # We calculate fitness for each agent individually to see if they improved
        # Note: In strict group optimization, fitness is often a group property.
        # However, GTO requires individual fitness to decide if an agent 'moves'.
        # We interpret individual fitness as: "How well does this agent's opinion 
        # align with the group goals?"
        
        # This is a simplification: We calculate the GLOBAL fitness if the group 
        # adopted this agent's opinion as the consensus.
        fitness_scores = []
        for i in range(len(opinions)):
            # Create a hypothetical group where everyone agrees with Agent i
            # Or evaluate Agent i's contribution. 
            # For this simulation, we use the global fitness function on the current state
            # but usually GTO needs a scalar per agent.
            
            # APPROACH: Fitness = How close is this agent to the desired state?
            # We use the global metric function defined in metrics.py
            # But we pass the Current Group State to evaluate the *Group's* performance
            # This part is tricky in Social Modeling. 
            # We will use the standard metric: Fitness of the current CONFIGURATION.
            pass
        
        # BETTER APPROACH FOR SOCIAL GTO:
        # Fitness is calculated for the *Current Group State*.
        # But to determine the Silverback, we need to know who has the "Best Opinion".
        # We assume the 'Silverback' is the agent whose opinion maximizes the fitness
        # if they were the leader.
        
        scores = []
        for i, op in enumerate(opinions):
            # Create a temporary view where this agent is the 'center'
            # For simplicity, we just use the global fitness function on the current
            # group state, but weighted by this agent's influence.
            
            # Let's trust the metric function. 
            # We pass the whole group opinions, but we need a score PER AGENT.
            # We will define Agent Fitness as: Influence * (Similarity to Mean) - Conflict
            # This identifies high-status agents who represent the consensus.
            
            # Simple Proxy for GTO selection:
            # An agent is a "better" gorilla if they have high influence 
            # and are close to the group center (Low Conflict).
            mean_op = np.mean(opinions, axis=0)
            dist = np.linalg.norm(op - mean_op)
            
            # Score = Influence - Distance (Normalized)
            score = influences[i] - dist
            scores.append(score)
            
        return np.array(scores)

    def calculate_global_fitness(self, group):
        """Calculates the fitness of the ENTIRE group state (for RL/Monitoring)."""
        ops = group.get_opinions_matrix()
        infs = group.get_influences_vector()
        return calculate_fitness(ops, infs, self.weights)

    def step(self, group):
        """
        Executes ONE iteration of the GTO algorithm.
        Updates agent positions (opinions) based on the Silverback and competition.
        """
        # 1. Extract Data
        current_positions = group.get_opinions_matrix() # Shape (N, D)
        influences = group.get_influences_vector()      # Shape (N,)
        N, D = current_positions.shape
        
        # 2. Identify Silverback (The Leader)
        # We use the global fitness function logic to find the 'best' agent
        fitness_scores = self._get_fitness(current_positions, influences)
        
        best_agent_idx = np.argmax(fitness_scores)
        current_best_score = fitness_scores[best_agent_idx]
        current_silverback = current_positions[best_agent_idx]

        # Update global memory of the Silverback
        if current_best_score > self.silverback_score:
            self.silverback_score = current_best_score
            self.silverback_position = current_silverback

        # 3. GTO LOGIC: Exploration & Exploitation
        # Create a container for the next positions
        new_positions = np.zeros_like(current_positions)
        
        # Factor 'a' decreases over iterations (requires passing iter or keeping state)
        # For a dynamic system, we keep 'a' constant or randomize it slightly
        a = np.random.uniform(-1, 1) 
        
        for i in range(N):
            # Random number for decision making
            r = np.random.rand()
            
            # --- PHASE 1: EXPLORATION (Following the Silverback) ---
            if r < self.p:
                # Migration: Random jump to known or unknown space
                # "Social Interpretation": The agent leaves the debate and thinks independently
                rand_idx = np.random.randint(0, N)
                migration_target = current_positions[rand_idx]
                
                # Math: X_new = (Lower + rand * (Upper - Lower))
                # We limit this to staying within opinion bounds [0,1]
                new_positions[i] = np.random.rand(D)
                
            else:
                # Normal GTO Movement (Following Leadership)
                # "Social Interpretation": Compromising between own view and Leader's view
                
                # Calculate distance to Silverback
                dist = np.abs(self.silverback_position - current_positions[i])
                
                # M and L are GTO coefficients simulating acceleration
                M = np.array([np.random.normal(0, 1) for _ in range(D)]) # Random standard normal
                L = M * dist 
                
                # The Core Update Equation:
                # New = (Silverback - X) * C + X
                # We interpret 'C' as the susceptibility/influence balance
                C = influences[i] # High influence agents might move LESS? Or use raw GTO math.
                
                # Standard GTO simplified:
                # X_new = X_silverback - L * (L * (X_old - X_silverback) + X_old) 
                # Let's use a cleaner social dynamic version:
                # Move towards Silverback + Noise
                
                noise = np.random.uniform(-1, 1, D)
                new_positions[i] = current_positions[i] + a * (self.silverback_position - current_positions[i]) + (self.config['gto']['silverback_bonus'] * noise * 0.01)

            # Clip to ensure opinions stay valid (0.0 to 1.0)
            new_positions[i] = np.clip(new_positions[i], 0.0, 1.0)

        # --- PHASE 2: EXPLOITATION (Competition / Follower Interaction) ---
        # Agents compare themselves to random peers
        for i in range(N):
            # Pick a random partner to compare with
            rand_idx = np.random.randint(0, N)
            partner = current_positions[rand_idx]
            
            # If partner is 'better' (has higher influence/fitness), move towards them
            if fitness_scores[rand_idx] > fitness_scores[i]:
                # Move towards partner
                step_size = np.random.rand() # Random social pressure
                diff = partner - new_positions[i]
                new_positions[i] += step_size * diff
            else:
                # Move away or stay (Social Distancing from bad ideas)
                # For stability, we often just do nothing here or move slightly away
                pass
                
            new_positions[i] = np.clip(new_positions[i], 0.0, 1.0)

        # 4. Apply Updates
        # In pure optimization, we only accept improvements.
        # In Social Simulation, opinions CHANGE even if they get 'worse' (people make mistakes).
        # So we update the group unconditionally (or with a high probability).
        group.set_opinions(new_positions)
        
        return self.calculate_global_fitness(group)