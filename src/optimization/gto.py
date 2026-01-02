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
        """
        Helper to calculate fitness for a batch of agents (Vectorized).
        Returns: A numpy array of scores of shape (N,)
        """
        # Vectorized calculation of distance from mean
        # mean_op shape: (D,)
        mean_op = np.mean(opinions, axis=0)
        
        # dists shape: (N,) - Norm of every row relative to mean
        dists = np.linalg.norm(opinions - mean_op, axis=1)
        
        # Score = Influence - Distance (Normalized)
        # This identifies high-status agents who represent the consensus.
        scores = influences - dists
            
        return scores

    def calculate_global_fitness(self, group):
        """Calculates the fitness of the ENTIRE group state (for RL/Monitoring)."""
        ops = group.get_opinions_matrix()
        infs = group.get_influences_vector()
        return calculate_fitness(ops, infs, self.weights)

    def step(self, group):
        """
        Executes ONE iteration of the GTO algorithm.
        Updates agent positions (opinions) based on the Silverback and competition.
        (Vectorized for Large-Scale Simulation)
        """
        # 1. Extract Data
        current_positions = group.get_opinions_matrix() # Shape (N, D)
        influences = group.get_influences_vector()      # Shape (N,)
        N, D = current_positions.shape
        
        # 2. Identify Silverback (The Leader)
        fitness_scores = self._get_fitness(current_positions, influences)
        
        best_agent_idx = np.argmax(fitness_scores)
        current_best_score = fitness_scores[best_agent_idx]
        current_silverback = current_positions[best_agent_idx]

        # Update global memory of the Silverback
        if current_best_score > self.silverback_score:
            self.silverback_score = current_best_score
            self.silverback_position = current_silverback

        # 3. GTO LOGIC: Exploration & Exploitation
        # Initialize new positions array
        new_positions = np.zeros_like(current_positions)
        
        # Factor 'a' for this iteration
        a = np.random.uniform(-1, 1) 
        silverback_bonus = self.config['gto']['silverback_bonus']

        # --- VECTORIZED PHASE 1: EXPLORATION (Following vs Migration) ---
        
        # Generate decision mask for all agents at once
        # True = Migrate, False = Follow Silverback
        r = np.random.rand(N)
        migrate_mask = r < self.p
        follow_mask = ~migrate_mask
        
        # A. Migration Logic (Random jumps)
        # Assign random positions to all migrating agents
        if np.any(migrate_mask):
            n_migrating = np.sum(migrate_mask)
            new_positions[migrate_mask] = np.random.rand(n_migrating, D)
            
        # B. Following Logic (Moving towards Silverback)
        if np.any(follow_mask):
            n_following = np.sum(follow_mask)
            followers_pos = current_positions[follow_mask]
            
            # Generate noise for all followers at once
            noise = np.random.uniform(-1, 1, (n_following, D))
            
            # Vectorized Update Equation:
            # New = Old + a * (Silverback - Old) + Noise_Bonus
            # (Silverback position broadcasts across the batch)
            update_step = a * (self.silverback_position - followers_pos)
            noise_step = (silverback_bonus * noise * 0.01)
            
            new_positions[follow_mask] = followers_pos + update_step + noise_step

        # Clip after Phase 1 to keep valid inputs for Phase 2
        new_positions = np.clip(new_positions, 0.0, 1.0)

        # --- VECTORIZED PHASE 2: EXPLOITATION (Competition) ---
        # Agents compare themselves to random peers
        
        # Select random partners for every agent
        rand_indices = np.random.randint(0, N, N)
        partners = current_positions[rand_indices]
        partner_scores = fitness_scores[rand_indices]
        
        # Identify which agents are "losing" the competition (and thus need to move)
        # We move IF partner_score > my_score
        move_mask = partner_scores > fitness_scores
        
        if np.any(move_mask):
            n_moving = np.sum(move_mask)
            
            # Extract data for moving agents
            moving_pos = new_positions[move_mask]
            winning_partners = partners[move_mask]
            
            # --- NEW UPDATE: Dynamic Influence (Rich-get-Richer) ---
            # Identify the specific agents who won (their index in the original list)
            winning_indices = rand_indices[move_mask]
            
            # Boost their influence because they successfully attracted a follower
            group.update_agent_influence(winning_indices)
            # -------------------------------------------------------
            
            # Random step sizes for social pressure
            step_sizes = np.random.rand(n_moving, 1) # Shape (n_moving, 1) for broadcasting
            
            # Calculate difference
            diff = winning_partners - moving_pos
            
            # Apply update
            new_positions[move_mask] += step_sizes * diff

        # Final Clip
        new_positions = np.clip(new_positions, 0.0, 1.0)

        # 4. Apply Updates
        # Bulk update the group state
        group.set_opinions(new_positions)
        
        return self.calculate_global_fitness(group)