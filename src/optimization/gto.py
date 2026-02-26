import numpy as np
import copy
from src.core.metrics import calculate_fitness

class GorillaTroopsOptimizer:
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        self.silverback_score = -np.inf
        self.silverback_position = None

    def _get_fitness(self, opinions, influences):
        mean_op = np.mean(opinions, axis=0)
        dists = np.linalg.norm(opinions - mean_op, axis=1)
        
        alpha = self.weights.get('alpha', 0.33)
        beta = self.weights.get('beta', 0.33)
        
        # AI balances Influence vs Consensus
        scores = (alpha * influences) - (beta * dists)
        return scores

    def calculate_global_fitness(self, group):
        ops = group.get_opinions_matrix()
        infs = group.get_influences_vector()
        return calculate_fitness(ops, infs, self.weights)

    def step(self, group):
        current_positions = group.get_opinions_matrix()
        influences = group.get_influences_vector()
        N, D = current_positions.shape
        
        fitness_scores = self._get_fitness(current_positions, influences)
        
        best_agent_idx = np.argmax(fitness_scores)
        current_best_score = fitness_scores[best_agent_idx]
        current_silverback = current_positions[best_agent_idx]

        if current_best_score > self.silverback_score:
            self.silverback_score = current_best_score
            self.silverback_position = current_silverback

        new_positions = np.zeros_like(current_positions)
        silverback_bonus = self.config['gto']['silverback_bonus']

        # --- AI Controls Gas Pedal & Steering ---
        gamma = self.weights.get('gamma', 0.33)
        beta = self.weights.get('beta', 0.33)
        
        # Gamma controls migration wandering (0% to 10%)
        self.p = gamma * 0.10
        
        # --- FIX: Cap the Accelerator ---
        # Strictly limits how fast followers can jump toward the leader.
        # Max persuasion (beta=1.0) now only covers 20% of the distance per step.
        a = np.random.uniform(0.0, beta * 0.2)
        
        r = np.random.rand(N)
        migrate_mask = r < self.p
        follow_mask = ~migrate_mask
        
        if np.any(migrate_mask):
            n_migrating = np.sum(migrate_mask)
            new_positions[migrate_mask] = np.random.uniform(-1.0, 1.0, (n_migrating, D))
            
        if np.any(follow_mask):
            n_following = np.sum(follow_mask)
            followers_pos = current_positions[follow_mask]
            noise = np.random.uniform(-1, 1, (n_following, D))
            
            update_step = a * (self.silverback_position - followers_pos)
            noise_step = (silverback_bonus * noise * 0.01)
            
            new_positions[follow_mask] = followers_pos + update_step + noise_step

        new_positions = np.clip(new_positions, -1.0, 1.0)

        # --- EXPLOITATION PHASE ---
        rand_indices = np.random.randint(0, N, N)
        partners = current_positions[rand_indices]
        partner_scores = fitness_scores[rand_indices]
        
        move_mask = partner_scores > fitness_scores
        
        if np.any(move_mask):
            n_moving = np.sum(move_mask)
            moving_pos = new_positions[move_mask]
            winning_partners = partners[move_mask]
            
            winning_indices = rand_indices[move_mask]
            group.update_agent_influence(winning_indices)
            
            step_sizes = np.random.rand(n_moving, 1) 
            diff = winning_partners - moving_pos
            
            # --- FIX: Cap Peer Pressure ---
            # Prevents agents from instantly overwriting their opinions based on neighbors
            new_positions[move_mask] += (step_sizes * beta * 0.2) * diff

        new_positions = np.clip(new_positions, -1.0, 1.0)
        group.set_opinions(new_positions)
        
        return self.calculate_global_fitness(group)