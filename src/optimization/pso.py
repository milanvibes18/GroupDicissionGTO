import numpy as np
from src.core.metrics import calculate_fitness

class ParticleSwarmOptimizer:
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        # FIX: Updated to Clerc's Constriction Parameters as claimed in the paper
        self.w = 0.729  # Inertia
        self.c1 = 1.49  # Cognitive (Personal best)
        self.c2 = 1.49  # Social (Global best)
        
        self.velocities = None
        self.personal_bests = None
        self.personal_best_scores = None
        self.global_best = None
        self.global_best_score = -np.inf

    def _get_fitness(self, opinions, influences):
        mean_op = np.mean(opinions, axis=0)
        dists = np.linalg.norm(opinions - mean_op, axis=1)
        return influences - dists

    def step(self, group):
        current_positions = group.get_opinions_matrix()
        influences = group.get_influences_vector()
        N, D = current_positions.shape
        
        if self.velocities is None:
            self.velocities = np.zeros_like(current_positions)
            self.personal_bests = np.copy(current_positions)
            self.personal_best_scores = np.full(N, -np.inf)

        fitness_scores = self._get_fitness(current_positions, influences)

        for i in range(N):
            if fitness_scores[i] > self.personal_best_scores[i]:
                self.personal_best_scores[i] = fitness_scores[i]
                self.personal_bests[i] = current_positions[i]
            
            if fitness_scores[i] > self.global_best_score:
                self.global_best_score = fitness_scores[i]
                self.global_best = current_positions[i]

        r1 = np.random.rand(N, D)
        r2 = np.random.rand(N, D)
        
        self.velocities = (self.w * self.velocities + 
                           self.c1 * r1 * (self.personal_bests - current_positions) + 
                           self.c2 * r2 * (self.global_best - current_positions))
        
        new_positions = current_positions + self.velocities
        new_positions = np.clip(new_positions, -1.0, 1.0)
        
        group.set_opinions(new_positions)
        
        from src.core.metrics import calculate_fitness
        return calculate_fitness(new_positions, influences, self.weights)