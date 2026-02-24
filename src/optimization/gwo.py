import numpy as np

class GreyWolfOptimizer:
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        
        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None
        
        self.alpha_score = -np.inf
        self.beta_score = -np.inf
        self.delta_score = -np.inf

    def _get_fitness(self, opinions, influences):
        mean_op = np.mean(opinions, axis=0)
        dists = np.linalg.norm(opinions - mean_op, axis=1)
        return influences - dists

    def step(self, group):
        current_positions = group.get_opinions_matrix()
        influences = group.get_influences_vector()
        N, D = current_positions.shape
        
        fitness_scores = self._get_fitness(current_positions, influences)

        # Identify Alpha, Beta, and Delta wolves
        for i in range(N):
            if fitness_scores[i] > self.alpha_score:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos
                self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos
                self.alpha_score, self.alpha_pos = fitness_scores[i], current_positions[i].copy()
            elif fitness_scores[i] > self.beta_score:
                self.delta_score, self.delta_pos = self.beta_score, self.beta_pos
                self.beta_score, self.beta_pos = fitness_scores[i], current_positions[i].copy()
            elif fitness_scores[i] > self.delta_score:
                self.delta_score, self.delta_pos = fitness_scores[i], current_positions[i].copy()

        # Fallback if beta/delta haven't been assigned yet
        if self.beta_pos is None: self.beta_pos = self.alpha_pos
        if self.delta_pos is None: self.delta_pos = self.alpha_pos

        a = 2.0 - 2.0 * (np.random.rand()) # linearly decreased from 2 to 0 theoretically, kept stochastic here

        new_positions = np.zeros_like(current_positions)
        
        # GWO Update Math
        for i in range(N):
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = np.abs(C1 * self.alpha_pos - current_positions[i])
            X1 = self.alpha_pos - A1 * D_alpha
            
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = np.abs(C2 * self.beta_pos - current_positions[i])
            X2 = self.beta_pos - A2 * D_beta
            
            r1, r2 = np.random.rand(D), np.random.rand(D)
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = np.abs(C3 * self.delta_pos - current_positions[i])
            X3 = self.delta_pos - A3 * D_delta
            
            new_positions[i] = (X1 + X2 + X3) / 3.0

        new_positions = np.clip(new_positions, -1.0, 1.0)
        group.set_opinions(new_positions)
        
        from src.core.metrics import calculate_fitness
        return calculate_fitness(new_positions, influences, self.weights)