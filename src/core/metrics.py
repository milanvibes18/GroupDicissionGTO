import numpy as np

def calculate_consensus(opinions):
    """
    Consensus is high when variance (disagreement) is low.
    Returns: Float (0.0 to 1.0)
    """
    # Variance across the population for each dimension
    variance = np.var(opinions, axis=0) 
    avg_variance = np.mean(variance)
    
    # Transform variance to a [0, 1] score where 1 is perfect consensus
    # epsilon prevents division by zero
    epsilon = 1e-6
    consensus_score = 1.0 / (1.0 + avg_variance + epsilon)
    return consensus_score

def calculate_conflict(opinions):
    """
    Conflict is the average distance of agents from the group mean.
    Returns: Float (Lower is better, but we return raw value for penalty)
    """
    mean_opinion = np.mean(opinions, axis=0)
    # Euclidean distance of every agent from the mean
    distances = np.linalg.norm(opinions - mean_opinion, axis=1)
    return np.mean(distances)

def calculate_fitness(opinions, influences, weights):
    """
    The Master Formula: 
    F = α·Consensus + β·InfluenceSum - γ·Conflict
    """
    alpha = weights['alpha']
    beta = weights['beta']
    gamma = weights['gamma']

    # 1. Consensus Score
    cons = calculate_consensus(opinions)
    
    # 2. Influence Score (Weighted average of opinions magnitude)
    # We want high influence agents to be happy (Leadership emergence)
    # This is a simplification; in GTO, fitness usually drives the Silverback selection.
    inf_score = np.mean(influences) 

    # 3. Conflict Penalty
    conf = calculate_conflict(opinions)

    # Final Fitness
    fitness = (alpha * cons) + (beta * inf_score) - (gamma * conf)
    return fitness