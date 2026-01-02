import numpy as np

class Agent:
    def __init__(self, id, dimension, influence_range=(0.1, 1.0)):
        """
        Initialize a single agent in the social system.
        """
        self.id = id
        self.dimension = dimension
        
        # Opinion: A vector representing stance on 'dimension' topics
        # Values between 0 (strongly disagree) and 1 (strongly agree)
        self.opinion = np.random.rand(dimension)
        
        # Influence: How much weight this agent carries (Social Status)
        self.influence = np.random.uniform(influence_range[0], influence_range[1])
        
        # Susceptibility: How likely they are to follow others (Inverse of influence usually)
        self.susceptibility = 1.0 - (self.influence * 0.5)

    def update_opinion(self, target_opinion, step_size=0.1):
        """
        Move opinion towards a target (Leader/Silverback).
        """
        # Simple vector movement: New = Old + Step * (Target - Old)
        diff = target_opinion - self.opinion
        noise = np.random.normal(0, 0.01, self.dimension) # Add slight realism noise
        self.opinion += (step_size * self.susceptibility * diff) + noise
        
        # Clip values to stay within valid opinion range [0, 1]
        self.opinion = np.clip(self.opinion, 0.0, 1.0)

    def __repr__(self):
        return f"Agent(id={self.id}, infl={self.influence:.2f})"