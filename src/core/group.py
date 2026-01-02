import numpy as np
from src.core.agent import Agent

class Group:
    def __init__(self, n_agents, dimension, influence_range=(0.1, 1.0)):
        """
        Manages the population of agents (The Troop).
        """
        self.n_agents = n_agents
        self.dimension = dimension
        self.influence_range = influence_range
        self.agents = []
        
        # Build the population immediately
        self._initialize_agents()

    def _initialize_agents(self):
        """Creates the initial population of agents."""
        for i in range(self.n_agents):
            agent = Agent(id=i, dimension=self.dimension, influence_range=self.influence_range)
            self.agents.append(agent)

    def get_opinions_matrix(self):
        """
        Returns a (N, D) numpy array of all agent opinions.
        Crucial for vectorised calculations (much faster than loops).
        """
        return np.array([agent.opinion for agent in self.agents])

    def get_influences_vector(self):
        """
        Returns a (N,) numpy array of all agent influence scores.
        """
        return np.array([agent.influence for agent in self.agents])
    
    def set_opinions(self, new_opinions):
        """
        Bulk update of all agent opinions.
        The GTO optimizer will calculate new positions for everyone, 
        and this method applies them back to the agents.
        
        Args:
            new_opinions: A (N, D) numpy array of the new positions.
        """
        # Safety check to ensure dimensions match
        if new_opinions.shape != (self.n_agents, self.dimension):
            raise ValueError(f"Shape mismatch: Expected ({self.n_agents}, {self.dimension}), got {new_opinions.shape}")

        for i, agent in enumerate(self.agents):
            agent.opinion = new_opinions[i]

    def __len__(self):
        return self.n_agents