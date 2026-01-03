import numpy as np
from src.core.agent import Agent

class Group:
    def __init__(self, n_agents=0, dimension=384, influence_range=(0.1, 1.0)):
        """
        Manages the population of agents.
        Updated for the App: Can start empty and accept real users.
        
        Args:
            n_agents: Number of random agents to generate (default 0 for App mode).
            dimension: Size of the opinion vector (384 for NLP).
            influence_range: Min/Max social influence.
        """
        self.n_agents = n_agents
        self.dimension = dimension
        self.influence_range = influence_range
        self.agents = []
        
        # Only create fake agents if requested (e.g. for testing)
        if self.n_agents > 0:
            self._initialize_agents()

    def _initialize_agents(self):
        """Creates the initial population of random agents (Simulation Mode)."""
        for i in range(self.n_agents):
            agent = Agent(id=i, dimension=self.dimension, influence_range=self.influence_range)
            self.agents.append(agent)

    def add_user_agent(self, text_content, vector_opinion):
        """
        Adds a REAL user to the group (App Mode).
        
        Args:
            text_content (str): The user's original text input.
            vector_opinion (np.array): The NLP vector (384,) representation.
        """
        new_id = len(self.agents)
        
        # Create agent using the modified Agent class
        # (This uses the 'initial_vector' & 'text_content' fields we added to Agent)
        agent = Agent(
            id=new_id, 
            dimension=self.dimension, 
            influence_range=self.influence_range,
            initial_vector=vector_opinion,
            text_content=text_content
        )
        
        self.agents.append(agent)
        self.n_agents += 1  # Keep the count in sync!

    def get_opinions_matrix(self):
        """
        Returns a (N, D) numpy array of all agent opinions.
        """
        if not self.agents:
            return np.empty((0, self.dimension))
        return np.array([agent.opinion for agent in self.agents])

    def get_influences_vector(self):
        """
        Returns a (N,) numpy array of all agent influence scores.
        """
        if not self.agents:
            return np.array([])
        return np.array([agent.influence for agent in self.agents])
    
    def set_opinions(self, new_opinions):
        """
        Bulk update of all agent opinions.
        The GTO optimizer uses this to apply the new math positions.
        """
        # Safety check to ensure dimensions match current agent count
        current_count = len(self.agents)
        if new_opinions.shape != (current_count, self.dimension):
            raise ValueError(f"Shape mismatch: Expected ({current_count}, {self.dimension}), got {new_opinions.shape}")

        for i, agent in enumerate(self.agents):
            agent.opinion = new_opinions[i]

    def update_agent_influence(self, indices, amount=0.01):
        """
        Updates the influence scores of specific agents (Rich-get-Richer).
        """
        if isinstance(indices, (int, np.integer)):
            indices = [indices]
            
        max_infl = self.influence_range[1]

        for idx in indices:
            # Ensure index is valid
            if 0 <= idx < len(self.agents):
                agent = self.agents[idx]
                
                # Boost influence
                agent.influence += amount
                agent.influence = min(agent.influence, max_infl)
                
                # Recalculate susceptibility (Inverse of influence)
                agent.susceptibility = 1.0 - (agent.influence * 0.5)

    def __len__(self):
        return len(self.agents)