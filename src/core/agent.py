import numpy as np

class Agent:
    def __init__(self, id, dimension, influence_range=(0.1, 1.0), initial_vector=None, text_content=""):
        """
        Initialize a single agent in the social system.
        
        Args:
            id (int): Unique identifier.
            dimension (int): Size of the opinion vector (e.g., 384 for NLP).
            influence_range (tuple): Min and Max social influence.
            initial_vector (np.array): Optional pre-computed opinion (from NLP).
            text_content (str): The original user text (e.g., "I think...").
        """
        self.id = id
        self.dimension = dimension
        self.text_content = text_content  # 📝 STORE THE USER'S TEXT
        
        # Opinion: A vector representing stance
        if initial_vector is not None:
            # ✅ USE REAL DATA: Load the NLP vector
            # We ensure it's a float array
            self.opinion = np.array(initial_vector, dtype=np.float32)
        else:
            # 🎲 FALLBACK: Random generation (for testing/filler agents)
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
        
        # ⚠️ CRITICAL UPDATE FOR NLP:
        # Standard GTO clips to [0, 1]. 
        # NLP vectors (MiniLM) have negative values, so we must allow [-1, 1].
        self.opinion = np.clip(self.opinion, -1.0, 1.0)

    def __repr__(self):
        # Shows text snippet for debugging
        snippet = (self.text_content[:20] + '...') if self.text_content else "Random"
        return f"Agent(id={self.id}, infl={self.influence:.2f}, text='{snippet}')"