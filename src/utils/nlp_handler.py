import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class NLPHandler:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the NLP engine.
        We use a small, fast model (MiniLM) perfect for running on a laptop.
        It converts any sentence into a vector of 384 numbers.
        """
        print(f"🧠 Loading NLP Model ({model_name})...")
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("💡 Tip: Did you install it? Run: pip install sentence-transformers")
            raise

    def text_to_vector(self, text):
        """
        Converts a single string (User Opinion) into a numpy vector.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input must be a valid text string.")
            
        # Encode returns a numpy array directly
        vector = self.model.encode(text)
        return vector

    def find_closest_opinion(self, target_vector, agents):
        """
        Takes the mathematical 'Consensus Vector' found by GTO
        and finds which real human opinion is closest to it.
        
        Args:
            target_vector: The (384,) array output by the Gorilla Optimizer.
            agents: List of Agent objects (which contain real user text).
            
        Returns:
            The text content of the closest agent.
        """
        best_similarity = -1.0
        best_text = "No consensus found."
        
        # Reshape target for scikit-learn (1, 384)
        target_vector = target_vector.reshape(1, -1)
        
        for agent in agents:
            # Skip agents with no text (if any)
            if not hasattr(agent, 'text_content') or not agent.text_content:
                continue
                
            # Reshape agent opinion (1, 384)
            agent_vector = agent.opinion.reshape(1, -1)
            
            # Calculate Cosine Similarity (1.0 = Perfect Match, 0.0 = Unrelated)
            sim = cosine_similarity(target_vector, agent_vector)[0][0]
            
            if sim > best_similarity:
                best_similarity = sim
                best_text = agent.text_content
                
        return best_text, best_similarity

if __name__ == "__main__":
    # Quick Test to verify it works
    nlp = NLPHandler()
    
    # Test Data
    v1 = nlp.text_to_vector("I think we should save money.")
    v2 = nlp.text_to_vector("We need to cut costs immediately.")
    v3 = nlp.text_to_vector("Let's buy a new expensive car.")
    
    # Check dimensions
    print(f"Vector Dimension: {v1.shape}") # Should be (384,)
    
    # Check Similarity (v1 should be close to v2, far from v3)
    sim_1_2 = cosine_similarity([v1], [v2])[0][0]
    sim_1_3 = cosine_similarity([v1], [v3])[0][0]
    
    print(f"Similarity (Save vs Cut Costs): {sim_1_2:.4f}") # High (e.g., 0.7+)
    print(f"Similarity (Save vs Buy Car):   {sim_1_3:.4f}") # Low (e.g., 0.2)