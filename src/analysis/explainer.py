import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.core.metrics import calculate_fitness

class XAIExplainer:
    def __init__(self, model_weights):
        """
        Explains why a specific agent became the Silverback.
        Args:
            model_weights: The alpha/beta/gamma weights used in the final decision.
        """
        self.weights = model_weights

    def predict_fitness(self, X):
        """
        Proxy function for SHAP. 
        SHAP generates fake agents (X) and asks: "How fit would these be?"
        
        X shape: (samples, features)
        Features: [Opinion_1, Opinion_2, ..., Opinion_D, Influence]
        """
        # separate opinions and influence
        # assuming last column is Influence
        opinions = X[:, :-1]
        influences = X[:, -1]
        
        # We need a 'dummy' group context to calculate conflict
        # For XAI, we assume the group mean is fixed to the Silverback's opinion
        # to see intrinsic value of the agent's traits.
        
        # Calculate fitness using the same logic as metrics.py
        # But vectorized for SHAP
        scores = []
        for i in range(len(X)):
            op = opinions[i]
            inf = influences[i]
            
            # Simplified fitness for explanation:
            # Score = Influence - Distance_from_Ideal (assumed 0.5 for neutral analysis)
            # This allows us to see if Influence or Opinion matters more
            
            # Reconstruct the exact fitness logic used in GTO
            # Fitness = alpha*Cons + beta*Inf - gamma*Conf
            # Since Cons and Conf are GROUP properties, we approximate individual contribution:
            
            contribution = (self.weights['beta'] * inf) - (self.weights['gamma'] * np.mean(np.abs(op - 0.5)))
            scores.append(contribution)
            
        return np.array(scores)

    def explain_simulation(self, group):
        """
        Generates SHAP plots for the current group state.
        """
        print("🔍 Generating Explanations...")
        
        # 1. Prepare Data
        # Combine Opinions + Influence into one matrix
        ops = group.get_opinions_matrix()
        infs = group.get_influences_vector().reshape(-1, 1)
        
        # Feature Matrix: [Op1, Op2, ..., Op5, Influence]
        X = np.hstack([ops, infs])
        
        # Feature Names
        feature_names = [f"Topic_{i+1}" for i in range(ops.shape[1])] + ["Influence"]
        
        # 2. Create Explainer
        # We use a KernelExplainer (works for any function)
        # We pass a small background dataset (median agent) as reference
        background = np.median(X, axis=0).reshape(1, -1)
        explainer = shap.KernelExplainer(self.predict_fitness, background)
        
        # 3. Calculate SHAP values
        shap_values = explainer.shap_values(X)
        
        # 4. Plot
        plt.figure()
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        
        save_path = "outputs/plots/shap_explanation.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 XAI Plot saved to: {save_path}")