import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text
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
        Proxy function for SHAP and Decision Tree. 
        Calculates fitness for a batch of agents (X).
        
        X shape: (samples, features)
        Features: [Opinion_1, ... Opinion_D, Influence]
        """
        # Separate opinions and influence
        # Assuming last column is Influence
        opinions = X[:, :-1]
        influences = X[:, -1]
        
        # VECTORIZED CALCULATION (Optimized for Large Scale)
        # Fitness = alpha*Cons + beta*Inf - gamma*Conf
        
        # For XAI, we approximate individual contribution relative to a neutral center (0.5)
        # Influence contribution (Maximize this)
        term_influence = self.weights['beta'] * influences
        
        # Conflict contribution (Minimize distance from neutral 0.5)
        # We calculate mean absolute distance per agent
        dist_from_neutral = np.mean(np.abs(opinions - 0.5), axis=1)
        term_conflict = self.weights['gamma'] * dist_from_neutral
        
        # Final Score
        scores = term_influence - term_conflict
            
        return scores

    def explain_with_rules(self, X, feature_names):
        """
        Fits a surrogate Decision Tree to explain the GTO fitness logic 
        in simple, human-readable rules.
        """
        # 1. Generate target labels (Fitness) using our proxy function
        y = self.predict_fitness(X)
        
        # 2. Fit a simple Decision Tree (Depth 3 for readability)
        tree = DecisionTreeRegressor(max_depth=3)
        tree.fit(X, y)
        
        # 3. Export and Print Rules
        # Note: We truncate feature names for the printout if they are too many
        rules = export_text(tree, feature_names=feature_names)
        print("\n📜 Rule-Based Explanation (Decision Tree Surrogate):")
        print("-----------------------------------------------------")
        print(rules)
        print("-----------------------------------------------------")

    def explain_simulation(self, group):
        """
        Generates SHAP plots and Rule-based explanations for the current group state.
        UPDATED: Aggregates all text features into one 'Content' bar for better visualization.
        """
        print("🔍 Generating Explanations...")
        
        # 1. Prepare Data
        # Combine Opinions + Influence into one matrix
        ops = group.get_opinions_matrix()
        infs = group.get_influences_vector().reshape(-1, 1)
        
        # Feature Matrix: [Op1, Op2, ..., Op384, Influence]
        X = np.hstack([ops, infs])
        
        # Feature Names
        feature_names = [f"Topic_{i+1}" for i in range(ops.shape[1])] + ["Influence"]
        
        # 2. Run Decision Tree Explanation (Keep detailed for logs)
        self.explain_with_rules(X, feature_names)
        
        # 3. Run SHAP Explanation
        # We use a KernelExplainer (works for any function)
        # We pass a small background dataset (median agent) as reference to speed it up
        background = np.median(X, axis=0).reshape(1, -1)
        explainer = shap.KernelExplainer(self.predict_fitness, background)
        
        # Calculate raw SHAP values for all 385 features
        shap_values = explainer.shap_values(X)
        
        # --- STRATEGY 1: AGGREGATION ---
        # Instead of plotting 384 tiny bars, we sum them into one "Content" score.
        
        # A. Aggregate SHAP values (Impact on output)
        # Sum columns 0 to 383 (The Text Topics)
        content_shap = np.sum(shap_values[:, :-1], axis=1)
        # Get column 384 (The Influence)
        influence_shap = shap_values[:, -1]
        
        # Create simplified SHAP matrix (N x 2)
        shap_values_agg = np.column_stack((content_shap, influence_shap))
        
        # B. Aggregate Feature Values (For Color Coding: Red=High, Blue=Low)
        # For "Content", we use the "Conflict/Distance" as the feature value.
        # (High Distance = Bad Content Match)
        content_feature_val = np.mean(np.abs(ops - 0.5), axis=1)
        influence_feature_val = infs.flatten()
        
        X_agg = np.column_stack((content_feature_val, influence_feature_val))
        
        # C. Define New Names
        feature_names_agg = ["Content Conflict (Distance)", "Social Influence"]
        
        # 4. Plot the Simplified Graph
        plt.figure()
        plt.title("Consensus Drivers: Social Status vs. Argument Quality")
        shap.summary_plot(shap_values_agg, X_agg, feature_names=feature_names_agg, show=False)
        
        save_path = "outputs/plots/shap_explanation.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"📊 XAI Plot saved to: {save_path}")