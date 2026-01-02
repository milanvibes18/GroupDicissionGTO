import unittest
import numpy as np
from src.core.group import Group
from src.optimization.gto import GorillaTroopsOptimizer
from src.core.metrics import calculate_consensus, calculate_fitness

class TestGTO(unittest.TestCase):

    def setUp(self):
        """Setup a basic simulation environment for testing."""
        self.config = {
            'simulation': {
                'n_agents': 10,
                'dimension': 5,
                'iterations': 10,
                'influence_range': [0.1, 1.0]
            },
            'gto': {
                'silverback_bonus': 1.2
            }
        }
        self.weights = {'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2}
        
        # Create a group instance
        self.group = Group(
            n_agents=self.config['simulation']['n_agents'],
            dimension=self.config['simulation']['dimension'],
            influence_range=self.config['simulation']['influence_range']
        )
        
        # Create an optimizer instance
        self.optimizer = GorillaTroopsOptimizer(self.config, self.weights)

    def test_consensus_metric(self):
        """Test if consensus score is always between 0 and 1."""
        ops = self.group.get_opinions_matrix()
        score = calculate_consensus(ops)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test perfect consensus
        perfect_ops = np.zeros((10, 5))
        perfect_score = calculate_consensus(perfect_ops)
        self.assertAlmostEqual(perfect_score, 1.0, places=1)

    def test_agent_bounds(self):
        """Test if agent opinions stay within [0, 1] after initialization."""
        ops = self.group.get_opinions_matrix()
        self.assertTrue(np.all(ops >= 0.0))
        self.assertTrue(np.all(ops <= 1.0))

    def test_dynamic_influence_update(self):
        """Test if influence increases correctly and respects max bound."""
        # Pick the first agent
        target_idx = 0
        initial_influence = self.group.agents[target_idx].influence
        
        # Apply update
        self.group.update_agent_influence([target_idx], amount=0.1)
        new_influence = self.group.agents[target_idx].influence
        
        # Check increase
        expected = min(initial_influence + 0.1, 1.0)
        self.assertAlmostEqual(new_influence, expected)
        
        # Test capping at max influence (1.0)
        self.group.update_agent_influence([target_idx], amount=10.0)
        self.assertEqual(self.group.agents[target_idx].influence, 1.0)

    def test_optimizer_step(self):
        """Test if a GTO step maintains valid shapes and value ranges."""
        initial_ops = self.group.get_opinions_matrix()
        
        # Run one step
        fitness = self.optimizer.step(self.group)
        
        new_ops = self.group.get_opinions_matrix()
        
        # Check Shape
        self.assertEqual(initial_ops.shape, new_ops.shape)
        
        # Check Bounds (Crucial: GTO must clip values)
        self.assertTrue(np.all(new_ops >= 0.0), "Opinions below 0 detected!")
        self.assertTrue(np.all(new_ops <= 1.0), "Opinions above 1 detected!")
        
        # Check Fitness output
        self.assertIsInstance(fitness, float)

if __name__ == '__main__':
    unittest.main()