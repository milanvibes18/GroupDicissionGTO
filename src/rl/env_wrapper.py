import gymnasium as gym
from gymnasium import spaces
import numpy as np
from src.core.group import Group
from src.optimization.gto import GorillaTroopsOptimizer
from src.core.metrics import calculate_consensus, calculate_conflict

class GroupDecisionEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    The RL Agent controls the weights (Alpha, Beta, Gamma) to optimize group convergence.
    """
    def __init__(self, config):
        super(GroupDecisionEnv, self).__init__()
        self.config = config
        self.sim_settings = config['simulation']
        
        # ACTION SPACE: The AI can output 3 numbers (Alpha, Beta, Gamma)
        # Values are between 0.0 and 1.0
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # OBSERVATION SPACE: What the AI 'sees'
        # [Current Consensus, Current Conflict, Progress (t/T)]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        
        # Internal State
        self.group = None
        self.optimizer = None
        self.current_step = 0
        self.max_steps = self.sim_settings['iterations']
        self.previous_fitness = 0.0

    def reset(self, seed=None, options=None):
        """Resets the simulation to start a new episode."""
        super().reset(seed=seed)
        
        # Create fresh group and optimizer
        self.group = Group(
            n_agents=self.sim_settings['n_agents'],
            dimension=self.sim_settings['dimension'],
            influence_range=self.sim_settings['influence_range']
        )
        
        # Initial default weights
        initial_weights = self.config['weights']
        self.optimizer = GorillaTroopsOptimizer(self.config, initial_weights)
        
        self.current_step = 0
        self.previous_fitness = 0.0
        
        return self._get_obs(), {}

    def step(self, action):
        """
        The AI takes an action (adjusts weights), and we run one GTO step.
        """
        # 1. Apply AI's Action
        # Action is [Alpha, Beta, Gamma]
        new_weights = {
            'alpha': float(action[0]),
            'beta':  float(action[1]),
            'gamma': float(action[2])
        }
        self.optimizer.weights = new_weights
        
        # 2. Run GTO Logic
        current_fitness = self.optimizer.step(self.group)
        
        # 3. Calculate Reward (Did the AI improve the situation?)
        # Reward = Improvement in Fitness + Bonus for high Consensus - Penalty for Time
        reward = (current_fitness - self.previous_fitness) * 10.0
        
        ops = self.group.get_opinions_matrix()
        cons = calculate_consensus(ops)
        conf = calculate_conflict(ops)
        
        # Add bonus if consensus is very high (The Goal)
        if cons > 0.95:
            reward += 1.0
            
        self.previous_fitness = current_fitness
        self.current_step += 1
        
        # 4. Check if done
        terminated = bool(self.current_step >= self.max_steps)
        truncated = False
        
        # 5. Return Info
        info = {
            "consensus": cons,
            "conflict": conf,
            "weights": new_weights
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        """Returns the current state of the environment."""
        ops = self.group.get_opinions_matrix()
        cons = calculate_consensus(ops)
        conf = calculate_conflict(ops)
        progress = self.current_step / self.max_steps
        
        # Normalize conflict (roughly) for the neural net
        # Assuming max distance is sqrt(dimension)
        norm_conf = min(conf, 1.0) 
        
        return np.array([cons, norm_conf, progress], dtype=np.float32)