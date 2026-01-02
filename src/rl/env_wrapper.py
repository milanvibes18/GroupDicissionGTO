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
        
        # Reward Tracking State
        self.previous_fitness = 0.0
        self.previous_cons = 0.0
        self.previous_conf = 0.0

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
        
        # Initialize previous metrics for reward calculation
        initial_ops = self.group.get_opinions_matrix()
        self.previous_cons = calculate_consensus(initial_ops)
        self.previous_conf = calculate_conflict(initial_ops)
        
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
        # This updates the group's opinions internally
        current_fitness = self.optimizer.step(self.group)
        
        # 3. Calculate Reward Decomposition
        ops = self.group.get_opinions_matrix()
        current_cons = calculate_consensus(ops)
        current_conf = calculate_conflict(ops)
        
        # Component 1: Reward for increasing Consensus
        # Multiplied by 10 to give it significant weight in the gradient
        reward_cons_part = (current_cons - self.previous_cons) * 10.0
        
        # Component 2: Reward for reducing Conflict
        # Negative delta means conflict went down (Good), so we negate it to make reward positive
        reward_conf_part = -(current_conf - self.previous_conf) * 5.0
        
        # Total Reward
        reward = reward_cons_part + reward_conf_part
        
        # Bonus for achieving high consensus (The Goal)
        if current_cons > 0.95:
            reward += 1.0
            
        # Update history state
        self.previous_fitness = current_fitness
        self.previous_cons = current_cons
        self.previous_conf = current_conf
        self.current_step += 1
        
        # 4. Check if done
        terminated = bool(self.current_step >= self.max_steps)
        truncated = False
        
        # 5. Return Info with Decomposition Data
        info = {
            "consensus": current_cons,
            "conflict": current_conf,
            "weights": new_weights,
            "reward_cons_part": reward_cons_part,
            "reward_conf_part": reward_conf_part
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