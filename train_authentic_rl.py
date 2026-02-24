import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from src.core.group import Group
from src.core.metrics import calculate_consensus
from src.optimization.gto import GorillaTroopsOptimizer

class AuthenticGTOEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # Actions: alpha, beta, gamma
        self.action_space = spaces.Box(low=1e-6, high=1.0, shape=(3,), dtype=np.float32)
        # Obs: [consensus, delta_consensus, time_fraction]
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(3,), dtype=np.float32)
        
        self.max_iters = 75
        self.max_step = 0.055 # The exact physics speed limit for Smart GTO
        self.config = {'simulation': {'n_agents': 50, 'dimension': 384, 'influence_range': [0.1, 1.0]}, 'gto': {'silverback_bonus': 1.2}}
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        self.group = Group(n_agents=50, dimension=384)
        initial_ops = np.random.uniform(-0.8, 0.8, (50, 384))
        initial_infs = np.random.uniform(0.1, 0.3, 50)
        self.group.set_opinions(initial_ops)
        for idx, a in enumerate(self.group.agents): 
            a.influence = initial_infs[idx]
            
        self.optimizer = GorillaTroopsOptimizer(self.config, {'alpha': 0.33, 'beta': 0.33, 'gamma': 0.33})
        self.current_step = 0
        self.last_cons = calculate_consensus(self.group.get_opinions_matrix())
        
        return np.array([self.last_cons, 0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        # Normalize actions
        total = sum(action) + 1e-6
        self.optimizer.weights = {'alpha': action[0]/total, 'beta': action[1]/total, 'gamma': action[2]/total}
        
        old_ops = self.group.get_opinions_matrix()
        self.optimizer.step(self.group)
        raw_new_ops = self.group.get_opinions_matrix()
        
        # Apply strict physics (speed limit + noise)
        delta = np.clip(raw_new_ops - old_ops, -self.max_step, self.max_step)
        noise = np.random.normal(0, 0.01, delta.shape)
        self.group.set_opinions(np.clip(old_ops + delta + noise, -1.0, 1.0))
        
        self.current_step += 1
        new_cons = calculate_consensus(self.group.get_opinions_matrix())
        
        # Authentic Reward Shaping from Paper
        delta_cons = new_cons - self.last_cons
        reward = delta_cons * 10.0  
        if new_cons > 0.95:
            reward += 1.0 # Bonus for reaching consensus
            
        self.last_cons = new_cons
        done = bool(new_cons > 0.95 or self.current_step >= self.max_iters)
        
        obs = np.array([new_cons, delta_cons, self.current_step / self.max_iters], dtype=np.float32)
        return obs, reward, done, False, {}

if __name__ == "__main__":
    print("🧠 Initiating Authentic PPO Training...")
    env = AuthenticGTOEnv()
    
    # Exact hyperparameters from Table 2 in your paper
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    
    model = PPO("MlpPolicy", env, learning_rate=3e-4, clip_range=0.2, 
                batch_size=64, n_epochs=10, ent_coef=0.01, 
                policy_kwargs=policy_kwargs, verbose=1)
    
    # Train the model!
    TIMESTEPS = 100000 
    model.learn(total_timesteps=TIMESTEPS)
    
    os.makedirs("outputs/models", exist_ok=True)
    model.save("outputs/models/ppo_gto_agent")
    print("✅ Authentic RL Model Trained and Saved to outputs/models/ppo_gto_agent.zip")