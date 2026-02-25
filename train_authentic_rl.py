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
        # FIX: Action space allows native PPO range
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        # FIX: 4 numbers (Eyes are open)
        self.observation_space = spaces.Box(low=-2.0, high=2.0, shape=(4,), dtype=np.float32)
        
        self.max_iters = 75
        self.max_step = 0.055 
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
        initial_std = np.std(self.group.get_opinions_matrix())
        
        return np.array([self.last_cons, 0.0, 0.0, initial_std], dtype=np.float32), {}

    def step(self, action):
        # --- THE FIX: Temperature Scaling (Multiplier) ---
        scaled_action = action * 5.0
        exp_action = np.exp(scaled_action)
        softmax_action = exp_action / np.sum(exp_action)
        
        self.optimizer.weights = {
            'alpha': softmax_action[0], 
            'beta': softmax_action[1], 
            'gamma': softmax_action[2]
        }
        
        old_ops = self.group.get_opinions_matrix()
        self.optimizer.step(self.group)
        raw_new_ops = self.group.get_opinions_matrix()
        
        delta = np.clip(raw_new_ops - old_ops, -self.max_step, self.max_step)
        noise = np.random.normal(0, 0.01, delta.shape)
        self.group.set_opinions(np.clip(old_ops + delta + noise, -1.0, 1.0))
        
        self.current_step += 1
        new_cons = calculate_consensus(self.group.get_opinions_matrix())
        delta_cons = new_cons - self.last_cons
        current_std = np.std(self.group.get_opinions_matrix())
        
        # Pure Shortest Path Reward
        reward = -1.0  
        
        if new_cons > 0.95:
            reward += 100.0 
            done = True
        elif self.current_step >= self.max_iters:
            reward -= 50.0
            done = True
        else:
            done = False
            
        self.last_cons = new_cons
        
        obs = np.array([new_cons, delta_cons, self.current_step / self.max_iters, current_std], dtype=np.float32)
        return obs, reward, done, False, {}

if __name__ == "__main__":
    print("🧠 Initiating PERFECT PPO Training...")
    env = AuthenticGTOEnv()
    
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    
    model = PPO("MlpPolicy", env, learning_rate=0.0003, clip_range=0.2, 
                batch_size=128, n_epochs=20, ent_coef=0.05, 
                policy_kwargs=policy_kwargs, verbose=1)
    
    TIMESTEPS = 300000 
    
    print(f"🚀 Training for {TIMESTEPS} timesteps. Let it cook!")
    model.learn(total_timesteps=TIMESTEPS)
    
    os.makedirs("outputs/models", exist_ok=True)
    model.save("outputs/models/ppo_gto_agent")
    print("✅ Perfect RL Model Trained and Saved to outputs/models/ppo_gto_agent.zip")