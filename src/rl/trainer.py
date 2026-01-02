import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from src.rl.env_wrapper import GroupDecisionEnv

def load_config(path="configs/settings.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_agent():
    print("🤖 Starting RL Training Phase...")
    
    # 1. Setup
    config = load_config()
    
    # Create the environment
    env = GroupDecisionEnv(config)
    
    # 2. Define the Model (PPO is generally the best for continuous control)
    model = PPO(
        "MlpPolicy",  # Multi-Layer Perceptron (Standard Neural Net)
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
    )
    
    # 3. Train
    total_timesteps = 20000  # Start small (20k steps) to test. Increase to 100k later.
    print(f"Training for {total_timesteps} steps...")
    
    model.learn(total_timesteps=total_timesteps)
    
    # 4. Save
    os.makedirs("outputs/models", exist_ok=True)
    save_path = "outputs/models/ppo_gto_agent"
    model.save(save_path)
    print(f"✅ Model saved to {save_path}.zip")

if __name__ == "__main__":
    train_agent()