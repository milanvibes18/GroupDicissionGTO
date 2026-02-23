import os
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure # <--- ADD THIS
from src.rl.env_wrapper import GroupDecisionEnv

def load_config(path="configs/settings.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_agent():
    print("🤖 Starting RL Training Phase...")
    config = load_config()
    env = GroupDecisionEnv(config)
    
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
    )
    
    # --- NEW LOGGING CODE ---
    log_path = "outputs/logs/"
    os.makedirs(log_path, exist_ok=True)
    # Tell SB3 to output to console (stdout) AND a CSV file
    new_logger = configure(log_path, ["stdout", "csv"])
    model.set_logger(new_logger)
    # ------------------------
    
    total_timesteps = 20000 
    print(f"Training for {total_timesteps} steps...")
    
    model.learn(total_timesteps=total_timesteps)
    
    os.makedirs("outputs/models", exist_ok=True)
    save_path = "outputs/models/ppo_gto_agent"
    model.save(save_path)
    print(f"✅ Model saved to {save_path}.zip")
    print(f"📊 Training log saved to {log_path}progress.csv") # <-- Note the CSV location

if __name__ == "__main__":
    train_agent()