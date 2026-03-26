import os
import yaml
import torch
import polars as pl
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.simulator.market_env import BinanceMarketMakerEnv


from dotenv import load_dotenv

from pathlib import Path

# Replace the SCRIPT_DIR logic with this:
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # This runs if __file__ isn't defined (Notebook/REPL)
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))


def load_config(yaml_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def load_data_to_numpy(parquet_path: str) -> np.ndarray:
    # โค้ดโหลดข้อมูลเหมือนเดิม...
    print(f"📥 Loading processed features from: {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    feature_cols = ["price", "tfi", "volatility_60s", "vpin"]
    np_data = df.select(feature_cols).to_numpy().astype(np.float32)
    return np_data

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Initializing Neural Network on device: {device.type.upper()}")
    
    # ---------------------------------------------------------
    # ⚙️ 1. Load Configurations from YAML
    # ---------------------------------------------------------
    config_path = PROJECT_ROOT / "configs" / "hyperparameters.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config file at {config_path}")
        
    config = load_config(config_path)
    env_config = config['env']
    ppo_config = config['ppo']
    print("✅ Loaded hyperparameters from YAML")

    # 2. Data Preparation
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / "BTCUSDT_features.parquet"
    np_data = load_data_to_numpy(DATA_PATH)
    
    split_idx = int(len(np_data) * 0.8)
    train_data = np_data[:split_idx]
    eval_data = np_data[split_idx:]
    
    # 3. Environment Setup (ใช้ env_config จาก YAML)
    train_env = DummyVecEnv([lambda: BinanceMarketMakerEnv(train_data, env_config)])
    eval_env = DummyVecEnv([lambda: BinanceMarketMakerEnv(eval_data, env_config)])
    
    # 4. Agent Architecture & Hyperparameters (ใช้ ppo_config จาก YAML)
    policy_kwargs = dict(
        net_arch=dict(
            pi=ppo_config['net_arch']['pi'], 
            vf=ppo_config['net_arch']['vf']
        ),
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=ppo_config['learning_rate'],
        n_steps=ppo_config['n_steps'],
        batch_size=ppo_config['batch_size'],
        n_epochs=ppo_config['n_epochs'],
        gamma=ppo_config['gamma'],
        gae_lambda=ppo_config['gae_lambda'],
        clip_range=ppo_config['clip_range'],
        ent_coef=ppo_config['ent_coef'],
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log="./logs/ppo_market_maker/"
    )
    
    # 5. Training with Evaluation Callback
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path='./models/best_model',
        log_path='./logs/results',
        eval_freq=5000,
        deterministic=True, 
        render=False
    )
    
    print("🔥 Starting PPO Training...")
    model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)
    model.save("models/ppo_mm_final")
    print("🎉 Training Complete!")

if __name__ == "__main__":
    main()