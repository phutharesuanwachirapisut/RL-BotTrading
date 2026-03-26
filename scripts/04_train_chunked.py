import os
import sys
import glob
import random
import argparse # ⭐️ นำเข้าไลบรารีรับคำสั่งจาก Terminal
import polars as pl
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from pathlib import Path

# หา Path ของโปรเจกต์
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# อิมพอร์ต Environment ของคุณ
from src.simulator.market_env import BinanceMarketMakerEnv

def load_chunk_to_numpy(parquet_path: str) -> np.ndarray:
    df = pl.read_parquet(parquet_path)
    feature_cols = ["price", "volume", "volatility_60s", "tfi", "vpin"] 
    return df.select(feature_cols).to_numpy().astype(np.float32)

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # ⭐️ 1. รับคำสั่งชื่อเหรียญจาก Terminal
    parser = argparse.ArgumentParser(description="Multi-Asset Chunked Training Pipeline")
    parser.add_argument("--pair", type=str, required=True, help="เช่น btc, eth, sol")
    parser.add_argument("--epochs", type=int, default=10, help="จำนวน Epoch ในการเทรน")
    args = parser.parse_args()
    
    pair_name = args.pair.lower()
    print(f"🚀 Starting Chunked Training Pipeline for: {pair_name.upper()}/USDT")
    
    # ⭐️ 2. โหลด Config แยกตามเหรียญ
    HYPER_PATH = PROJECT_ROOT / "configs" / f"{pair_name}_hyperparameters.yaml"
    ENV_PATH = PROJECT_ROOT / "configs" / f"{pair_name}_trading_env.yaml"
    
    if not HYPER_PATH.exists() or not ENV_PATH.exists():
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ Config ของ {pair_name.upper()} (เช็คโฟลเดอร์ configs/)")
        
    hyper_config = load_config(str(HYPER_PATH))
    env_config = load_config(str(ENV_PATH))
    
    # ⭐️ 3. หาโฟลเดอร์ Chunks ให้ตรงกับชื่อเหรียญ
    CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / f"{pair_name}_chunks"
    chunk_files = glob.glob(os.path.join(CHUNK_DIR, "*.parquet"))
    
    if not chunk_files:
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ Chunk ของ {pair_name.upper()} ใน {CHUNK_DIR}")

    # ⭐️ 4. บันทึกโมเดลและ Log แยกชื่อเหรียญชัดเจน
    MODEL_SAVE_PATH = PROJECT_ROOT / "models" / f"ppo_{pair_name}_chunked"
    TENSORBOARD_LOG = PROJECT_ROOT / "logs" / f"tensorboard_{pair_name}/"
    
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    
    model = None 

    for epoch in range(1, args.epochs + 1):
        print(f"\n================ EPOCH {epoch}/{args.epochs} ================")
        random.shuffle(chunk_files) 
        
        for chunk_idx, chunk_path in enumerate(chunk_files):
            np_data = load_chunk_to_numpy(chunk_path)
            
            raw_env = DummyVecEnv([lambda: BinanceMarketMakerEnv(data=np_data, config=env_config)])
            env = VecMonitor(raw_env) 
            
            if model is None:
                print(f"🧠 กำลังสร้าง PPO Agent ตัวใหม่สำหรับ {pair_name.upper()}...")
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    learning_rate=hyper_config['ppo'].get('learning_rate', 0.00003), 
                    n_steps=hyper_config['ppo'].get('n_steps', 1024),
                    batch_size=hyper_config['ppo'].get('batch_size', 512),
                    ent_coef=hyper_config['ppo'].get('ent_coef', 0.01),
                    tensorboard_log=str(TENSORBOARD_LOG),
                    device="mps", 
                    verbose=1
                )
            else:
                print(f"🧠 อัปเดต Environment (Chunk {chunk_idx+1}/{len(chunk_files)}) ให้ Agent...")
                model.set_env(env)
            
            steps_in_chunk = len(np_data)
            if steps_in_chunk < model.n_steps:
                print(f"⚠️ Warning: Chunk size ({steps_in_chunk}) is smaller than n_steps ({model.n_steps}).")
                
            print(f"🔥 Training on {steps_in_chunk} steps...")
            model.learn(total_timesteps=steps_in_chunk, reset_num_timesteps=False)
            
            model.save(f"{MODEL_SAVE_PATH}_latest")
            
            del np_data
            del env

    print(f"\n✅✅✅ Aggressive Chunked Training Complete for {pair_name.upper()}! ✅✅✅")
    model.save(f"{MODEL_SAVE_PATH}_final")

if __name__ == "__main__":
    main()