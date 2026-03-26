import os
import sys
import glob
import random
import polars as pl
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
# เพิ่ม Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.simulator.market_env import BinanceMarketMakerEnv

from dotenv import load_dotenv
from pathlib import Path

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # This runs if __file__ isn't defined (Notebook/REPL)
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

def load_chunk_to_numpy(parquet_path: str) -> np.ndarray:
    df = pl.read_parquet(parquet_path)
    # ⭐️ เปลี่ยนมาใช้ TFI และ VPIN ที่มีอยู่แล้วใน Data
    feature_cols = [
        "price",           # คอลัมน์ 0 (เดี๋ยวโดนปิดตาใน Env)
        "volume",          # คอลัมน์ 1
        "volatility_60s",  # คอลัมน์ 2
        "tfi",             # คอลัมน์ 3 (ใช้แทน OBI)
        "vpin"             # คอลัมน์ 4 (เอาไว้ดูรายใหญ่เข้า)
    ] 
    return df.select(feature_cols).to_numpy().astype(np.float32)

def load_config(yaml_path: str) -> dict:
    """ฟังก์ชันสำหรับโหลดค่าจาก YAML"""
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    print("🚀 Starting Chunked Training Pipeline...")
    
    CONFIG_PATH = PROJECT_ROOT / "configs" / "hyperparameters.yaml"
    full_config = load_config(CONFIG_PATH)
    env_config = full_config['env'] 
    
    CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
    chunk_files = glob.glob(os.path.join(CHUNK_DIR, "*.parquet"))
    
    if not chunk_files:
        raise FileNotFoundError(f"❌ ไม่พบไฟล์ Chunk ใน {CHUNK_DIR}")

    MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "ppo_hft_chunked"
    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    os.makedirs(save_dir, exist_ok=True)
    
    TENSORBOARD_LOG = PROJECT_ROOT / "logs" / "tensorboard/" # ⭐️ ที่เก็บกราฟ
    TOTAL_EPOCHS = 10 
    model = None 

    for epoch in range(1, TOTAL_EPOCHS + 1):
        print(f"\n================ EPOCH {epoch}/{TOTAL_EPOCHS} ================")
        # สลับลำดับไฟล์นิดหน่อยในแต่ละ Epoch เพื่อป้องกัน AI จำแพทเทิร์น (Overfitting)
        random.shuffle(chunk_files) 
        
        for chunk_idx, chunk_path in enumerate(chunk_files):
            np_data = load_chunk_to_numpy(chunk_path)
            
            # ⭐️ ห่อ Env ด้วย VecMonitor เพื่อให้มันช่วยเก็บสถิติ Reward ให้
            raw_env = DummyVecEnv([lambda: BinanceMarketMakerEnv(data=np_data, config=env_config)])
            env = VecMonitor(raw_env) 
            
            if model is None:
                print("🧠 กำลังสร้าง PPO Agent ตัวใหม่...")
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    learning_rate=0.00003, 
                    n_steps=1024,
                    batch_size=512,
                    ent_coef=0.01,
                    tensorboard_log=TENSORBOARD_LOG, # ⭐️ เปิดใช้งาน TensorBoard
                    device="mps", # ถ้า Mac มีชิป M ซีรีส์ ลองเปลี่ยนเป็น "mps" เพื่อเร่งความเร็วได้ครับ
                    verbose=1
                )
            else:
                print(f"🧠 อัปเดต Environment (Chunk {chunk_idx+1}/{len(chunk_files)}) ให้ Agent ตัวเดิม...")
                model.set_env(env)
            
            steps_in_chunk = len(np_data)
            
            # ⭐️ ข้อควรระวัง: steps_in_chunk ควรมีขนาดใหญ่กว่า n_steps (1024)
            if steps_in_chunk < 1024:
                print(f"⚠️ Warning: Chunk size ({steps_in_chunk}) is smaller than n_steps (1024). PPO might struggle to update.")
                
            print(f"🔥 Training on {steps_in_chunk} steps...")
            model.learn(total_timesteps=steps_in_chunk, reset_num_timesteps=False)
            
            # เซฟตลอดย่อยเผื่อไฟดับ
            model.save(f"{MODEL_SAVE_PATH}_latest")
            
            del np_data
            del env

    print("\n✅✅✅ Aggressive Chunked Training Complete! ✅✅✅")
    model.save(f"{MODEL_SAVE_PATH}_final")

if __name__ == "__main__":
    main()