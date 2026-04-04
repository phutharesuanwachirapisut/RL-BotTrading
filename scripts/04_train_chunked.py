import os
import sys
import glob
import random
import argparse  
import gc
import polars as pl
import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
from src.simulator.market_env import BinanceMarketMakerEnv

def load_chunk_to_numpy(parquet_path: str) -> np.ndarray:
    df = pl.read_parquet(parquet_path)
    # ต้องตรงกับ Features ที่เราทำ Normalization ไว้ในกลุ่ม 1
    feature_cols = ["price", "quantity", "returns_pct_norm", "volatility_norm", "tfi_norm", "rsi_norm", "macd_raw_norm"] 
    
    # ดึงค่ามาเป็น Numpy
    raw_np = df.select(feature_cols).to_numpy().astype(np.float32)
    
    # ⭐️ 1. ป้องกันค่า NaN หรือ Infinity ใน Data (เปลี่ยนเป็น 0)
    raw_np = np.nan_to_num(raw_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # ⭐️ 2. ตัดหางข้อมูลที่กระโดดรุนแรงเกินไป (Outlier Clipping)
    # จำกัดฟีเจอร์ที่ทำ Z-Score มาแล้ว ไม่ให้เกิน +-10 ป้องกัน AI ช็อก
    raw_np = np.clip(raw_np, -10.0, 10.0)
    
    # บังคับให้ Memory เรียงตัวติดกันเป็นเส้นตรง ป้องกัน PyTorch C++ Crash
    return np.ascontiguousarray(raw_np)

def load_config(yaml_path: str) -> dict:
    if not os.path.exists(yaml_path): return {}
    with open(yaml_path, 'r') as file: return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Chunked Training Pipeline")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint")
    args = parser.parse_args()
    
    print(f"🚀 เริ่มต้น Universal RL Training Pipeline (Cross-Asset Pooled Data)")
    
    # ==========================================
    # 1. โหลด Config & GA Baseline
    # ==========================================
    HYPER_PATH = PROJECT_ROOT / "configs" / "btc_hyperparameters.yaml"
    hyper_config = load_config(str(HYPER_PATH))
    
    GA_PATH = PROJECT_ROOT / "configs" / "ga_optimized_baseline.yaml"
    ga_config = load_config(str(GA_PATH)).get("ga_optimized_env", {})
    
    env_config = {
        "max_inventory": 0.002, "order_size": 0.0002, "maker_fee": 0.0002, 
        "min_spread": 1.0, "frame_stack": 10, "initial_balance": 100.0
    }
    env_config.update(ga_config) 
    
    print(f"🧬 ใช้สมการ Baseline จาก GA: {ga_config}")

    # ==========================================
    # 2. จัดการ Pooled Dataset
    # ==========================================
    CHUNK_DIR = PROJECT_ROOT / "data" / "processed" / "pooled_chunks"
    chunk_files = glob.glob(os.path.join(CHUNK_DIR, "universal_pool_chunk_*.parquet"))
    
    if not chunk_files:
        print(f"❌ ไม่พบข้อมูลรวมใน {CHUNK_DIR} (กรุณารัน 01b ก่อน)")
        sys.exit(1)

    MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "ppo_universal_hft"
    TENSORBOARD_LOG = PROJECT_ROOT / "logs" / "tensorboard_universal/"
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    
    model = None 
    epochs = 10
    starting_epoch = 1

    latest_model_path = f"{MODEL_SAVE_PATH}_latest.zip"
    
    # ==========================================
    # ⭐️ 3. โหลดสมอง AI (เช็คจุดเซฟก่อนเริ่มลูป)
    # ==========================================
    if getattr(args, 'resume', False) and os.path.exists(latest_model_path):
        print(f"🔄 กำลังโหลดสมอง AI จากจุดเซฟล่าสุด: {os.path.basename(latest_model_path)}")
        
        # สร้างสภาพแวดล้อมจำลองชั่วคราวเพื่อให้โหลดโมเดลได้
        dummy_data = load_chunk_to_numpy(chunk_files[0])
        tmp_env = VecNormalize(VecMonitor(DummyVecEnv([lambda: BinanceMarketMakerEnv(data=dummy_data, config=env_config)])), norm_obs=False, norm_reward=True, clip_reward=10.0)
        
        model = PPO.load(
            latest_model_path, 
            env=tmp_env, 
            device="cpu",
            tensorboard_log=str(TENSORBOARD_LOG)
        )
        
        # คำนวณ Epoch จากจำนวนก้าวที่ AI เคยเดินไปแล้ว
        total_steps_trained = model.num_timesteps
        steps_per_epoch = len(chunk_files) * 500_000
        starting_epoch = int(total_steps_trained / steps_per_epoch) + 1
        
        print(f"📍 AI เคยฝึกไปแล้ว {total_steps_trained:,} ก้าว -> คำนวณแล้วตรงกับเริ่มรันต่อที่ EPOCH {starting_epoch}")
        
        del tmp_env
        del dummy_data
        gc.collect()

    # ==========================================
    # 4. Training Loop 
    # ==========================================
    # ⭐️ ให้ลูปเริ่มจากจุดที่คำนวณไว้
    for epoch in range(starting_epoch, epochs + 1):
        print(f"\n================ EPOCH {epoch}/{epochs} ================")
        
        random.shuffle(chunk_files) 
        
        for chunk_idx, chunk_path in enumerate(chunk_files):
            np_data = load_chunk_to_numpy(chunk_path)
            
            raw_env = DummyVecEnv([lambda: BinanceMarketMakerEnv(data=np_data, config=env_config)])
            env = VecMonitor(raw_env) 
            env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)
            
            if model is None:
                print(f"🧠 กำลังสร้าง PPO Agent (Universal Model) ขึ้นมาใหม่...")
                model = PPO(
                    "MlpPolicy", 
                    env, 
                    learning_rate=0.00001, 
                    n_steps=hyper_config['ppo'].get('n_steps', 2048),
                    batch_size=hyper_config['ppo'].get('batch_size', 256),
                    ent_coef=0.01, 
                    max_grad_norm=0.2, 
                    clip_range=0.1,
                    tensorboard_log=str(TENSORBOARD_LOG),
                    device="cpu", 
                    verbose=0 
                )
            else:
                model.set_env(env)
            
            steps_in_chunk = len(np_data)
            print(f"🔥 Training on {os.path.basename(chunk_path)} ({steps_in_chunk} steps)...")
            model.learn(total_timesteps=steps_in_chunk, reset_num_timesteps=False, progress_bar=True)
            
            # บันทึกโมเดลทุกๆ ครั้งที่จบ chunk
            model.save(f"{MODEL_SAVE_PATH}_latest")
            
            del np_data
            del raw_env
            del env
            gc.collect()

    print(f"\n✅✅✅ การฝึกสอนสมองระดับ Universal เสร็จสมบูรณ์! ✅✅✅")
    model.save(f"{MODEL_SAVE_PATH}_final")
if __name__ == "__main__":
    main()