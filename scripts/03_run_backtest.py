import os
import torch
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import yaml  # ⭐️ อย่าลืม import yaml
from stable_baselines3 import PPO
import sys

# เพิ่ม Path ให้ Python รู้จักโฟลเดอร์ src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.simulator.market_env import BinanceMarketMakerEnv


from dotenv import load_dotenv

from pathlib import Path
import sys

# Replace the SCRIPT_DIR logic with this:
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # This runs if __file__ isn't defined (Notebook/REPL)
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))


def load_eval_data(parquet_path: str) -> np.ndarray:
    """โหลดข้อมูลและตัดมาเฉพาะ 20% สุดท้าย (Out-of-sample)"""
    print(f"📥 Loading Parquet: {os.path.basename(parquet_path)}")
    df = pl.read_parquet(parquet_path)
    feature_cols = [
        "price",           # คอลัมน์ 0 (เดี๋ยวโดนปิดตาใน Env)
        "volume",          # คอลัมน์ 1
        "volatility_60s",  # คอลัมน์ 2
        "tfi",             # คอลัมน์ 3 (ใช้แทน OBI)
        "vpin"             # คอลัมน์ 4 (เอาไว้ดูรายใหญ่เข้า)
    ] 
    np_data = df.select(feature_cols).to_numpy().astype(np.float32)
    split_idx = int(len(np_data) * 0.8)
    return np_data[split_idx:]

# ⭐️ เพิ่มฟังก์ชันสำหรับอ่าน YAML (เหมือนสคริปต์ 04)
def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def run_backtest_for_regime(regime: str, model, env_config: dict, device) -> dict:
    # ... (โค้ดข้างในฟังก์ชันนี้เหมือนเดิมทุกประการ) ...
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"BTCUSDT_features_{regime}.parquet"
    
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ ข้าม {regime.upper()} - ไม่พบไฟล์ข้อมูล")
        return None

    eval_data = load_eval_data(DATA_PATH)
    env = BinanceMarketMakerEnv(eval_data, env_config)
    
    history = {
        "mid_price": [], "pnl": [], "inventory": [], 
        "spread_action": [], "skew_action": []
    }

    print(f"🚀 Running Backtest on '{regime.upper()}' Regime...")
    obs, info = env.reset()
    done = False
    
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        history["mid_price"].append(obs[0]) 
        history["pnl"].append(info["pnl"])
        history["inventory"].append(info["inventory"])
        history["spread_action"].append(action[0])
        history["skew_action"].append(action[1])

    for key in history:
        history[key] = np.array(history[key])

    print(f"✅ {regime.upper()} Finished! Final PnL: {history['pnl'][-1]:.2f} USD")
    return history

def main():
    # ⭐️ 1. ดึง Config จาก YAML โดยตรง
    CONFIG_PATH = PROJECT_ROOT / "configs" / "hyperparameters.yaml"
    full_config = load_config(CONFIG_PATH)
    env_config = full_config['env']
    print(f"⚙️ Loaded Env Config from YAML: {env_config}")

    MODEL_PATH = PROJECT_ROOT / "models" / "ppo_hft_chunked_final.zip"
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: ไม่พบไฟล์โมเดลที่ {MODEL_PATH}")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🤖 Loading PPO Agent from {os.path.basename(MODEL_PATH)} on {device.type.upper()}...")
    model = PPO.load(MODEL_PATH, device=device)

    # ==========================================
    # ⚙️ 2. รัน Backtest ทั้ง 3 สภาวะตลาด
    # ==========================================
    regimes = ["sideway", "trend", "toxic"]
    all_histories = {}
    
    for regime in regimes:
        hist = run_backtest_for_regime(regime, model, env_config, device)
        if hist is not None:
            all_histories[regime] = hist

    # ==========================================
    # 📉 2. พล็อตกราฟ Quant Dashboard (3x3 Grid)
    # ==========================================
    print("\n📈 Generating The Ultimate 3-Regime Dashboard...")
    
    # สร้าง Grid ขนาดใหญ่ 3 แถว (Metrics) x 3 คอลัมน์ (Regimes)
    num_regimes = len(all_histories)
    fig, axes = plt.subplots(3, num_regimes, figsize=(10 * num_regimes, 14), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle('HFT Market Maker - Multi-Regime Evaluation', fontsize=20, fontweight='bold')

    for col_idx, (regime, history) in enumerate(all_histories.items()):
        
        # [Quant Trick] Downsample ข้อมูลถ้ายาวเกินไป เพื่อให้กราฟไม่รก
        max_plot_points = 5000
        if len(history["pnl"]) > max_plot_points:
            step = len(history["pnl"]) // max_plot_points
            for key in history:
                history[key] = history[key][::step]

        time_steps = np.arange(len(history["pnl"]))

        # แถวที่ 1: Cumulative PnL & Mid Price
        ax1 = axes[0, col_idx] if num_regimes > 1 else axes[0]
        ax1.plot(time_steps, history["pnl"], color='green', label='PnL (USD)', linewidth=2)
        ax1.set_ylabel('PnL (USD)', color='green', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_steps, history["mid_price"], color='gray', alpha=0.5, label='BTC Price')
        if col_idx == num_regimes - 1: # โชว์ Label ราคาเฉพาะคอลัมน์ขวาสุด
            ax1_twin.set_ylabel('BTC Price (USD)', color='gray')
        ax1_twin.legend(loc='upper right')
        ax1.set_title(f"[{regime.upper()}] Profitability", fontweight='bold')

        # แถวที่ 2: Inventory Position
        ax2 = axes[1, col_idx] if num_regimes > 1 else axes[1]
        ax2.plot(time_steps, history["inventory"], color='blue', alpha=0.7)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(env_config["max_inventory"], color='red', linestyle=':')
        ax2.axhline(-env_config["max_inventory"], color='red', linestyle=':')
        if col_idx == 0:
            ax2.set_ylabel('Inventory (BTC)')
        ax2.set_title("Risk Management")
        ax2.grid(True, alpha=0.3)

        # แถวที่ 3: Agent Actions (Spread & Skew)
        ax3 = axes[2, col_idx] if num_regimes > 1 else axes[2]
        spread_usd = ((history["spread_action"] + 1.0) / 2.0) * env_config["max_spread"]
        skew_usd = history["skew_action"] * env_config["max_skew_usd"]

        ax3.plot(time_steps, spread_usd, color='purple', label='Spread', alpha=0.8)
        ax3.plot(time_steps, skew_usd, color='orange', label='Skew', alpha=0.8)
        ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
        if col_idx == 0:
            ax3.set_ylabel('USD')
        ax3.set_xlabel('Time Steps')
        ax3.set_title("Quoting Behavior")
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = PROJECT_ROOT / "notebooks" / "backtest_dashboard_all.png"
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Dashboard Saved Successfully at: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()