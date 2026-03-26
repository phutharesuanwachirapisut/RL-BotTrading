import os
import torch
import argparse # ⭐️ นำเข้าไลบรารีรับคำสั่ง
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import yaml
from stable_baselines3 import PPO
from pathlib import Path
import sys

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
from src.simulator.market_env import BinanceMarketMakerEnv

def load_eval_data(parquet_path: str) -> np.ndarray:
    print(f"📥 Loading Parquet: {os.path.basename(parquet_path)}")
    df = pl.read_parquet(parquet_path)
    feature_cols = ["price", "volume", "volatility_60s", "tfi", "vpin"] 
    np_data = df.select(feature_cols).to_numpy().astype(np.float32)
    split_idx = int(len(np_data) * 0.8)
    return np_data[split_idx:]

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def run_backtest_for_regime(regime: str, pair_name: str, model, env_config: dict, device) -> dict:
    # ⭐️ โหลดไฟล์ Parquet ให้ตรงกับชื่อเหรียญ
    DATA_PATH = PROJECT_ROOT / "data" / "processed" / f"{pair_name.upper()}USDT_features_{regime}.parquet"
    
    if not os.path.exists(DATA_PATH):
        print(f"⚠️ ข้าม {regime.upper()} - ไม่พบไฟล์ข้อมูล: {DATA_PATH}")
        return None

    eval_data = load_eval_data(str(DATA_PATH))
    env = BinanceMarketMakerEnv(eval_data, env_config)
    
    history = {"mid_price": [], "pnl": [], "inventory": [], "spread_action": [], "skew_action": []}

    print(f"🚀 Running Backtest on '{regime.upper()}' Regime for {pair_name.upper()}...")
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
    # ⭐️ 1. รับคำสั่งชื่อเหรียญ และ โฟลเดอร์ปลายทาง
    parser = argparse.ArgumentParser(description="Multi-Asset Backtest Evaluation")
    parser.add_argument("--pair", type=str, required=True, help="เช่น btc, eth")
    parser.add_argument("--save_dir", type=str, default=None, help="โฟลเดอร์ปลายทางสำหรับเซฟกราฟ")
    args = parser.parse_args()
    pair_name = args.pair.lower()

    # ⭐️ 2. โหลด Config แยกเหรียญ
    ENV_PATH = PROJECT_ROOT / "configs" / f"{pair_name}_trading_env.yaml"
    env_config = load_config(str(ENV_PATH))
    print(f"⚙️ Loaded Trading Env Config for {pair_name.upper()}")

    # ⭐️ 3. โหลด Model ให้ตรงเหรียญ
    MODEL_PATH = PROJECT_ROOT / "models" / f"ppo_{pair_name}_chunked_final.zip"
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: ไม่พบไฟล์โมเดลที่ {MODEL_PATH}")
        return

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🤖 Loading PPO Agent for {pair_name.upper()} on {device.type.upper()}...")
    model = PPO.load(str(MODEL_PATH), device=device)

    # รัน Backtest
    regimes = ["sideway", "trend", "toxic"]
    all_histories = {}
    for regime in regimes:
        hist = run_backtest_for_regime(regime, pair_name, model, env_config, device)
        if hist is not None:
            all_histories[regime] = hist

    if not all_histories:
        print("❌ ไม่มีการรันทดสอบใดๆ (ไม่พบไฟล์ข้อมูล)")
        return

    print(f"\n📈 Generating The Ultimate 3-Regime Dashboard for {pair_name.upper()}...")
    num_regimes = len(all_histories)
    fig, axes = plt.subplots(3, num_regimes, figsize=(10 * num_regimes, 14), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle(f'HFT Market Maker [{pair_name.upper()}/USDT] - Multi-Regime Evaluation', fontsize=20, fontweight='bold')

    for col_idx, (regime, history) in enumerate(all_histories.items()):
        max_plot_points = 5000
        if len(history["pnl"]) > max_plot_points:
            step = len(history["pnl"]) // max_plot_points
            for key in history:
                history[key] = history[key][::step]

        time_steps = np.arange(len(history["pnl"]))

        # แถว 1: PnL
        ax1 = axes[0, col_idx] if num_regimes > 1 else axes[0]
        ax1.plot(time_steps, history["pnl"], color='green', label='PnL (USD)', linewidth=2)
        ax1.set_ylabel('PnL (USD)', color='green', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        ax1_twin = ax1.twinx()
        ax1_twin.plot(time_steps, history["mid_price"], color='gray', alpha=0.5, label='Price')
        ax1_twin.legend(loc='upper right')
        ax1.set_title(f"[{regime.upper()}] Profitability", fontweight='bold')

        # แถว 2: Inventory (⭐️ ดึงค่าจาก risk.max_inventory)
        max_inv = env_config['risk']['max_inventory']
        ax2 = axes[1, col_idx] if num_regimes > 1 else axes[1]
        ax2.plot(time_steps, history["inventory"], color='blue', alpha=0.7)
        ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(max_inv, color='red', linestyle=':')
        ax2.axhline(-max_inv, color='red', linestyle=':')
        if col_idx == 0:
            ax2.set_ylabel(f'Inventory ({pair_name.upper()})')
        ax2.set_title("Risk Management")
        ax2.grid(True, alpha=0.3)

        # แถว 3: Agent Actions (⭐️ ดึงค่าจาก strategy)
        ax3 = axes[2, col_idx] if num_regimes > 1 else axes[2]
        spread_usd = ((history["spread_action"] + 1.0) / 2.0) * env_config['strategy']['max_spread']
        skew_usd = history["skew_action"] * env_config['strategy']['max_skew_usd']

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
    
    # ⭐️ ตรวจสอบว่าถูกสั่งรันจาก Pipeline (มี --save_dir) หรือรันมือปกติ
    if args.save_dir:
        save_path = Path(args.save_dir) / f"backtest_{pair_name}.png"
    else:
        save_path = PROJECT_ROOT / "notebooks" / f"backtest_dashboard_{pair_name}.png"
        
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Dashboard Saved Successfully at: {save_path}")
    
    # ถ้าโดนสั่งรันจาก Pipeline ไม่ต้องเด้งหน้าต่าง plt.show() ขึ้นมาขัดจังหวะ
    if not args.save_dir:
        plt.show()

if __name__ == "__main__":
    main()