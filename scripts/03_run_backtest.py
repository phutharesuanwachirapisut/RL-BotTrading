import os
import glob
import torch
import argparse
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

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def calculate_risk_metrics(pnl_history):
    if len(pnl_history) < 2: return 0.0, 0.0
    returns = np.diff(pnl_history)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    negative_returns = returns[returns < 0]
    downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-8
    scaling_factor = np.sqrt(10000) 
    sharpe_ratio = (mean_return / (std_return + 1e-8)) * scaling_factor
    sortino_ratio = (mean_return / (downside_std + 1e-8)) * scaling_factor
    return sharpe_ratio, sortino_ratio

def load_testset_data(pair_name: str) -> np.ndarray:
    """โหลดข้อมูลของเหรียญนั้นๆ และดึง 20% สุดท้ายมาเป็นข้อสอบ (Out-of-Sample)"""
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    
    # ดึงไฟล์ Feature ปกติของเหรียญนั้นๆ
    potential_files = glob.glob(str(DATA_DIR / f"{pair_name.upper()}USDT_features*.parquet"))
    
    if not potential_files:
        print(f"❌ Error: ไม่พบไฟล์ข้อมูลสำหรับ {pair_name.upper()} ในโฟลเดอร์ data/processed/")
        sys.exit(1)
        
    DATA_PATH = Path(potential_files[0]) # ใช้ไฟล์แรกที่เจอ
    print(f"📥 Loading Data for Test: {DATA_PATH.name}")
    
    df = pl.read_parquet(str(DATA_PATH))
    feature_cols = ["price", "quantity", "returns_pct_norm", "volatility_norm", "tfi_norm", "rsi_norm", "macd_raw_norm"]
    available_cols = [c for c in feature_cols if c in df.columns]
    
    raw_np = df.select(available_cols).to_numpy().astype(np.float32)
    
    # ⭐️ 1. ใส่เกราะป้องกันข้อมูล (ต้องเหมือนตอน Train เป๊ะ!)
    raw_np = np.nan_to_num(raw_np, nan=0.0, posinf=0.0, neginf=0.0)
    raw_np = np.clip(raw_np, -10.0, 10.0)
    
    # ⭐️ 2. ตัดข้อมูล 10,000 แถวสุดท้ายมาเป็น Out-of-Sample Test
    test_size = min(10000, len(raw_np) // 5) 
    test_np = raw_np[-test_size:]
    
    print(f"✂️ Sliced last {test_size} steps for Out-of-Sample Testing.")
    return np.ascontiguousarray(test_np)

def main():
    parser = argparse.ArgumentParser(description="Out-of-Sample Backtest Evaluation")
    parser.add_argument("--pair", type=str, required=True, help="เช่น btc, eth")
    parser.add_argument("--save_dir", type=str, default=None, help="โฟลเดอร์ปลายทางสำหรับเซฟกราฟ")
    args = parser.parse_args()
    
    pair_name = args.pair.lower()

    # 1. โหลด Configurations (Environment)
    ENV_PATH = PROJECT_ROOT / "configs" / f"{pair_name}_trading_env.yaml"
    if not os.path.exists(ENV_PATH):
        print(f"❌ Error: ไม่พบไฟล์ {ENV_PATH}")
        sys.exit(1)
        
    env_config = load_config(str(ENV_PATH))
    
    # ⭐️ 3. ดึง GA Baseline มาผสาน (เพื่อให้ Env ในห้องสอบเหมือนห้องเรียน)
    GA_PATH = PROJECT_ROOT / "configs" / "ga_optimized_baseline.yaml"
    if os.path.exists(GA_PATH):
        ga_config = load_config(str(GA_PATH)).get("ga_optimized_env", {})
        env_config.update(ga_config)
        print(f"🧬 Applied GA Baseline: {ga_config}")
        
    print(f"⚙️ Loaded Trading Env Config for {pair_name.upper()}")

    # 2. โหลด Model
    MODEL_PATH_FINAL = PROJECT_ROOT / "models" / "ppo_universal_hft_final.zip"
    MODEL_PATH_LATEST = PROJECT_ROOT / "models" / "ppo_universal_hft_latest.zip"
    
    if MODEL_PATH_FINAL.exists():
        MODEL_PATH = MODEL_PATH_FINAL
    elif MODEL_PATH_LATEST.exists():
        MODEL_PATH = MODEL_PATH_LATEST
    else:
        print(f"❌ [Error] Universal Model not found.")
        sys.exit(1)

    print(f"🤖 Loading Agent from {MODEL_PATH.name} on CPU...")
    model = PPO.load(str(MODEL_PATH), device="cpu")

    # 3. เตรียมข้อมูลและ Environment
    test_data = load_testset_data(pair_name)
    env = BinanceMarketMakerEnv(test_data, env_config)
    
    history = {"mid_price": [], "pnl": [], "inventory": [], "spread_action": [], "skew_action": []}

    print(f"🚀 Running Backtest on OUT-OF-SAMPLE Data ({len(test_data)} steps)...")
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

    # 4. คำนวณ Metrics เชิง Quant
    final_pnl = history['pnl'][-1]
    sharpe, sortino = calculate_risk_metrics(history['pnl'])
    print(f"✅ Backtest Finished!")
    print(f"💰 Final Net PnL (After Fees): {final_pnl:.2f} USD")
    print(f"📊 Sharpe Ratio: {sharpe:.2f} | Sortino Ratio: {sortino:.2f}")

    # 5. วาดกราฟ Dashboard
    print(f"\n📈 Generating Out-of-Sample Dashboard for {pair_name.upper()}...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [2, 1, 1]})
    fig.suptitle(f'HFT Market Maker [{pair_name.upper()}/USDT] - Out-of-Sample Evaluation', fontsize=18, fontweight='bold')

    time_steps = np.arange(len(history["pnl"]))

    # แถว 1: PnL
    ax1 = axes[0]
    ax1.plot(time_steps, history["pnl"], color='green', label='Net PnL (USD)', linewidth=2)
    ax1.set_ylabel('Net PnL (USD)', color='green', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    ax1_twin = ax1.twinx()
    ax1_twin.plot(time_steps, history["mid_price"], color='gray', alpha=0.5, label='Mid Price')
    ax1_twin.legend(loc='upper right')
    
    textstr = '\n'.join((
        f'Final Net PnL: ${final_pnl:.2f}',
        f'Sharpe Ratio: {sharpe:.2f}',
        f'Sortino Ratio: {sortino:.2f}'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    ax1.text(0.02, 0.05, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='bottom', bbox=props, fontweight='bold', color='#1f2937')
    ax1.set_title("Profitability & Market Price", fontweight='bold')

    # แถว 2: Inventory
    max_inv = env_config.get('max_inventory', 0.002)
    ax2 = axes[1]
    ax2.plot(time_steps, history["inventory"], color='blue', alpha=0.7)
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.axhline(max_inv, color='red', linestyle=':')
    ax2.axhline(-max_inv, color='red', linestyle=':')
    ax2.set_ylabel(f'Inventory ({pair_name.upper()})')
    ax2.set_title("Risk Management (Inventory Control)")
    ax2.grid(True, alpha=0.3)

    # แถว 3: Agent Actions (Spread / Skew)
    ax3 = axes[2]
    spread_usd = ((history["spread_action"] + 1.0) / 2.0) * env_config.get('max_spread_multiplier', 10.0)
    skew_usd = history["skew_action"] * env_config.get('max_skew_usd', 20.0)

    ax3.plot(time_steps, spread_usd, color='purple', label=r'Spread Control', alpha=0.8)
    ax3.plot(time_steps, skew_usd, color='orange', label='Micro-Skew (USD)', alpha=0.8)
    ax3.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_ylabel('Parameter Action')
    ax3.set_xlabel('Time Steps')
    ax3.set_title("Agent Parameter Tuning")
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    save_path = Path(args.save_dir) / f"backtest_{pair_name}_out_of_sample.png" if args.save_dir else PROJECT_ROOT / "results" / f"backtest_{pair_name}_out_of_sample.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
        
    plt.savefig(save_path, dpi=300)
    print(f"🎉 Dashboard Saved Successfully at: {save_path}")
    
    if not args.save_dir:
        plt.show()

if __name__ == "__main__":
    main()