import os
import sys
import argparse
import yaml
from pathlib import Path

# Setup Paths
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

# Import Pipeline Functions
from src.data_pipeline.binance_parser import process_raw_trades_to_parquet
from src.data_pipeline.features import generate_rl_state_features, calculate_vpin_and_merge

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def process_pipeline(regime_name, raw_csv, tick_parquet, feature_parquet):
    if not os.path.exists(raw_csv):
        print(f"❌ Error: ไม่พบไฟล์ {raw_csv} ข้ามไปทำไฟล์อื่น...")
        return

    print(f"\n[Step 1/2] Converting {regime_name.upper()} Raw CSV to Tick Parquet...")
    if not os.path.exists(tick_parquet):
        process_raw_trades_to_parquet(raw_csv, tick_parquet)
    else:
        print(f"⚠️ พบไฟล์ {os.path.basename(tick_parquet)} อยู่แล้ว ข้าม Step นี้...")

    print(f"\n[Step 2/2] Generating RL Features for {regime_name.upper()}...")
    df_features = generate_rl_state_features(str(tick_parquet), window_size="1s")
    df_features_with_vpin = calculate_vpin_and_merge(
        tick_parquet_path=str(tick_parquet),
        df_time_features=df_features,
        volume_bucket_size=10.0,
        window_size=50
    )
    
    df_features_with_vpin.write_parquet(str(feature_parquet))
    print(f"✅ Feature Pipeline Complete! Saved to {os.path.basename(feature_parquet)}")
    print(f"📊 Final Shape: {df_features_with_vpin.shape}")

if __name__ == "__main__":
    # ⭐️ 1. รับค่า Config ว่าจะรันเหรียญอะไร
    parser = argparse.ArgumentParser(description="Build Features Dataset")
    parser.add_argument("--config", type=str, default="downloadData.yaml", help="ชื่อไฟล์ Config")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / args.config
    if not config_path.exists():
        print(f"❌ ไม่พบไฟล์ Config ที่: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    asset = config.get("asset", "BTCUSDT") # ดึงชื่อเหรียญ (เช่น BTCUSDT)
    regimes = config.get("regimes", {}).keys()

    print(f"🔥 เริ่มสร้าง RL Features สำหรับ: {asset}")
    
    # ⭐️ 2. โฟลเดอร์ปลายทาง
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    PROC_DIR = PROJECT_ROOT / "data" / "processed"
    os.makedirs(PROC_DIR, exist_ok=True)
    
    for regime in regimes:
        print(f"\n{'='*50}")
        print(f"⚡ Processing Regime: {regime.upper()}")
        print(f"{'='*50}")
        
        # ⭐️ 3. ตั้งชื่อไฟล์ให้ตรงกับชื่อเหรียญอัตโนมัติ
        RAW_CSV_PATH = RAW_DIR / f"{asset}_{regime}.csv"
        TICK_PARQUET_PATH = PROC_DIR / f"{asset}_tick_{regime}.parquet"
        FEATURE_PARQUET_PATH = PROC_DIR / f"{asset}_features_{regime}.parquet"
        
        process_pipeline(regime, str(RAW_CSV_PATH), str(TICK_PARQUET_PATH), str(FEATURE_PARQUET_PATH))