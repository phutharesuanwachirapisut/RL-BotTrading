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

# ⭐️ แก้ไขการ Import ให้ดึงตัวใหม่มาใช้
from src.data_pipeline.binance_parser import process_raw_trades_to_parquet
from src.data_pipeline.features import generate_institutional_features 

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

    print(f"\n[Step 2/2] Generating Institutional Features for {regime_name.upper()}...")
    
    # ⭐️ เรียกใช้ฟังก์ชัน Z-Score Normalization ตัวใหม่
    df_features_normalized = generate_institutional_features(str(tick_parquet))
    
    df_features_normalized.write_parquet(str(feature_parquet))
    print(f"✅ Feature Pipeline Complete! Saved to {os.path.basename(feature_parquet)}")
    print(f"📊 Final Shape: {df_features_normalized.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Features Dataset")
    parser.add_argument("--config", type=str, default="downloadData.yaml", help="ชื่อไฟล์ Config")
    parser.add_argument("--override_regime", type=str, default=None, help="บังคับทำ Feature แค่ Regime เดียว")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / args.config
    if not config_path.exists():
        print(f"❌ ไม่พบไฟล์ Config ที่: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    asset = config.get("asset", "BTCUSDT") 
    
    if args.override_regime:
        regimes = [args.override_regime]
    else:
        regimes = config.get("regimes", {}).keys()

    print(f"🔥 เริ่มสร้าง RL Features สำหรับ: {asset}")
    
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    PROC_DIR = PROJECT_ROOT / "data" / "processed"
    os.makedirs(PROC_DIR, exist_ok=True)
    
    for regime in regimes:
        print(f"\n{'='*50}")
        print(f"⚡ Processing Regime: {regime.upper()}")
        print(f"{'='*50}")
        
        RAW_CSV_PATH = RAW_DIR / f"{asset}_{regime}.csv"
        TICK_PARQUET_PATH = PROC_DIR / f"{asset}_tick_{regime}.parquet"
        FEATURE_PARQUET_PATH = PROC_DIR / f"{asset}_features_{regime}.parquet"
        
        process_pipeline(regime, str(RAW_CSV_PATH), str(TICK_PARQUET_PATH), str(FEATURE_PARQUET_PATH))