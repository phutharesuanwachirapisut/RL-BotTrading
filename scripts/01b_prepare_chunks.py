import polars as pl
import argparse
import yaml
from pathlib import Path
import os
import sys

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    # ⭐️ 1. รับค่า Config 
    parser = argparse.ArgumentParser(description="Chunk Dataset by Days")
    parser.add_argument("--config", type=str, default="downloadData.yaml")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / args.config
    config = load_config(str(config_path))
    asset = config.get("asset", "BTCUSDT")
    pair_name = asset.replace("USDT", "").lower() # แปลง "BTCUSDT" เป็น "btc"
    regimes = config.get("regimes", {}).keys()

    print(f"🪓 Starting Chunking for {asset}...")
    
    # ⭐️ 2. สร้างโฟลเดอร์ Chunks แยกตามเหรียญ (เช่น btc_chunks)
    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / f"{pair_name}_chunks"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for regime in regimes:
        # ⭐️ 3. หาไฟล์ Features ของเหรียญนั้นๆ
        INPUT_FILE = PROJECT_ROOT / "data" / "processed" / f"{asset}_features_{regime}.parquet"
        
        if not os.path.exists(INPUT_FILE):
            print(f"⚠️ ข้าม {regime} - ไม่พบไฟล์ {INPUT_FILE}")
            continue

        print(f"\n📥 Loading {regime} dataset...")
        df = pl.read_parquet(str(INPUT_FILE))
        
        if "datetime" not in df.columns:
            print(f"❌ Error: ไม่พบคอลัมน์ 'datetime' ใน {regime}")
            continue
            
        df = df.with_columns([
            pl.col("datetime").dt.date().alias("date")
        ])
        
        print(f"🔪 Partitioning {regime} by days...")
        partitions = df.partition_by("date", as_dict=True)
        
        for (d,), df_chunk in partitions.items():
            df_chunk = df_chunk.drop("date") 
            output_filename = OUTPUT_DIR / f"chunk_{regime}_{d}.parquet"
            
            df_chunk.write_parquet(str(output_filename))
            print(f"  -> Saved {output_filename.name} ({len(df_chunk)} rows)")

if __name__ == "__main__":
    main()