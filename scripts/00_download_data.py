import os
import sys
import yaml
import zipfile
import requests
import pandas as pd
from pathlib import Path
import argparse

# ==========================================
# 📂 1. Setup Paths
# ==========================================
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True) # สร้างโฟลเดอร์ถ้ายังไม่มี

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

# ==========================================
# 📥 2. Core Functions
# ==========================================
def download_binance_data(asset: str, single_date: pd.Timestamp):
    """โหลดไฟล์ .zip จาก Binance Vision"""
    year = single_date.strftime('%Y')
    month = single_date.strftime('%m')
    day = single_date.strftime('%d')
    
    url = f"https://data.binance.vision/data/spot/daily/aggTrades/{asset}/{asset}-aggTrades-{year}-{month}-{day}.zip"
    save_path = RAW_DATA_DIR / f"{asset}-{year}-{month}-{day}.zip"
    
    if save_path.exists():
        print(f"⏩ ข้าม: {save_path.name} (มีไฟล์อยู่แล้ว)")
        return save_path

    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded: {save_path.name}")
        return save_path
    else:
        print(f"❌ File not found at: {url}")
        return None

def process_regime_to_csv(asset: str, regime_name: str, start_date: str, end_date: str):
    """โหลด Zip ตามช่วงเวลา แล้วแตกไฟล์รวมเป็น CSV เดียว"""
    date_range = pd.date_range(start=start_date, end=end_date)
    output_csv = RAW_DATA_DIR / f"{asset}_{regime_name}.csv"
    
    print(f"\n==========================================")
    print(f"🔎 กำลังสร้าง Dataset: [{regime_name.upper()}] ({start_date} ถึง {end_date})")
    print(f"==========================================")
    
    if output_csv.exists():
        os.remove(output_csv)
        print(f"🗑️ ลบไฟล์ {output_csv.name} ตัวเก่าทิ้งแล้ว")

    column_names = [
        'aggregate_trade_id', 'price', 'quantity', 
        'first_trade_id', 'last_trade_id', 'timestamp', 
        'is_buyer_maker', 'is_best_match'
    ]

    first_file = True

    for single_date in date_range:
        # 1. โหลดไฟล์ Zip
        zip_path = download_binance_data(asset, single_date)
        
        if not zip_path or not zip_path.exists():
            continue

        # 2. แตกไฟล์ Zip และ Append ใส่ CSV
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            csv_filename = zip_ref.namelist()[0]
            with zip_ref.open(csv_filename) as f:
                df = pd.read_csv(f, names=column_names)
                
                # แปลง Timestamp เป็น Datetime และหาว่าใครเป็นคนโยน (Side)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms') # Binance aggTrades ใช้ ms
                df['side'] = df['is_buyer_maker'].map({True: 'SELL', False: 'BUY'})
                
                df.to_csv(
                    output_csv, 
                    mode='w' if first_file else 'a', 
                    header=first_file, 
                    index=False
                )
                
                first_file = False
                print(f"🔨 Processed -> appended to {output_csv.name}")

    print(f"🎉 เสร็จสิ้นสภาวะ [{regime_name.upper()}]! เซฟไว้ที่: {output_csv.name}\n")

# ==========================================
# 🚀 3. Main Pipeline
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Download Historical Data from Binance")
    parser.add_argument("--config", type=str, default="downloadData.yaml", help="ชื่อไฟล์ Config")
    
    # ⭐️ เพิ่ม Arguments สำหรับ Sparse Asset Rotation
    parser.add_argument("--override_regime", type=str, default=None, help="ชื่อ Regime ที่บังคับรัน")
    parser.add_argument("--override_start", type=str, default=None, help="บังคับวันที่เริ่มต้น (YYYY-MM-DD)")
    parser.add_argument("--override_end", type=str, default=None, help="บังคับวันที่สิ้นสุด (YYYY-MM-DD)")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / "configs" / args.config
    
    if not config_path.exists():
        print(f"❌ ไม่พบไฟล์ Config ที่: {config_path}")
        sys.exit(1)

    config = load_config(str(config_path))
    asset = config.get("asset", "BTCUSDT")

    # ⭐️ ถ้าระบบหลักส่งค่า Override มาให้ทำแค่ส่วนที่โดนส่งมา (Time-Slicing)
    if args.override_regime and args.override_start and args.override_end:
        print(f"🚀 เริ่มดาวน์โหลด [Sliced Chunk] เหรียญ {asset} | {args.override_start} ถึง {args.override_end}")
        process_regime_to_csv(asset, args.override_regime, args.override_start, args.override_end)
    else:
        # ถ้ารันมือปกติ ก็ดึงข้อมูลตามไฟล์ YAML เหมาเข่งเหมือนเดิม
        regimes = config.get("regimes", {})
        print(f"🚀 เริ่มต้น Data Pipeline เหมาเข่งสำหรับเหรียญ: {asset}")
        for regime_name, details in regimes.items():
            start_date = details['start_date']
            end_date = details['end_date']
            process_regime_to_csv(asset, regime_name, start_date, end_date)

    print("✅✅✅ Data Download Complete! ✅✅✅")

if __name__ == "__main__":
    main()