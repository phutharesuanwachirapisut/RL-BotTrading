import os
import sys

# เพิ่ม Path ให้ Python รู้จักโฟลเดอร์ src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_pipeline.binance_parser import process_raw_trades_to_parquet
from src.data_pipeline.features import generate_rl_state_features, calculate_vpin_and_merge


from dotenv import load_dotenv

from pathlib import Path

# Replace the SCRIPT_DIR logic with this:
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # This runs if __file__ isn't defined (Notebook/REPL)
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))



def process_pipeline(regime_name, raw_csv, tick_parquet, feature_parquet):
    """
    ฟังก์ชันแกนหลัก: อ่าน CSV -> แปลงเป็น Tick Parquet -> สร้าง Features (VPIN/TFI)
    """
    # เช็คว่ามีไฟล์ Raw อยู่จริงไหม
    if not os.path.exists(raw_csv):
        print(f"❌ Error: ไม่พบไฟล์ {raw_csv} ข้ามไปทำไฟล์อื่น...")
        return

    # ---------------------------------------------------------
    # 🚀 Step 1: แปลง CSV เป็น Tick-level Parquet (ลดขนาดไฟล์ & RAM)
    # ---------------------------------------------------------
    print(f"\n[Step 1/2] Converting {regime_name.upper()} Raw CSV to Tick Parquet...")
    if not os.path.exists(tick_parquet):
        process_raw_trades_to_parquet(raw_csv, tick_parquet)
    else:
        print(f"⚠️ พบไฟล์ {tick_parquet} อยู่แล้ว ข้าม Step นี้...")

    # ---------------------------------------------------------
    # 🧠 Step 2: สร้าง RL Features (1-second bars + Microstructure)
    # ---------------------------------------------------------
    print(f"\n[Step 2/2] Generating RL Features for {regime_name.upper()}...")
    
    # 1. สร้าง Time features
    df_features = generate_rl_state_features(tick_parquet, window_size="1s")
    
    # 2. นำเข้าฟังก์ชัน VPIN เพื่อแปะลงใน Features
    df_features_with_vpin = calculate_vpin_and_merge(
        tick_parquet_path=tick_parquet,
        df_time_features=df_features,
        volume_bucket_size=10.0, # 10 BTC ต่อ 1 ถัง
        window_size=50           # ใช้ย้อนหลัง 50 ถัง
    )
    
    # 3. เซฟลง Parquet ตัวสุดท้าย (แยกตาม Regime)
    df_features_with_vpin.write_parquet(feature_parquet)
    print(f"✅ Feature Pipeline Complete! Saved to {feature_parquet}")
    print(f"📊 Final Shape: {df_features_with_vpin.shape}")


if __name__ == "__main__":
    # รายชื่อ Regime (สภาวะตลาด) ที่เตรียมไฟล์ CSV ไว้แล้วใน raw/
    regimes = ["sideway", "trend", "toxic"]
    
    for regime in regimes:
        print(f"\n{'='*50}")
        print(f"🔥 เริ่มประมวลผลสภาวะตลาด: {regime.upper()}")
        print(f"{'='*50}")
        
        # ⭐️ เปลี่ยนชื่อไฟล์เป้าหมายให้มีคำว่า sideway/trend/toxic ต่อท้าย (จะได้ไม่ทับกัน)
        RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / f"BTCUSDT_{regime}.csv"
        TICK_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / f"BTCUSDT_tick_{regime}.parquet"
        FEATURE_PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / f"BTCUSDT_features_{regime}.parquet"
        
        # โยนเข้าเตาอบทีละไฟล์
        process_pipeline(
            regime_name=regime, 
            raw_csv=RAW_CSV_PATH, 
            tick_parquet=TICK_PARQUET_PATH, 
            feature_parquet=FEATURE_PARQUET_PATH
        )
        
        print(f"🎯 จบกระบวนการของ {regime.upper()}")