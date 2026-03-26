import polars as pl
from dotenv import load_dotenv

from pathlib import Path
import os
import sys

# Replace the SCRIPT_DIR logic with this:
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    # This runs if __file__ isn't defined (Notebook/REPL)
    SCRIPT_DIR = Path(os.getcwd())

PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

def main():
    print("🪓 Starting Domain Randomization Chunking (Daily)...")
    OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "chunks"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    regimes = ["sideway", "trend", "toxic"]
    
    for regime in regimes:
        INPUT_FILE = PROJECT_ROOT / "data" / "processed" / f"BTCUSDT_features_{regime}.parquet"
        
        if not os.path.exists(INPUT_FILE):
            print(f"⚠️ ข้าม {regime} - ไม่พบไฟล์ {INPUT_FILE}")
            continue

        print(f"\n📥 Loading {regime} dataset...")
        df = pl.read_parquet(INPUT_FILE)
        
        # ตรวจสอบเผื่อกรณีที่ลืมแปลง datetime เป็นประเภท Date/Datetime
        if "datetime" not in df.columns:
            print(f"❌ Error: ไม่พบคอลัมน์ 'datetime' ใน {regime}")
            continue
            
        # สร้างคอลัมน์ date
        df = df.with_columns([
            pl.col("datetime").dt.date().alias("date")
        ])
        
        print(f"🔪 Partitioning {regime} by days...")
        # ⭐️ เปลี่ยนมาใช้ partition_by แทนการ for loop filter (เร็วกว่ามาก)
        # as_dict=True จะคืนค่ากลับมาเป็น Dictionary { (วันที่,): DataFrame_ของวันนั้น }
        partitions = df.partition_by("date", as_dict=True)
        
        for (d,), df_chunk in partitions.items():
            # ลบคอลัมน์วันที่ทิ้งไปเมื่อใช้เสร็จ
            df_chunk = df_chunk.drop("date") 
            
            # ตั้งชื่อไฟล์ตาม Regime และ วันที่
            output_filename = os.path.join(OUTPUT_DIR, f"chunk_{regime}_{d}.parquet")
            
            df_chunk.write_parquet(output_filename)
            print(f"  -> Saved {output_filename} ({len(df_chunk)} rows)")

if __name__ == "__main__":
    main()