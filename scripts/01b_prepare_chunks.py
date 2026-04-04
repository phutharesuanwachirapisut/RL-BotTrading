import polars as pl
import argparse
import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("🪓 [DATA POOLING] เริ่มต้นสร้าง Cross-Asset Replay Buffer...")
    
    PROC_DIR = PROJECT_ROOT / "data" / "processed"
    OUTPUT_DIR = PROC_DIR / "pooled_chunks"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. กวาดหาไฟล์ features ทั้งหมด
    feature_files = glob.glob(os.path.join(PROC_DIR, "*_features_*.parquet"))
    
    if not feature_files:
        print("❌ ไม่พบไฟล์ Features กรุณารัน Data Pipeline ก่อน")
        sys.exit(1)

    print(f"📥 พบไฟล์ข้อมูลทั้งหมด {len(feature_files)} ไฟล์. กำลังคัดกรองและเทรวมกัน (Pooling)...")
    
    # ⭐️ 2. กำหนดคอลัมน์มาตรฐานของระบบใหม่ (7 คอลัมน์)
    EXPECTED_COLS = [
        "price", "quantity", "returns_pct_norm", 
        "volatility_norm", "tfi_norm", "rsi_norm", "macd_raw_norm"
    ]
    
    dfs = []
    for f in feature_files:
        df = pl.read_parquet(f)
        
        # ⭐️ ตรวจสอบว่าไฟล์นี้มีคอลัมน์ครบตามระบบใหม่หรือไม่
        if all(col in df.columns for col in EXPECTED_COLS):
            # ดึงมาเฉพาะคอลัมน์ที่ต้องการ เรียงให้ตรงกัน
            dfs.append(df.select(EXPECTED_COLS))
        else:
            print(f"⚠️ ข้ามไฟล์เก่า/ผิดรูปแบบ: {os.path.basename(f)} (มี {len(df.columns)} คอลัมน์)")
            
    if not dfs:
        print("❌ ไม่พบไฟล์ข้อมูลที่ถูกต้องตามโครงสร้างใหม่เลย (กรุณารัน Step 1 & 2 ใหม่)")
        sys.exit(1)
        
    # 3. รวมร่าง
    pooled_df = pl.concat(dfs)
    total_rows = len(pooled_df)
    print(f"🌊 Pooled Dataset Size: {total_rows:,} rows")
    
    # 4. สับข้อมูลเป็น Chunks
    chunk_size = 500_000 
    num_chunks = (total_rows // chunk_size) + 1
    
    print(f"🔪 กำลังสับเป็น {num_chunks} Chunks...")
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, total_rows)
        
        chunk = pooled_df[start_idx:end_idx]
        if len(chunk) > 0:
            out_name = OUTPUT_DIR / f"universal_pool_chunk_{i+1:03d}.parquet"
            chunk.write_parquet(str(out_name))
            print(f"  -> Saved {out_name.name} ({len(chunk):,} rows)")
            
    print("✅✅✅ การสร้าง Cross-Asset Pooled Dataset เสร็จสมบูรณ์! พร้อมส่งให้ AI เรียนรู้แบบสากลแล้ว")

if __name__ == "__main__":
    main()