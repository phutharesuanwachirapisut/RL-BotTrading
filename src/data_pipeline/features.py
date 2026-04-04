import polars as pl
import numpy as np

def generate_institutional_features(parquet_path: str) -> pl.DataFrame:
    """
    โหลดข้อมูล Tick, สร้าง Features ทางเทคนิคและโครงสร้างจุลภาค, 
    ทำ Z-score Normalization และตัดเวลาทิ้ง
    """
    print(f"⚙️ สกัดฟีเจอร์ระดับสถาบันจาก: {parquet_path}")
    df = pl.read_parquet(parquet_path)
    
    # 1. จัดเรียงตามเวลา
    df = df.sort("datetime")
    
    # 2. คำนวณ Microstructure Features พื้นฐาน
    # (หมายเหตุ: OFI แท้ต้องใช้ Orderbook L2, ถ้าใช้ aggTrades เราจะใช้ TFI เป็นตัวแทนที่ทรงพลัง)
    df = df.with_columns([
        (pl.col("price").pct_change() * 100).alias("returns_pct"),
        pl.when(pl.col("side") == "BUY").then(pl.col("quantity")).otherwise(-pl.col("quantity")).alias("trade_flow")
    ])
    
    # 3. คำนวณ Rolling Features (เช่น 50 ticks)
    window = 50
    df = df.with_columns([
        # Volatility
        pl.col("returns_pct").rolling_std(window_size=window).alias("volatility"),
        # TFI (Trade Flow Imbalance สะสม)
        pl.col("trade_flow").rolling_sum(window_size=window).alias("tfi"),
        
        # Simple RSI (Proxy แบบรวดเร็ว)
        pl.col("returns_pct").clip(lower_bound=0).rolling_mean(window_size=window).alias("gain"),
        (-pl.col("returns_pct").clip(upper_bound=0)).rolling_mean(window_size=window).alias("loss")
    ])
    
    # คำนวณ RSI & MACD Proxy
    df = df.with_columns([
        (100.0 - (100.0 / (1.0 + (pl.col("gain") / (pl.col("loss") + 1e-8))))).alias("rsi"),
        (pl.col("price").ewm_mean(span=12) - pl.col("price").ewm_mean(span=26)).alias("macd_raw")
    ])
    
    # Drop rows ที่เกิด NaN จากการทำ Rolling
    df = df.drop_nulls()

    # ==========================================
    # ⭐️ 4. Z-SCORE NORMALIZATION (แยกตามเหรียญ)
    # ==========================================
    # ฟีเจอร์ที่ต้องการนำเข้า AI
    feature_cols = ["returns_pct", "volatility", "tfi", "rsi", "macd_raw"] # เพิ่ม VPIN ได้ถ้ามีฟังก์ชัน
    
    norm_exprs = []
    for col in feature_cols:
        col_mean = df.select(pl.col(col).mean()).item()
        col_std = df.select(pl.col(col).std()).item()
        # ป้องกันหารด้วย 0
        if col_std == 0: col_std = 1e-8 
        
        norm_exprs.append(
            ((pl.col(col) - col_mean) / col_std).alias(f"{col}_norm")
        )
        
    df = df.with_columns(norm_exprs)
    
    # ==========================================
    # ⭐️ 5. คัดกรองและทำลายความเชื่อมโยงของเวลา
    # ==========================================
    # เลือกเฉพาะคอลัมน์ Price (ไว้คำนวณ PnL ใน Env), Volume, และคอลัมน์ที่ Normalize แล้ว
    final_cols = ["price", "quantity"] + [f"{c}_norm" for c in feature_cols]
    df_final = df.select(final_cols)
    
    return df_final