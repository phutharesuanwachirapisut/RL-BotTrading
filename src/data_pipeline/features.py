import polars as pl
import numpy as np

def generate_rl_state_features(tick_parquet_path: str, window_size: str = "1s") -> pl.DataFrame:
    """
    ยุบข้อมูลระดับ Tick ให้กลายเป็นแท่งเทียนระดับ 1 วินาที (1-second timebars)
    พร้อมคำนวณ Volume ฝั่งซื้อ/ขาย และ Volatility
    """
    print("⏳ Grouping ticks into 1-second timebars...")
    df = pl.read_parquet(tick_parquet_path)
    
    # 1. ยุบรวมข้อมูลทุกๆ 1 วินาที (Group by dynamic)
    df_1s = df.group_by_dynamic("datetime", every=window_size).agg([
        pl.col("price").last().alias("price"),
        pl.col("quantity").sum().alias("volume"),
        # แยก Volume ฝั่งคนเคาะซื้อ (BUY) และคนเคาะขาย (SELL)
        pl.when(pl.col("side") == "BUY").then(pl.col("quantity")).otherwise(0).sum().alias("buy_vol"),
        pl.when(pl.col("side") == "SELL").then(pl.col("quantity")).otherwise(0).sum().alias("sell_vol")
    ])

    # 2. คำนวณความผันผวนย้อนหลัง 60 วินาที (Rolling Volatility)
    print("⏳ Calculating 60s Rolling Volatility...")
    df_1s = df_1s.with_columns([
        (pl.col("price").pct_change().fill_null(0)).alias("return")
    ]).with_columns([
        pl.col("return").rolling_std(window_size=60).fill_null(0).alias("volatility_60s")
    ])
    
    return df_1s

def calculate_vpin_and_merge(tick_parquet_path: str, df_time_features: pl.DataFrame, volume_bucket_size=10.0, window_size=50) -> pl.DataFrame:
    """
    คำนวณ TFI (Trade Flow Imbalance) และ VPIN (Volume-Synchronized Probability of Informed Trading)
    """
    print("⏳ Calculating TFI and Time-based VPIN...")
    
    # 1. คำนวณ TFI (แรงซื้อสุทธิในวินาทีนั้น)
    # สูตร: (Buy Vol - Sell Vol) / Total Vol
    df = df_time_features.with_columns([
        ((pl.col("buy_vol") - pl.col("sell_vol")) / (pl.col("volume") + 1e-8)).alias("tfi")
    ])
    
    # 2. คำนวณ VPIN (สัดส่วนความไม่สมดุลของออเดอร์ย้อนหลัง 50 วินาที)
    # เอาไว้จับสัญญาณว่า "รายใหญ่" กำลังไล่กวาดฝั่งใดฝั่งหนึ่งอยู่หรือไม่
    df = df.with_columns([
        (
            (pl.col("buy_vol").rolling_sum(window_size=window_size) - pl.col("sell_vol").rolling_sum(window_size=window_size)).abs() 
            / (pl.col("volume").rolling_sum(window_size=window_size) + 1e-8)
        ).fill_null(0).alias("vpin")
    ])
    
    # ทำความสะอาดข้อมูล (ลบแถวที่คำนวณช่วงแรกๆ ไม่ได้ หรือค่าเป็น NaN)
    df = df.drop(["return", "buy_vol", "sell_vol"]) # ลบคอลัมน์ขยะทิ้ง
    df = df.fill_nan(0).fill_null(0)
    
    return df