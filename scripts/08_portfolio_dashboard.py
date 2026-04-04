import streamlit as st
import json
import glob
import os
import pandas as pd
from pathlib import Path
import time

# ตั้งค่าหน้าจอให้กว้างแบบ Dashboard
st.set_page_config(page_title="HFT Command Center", page_icon="📈", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATUS_DIR = PROJECT_ROOT / "logs" / "live_status"

# ลิสต์เหรียญมาตรฐานเพื่อใช้คำนวณพอร์ตให้ตรงกับที่ตั้งไว้ใน hft_market_maker.py
DEFAULT_ASSETS = ["btc", "eth", "bnb", "sol", "xrp", "ltc", "ada"]
PORT_MAPPING = {asset.upper(): 8001 + idx for idx, asset in enumerate(DEFAULT_ASSETS)}

def load_data():
    if not STATUS_DIR.exists(): 
        return []
    
    files = glob.glob(os.path.join(STATUS_DIR, "*_status.json"))
    data = []
    for f in files:
        try:
            with open(f, 'r') as file:
                data.append(json.load(file))
        except Exception:
            pass
    return data

st.markdown("# 🌐 HFT Universal Command Center")
st.markdown("ศูนย์บัญชาการหลัก: มอนิเตอร์สถานะและกำไรของบอททุกตัวแบบ Real-time")

placeholder = st.empty()

while True:
    data = load_data()
    
    with placeholder.container():
        if not data:
            st.warning("⏳ รอการเชื่อมต่อข้อมูล... (โปรดรัน Deploy บอทอย่างน้อย 1 ตัว)")
        else:
            df = pd.DataFrame(data)
            
            # ---------------------------------------------------------
            # 🔗 1. NAVIGATION TABS (ปุ่มกดไปยังหน้า Asset)
            # ---------------------------------------------------------
            st.markdown("### 🚀 Quick Navigation")
            
            # สร้างคอลัมน์ตามจำนวนเหรียญที่กำลังรันอยู่
            cols = st.columns(len(df))
            for idx, row in df.iterrows():
                asset = row['Asset']
                # หา Port จาก Dictionary ถ้าไม่มีให้เดาว่าเป็นพอร์ต Custom (8090+)
                port = PORT_MAPPING.get(asset, 8090 + idx) 
                url = f"http://localhost:{port}"
                
                # สร้างปุ่ม Link Button
                cols[idx].link_button(f"🖥️ {asset} Dashboard", url, use_container_width=True)
            
            st.markdown("---")
            
            # ---------------------------------------------------------
            # 📊 2. PORTFOLIO OVERVIEW (สรุปยอดรวม)
            # ---------------------------------------------------------
            total_realized = df.get('Realized_PnL', pd.Series([0])).sum()
            total_unrealized = df.get('Unrealized_PnL', pd.Series([0])).sum()
            net_pnl = total_realized + total_unrealized
            
            st.markdown("### 🏆 Total Portfolio Performance")
            metric_cols = st.columns(3)
            metric_cols[0].metric("Realized PnL (กำไรรับรู้แล้ว)", f"${total_realized:,.4f}")
            metric_cols[1].metric("Unrealized PnL (กำไรลอยตัว)", f"${total_unrealized:,.4f}")
            metric_cols[2].metric("Net PnL (กำไรสุทธิ)", f"${net_pnl:,.4f}")
            
            # ---------------------------------------------------------
            # 📈 3. VISUAL OVERVIEW (กราฟแท่งเปรียบเทียบ)
            # ---------------------------------------------------------
            st.markdown("### 📊 PnL Breakdown by Asset")
            chart_col, table_col = st.columns([1, 1]) # แบ่งครึ่งหน้าจอ ซ้ายกราฟ ขวาตาราง
            
            with chart_col:
                # สร้าง Dataframe สำหรับกราฟโดยเฉพาะ
                chart_data = df[['Asset', 'Realized_PnL', 'Unrealized_PnL']].set_index('Asset')
                st.bar_chart(chart_data, color=["#00FF00", "#FFFF00"]) # สีเขียว=Realized, สีเหลือง=Unrealized
                
            with table_col:
                # ปรับแต่งตารางให้ดูง่าย
                display_df = df[['Asset', 'Mode', 'Inventory', 'Realized_PnL', 'Unrealized_PnL', 'Status']]
                
                # ไฮไลท์สีในตาราง (ถ้ายอด PnL มากกว่า 0 ให้เป็นสีเขียว)
                st.dataframe(
                    display_df.style.applymap(
                        lambda x: 'color: lime;' if isinstance(x, (int, float)) and x > 0 else ('color: red;' if isinstance(x, (int, float)) and x < 0 else ''),
                        subset=['Realized_PnL', 'Unrealized_PnL']
                    ),
                    use_container_width=True, 
                    hide_index=True
                )
            
            st.caption(f"Last updated: {df['Last_Update'].iloc[0]}")
            
    time.sleep(1.0) # รีเฟรชข้อมูลอัตโนมัติ