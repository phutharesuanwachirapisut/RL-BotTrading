นี่คือไฟล์ **README.md** สำหรับโปรเจกต์ของคุณที่ได้รับการออกแบบให้ดูเป็นมืออาชีพ มีความชัดเจนในเชิงเทคนิค และครอบคลุมขั้นตอนการใช้งานทั้งหมดตามที่คุณต้องการครับ

---

# HFT Quant Elite: RL-Driven Market Maker v3.2

ระบบเทรดความถี่สูง (High-Frequency Trading) ที่ขับเคลื่อนด้วยการเรียนรู้เชิงเสริมกำลัง (**Reinforcement Learning - PPO**) และโมเดล **Avellaneda-Stoikov** ออกแบบมาเพื่อทำกำไรจาก Spread ในตลาด Binance Futures โดยเฉพาะ

## 🚀 Key Features

* **All-in-One Orchestrator:** ควบคุมทุกขั้นตอน (Data -> Train -> Backtest -> Live) ผ่านไฟล์เดียว
* **Adaptive AS-PPO Strategy:** ใช้ AI ปรับค่า Spread และ Skew ตามความผันผวน (Volatility) และความเป็นพิษของ Order Flow (VPIN)
* **Advanced Real-time Dashboard:** แสดงผล Orderbook L2, Inventory Risk, Latency History และกำไรสุทธิหักค่าธรรมเนียมจริง (Maker/Taker Fees)
* **Safety First:** ระบบ **Kill Switch** พร้อมรหัส PIN 4 หลัก เพื่อล้างพอร์ตและยกเลิกออเดอร์ทั้งหมดทันที
* **Resource Optimized:** ออกแบบมาเพื่อรันบน Apple Silicon (M3) โดยใช้ `Polars` และ `PyTorch MPS`

---

## 🛠 Installation

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/your-repo/hft-rl.git
    cd hft-rl
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment:**
    สร้างไฟล์ `.env` ที่ Root Directory และใส่ API Keys:
    ```env
    BINANCE_API_KEY=your_live_key
    BINANCE_SECRET_KEY=your_live_secret
    BINANCE_DEMO_API_KEY=your_testnet_key
    BINANCE_DEMO_SECRET_KEY=your_testnet_secret
    ```

---

## How to Run

เริ่มต้นใช้งานผ่านระบบ Pipeline หลักด้วยคำสั่งเดียว:

```bash
python3 hft_market_maker.py
```

### Pipeline Workflow:
1.  **Model Inventory:** ระบบจะสแกนโฟลเดอร์ `models/` เพื่อหาโมเดลที่มีอยู่แล้ว
2.  **Main Menu:**
    * **Deploy:** เลือกรัน Dashboard จากโมเดลที่มี (เลือกได้ทั้ง Sandbox หรือ Live)
    * **Train New:** เริ่มกระบวนการปั้นโมเดลใหม่
3.  **Regime Configuration (สำหรับ Train):** ระบุวันที่เริ่มต้นของ 3 สภาวะตลาด (Mean-Reverting, Trending, High Volatility)
4.  **Auto Cleanup:** ระบบจะลบไฟล์ Raw Data ทันทีหลังเทรนเสร็จเพื่อประหยัดพื้นที่ SSD

---

## 📊 Dashboard Insight

| Metric | Description |
| :--- | :--- |
| **BBA Distance** | ระยะห่างระหว่างออเดอร์ของเรากับคิวแรกของกระดาน (Best Bid/Ask) |
| **ROC %** | Return on Capital คำนวณจากกำไรเทียบกับ Max Inventory Risk |
| **VPIN** | ดัชนีชี้วัดความเสี่ยงจากการโดนจับคู่โดยผู้เล่นที่รู้ข้อมูลภายใน (Informed Traders) |
| **T2T Latency** | Tick-to-Trade Latency วัดความเร็วตั้งแต่ราคาขยับจนถึงส่งคำสั่งสำเร็จ |
| **Gross vs Fee** | แสดงกำไรดิบแยกกับค่าธรรมเนียมสะสม (Maker 0.02% / Taker 0.05%) |

---

## 📂 Project Structure

* `hft_market_maker.py`: ไฟล์หลักสำหรับควบคุมระบบทั้งหมด (Orchestrator)
* `scripts/`: รวมสคริปต์ย่อย (Download, Train, Backtest, Live Dashboard)
* `models/`: พื้นที่เก็บโมเดล AI (`.zip`)
* `configs/`: ไฟล์ตั้งค่าคู่เหรียญและ Hyperparameters
* `results/`: เก็บผลการ Backtest และรายงาน Performance

---

## ⚠️ Disclaimer

การเทรดความถี่สูงมีความเสี่ยงสูงมาก ผู้พัฒนาไม่รับผิดชอบต่อความสูญเสียทางการเงินที่เกิดขึ้น โปรดทดสอบในโหมด **Sandbox** จนมั่นใจก่อนเข้าสู่ตลาดจริง

---
*Developed with Empathy and Candor for Quant Developers.*