# ⛳️ HFT Quant Elite: RL-Driven Universal Market Maker v4.0

A High-Frequency Trading (HFT) system driven by **Reinforcement Learning (PPO)** and the **Avellaneda-Stoikov (AS)** model, specifically designed to capture spread profits and manage inventory risk in the Binance Futures market across multiple assets.

## 🚀 Key Features

* **Universal AI Model:** Train a single, robust PPO agent on cross-asset pooled data (e.g., BTC, ETH, BNB, SOL) to build generalized "muscle memory" capable of handling any market condition.
* **Master Portfolio Dashboard:** A centralized Streamlit command center (Port 8000) that monitors Realized/Unrealized PnL across all active bots in real-time, featuring quick-navigation tabs to individual asset dashboards.
* **Interactive Orchestrator & Static Port Mapping:** Manage the entire pipeline via a single terminal interface with an interactive multi-choice asset selector. Automatically assigns predictable static ports (e.g., 8001 for BTC, 8002 for ETH) for organized deployment.
* **Adaptive AS-PPO & GA Optimization:** Utilizes a Genetic Algorithm (GA) to find the optimal baseline parameters, while PPO dynamically adjusts Spread and Skew based on Volatility and Order Flow Toxicity (VPIN).
* **Fault-Tolerant Training:** Advanced data sanitization, gradient clipping, reward normalization, and C-contiguous memory management to strictly prevent PyTorch Segmentation Faults (SIGSEGV) and Exploding Gradients (NaN) on Apple Silicon.
* **Safety First:** Emergency **Kill Switch** protected by a 4-digit PIN (default: 1234) to immediately flatten positions and cancel all active orders.

---

## 🛠 Installation

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/your-repo/hft-rl.git](https://github.com/your-repo/hft-rl.git)
   cd hft-rl
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment Variables:**
   Create a `.env` file in the root directory and configure your API keys:
   ```env
   BINANCE_API_KEY=your_live_key
   BINANCE_SECRET_KEY=your_live_secret
   BINANCE_DEMO_API_KEY=your_testnet_key
   BINANCE_DEMO_SECRET_KEY=your_testnet_secret
   ```

---

## 🚩 How to Run

Launch the centralized orchestrator with a single command:

```bash
python3 hft_market_maker.py
```

### Main Menu Commands:
* **`[1]` Deploy Dashboard (Single / Portfolio Overview):** Launch the trading bots (Sandbox or Live). You can choose to run a single asset or deploy multiple assets simultaneously. The system will automatically open the Master Portfolio Dashboard.
* **`[2]` Run Out-of-Sample Backtest:** Evaluate the AI's performance on unseen data.
* **`[3]` Train UNIVERSAL Model (Multi-Asset Selection):** Start the data preparation, GA optimization, and PPO training pipeline. Features an interactive asset selector to pool data from multiple coins and a smart resume feature.

---

## 📊 Dashboard Insight

| Metric | Description |
| :--- | :--- |
| **Realized / Unrealized PnL** | Tracks both locked-in profits and floating profits/losses in real-time across the entire portfolio. |
| **BBA Distance** | The distance between our orders and the Best Bid/Ask (Top of the order book). |
| **ROC %** | Return on Capital, calculated from the net profit against the allocated maximum inventory risk. |
| **VPIN** | Volume-Synchronized Probability of Informed Trading. An indicator to detect toxic order flows. |
| **T2T Latency** | Tick-to-Trade Latency. Measures the round-trip speed from market data arrival to order execution. |

---

## 📂 Project Structure

* `hft_market_maker.py`: The main orchestrator script for managing the entire system workflow.
* `scripts/`: Contains modular sub-scripts (Download, Build Dataset, Train, GA Optimize, Live Real Money, Portfolio Dashboard).
* `models/`: Storage for the trained AI models (`.zip`).
* `configs/`: YAML files containing hyperparameters and asset-specific configurations.
* `data/`: Storage for raw CSVs and processed Parquet feature pools.
* `logs/live_status/`: Real-time JSON status updates sent by active bots to the Master Dashboard.

---

## ⚠️ Disclaimer

High-Frequency Trading is inherently highly risky. The developers are not responsible for any financial losses incurred. Please rigorously test the system in **Sandbox** mode before deploying with real capital.