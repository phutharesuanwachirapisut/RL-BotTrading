import os
import sys
import yaml
import time
import json
import asyncio
import websockets
import numpy as np
from collections import deque
from datetime import datetime
from stable_baselines3 import PPO
import ccxt.async_support as ccxt
from dotenv import load_dotenv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=ENV_PATH)

def load_config(yaml_path: str) -> dict:
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

# ==========================================
# ⚙️ 1. Live Feature Engine (รับข้อมูลสด + คำนวณ Technical Indicators แบบ Real-time)
# ==========================================
class LiveFeatureEngine:
    def __init__(self):
        print("⚙️ Initializing Live Feature Engine with RSI/MACD...")
        self.price_history_60s = deque(maxlen=60)
        self.buy_volume_history = deque(maxlen=60)
        self.sell_volume_history = deque(maxlen=60)
        
        self.current_sec_buy_vol = 0.0
        self.current_sec_sell_vol = 0.0
        self.last_trade_price = 0.0
        self.mid_price = 0.0 
        self.orderbook = {"bids": [], "asks": []}
        
        # ⭐️ เพิ่มหน่วยความจำสำหรับ RSI และ MACD
        self.long_price_history = deque(maxlen=300)

    def process_agg_trade(self, data):
        price = float(data['p'])
        qty = float(data['q'])
        is_buyer_maker = data['m']
        
        self.last_trade_price = price
        if is_buyer_maker:
            self.current_sec_sell_vol += qty
        else:
            self.current_sec_buy_vol += qty

    def process_book_ticker(self, data):
        best_bid = float(data['b'])
        best_ask = float(data['a'])
        self.mid_price = (best_bid + best_ask) / 2.0

    def calculate_rsi(self, periods=14):
        if len(self.long_price_history) < periods + 1: return 0.5 
        
        prices = np.array(self.long_price_history)
        deltas = np.diff(prices)
        
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-periods:])
        avg_loss = np.mean(losses[-periods:])
        
        if avg_loss == 0: return 1.0 
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi / 100.0

    def calculate_macd(self):
        if len(self.long_price_history) < 26: return 0.0
        
        prices = np.array(self.long_price_history)
        
        ema12 = np.mean(prices[-12:])
        ema26 = np.mean(prices[-26:])
        
        macd_raw = ema12 - ema26
        return np.clip(macd_raw / (self.mid_price * 0.005), -1.0, 1.0) if self.mid_price > 0 else 0.0

    def tick_every_second(self):
        current_price = self.last_trade_price if self.last_trade_price > 0 else self.mid_price
        self.price_history_60s.append(current_price)
        self.long_price_history.append(current_price)
        
        self.buy_volume_history.append(self.current_sec_buy_vol)
        self.sell_volume_history.append(self.current_sec_sell_vol)
        
        self.current_sec_buy_vol = 0.0
        self.current_sec_sell_vol = 0.0

        if len(self.price_history_60s) > 1:
            prices = np.array(self.price_history_60s)
            returns_pct = (prices[-1] - prices[-2]) / prices[-2]
            returns_pct_norm = np.clip(returns_pct * 100, -1.0, 1.0)
            
            returns_array = np.diff(prices) / prices[:-1]
            volatility_norm = np.std(returns_array) * 100 if len(returns_array) > 0 else 0.0
        else:
            returns_pct_norm = 0.0
            volatility_norm = 0.0
            
        buy_v = self.buy_volume_history[-1]
        sell_v = self.sell_volume_history[-1]
        total_v = buy_v + sell_v
        tfi_norm = (buy_v - sell_v) / total_v if total_v > 0 else 0.0
        
        rsi_norm = self.calculate_rsi()
        macd_raw_norm = self.calculate_macd()

        return returns_pct_norm, volatility_norm, tfi_norm, rsi_norm, macd_raw_norm

    def get_live_observation(self, current_inventory, max_inventory):
        ret, vola, tfi, rsi, macd = self.tick_every_second()
        inv_ratio = current_inventory / max_inventory if max_inventory > 0 else 0.0
        
        obs_array = np.array([ret, vola, tfi, rsi, macd, inv_ratio], dtype=np.float32)
        return np.nan_to_num(obs_array, nan=0.0, posinf=0.0, neginf=0.0)

# ==========================================
# 🤖 2. Production Market Maker (Async Edition)
# ==========================================
class ProductionMarketMaker:
    def __init__(self, model_path: str, hyper_config: dict, trading_config: dict, mode: str = "demo"):
        print(f"🚀 Initializing Async Production Market Maker ({mode.upper()} MODE)...")
        
        self.hyper_config = hyper_config 
        self.trading_config = trading_config
        self.mode = mode
        
        self.model = PPO.load(model_path, device="cpu")
        self.feature_engine = LiveFeatureEngine()
        
        self.inventory = 0.0
        
        self.symbol = self.trading_config['exchange']['symbol']
        self.order_size = self.trading_config['risk']['order_size'] 
        self.max_inventory = self.trading_config['risk']['max_inventory']
        self.min_spread = self.trading_config['strategy']['min_spread'] 
        self.max_spread = self.trading_config['strategy']['max_spread']
        self.vol_multiplier = self.trading_config['strategy']['vol_multiplier']
        self.max_skew_usd = self.trading_config['strategy']['max_skew_usd']
        self.order_update_threshold = self.trading_config['strategy']['order_update_threshold']
        
        # ⭐️ ----------------------------------------------------
        # 🧠 SMART SHAPE DETECTION (บอกลาปัญหา 30 vs 60 ตลอดกาล)
        # ----------------------------------------------------
        # ดึงขนาด Observation Space ที่ AI คาดหวังออกมาโดยตรง (เช่น 60)
        expected_shape = self.model.observation_space.shape[0]
        
        # หารด้วยจำนวน Feature ที่เราสกัดออกมา (6 ตัวแปร) เพื่อหาความยาว Stack
        self.stack_size = expected_shape // 6 
        print(f"🧠 [Smart Sync] Auto-adjusted frame stack to {self.stack_size} (Matches Expected Shape: {expected_shape})")
        
        # สร้างกระปุกความจำตามขนาดที่ AI ต้องการเป๊ะๆ
        self.frames = deque(maxlen=self.stack_size)
        
        if mode == "live":
            api_key = os.getenv('BINANCE_API_KEY')
            secret_key = os.getenv('BINANCE_SECRET_KEY')
            enable_demo = False
            print("🚨 WARNING: RUNNING IN LIVE REAL-MONEY MODE 🚨")
        else:
            api_key = os.getenv('BINANCE_DEMO_API_KEY')
            secret_key = os.getenv('BINANCE_DEMO_SECRET_KEY')
            enable_demo = True
            print("🟢 RUNNING IN SANDBOX DEMO MODE 🟢")

        self.exchange = ccxt.binance({
            'apiKey': api_key,       
            'secret': secret_key,    
            'enableRateLimit': True,
            'timeout': 30000,
            'options': {
                'defaultType': self.trading_config['exchange']['market_type'],
            }
        })
        
        self.exchange.enable_demo_trading(enable_demo) 
        self.current_open_bid = 0.0
        self.current_open_ask = 0.0
        self.order_update_threshold = 2.0

    def calculate_prices(self, mid_price: float, action: np.ndarray, volatility: float, vpin: float) -> tuple:
        spread_action = action[0]
        skew_action = action[1]

        base_half_spread = self.min_spread + ((spread_action + 1.0) / 2.0) * (self.max_spread - self.min_spread)
        final_half_spread = base_half_spread + (volatility * self.vol_multiplier)

        ai_skew = skew_action * self.max_skew_usd
        risk_skew = (self.inventory / self.max_inventory) * self.max_skew_usd
        final_skew = np.clip(ai_skew + risk_skew, -(final_half_spread - 0.5), (final_half_spread - 0.5))

        if vpin > 0.8:
            final_half_spread += 10.0

        my_bid = mid_price - final_half_spread - final_skew
        my_ask = mid_price + final_half_spread - final_skew
        
        return my_bid, my_ask

    async def listen_binance_ws(self):
        base_symbol = self.symbol.split(":")[0] 
        symbol_stream = base_symbol.replace("/", "").lower()
        
        base_url = "wss://stream.binance.com:9443/stream?streams="
        streams = f"{symbol_stream}@aggTrade/{symbol_stream}@bookTicker/{symbol_stream}@depth20@100ms"
        uri = base_url + streams
        
        while True:
            try:
                print(f"📡 [WS] Connecting to Binance Multi-Stream: {symbol_stream}...")
                async with websockets.connect(uri) as websocket:
                    print(f"✅ [WS] Connected! Listening for real-time order flow...")
                    while True:
                        message = await websocket.recv()
                        msg = json.loads(message)
                        
                        if "data" in msg:
                            stream_name = msg["stream"]
                            data = msg["data"]
                            
                            if "@aggTrade" in stream_name:
                                self.feature_engine.process_agg_trade(data)
                            elif "@bookTicker" in stream_name:
                                self.feature_engine.process_book_ticker(data)
                            elif "@depth20" in stream_name:
                                self.feature_engine.orderbook = {
                                    "bids": data.get('bids', []), 
                                    "asks": data.get('asks', [])
                                }
                                
            except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
                print(f"⚠️ WebSocket Disconnected: {e}. Reconnecting in 3 seconds...")
                await asyncio.sleep(3)
            except Exception as e:
                print(f"🚨 Unexpected WebSocket Error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def emergency_flatten(self):
        print("🚨 [KILL SWITCH] ACTIVATED: FLATTENING POSITIONS AND CANCELLING ALL ORDERS!")
        try:
            await self.exchange.cancel_all_orders(self.symbol)
            print("✅ [KILL SWITCH] Canceled all active quotes.")
            
            positions = await self.exchange.fetch_positions([self.symbol])
            if positions and float(positions[0]['info']['positionAmt']) != 0:
                amt = float(positions[0]['info']['positionAmt'])
                side = 'sell' if amt > 0 else 'buy'
                
                await self.exchange.create_market_order(self.symbol, side, abs(amt))
                base_coin = self.symbol.split('/')[0]
                print(f"✅ [KILL SWITCH] Market {side.upper()} order executed to flatten {abs(amt)} {base_coin}.")
                
            self.inventory = 0.0 
            print("🛑 [KILL SWITCH] PORTFOLIO FLATTENED COMPLETELY.")
        except Exception as e:
            print(f"⚠️ [KILL SWITCH ERROR] {e}")
            
    async def execute_orders(self, my_bid: float, my_ask: float):
        try:
            bid_diff = abs(my_bid - self.current_open_bid)
            ask_diff = abs(my_ask - self.current_open_ask)
            
            if bid_diff < self.order_update_threshold and ask_diff < self.order_update_threshold:
                return

            await self.exchange.cancel_all_orders(self.symbol)
            target_notional_usd = 105.0
            min_required_size = target_notional_usd / my_bid
            dynamic_size = max(self.order_size, round(min_required_size + 0.0005, 3))
            
            orders_to_create = []
            if self.inventory < self.max_inventory:
                orders_to_create.append(self.exchange.create_limit_buy_order(self.symbol, dynamic_size, my_bid))
            
            if self.inventory > -self.max_inventory:
                orders_to_create.append(self.exchange.create_limit_sell_order(self.symbol, dynamic_size, my_ask))
                
            if orders_to_create:
                await asyncio.gather(*orders_to_create)
                
            self.current_open_bid = my_bid
            self.current_open_ask = my_ask
            base_coin = self.symbol.split('/')[0]
            print(f"🔫 [Executed] Size: {dynamic_size:.3f} {base_coin} | Bid: {my_bid:.2f} | Ask: {my_ask:.2f}")

        except Exception as e:
            print(f"⚠️ [API Error] {str(e)}")