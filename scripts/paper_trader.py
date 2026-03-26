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
# ⚙️ 1. Live Feature Engine (รับข้อมูลสด)
# ==========================================
class LiveFeatureEngine:
    def __init__(self):
        print("⚙️ Initializing Live Feature Engine...")
        self.price_history_60s = deque(maxlen=60)
        self.buy_volume_history = deque(maxlen=60)
        self.sell_volume_history = deque(maxlen=60)
        
        self.current_sec_buy_vol = 0.0
        self.current_sec_sell_vol = 0.0
        self.last_trade_price = 0.0
        self.mid_price = 0.0 
        self.orderbook = {"bids": [], "asks": []} # ⭐️ เพิ่มตัวแปรเก็บ Orderbook

    def process_agg_trade(self, data):
        """ประมวลผลข้อมูลคนเคาะซื้อ/ขาย (@aggTrade)"""
        price = float(data['p'])
        qty = float(data['q'])
        is_buyer_maker = data['m'] # True = Sell Taker, False = Buy Taker
        
        self.last_trade_price = price
        if is_buyer_maker:
            self.current_sec_sell_vol += qty
        else:
            self.current_sec_buy_vol += qty

    def process_book_ticker(self, data):
        """ประมวลผลข้อมูล Orderbook BBO (@bookTicker)"""
        best_bid = float(data['b'])
        best_ask = float(data['a'])
        # คำนวณราคากลาง (Mid Price)
        self.mid_price = (best_bid + best_ask) / 2.0

    def tick_every_second(self):
        """แพ็กข้อมูลทุกๆ 1 วินาที เพื่อส่งให้ AI"""
        self.price_history_60s.append(self.last_trade_price if self.last_trade_price > 0 else self.mid_price)
        self.buy_volume_history.append(self.current_sec_buy_vol)
        self.sell_volume_history.append(self.current_sec_sell_vol)
        
        total_vol_this_sec = self.current_sec_buy_vol + self.current_sec_sell_vol
        self.current_sec_buy_vol = 0.0
        self.current_sec_sell_vol = 0.0

        if len(self.price_history_60s) > 1:
            prices = np.array(self.price_history_60s)
            returns = np.diff(prices) / prices[:-1]
            vola_60s = np.std(returns) if len(returns) > 0 else 0.0
        else:
            vola_60s = 0.0
            
        buy_v = self.buy_volume_history[-1]
        sell_v = self.sell_volume_history[-1]
        total_v = buy_v + sell_v
        tfi = (buy_v - sell_v) / total_v if total_v > 0 else 0.0
        
        total_buy_60s = sum(self.buy_volume_history)
        total_sell_60s = sum(self.sell_volume_history)
        total_vol_60s = total_buy_60s + total_sell_60s
        vpin = abs(total_buy_60s - total_sell_60s) / total_vol_60s if total_vol_60s > 0 else 0.0

        return total_vol_this_sec, vola_60s, tfi, vpin

    def get_live_observation(self, current_inventory, max_inventory):
        vol, vola_60s, tfi, vpin = self.tick_every_second()
        inv_ratio = current_inventory / max_inventory
        
        now = datetime.now()
        seconds_since_midnight = now.hour * 3600 + now.minute * 60 + now.second
        time_ratio = seconds_since_midnight / 86400.0
        
        return np.array([vol, vola_60s, tfi, vpin, inv_ratio, time_ratio], dtype=np.float32)

# ==========================================
# 🤖 2. Production Market Maker (Async Edition)
# ==========================================
class ProductionMarketMaker:
    def __init__(self, model_path: str, config: dict):
        print("🚀 Initializing Async Production Market Maker...")
        self.config = config['env']
        self.model = PPO.load(model_path, device="cpu")
        self.feature_engine = LiveFeatureEngine()
        
        # ⭐️ เอากลับมาแล้ว! ตัวแปรเก่าที่โดนทับไป
        self.inventory = 0.0
        self.max_inventory = self.config.get("max_inventory", 0.0004)
        self.order_size = self.config.get("order_size", 0.0001)
        self.min_spread = self.config.get("min_spread", 5.0)
        self.max_spread = self.config.get("max_spread", 30.0)
        self.vol_multiplier = self.config.get("vol_multiplier", 10.0)
        self.max_skew_usd = self.config.get("max_skew_usd", 30.0)
        
        self.stack_size = self.config.get("frame_stack", 5)
        self.frames = deque(maxlen=self.stack_size)
        
        # ⭐️ ส่วนของการเชื่อมต่อ CCXT
        self.symbol = 'BTC/USDT:USDT' # สัญลักษณ์สำหรับ Futures
        
        self.exchange = ccxt.binance({
            'apiKey': os.getenv('BINANCE_DEMO_API_KEY'),       
            'secret': os.getenv('BINANCE_DEMO_SECRET_KEY'),    
            'enableRateLimit': True,
            'options': {
                'defaultType': 'future', 
            }
        })
        self.exchange.enable_demo_trading(True)
        
        # ⭐️ 2. ตัวแปรจำสถานะออเดอร์ปัจจุบัน (กันการ Spam API)
        self.current_open_bid = 0.0
        self.current_open_ask = 0.0
        self.order_update_threshold = 2.0 # ถ้าราคาขยับไม่ถึง 2 USD จะยังไม่แก้ตั๋ว (ประหยัด Rate Limit)

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

    # ⭐️ โค้ดที่ต้องแก้ไขในไฟล์ paper_trader.py
    async def listen_binance_ws(self):
        """หูทิพย์: รอรับข้อมูลจาก Binance ตลอดเวลา (พร้อมระบบ Auto-Reconnect)"""
        uri = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade/btcusdt@bookTicker/btcusdt@depth20@100ms"
        
        while True: # ⭐️ เพิ่ม While Loop วงนอกสุด เพื่อให้มันพยายามต่อใหม่เรื่อยๆ
            try:
                print(f"📡 Connecting to Binance WebSocket...")
                async with websockets.connect(uri) as websocket:
                    print(f"✅ Connected! Listening for real-time order flow...")
                    while True:
                        message = await websocket.recv()
                        data = json.loads(message)
                        
                        if 'e' in data and data['e'] == 'aggTrade':
                            self.feature_engine.process_agg_trade(data)
                        elif 'bids' in data and 'asks' in data:
                            self.feature_engine.orderbook = {"bids": data['bids'], "asks": data['asks']}
                        elif 'u' in data: 
                            self.feature_engine.process_book_ticker(data)
                            
            except (websockets.exceptions.ConnectionClosedError, ConnectionResetError) as e:
                # ⭐️ ดักจับ Error ตอนสายหลุด
                print(f"⚠️ WebSocket Disconnected: {e}. Reconnecting in 3 seconds...")
                await asyncio.sleep(3) # รอ 3 วินาทีแล้ววนกลับไปต่อใหม่
            except Exception as e:
                # เผื่อ Error อื่นๆ ที่ไม่ได้คาดคิด
                print(f"🚨 Unexpected WebSocket Error: {e}. Reconnecting in 5 seconds...")
                await asyncio.sleep(5)

    async def trading_loop(self):
        """[Task 2] สมองสั่งการ: ทำงานทุกๆ 1 วินาทีเป๊ะๆ"""
        print("🟢 Starting AI Trading Loop...")
        
        # รอให้ WebSocket ดึงข้อมูล Mid Price ก้อนแรกมาก่อน ค่อยเริ่มเทรด
        while self.feature_engine.mid_price == 0:
            await asyncio.sleep(0.1)
            
        try:
            while True:
                loop_start = time.time()
                
                # 1. ดึง Mid Price ของจริงที่อัปเดตจาก WS
                mid_price = self.feature_engine.mid_price
                
                # 2. สร้าง Observation จากกระแสข้อมูลจริง
                live_features = self.feature_engine.get_live_observation(self.inventory, self.max_inventory)
                
                if len(self.frames) == 0:
                    for _ in range(self.stack_size):
                        self.frames.append(live_features)
                else:
                    self.frames.append(live_features)
                    
                obs = np.concatenate(list(self.frames))
                
                # 3. ให้ AI คิด
                action, _ = self.model.predict(obs, deterministic=True)
                
                # 4. แปลงเป็นราคาออเดอร์
                volatility = live_features[1]
                vpin = live_features[3]
                my_bid, my_ask = self.calculate_prices(mid_price, action, volatility, vpin)
                
                # 5. แสดงผล (ถ้าของจริงจะยิง API ตรงนี้)
                print(f"⚡ [LIVE] Mid: {mid_price:.2f} | Bid: {my_bid:.2f} | Ask: {my_ask:.2f} | Spread: {(my_ask-my_bid):.2f} | Vol 1s: {live_features[0]:.4f}")
                
                # ควบคุม Loop ให้ AI ทำงาน 1 ครั้งต่อวินาที (ไม่ไปบล็อก WebSocket)
                elapsed = time.time() - loop_start
                sleep_time = max(0.0, 1.0 - elapsed)
                await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            print("\n🛑 AI Trading Loop Stopped.")

    async def run(self):
        """รัน 2 Tasks คู่ขนานกัน"""
        # ใช้ asyncio.gather เพื่อให้หูรับข้อมูล กับ สมองสั่งการ ทำงานพร้อมกัน
        await asyncio.gather(
            self.listen_binance_ws(),
            self.trading_loop()
        )
    async def execute_orders(self, my_bid: float, my_ask: float):
        """ระบบยิงออเดอร์อัจฉริยะ (Smart Execution)"""
        try:
            # เช็คว่าราคาเป้าหมายใหม่ ห่างจากออเดอร์ที่ตั้งไว้เดิมเกิน Threshold หรือไม่?
            bid_diff = abs(my_bid - self.current_open_bid)
            ask_diff = abs(my_ask - self.current_open_ask)
            
            # ถ้าราคาแทบไม่ขยับเลย ให้ข้ามไป ไม่ต้องรบกวน API (ลด Latency)
            if bid_diff < self.order_update_threshold and ask_diff < self.order_update_threshold:
                return

            # ถ้าราคาเปลี่ยนเยอะ -> ยกเลิกออเดอร์เก่าทั้งหมดก่อน
            await self.exchange.cancel_all_orders(self.symbol)
            
            orders_to_create = []
            
            # ถ้าของยังไม่เต็มมือฝั่งซื้อ ให้ตั้ง Bid
            if self.inventory < self.max_inventory:
                orders_to_create.append(
                    self.exchange.create_limit_buy_order(self.symbol, self.order_size, my_bid)
                )
            
            # ถ้าของยังไม่เต็มมือฝั่งขาย (ยังไม่ Short ล้น) ให้ตั้ง Ask
            if self.inventory > -self.max_inventory:
                orders_to_create.append(
                    self.exchange.create_limit_sell_order(self.symbol, self.order_size, my_ask)
                )
                
            # ยิง 2 ออเดอร์ (Bid/Ask) ไปที่กระดานพร้อมๆ กันแบบขนาน (Parallel)
            if orders_to_create:
                await asyncio.gather(*orders_to_create)
                
            # จำราคาที่เพิ่งตั้งไป
            self.current_open_bid = my_bid
            self.current_open_ask = my_ask
            print(f"🔫 [Executed] New Limit Orders -> Bid: {my_bid:.2f} | Ask: {my_ask:.2f}")

        except Exception as e:
            print(f"⚠️ [API Error] {str(e)}")

if __name__ == "__main__":
    CONFIG_PATH = PROJECT_ROOT / "configs" / "hyperparameters.yaml"
    MODEL_PATH = PROJECT_ROOT / "models" / "ppo_hft_chunked_final.zip"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ไม่พบไฟล์โมเดลที่: {MODEL_PATH}")
        sys.exit(1)
        
    config = load_config(CONFIG_PATH)
    bot = ProductionMarketMaker(MODEL_PATH, config)
    
    # รัน Async Event Loop
    try:
        asyncio.run(bot.run())
    except KeyboardInterrupt:
        print("\n🛑 Bot Terminated by User.")