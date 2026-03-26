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
    def __init__(self, model_path: str, hyper_config: dict, trading_config: dict, mode: str = "demo"):
        print(f"🚀 Initializing Async Production Market Maker ({mode.upper()} MODE)...")
        
        self.hyper_config = hyper_config # ✅ เก็บไว้เฉยๆ ไม่ต้องเจาะหา 'env' แล้ว
        self.trading_config = trading_config
        
        self.model = PPO.load(model_path, device="cpu")
        self.feature_engine = LiveFeatureEngine()
        
        self.inventory = 0.0
        
        # ⭐️ 2. โยงค่าจากไฟล์ trading_env.yaml มาใช้แทนการพิมพ์ตัวเลขตรงๆ!
        self.symbol = self.trading_config['exchange']['symbol']    # ดึงมาจากหมวด exchange
        
        self.order_size = self.trading_config['risk']['order_size'] # ดึงมาจากหมวด risk
        self.max_inventory = self.trading_config['risk']['max_inventory']
        
        self.min_spread = self.trading_config['strategy']['min_spread'] # ดึงมาจากหมวด strategy
        self.max_spread = self.trading_config['strategy']['max_spread']
        self.vol_multiplier = self.trading_config['strategy']['vol_multiplier']
        self.max_skew_usd = self.trading_config['strategy']['max_skew_usd']
        self.order_update_threshold = self.trading_config['strategy']['order_update_threshold']
        
        self.stack_size = self.trading_config['features']['frame_stack']
        self.frames = deque(maxlen=self.stack_size)
        
        # ⭐️ 3. ตั้งค่า CCXT เหมือนเดิม
        # ตัวอย่างการตั้งค่า API ใน __init__ ของ ProductionMarketMaker
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
        
        self.exchange.enable_demo_trading(enable_demo) # เปิด/ปิดโหมด Testnet ของ CCXT
        
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
        # 1. จัดการชื่อ Symbol ให้ถูกต้อง เช่น ETH/USDT, ETH/USDT:USDT -> ethusdt
        base_symbol = self.symbol.split(":")[0]  # ตัด :USDT ออก (ถ้ามี)
        symbol_stream = base_symbol.replace("/", "").lower()
        
        # 2. ใช้ Endpoint สำหรับ Multi-stream
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
                        
                        # 3. ⭐️ สำคัญที่สุด: แกะ data ออกมาจากซองจดหมายของ Multi-stream
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
    
    # === เพิ่ม Method นี้ใน class ProductionMarketMaker ===
    async def emergency_flatten(self):
        """ระบบฉุกเฉิน (Kill Switch): ยกเลิกทุก Order และโยน Market Order เพื่อล้างพอร์ตทันที"""
        print("🚨 [KILL SWITCH] ACTIVATED: FLATTENING POSITIONS AND CANCELLING ALL ORDERS!")
        try:
            # 1. ยกเลิก Limit Orders เดิมทั้งหมด
            await self.exchange.cancel_all_orders(self.symbol)
            print("✅ [KILL SWITCH] Canceled all active quotes.")
            
            # 2. เช็ค Position เพื่อปิด
            positions = await self.exchange.fetch_positions([self.symbol])
            if positions and float(positions[0]['info']['positionAmt']) != 0:
                amt = float(positions[0]['info']['positionAmt'])
                side = 'sell' if amt > 0 else 'buy'
                
                # โยน Market Order สวนทางเพื่อปิด Position
                await self.exchange.create_market_order(self.symbol, side, abs(amt))
                
                # ⭐️ ดึงชื่อเหรียญอัตโนมัติ
                base_coin = self.symbol.split('/')[0]
                print(f"✅ [KILL SWITCH] Market {side.upper()} order executed to flatten {abs(amt)} {base_coin}.")
                
            self.inventory = 0.0 # รีเซ็ตสถานะ
            print("🛑 [KILL SWITCH] PORTFOLIO FLATTENED COMPLETELY.")
        except Exception as e:
            print(f"⚠️ [KILL SWITCH ERROR] {e}")
            
    async def execute_orders(self, my_bid: float, my_ask: float):
        """ระบบยิงออเดอร์อัจฉริยะ (Smart Execution) + Auto-Size Notional"""
        try:
            # เช็คว่าราคาเป้าหมายใหม่ ห่างจากออเดอร์ที่ตั้งไว้เดิมเกิน Threshold หรือไม่?
            bid_diff = abs(my_bid - self.current_open_bid)
            ask_diff = abs(my_ask - self.current_open_ask)
            
            # ถ้าราคาแทบไม่ขยับเลย ให้ข้ามไป ไม่ต้องรบกวน API (ลด Latency)
            if bid_diff < self.order_update_threshold and ask_diff < self.order_update_threshold:
                return

            # ถ้าราคาเปลี่ยนเยอะ -> ยกเลิกออเดอร์เก่าทั้งหมดก่อน
            await self.exchange.cancel_all_orders(self.symbol)
            
            # ==========================================
            # ⭐️ ระบบ AUTO-SIZE NOTIONAL (แก้บั๊ก -4164)
            # ==========================================
            # 1. ตั้งเป้าหมายมูลค่าที่ 105 USD (เผื่อบัฟเฟอร์ 5 ดอลลาร์ กันราคาร่วงกระทันหัน)
            target_notional_usd = 105.0
            
            # 2. เอาเป้าหมายมาหารด้วยราคาประมูล (Bid) เพื่อหาว่าต้องใช้ขั้นต่ำกี่เหรียญ
            min_required_size = target_notional_usd / my_bid
            
            # 3. เลือกขนาดที่ใหญ่ที่สุด ระหว่าง 'ค่าในไฟล์ yaml' กับ 'ค่าที่ระบบคำนวณให้'
            # (ปัดเศษทศนิยม 3 ตำแหน่งเพื่อให้ตรงกับกฎ Tick Size ของ BTC/USDT)
            dynamic_size = max(self.order_size, round(min_required_size + 0.0005, 3))
            
            orders_to_create = []
            
            # ⭐️ ใช้ตัวแปร dynamic_size แทน self.order_size ในการส่งคำสั่ง
            # ถ้าของยังไม่เต็มมือฝั่งซื้อ ให้ตั้ง Bid
            if self.inventory < self.max_inventory:
                orders_to_create.append(
                    self.exchange.create_limit_buy_order(self.symbol, dynamic_size, my_bid)
                )
            
            # ถ้าของยังไม่เต็มมือฝั่งขาย (ยังไม่ Short ล้น) ให้ตั้ง Ask
            if self.inventory > -self.max_inventory:
                orders_to_create.append(
                    self.exchange.create_limit_sell_order(self.symbol, dynamic_size, my_ask)
                )
                
            # ยิง 2 ออเดอร์ (Bid/Ask) ไปที่กระดานพร้อมๆ กันแบบขนาน (Parallel)
            if orders_to_create:
                await asyncio.gather(*orders_to_create)
                
            # จำราคาที่เพิ่งตั้งไป
            self.current_open_bid = my_bid
            self.current_open_ask = my_ask
            
            # ⭐️ ดึงชื่อเหรียญด้านหน้าออกมา เช่น BNB/USDT -> BNB
            base_coin = self.symbol.split('/')[0]
            
            # อัปเดต Log ให้โชว์ไซส์ที่ยิงไป พร้อมชื่อเหรียญที่ถูกต้อง
            print(f"🔫 [Executed] Size: {dynamic_size:.3f} {base_coin} | Bid: {my_bid:.2f} | Ask: {my_ask:.2f}")

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