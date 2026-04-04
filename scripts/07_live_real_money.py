import os
import time
import asyncio
import json
import traceback
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import ccxt.async_support as ccxt
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "scripts"))

from scripts.paper_trader import ProductionMarketMaker, load_config
import argparse

os.chdir(PROJECT_ROOT / "scripts")
app = FastAPI()

# ⭐️ 1. เพิ่ม Metrics ที่จำเป็นลงใน BOT_STATE
BOT_STATE = {
    "symbol": "--",
    "mid_price": 0.0, "inventory": 0.0, "max_inventory": 0.0, 
    "pos_cost": 0.0, "pnl_pct": 0.0, "pnl": 0.0, "gross_pnl": 0.0,
    "realized_pnl": 0.0, "est_fees": 0.0,  # <--- เช็คว่ามี realized_pnl อยู่ตรงนี้
    "roc_pct": 0.0, "bid_dist": 0.0, "ask_dist": 0.0,
    "spread": 0.0, "skew": 0.0, "volatility": 0.0, "bid": 0.0, "ask": 0.0,
    "action": [0.0, 0.0], "tfi": 0.0, "vpin": 0.0, 
    "orderbook": {"bids": [], "asks": []},
    # New Metrics
    "latency_ping": 0, "latency_t2t": 0.0, 
    "recent_trades": [], "volume_24h": 0.0, "quote_win_rate": 0.0,
    "_total_quotes": 0, "_total_fills": 0
}

connected_clients = []

@app.get("/")
async def get_dashboard():
    with open(PROJECT_ROOT / "scripts" / "dashboard.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.25)
                payload = json.loads(data)
                
                # ⭐️ ปรับปรุงระบบตรวจสอบ Kill Switch ให้เก็บ State ชัดเจนขึ้น
                if payload.get("action") == "kill_switch":
                    if payload.get("pin") == "1234":
                        print("🔥 [SECURITY] KILL SWITCH INITIATED FROM DASHBOARD 🔥")
                        # เซ็ต State เพื่อสั่งให้ Loop ทำงานชอตฉุกเฉิน
                        app.state.trigger_flatten = True 
                    else:
                        print("❌ [SECURITY] Invalid Kill Switch PIN entered!")
            except asyncio.TimeoutError:
                pass # วนลูปส่งข้อมูลปกติถ้าไม่มีการกดปุ่มจากหน้าเว็บ
            except Exception as e:
                pass # ป้องกัน Error เวลาพาร์ส JSON ไม่ผ่าน
            
            await websocket.send_text(json.dumps(BOT_STATE))
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def dashboard_trading_loop(bot: ProductionMarketMaker):
    print("🟢 Starting Dashboard Trading Loop...")
    app.state.trigger_flatten = False
    await bot.exchange.load_markets()
    
    while bot.feature_engine.mid_price == 0:
        await asyncio.sleep(0.1)
        
    try:
        while True:
            loop_start = asyncio.get_event_loop().time()
            
            # 🚨 ดักจับสัญญาณ Kill Switch อย่างปลอดภัย
            if getattr(app.state, 'trigger_flatten', False):
                print("🚨 Triggering Emergency Flatten Sequence...")
                app.state.is_emergency_flattening = True
                try:
                    await bot.emergency_flatten()
                    print("✅ Emergency Flatten Completed.")
                except Exception as e:
                    print(f"⚠️ [FATAL] Error during emergency flatten: {e}")
                finally:
                    # รีเซ็ตสถานะกลับเป็นปกติเสมอ ป้องกันบอทค้าง
                    app.state.trigger_flatten = False
                    app.state.is_emergency_flattening = False
                continue # ข้ามการ Quote ในเสี้ยววินาทีนี้ไปก่อน
                
            try:
                # ⭐️ 3. วัด Tick-to-Trade Latency
                t2t_start = time.perf_counter()
                
                mid_price = bot.feature_engine.mid_price
                live_features = bot.feature_engine.get_live_observation(bot.inventory, bot.max_inventory)
                
                # Mock Ping (จำลอง Exchange Ping สำหรับสาธิต ถ้าของจริงต้องเทียบ timestamp จาก Stream)
                import random
                BOT_STATE["latency_ping"] = random.randint(15, 35) 
                
                if len(bot.frames) == 0:
                    for _ in range(bot.stack_size):
                        bot.frames.append(live_features)
                else:
                    bot.frames.append(live_features)
                    
                obs = np.concatenate(list(bot.frames))
                action, _ = bot.model.predict(obs, deterministic=True)
                
                volatility = live_features[1]
                vpin = live_features[3]
                my_bid, my_ask = bot.calculate_prices(mid_price, action, volatility, vpin)
                
                # ยิงออเดอร์
                BOT_STATE["_total_quotes"] += 2
                await bot.execute_orders(my_bid, my_ask)
                
                BOT_STATE["latency_t2t"] = (time.perf_counter() - t2t_start) * 1000

                # ⭐️ ปรับปรุงระบบคำนวณ PNL และ Fee
                pos_cost, pnl_pct, gross_pnl, net_pnl = 0.0, 0.0, 0.0, 0.0
                
                try:
                    positions = await bot.exchange.fetch_positions([bot.symbol])
                    if positions and float(positions[0]['info']['positionAmt']) != 0:
                        pos = positions[0]
                        entry_price = float(pos['entryPrice'])
                        actual_inv = float(pos['info']['positionAmt'])                        
                        
                        MAKER_FEE_RATE = 0.00020
                        TAKER_FEE_RATE = 0.00050
                        
                        # ระบบตรวจจับการ Match Order และคิดค่าธรรมเนียมสะสม
                        if actual_inv != bot.inventory:
                            diff = actual_inv - bot.inventory
                            side = "BUY" if diff > 0 else "SELL"
                            qty = abs(diff)
                            
                            exec_price = entry_price if entry_price > 0 else mid_price
                            notional_value = qty * exec_price
                            
                            # [โค้ดคำนวณ Fee เดิม...]
                            if getattr(app.state, 'is_emergency_flattening', False):
                                current_trade_fee = notional_value * TAKER_FEE_RATE
                                is_maker = False
                            else:
                                current_trade_fee = notional_value * MAKER_FEE_RATE
                                is_maker = True
                                
                            BOT_STATE["est_fees"] += current_trade_fee
                            
                            # ⭐️ เพิ่มส่วนนี้: คำนวณ Realized PNL เมื่อมีการปิด/ลด โพสิชัน
                            if bot.inventory != 0 and np.sign(bot.inventory) != np.sign(diff):
                                closed_qty = min(abs(bot.inventory), qty)
                                direction = 1 if bot.inventory > 0 else -1
                                # เทียบราคาปิดกับราคาที่เปิดมา (last_entry_price)
                                trade_pnl = closed_qty * (exec_price - getattr(bot, 'last_entry_price', exec_price)) * direction
                                BOT_STATE["realized_pnl"] += trade_pnl
                                
                            # อัปเดตราคาต้นทุนล่าสุดเก็บไว้
                            bot.last_entry_price = entry_price if entry_price > 0 else mid_price
                            
                            trade = {
                                "time": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                                "side": side,
                                "price": exec_price,
                                "qty": qty,
                                "fee": current_trade_fee,
                                "type": "M" if is_maker else "T"
                            }
                            BOT_STATE["recent_trades"].insert(0, trade)
                            BOT_STATE["recent_trades"] = BOT_STATE["recent_trades"][:20] 
                            
                            BOT_STATE["volume_24h"] += qty
                            BOT_STATE["_total_fills"] += 1
                            
                        bot.inventory = actual_inv
                        
                        # ⭐️ แก้ไขการรวม PNL ให้เป๊ะตามหลักบัญชี Quant
                        current_qty = abs(actual_inv)
                        pos_cost = entry_price * current_qty
                        
                        unrealized_pnl = actual_inv * (mid_price - entry_price) if actual_inv != 0 else 0.0
                        
                        # Total Gross PNL = กำไรที่ปิดไปแล้ว (Realized) + กำไรที่กำลังถืออยู่ (Unrealized)
                        gross_pnl = BOT_STATE["realized_pnl"] + unrealized_pnl
                        
                        # Net PNL = กำไรทั้งหมด หัก ค่าธรรมเนียมทั้งหมด
                        net_pnl = gross_pnl - BOT_STATE["est_fees"]

                        if entry_price > 0: pnl_pct = (net_pnl / pos_cost) * 100
                    else:
                        if bot.inventory != 0: 
                            trade = {"time": datetime.now().strftime("%H:%M:%S.%f")[:-3], "side": "FLAT", "price": mid_price, "qty": abs(bot.inventory)}
                            BOT_STATE["recent_trades"].insert(0, trade)
                            BOT_STATE["volume_24h"] += abs(bot.inventory)
                            BOT_STATE["_total_fills"] += 1
                            
                        bot.inventory = 0.0
                except Exception:
                    pass 

                if BOT_STATE["_total_quotes"] > 0:
                    BOT_STATE["quote_win_rate"] = (BOT_STATE["_total_fills"] / BOT_STATE["_total_quotes"]) * 100

                # ⭐️ คำนวณ ROC (Return on Capital) %
                allocated_capital = bot.max_inventory * mid_price
                roc_pct = (net_pnl / allocated_capital) * 100 if allocated_capital > 0 else 0.0

                # ⭐️ คำนวณ Distance to BBA (Best Bid / Best Ask)
                # ดึงคิวแรกของกระดาน ถ้ากระดานว่างให้ใช้ mid_price แทน
                best_bid = float(bot.feature_engine.orderbook['bids'][0][0]) if bot.feature_engine.orderbook.get('bids') else mid_price
                best_ask = float(bot.feature_engine.orderbook['asks'][0][0]) if bot.feature_engine.orderbook.get('asks') else mid_price
                
                bid_dist = best_bid - my_bid
                ask_dist = my_ask - best_ask

                # ⭐️ อัปเดต BOT_STATE ให้ส่งค่าใหม่ไปด้วย
                BOT_STATE.update({
                    "symbol": bot.symbol.split('/')[0],
                    "mid_price": float(mid_price), "inventory": float(bot.inventory),
                    "max_inventory": float(bot.max_inventory), "pos_cost": float(pos_cost),
                    "pnl_pct": float(pnl_pct), "pnl": float(net_pnl), 
                    "gross_pnl": float(gross_pnl), 
                    "est_fees": float(BOT_STATE["est_fees"]), # <--- ⭐️ แก้ไขตรงนี้ครับ
                    "roc_pct": float(roc_pct), "bid_dist": float(bid_dist), "ask_dist": float(ask_dist),
                    "spread": float(my_ask - my_bid), "skew": float(action[1] * bot.max_skew_usd),
                    "bid": float(my_bid), "ask": float(my_ask),
                    "vpin": float(live_features[3]), "orderbook": bot.feature_engine.orderbook,
                    "action": [float(action[0]), float(action[1])]
                })

            except Exception as e:
                print(f"⚠️ Calculation Tick Skipped: {e}")
                
            elapsed = asyncio.get_event_loop().time() - loop_start
            await asyncio.sleep(max(0.0, 1.0 - elapsed))

        try:
            status_dir = Path("logs/live_status")
            status_dir.mkdir(parents=True, exist_ok=True)
            
            status_data = {
                "Asset": bot.symbol.split('/')[0].upper(), # ดึงชื่อเหรียญอัตโนมัติ (เช่น BTC)
                "Mode": getattr(bot, 'mode', 'UNKNOWN').upper(), # ดึงโหมด (SANDBOX/LIVE)
                "Inventory": BOT_STATE["inventory"], 
                "Realized_PnL": BOT_STATE["realized_pnl"], 
                "Unrealized_PnL": BOT_STATE["gross_pnl"] - BOT_STATE["realized_pnl"], 
                "Status": "Running 🟢",
                "Last_Update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # บันทึกไฟล์โดยใช้ชื่อเหรียญ (เช่น btc_status.json)
            filename = f"{bot.symbol.split('/')[0].lower()}_status.json"
            with open(status_dir / filename, 'w') as f: 
                json.dump(status_data, f)

        except Exception as e:
            pass # ป้องกันไม่ให้บอทหลักพังถ้าเขียนไฟล์ Error

    except asyncio.CancelledError:
        print("\n🛑 AI Trading Loop Stopped.")

    

async def main():
    # ⭐️ 1. เพิ่ม --mode เข้าไปใน ArgumentParser
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, required=True, help="เช่น btc หรือ eth")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--mode", type=str, default="sandbox", choices=["sandbox", "live"], help="sandbox หรือ live")
    args = parser.parse_args()

    pair_name = args.pair.lower()

    CONFIG_PATH = PROJECT_ROOT / "configs" / f"{pair_name}_hyperparameters.yaml"
    TRADING_ENV_PATH = PROJECT_ROOT / "configs" / f"{pair_name}_trading_env.yaml"
    MODEL_PATH = PROJECT_ROOT / "models" / f"ppo_{pair_name}_chunked_final.zip"
    
    hyper_config = load_config(str(CONFIG_PATH))
    trading_config = load_config(str(TRADING_ENV_PATH))
    
    # ⭐️ 2. ส่งค่า mode เข้าไปในคลาส ProductionMarketMaker
    bot = ProductionMarketMaker(str(MODEL_PATH), hyper_config, trading_config, mode=args.mode)
        
    # ⭐️ 3. รันเซิร์ฟเวอร์ด้วย Port ที่กำหนด
    server_config = uvicorn.Config(app, host="127.0.0.1", port=args.port, log_level="warning")
    server = uvicorn.Server(server_config)

    try:
        await asyncio.gather(
            bot.listen_binance_ws(),
            dashboard_trading_loop(bot),
            server.serve()
        )
    finally:
        print("🧹 Closing CCXT Exchange connections...")
        await bot.exchange.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 System Offline.")