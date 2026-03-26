import os
import asyncio
import json
import traceback
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import ccxt.async_support as ccxt
import numpy as np
from pathlib import Path
import sys
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

folder_path = PROJECT_ROOT / "scripts"   # class path
sys.path.append(folder_path)
from scripts.paper_trader import ProductionMarketMaker, load_config

os.chdir(PROJECT_ROOT / "scripts")
app = FastAPI()

# ⭐️ FIX: เพิ่มคีย์ gross_pnl, est_fees และ max_inventory ให้ครบ
BOT_STATE = {
    "mid_price": 0.0,
    "inventory": 0.0,
    "max_inventory": 0.0, 
    "pos_cost": 0.0,
    "pnl_pct": 0.0,
    "pnl": 0.0,
    "gross_pnl": 0.0,
    "est_fees": 0.0,
    "spread": 0.0,
    "skew": 0.0,
    "volatility": 0.0,
    "bid": 0.0,
    "ask": 0.0,
    "action": [0.0, 0.0],
    "tfi": 0.0,
    "vpin": 0.0,
    "orderbook": {"bids": [], "asks": []}
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
            await websocket.send_text(json.dumps(BOT_STATE))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def dashboard_trading_loop(bot: ProductionMarketMaker):
    print("🟢 Starting Dashboard Trading Loop...")
    
    await bot.exchange.load_markets()
    
    while bot.feature_engine.mid_price == 0:
        await asyncio.sleep(0.1)
        
    try:
        while True:
            loop_start = asyncio.get_event_loop().time()
            
            try:
                # ⭐️ FIX 1: กู้คืนสมอง AI และระบบยิงออเดอร์กลับมา
                mid_price = bot.feature_engine.mid_price
                live_features = bot.feature_engine.get_live_observation(bot.inventory, bot.max_inventory)
                
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
                
                # สั่งยิงออเดอร์
                asyncio.create_task(bot.execute_orders(my_bid, my_ask))

                # ⭐️ FIX 2: คำนวณค่าธรรมเนียมและ PnL
                pos_cost = 0.0
                pnl_pct = 0.0
                entry_price = 0.0
                gross_pnl = 0.0
                net_pnl = 0.0
                est_fees = 0.0
                MAKER_FEE_RATE = 0.0002 # Maker Fee 0.02%
                
                try:
                    positions = await bot.exchange.fetch_positions([bot.symbol])
                    if positions and float(positions[0]['info']['positionAmt']) != 0:
                        pos = positions[0]
                        entry_price = float(pos['entryPrice'])
                        actual_inv = float(pos['info']['positionAmt'])
                        current_qty = abs(actual_inv)
                        
                        pos_cost = entry_price * current_qty
                        gross_pnl = actual_inv * (mid_price - entry_price)
                        
                        open_fee = pos_cost * MAKER_FEE_RATE
                        close_fee = (current_qty * mid_price) * MAKER_FEE_RATE
                        est_fees = open_fee + close_fee
                        
                        net_pnl = gross_pnl - est_fees
                        
                        if entry_price > 0:
                            pnl_pct = (net_pnl / pos_cost) * 100
                            
                        bot.inventory = actual_inv
                    else:
                        bot.inventory = 0.0
                except Exception as e:
                    pass # ถ้า API ดึงไม่ได้ให้ข้ามไป

                # ⭐️ FIX 3: จัดการ Indent ของ BOT_STATE ให้อยู่ถูกที่
                BOT_STATE["mid_price"] = float(mid_price)
                BOT_STATE["inventory"] = float(bot.inventory)
                BOT_STATE["max_inventory"] = float(bot.max_inventory)
                BOT_STATE["pos_cost"] = float(pos_cost) 
                BOT_STATE["pnl_pct"] = float(pnl_pct)   
                BOT_STATE["pnl"] = float(net_pnl) 
                BOT_STATE["gross_pnl"] = float(gross_pnl) 
                BOT_STATE["est_fees"] = float(est_fees)   
                BOT_STATE["spread"] = float(my_ask - my_bid)
                BOT_STATE["skew"] = float(action[1] * bot.max_skew_usd) 
                BOT_STATE["volatility"] = float(volatility)
                BOT_STATE["bid"] = float(my_bid)
                BOT_STATE["ask"] = float(my_ask)
                BOT_STATE["action"] = [float(action[0]), float(action[1])]
                BOT_STATE["tfi"] = float(live_features[2])
                BOT_STATE["vpin"] = float(live_features[3])
                BOT_STATE["orderbook"] = bot.feature_engine.orderbook
                
                print(f"[Quoting] Spread: {(my_ask-my_bid):.2f} | Bid: {my_bid:.2f} | Ask: {my_ask:.2f} | Inv: {bot.inventory:.4f} | PnL: {net_pnl:.2f}")
            
            except Exception as e:
                print(f"⚠️ Calculation Tick Skipped: {e}")
                traceback.print_exc()
                
            elapsed = asyncio.get_event_loop().time() - loop_start
            await asyncio.sleep(max(0.0, 1.0 - elapsed))
            
    except asyncio.CancelledError:
        print("\n🛑 AI Trading Loop Stopped.")

async def main():
    CONFIG_PATH = PROJECT_ROOT / "configs" / "hyperparameters.yaml"
    MODEL_PATH = PROJECT_ROOT / "models" / "ppo_hft_chunked_final.zip"
    config = load_config(CONFIG_PATH)
    bot = ProductionMarketMaker(MODEL_PATH, config)

    server_config = uvicorn.Config(app, host="127.0.0.1", port=8123, log_level="warning")
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