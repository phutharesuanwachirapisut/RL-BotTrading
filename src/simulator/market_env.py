import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple
import collections
from .matching_engine import run_fast_matching_engine

class BinanceMarketMakerEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, data: np.ndarray, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.data = data
        self.n_steps = len(self.data)

        # ==========================================
        # 1. Config & Constraints
        # ==========================================
        self.max_inventory = config.get("max_inventory", 0.002) # สมมติเป็น BTC
        self.order_size = config.get("order_size", 0.0002)
        self.maker_fee = config.get("maker_fee", 0.0002) # 0.02%
        self.initial_balance = config.get("initial_balance", 100.0)

        # AS Parameters Limits
        self.min_spread = config.get("min_spread", 1.0)
        self.max_spread_multiplier = config.get("max_spread_multiplier", 10.0)
        self.max_skew_usd = config.get("max_skew_usd", 20.0)

        # ==========================================
        # 2. Action Space (The AS-Avatar)
        # ==========================================
        # [0]: Risk Aversion (Gamma) -> ให้ AI เลือกความกลัวตั้งแต่ -1 ถึง 1
        # [1]: Micro-Skew -> ให้ AI บิดราคาเสนอซื้อขายเล็กน้อย
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # ==========================================
        # 3. Observation Space (Normalized Features)
        # ==========================================
        self.stack_size = config.get("frame_stack", 10) # เพิ่ม Stack ให้เห็น Temporal Pattern
        self.frames = collections.deque(maxlen=self.stack_size)

        # Features: [vol_norm, tfi_norm, rsi_norm, macd_norm, inv_ratio, time_ratio] = 6 ตัว
        self.n_features_per_frame = 6
        total_features = self.stack_size * self.n_features_per_frame
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_features,), dtype=np.float32)

        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self.current_step = 0
        
        # Portfolio Management
        self.inventory = 0.0
        self.cash = self.initial_balance
        self.avg_entry_price = 0.0
        
        # Trackers
        self.steps_holding = 0
        self.realized_pnl = 0.0
        self.total_fees_paid = 0.0

        self.frames.clear()
        first_frame = self._get_current_frame()
        for _ in range(self.stack_size):
            self.frames.append(first_frame)

        return self._get_stacked_observation(), self._get_info()

    def _get_current_frame(self) -> np.ndarray:
        # ดึง Feature ที่ผ่านการ Normalize (สมมติว่าอยู่ index 2 ถึง 5 ใน data)
        # Index 0 = Price, Index 1 = Quantity, Index 2+ = Features
        market_obs = self.data[self.current_step, 2:6] 
        inv_ratio = self.inventory / self.max_inventory
        time_ratio = self.current_step / self.n_steps
        return np.append(market_obs, [inv_ratio, time_ratio]).astype(np.float32)

    def _get_stacked_observation(self) -> np.ndarray:
        return np.concatenate(list(self.frames))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self.current_step >= self.n_steps - 1:
            return self._get_stacked_observation(), 0.0, True, False, self._get_info()

        # ตึงข้อมูลดิบสำหรับคำนวณบัญชี (Price, Volume)
        mid_price = self.data[self.current_step, 0]
        volume = self.data[self.current_step, 1]
        next_mid = self.data[self.current_step + 1, 0]
        
        # ดึง Volatility (สมมติอยู่ index 2 และคืนค่ากลับมาเป็นสเกลบวก)
        volatility_norm = abs(self.data[self.current_step, 2]) 

        # ==========================================
        # 🧠 1. AS-Avatar (แปลง Action เป็น Bid/Ask)
        # ==========================================
        gamma = 0.01 + ((action[0] + 1.0) / 2.0) * 0.19
        ai_skew = action[1] * self.max_skew_usd

        # คำนวณ Reservation Price (จุดศูนย์กลางใหม่ที่เบ้หนี Inventory)
        inv_ratio = self.inventory / self.max_inventory
        reservation_price = mid_price - (inv_ratio * gamma * self.max_skew_usd)

        # คำนวณ Optimal Spread (ผันแปรตามความกลัวและความผันผวน)
        base_half_spread = (self.min_spread + (volatility_norm * self.max_spread_multiplier)) / 2.0
        optimal_half_spread = base_half_spread * (1.0 + gamma)

        my_bid = reservation_price - optimal_half_spread + ai_skew
        my_ask = reservation_price + optimal_half_spread + ai_skew

        # ==========================================
        # 🛡️ 2. No-Trade Regime (Red Queen's Trap Defense)
        # ==========================================
        # หาก Spread แคบกว่าค่าธรรมเนียมที่ต้องจ่าย 2 ฝั่ง (ซื้อเข้า+ขายออก) ให้ดึงออเดอร์หนี
        min_profitable_spread = (mid_price * self.maker_fee) * 1.1 # เผื่อกำไร 0.5
        calculated_spread = my_ask - my_bid
        
        if calculated_spread < min_profitable_spread:
            my_bid = 0.0
            my_ask = float('inf') # ระงับการทำ Market Making ชั่วคราว

        # Circuit Breakers ปกติ
        if self.inventory >= self.max_inventory: my_bid = 0.0
        if self.inventory <= -self.max_inventory: my_ask = float('inf')

        # ==========================================
        # ⚙️ 3. Execution & Portfolio Update
        # ==========================================
        sim_trades = np.array([[mid_price, volume * 1.5, 1], [next_mid, volume * 1.5, -1]])
        new_inv, cash_flow, _, _ = run_fast_matching_engine(
            my_bid, my_ask, self.order_size, sim_trades, self.inventory, self.max_inventory, self.maker_fee
        )

        step_realized_pnl = 0.0
        trade_volume = abs(new_inv - self.inventory)
        
        if trade_volume > 0:
            # คำนวณค่าธรรมเนียมของ Step นี้
            exec_price = mid_price # ประมาณการราคา execution
            step_fee = trade_volume * exec_price * self.maker_fee
            self.total_fees_paid += step_fee
            
            # การอัปเดต Avg Entry Price และคำนวณ Realized PnL (Round-trip completion)
            if self.inventory == 0:
                self.avg_entry_price = exec_price
            elif np.sign(self.inventory) == np.sign(new_inv - self.inventory):
                # ถัวเฉลี่ยต้นทุนถ้าซื้อเพิ่มฝั่งเดิม
                total_cost = (abs(self.inventory) * self.avg_entry_price) + (trade_volume * exec_price)
                self.avg_entry_price = total_cost / abs(new_inv)
            else:
                # ถ้าทอนของออก (ปิด position บางส่วนหรือทั้งหมด) -> รับรู้กำไร/ขาดทุน
                direction = 1 if self.inventory > 0 else -1
                step_realized_pnl = trade_volume * (exec_price - self.avg_entry_price) * direction
                self.realized_pnl += step_realized_pnl
                
                # รีเซ็ตต้นทุนถ้าปิดหมด
                if new_inv == 0: self.avg_entry_price = 0.0

        self.inventory = new_inv
        self.cash += cash_flow

        # ==========================================
        # 📈 4. Round-trip & Asymmetric Reward
        # ==========================================
        reward = 0.0
        
        # A. Trade Completion Reward (ให้รางวัล/ทำโทษ เฉพาะที่รับรู้ PnL แล้ว)
        if step_realized_pnl > 0:
            reward += step_realized_pnl * 50.0  # ได้กำไร ชมเชย
        elif step_realized_pnl < 0:
            reward += step_realized_pnl * 150.0 # ขาดทุน ทำโทษหนักกว่า (Asymmetric)

        # B. Holding Penalty (ลงโทษถ้าถือของค้างนานเกินไป ป้องกันการเดาทิศทางลากยาว)
        if self.inventory != 0:
            self.steps_holding += 1
            holding_penalty = (self.steps_holding * 0.001) * abs(self.inventory / self.max_inventory)
            reward -= holding_penalty
        else:
            self.steps_holding = 0 # รีเซ็ตเมื่อพอร์ตว่าง
            
        # C. Unrealized Loss Penalty (ทำโทษถ้าโดนลากจนพอร์ตแดง แต่ไม่ให้รางวัลถ้าเขียว)
        if self.inventory != 0:
            unrealized_pnl = abs(self.inventory) * (next_mid - self.avg_entry_price) * (1 if self.inventory > 0 else -1)
            if unrealized_pnl < 0:
                reward += unrealized_pnl * 10.0 # ตอดคะแนนเบาๆ เพื่อให้รีบตัดขาดทุน

        self.current_step += 1
        portfolio_value = self.cash + (self.inventory * next_mid)
        terminated = portfolio_value < (self.initial_balance * 0.5)
        
        if terminated: reward -= 500.0

        if not terminated:
            self.frames.append(self._get_current_frame())

        return self._get_stacked_observation(), float(reward), terminated, False, self._get_info()

    def _get_info(self) -> Dict:
        mid_price = self.data[self.current_step, 0]
        unrealized = 0.0
        if self.inventory != 0:
            unrealized = abs(self.inventory) * (mid_price - self.avg_entry_price) * (1 if self.inventory > 0 else -1)
        
        mtm_balance = self.cash + (self.inventory * mid_price)
        return {
            "inventory": self.inventory,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "fees_paid": self.total_fees_paid,
            "portfolio_value": mtm_balance,
            "pnl": mtm_balance - self.initial_balance
        }