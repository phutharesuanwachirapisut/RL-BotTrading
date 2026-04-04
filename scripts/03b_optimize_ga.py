import os
import sys
import numpy as np
import polars as pl
import yaml
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.simulator.market_env import BinanceMarketMakerEnv

class GeneticAlgorithmOptimizer:
    def __init__(self, data_chunk: np.ndarray, base_config: dict, pop_size=20, generations=10):
        self.data = data_chunk
        self.base_config = base_config
        self.pop_size = pop_size
        self.generations = generations
        self.population = self._initialize_population()

    def _initialize_population(self):
        """
        สร้างประชากรเริ่มต้น (สุ่มค่ายีนส์)
        Gene 0: custom_parameter_1 [-3.0 to 3.0]
        Gene 1: max_spread_multiplier [5.0 - 50.0]
        Gene 2: max_skew_usd [5.0 - 50.0]
        """
        return np.column_stack((
            np.random.uniform(-3.0, 3.0, self.pop_size),
            np.random.uniform(5.0, 50.0, self.pop_size),
            np.random.uniform(5.0, 50.0, self.pop_size)
        ))

    def _evaluate_fitness(self, individual):
        """วัดความแข็งแกร่งด้วย Sharpe Ratio (ผลตอบแทนเทียบกับความเสี่ยง)"""
        # อัปเดต Config ด้วยยีนส์ของตัวนี้
        env_config = self.base_config.copy()
        env_config["eta"] = float(individual[0])
        env_config["max_spread_multiplier"] = float(individual[1])
        env_config["max_skew_usd"] = float(individual[2])

        env = BinanceMarketMakerEnv(self.data, env_config)
        obs, _ = env.reset()
        done = False
        
        step_returns = []
        last_pnl = 0.0

        while not done:
            # สุ่ม Action เบาๆ เพื่อจำลองว่า AI เริ่มออกแรงปรับ Skew/Spread
            random_action = np.random.uniform(-0.5, 0.5, size=(2,))
            obs, reward, terminated, truncated, info = env.step(random_action)
            done = terminated or truncated
            
            current_pnl = info['pnl']
            step_returns.append(current_pnl - last_pnl)
            last_pnl = current_pnl

        returns_arr = np.array(step_returns)
        
        # ⭐️ เพิ่มการตรวจสอบจำนวนก้าวที่มีการเทรดจริงๆ (กำไร/ขาดทุนไม่เป็นศูนย์)
        trades_executed = np.count_nonzero(returns_arr)
        
        if len(returns_arr) < 2 or trades_executed == 0:
            # ถ้าไม่เกิดการแมตช์ออเดอร์เลย (returns เป็น 0 หมด) ให้คะแนนติดลบ
            # print(f"  [Debug] ยีนส์ {individual} ไม่มีการเทรดเลย (Trades: 0) -> ลงโทษ -10.0")
            return -10.0 
            
        # คำนวณ Sharpe Ratio
        sharpe = (np.mean(returns_arr) / (np.std(returns_arr) + 1e-8)) * np.sqrt(len(returns_arr))
        
        # ป้องกันค่า Sharpe ระเบิดกรณี Std ต่ำมากแตะขีดจำกัด
        return float(np.clip(sharpe, -100.0, 100.0))

    def optimize(self):
        print(f"🧬 เริ่มกระบวนการวิวัฒนาการ: {self.generations} รุ่น, ประชากรรุ่นละ {self.pop_size} ตัว")
        best_overall = None
        best_fitness = -np.inf

        for gen in range(self.generations):
            fitness_scores = np.array([self._evaluate_fitness(ind) for ind in self.population])
            
            # หาตัวที่ดีที่สุดของรุ่น
            best_idx = np.argmax(fitness_scores)
            if fitness_scores[best_idx] > best_fitness:
                best_fitness = fitness_scores[best_idx]
                best_overall = self.population[best_idx]
                
            print(f"  -> Generation {gen+1:02d} | Best Sharpe: {np.max(fitness_scores):.4f} | Avg Sharpe: {np.mean(fitness_scores):.4f}")

            # 1. การคัดเลือก (Selection): เลือกครึ่งนึงที่แข็งแกร่งรอดไปเป็นพ่อแม่
            parents = self.population[np.argsort(fitness_scores)[-self.pop_size//2:]]
            
            # 2. การผสมพันธุ์ (Crossover): สลับยีนส์ระหว่างพ่อแม่
            offspring = []
            while len(offspring) < self.pop_size - len(parents):
                p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]
                split = np.random.randint(1, 3)
                child = np.concatenate((p1[:split], p2[split:]))
                offspring.append(child)
                
            # 3. การกลายพันธุ์ (Mutation): ป้องกันสายเลือดชิด 
            offspring = np.array(offspring)
            mutation_mask = np.random.rand(*offspring.shape) < 0.2 # โอกาสกลายพันธุ์ 20%
            noise = np.random.normal(0, 2.0, offspring.shape)
            offspring = np.where(mutation_mask, offspring + noise, offspring)
            
            # สร้างประชากรรุ่นถัดไป (พ่อแม่ + ลูก)
            self.population = np.vstack((parents, offspring))
            
            # ป้องกันยีนส์ติดลบ
            self.population = np.clip(self.population, [-3.0, 5.0, 5.0], [3.0, 50.0, 50.0])

        print(f"\n🏆 ค้นพบยีนส์ที่แข็งแกร่งที่สุด! (Sharpe Ratio: {best_fitness:.4f})")
        return {
            "eta": float(best_overall[0]),
            "max_spread_multiplier": float(best_overall[1]),
            "max_skew_usd": float(best_overall[2])
        }

def main():
    # 1. โหลดข้อมูลเล็กๆ มา 1 Chunk เพื่อใช้ประเมิน GA (ประหยัดเวลา)
    pool_dir = PROJECT_ROOT / "data" / "processed" / "pooled_chunks"
    chunk_files = glob.glob(os.path.join(pool_dir, "*.parquet"))
    
    if not chunk_files:
        print("❌ ไม่พบข้อมูล Pooled Chunks (กรุณารัน 01b_prepare_chunks.py ก่อน)")
        sys.exit(1)
        
    print(f"📥 กำลังโหลดข้อมูลทดสอบสำหรับ GA จาก {os.path.basename(chunk_files[0])}...")
    df = pl.read_parquet(chunk_files[0])
    # ใช้แค่ 10,000 แถวแรกเพื่อความเร็วในการรัน GA
    eval_data = df.head(10000).select(["price", "quantity", "returns_pct_norm", "volatility_norm", "tfi_norm", "rsi_norm", "macd_raw_norm"]).to_numpy().astype(np.float32)

    base_config = {
        "max_inventory": 0.002, "order_size": 0.0002, "maker_fee": 0.0002, "min_spread": 1.0, "frame_stack": 10, "initial_balance": 100.0
    }

    # 2. รัน Genetic Algorithm
    optimizer = GeneticAlgorithmOptimizer(eval_data, base_config, pop_size=20, generations=15)
    best_params = optimizer.optimize()

    # 3. เซฟ Baseline ที่ดีที่สุดลงไฟล์ เพื่อให้ RL เอาไปใช้ต่อ
    save_path = PROJECT_ROOT / "configs" / "ga_optimized_baseline.yaml"
    with open(save_path, 'w') as f:
        yaml.dump({"ga_optimized_env": best_params}, f)
        
    print(f"💾 บันทึก Baseline Parameter ลงใน {save_path} เรียบร้อยแล้ว!")

if __name__ == "__main__":
    main()