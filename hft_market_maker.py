import os
import sys
import time
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# ถอยหลังไปที่โฟลเดอร์หลักของโปรเจกต์ (ปรับระดับให้ตรงกับเครื่องของคุณ)
PROJECT_ROOT = Path(__file__).resolve().parent

# โค้ดสีสำหรับ Terminal
C_HEADER = '\033[95m'
C_BLUE = '\033[94m'
C_GREEN = '\033[92m'
C_YELLOW = '\033[93m'
C_RED = '\033[91m'
C_CYAN = '\033[96m'
C_RESET = '\033[0m'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(text):
    print(f"\n{C_CYAN}{'='*60}")
    print(f" {text}")
    print(f"{'='*60}{C_RESET}\n")

def get_user_input(prompt, valid_options=None, cast_type=str):
    while True:
        try:
            user_input = input(f"{C_YELLOW}{prompt}:{C_RESET} ").strip().lower()
            if not user_input:
                continue
            value = cast_type(user_input)
            if valid_options and value not in valid_options:
                print(f"{C_RED}[!] Please choose from: {valid_options}{C_RESET}")
                continue
            return value
        except ValueError:
            print(f"{C_RED}[!] Invalid format. Please try again.{C_RESET}")

def get_date_input(regime_name, timeframe_days):
    while True:
        prompt = f"{C_YELLOW}Start Date for [{regime_name}] (Format: YYYY-MM-DD):{C_RESET} "
        user_input = input(prompt).strip()
        try:
            start_date = datetime.strptime(user_input, "%Y-%m-%d")
            end_date = start_date + timedelta(days=timeframe_days)
            return start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
        except ValueError:
            print(f"{C_RED}[!] Invalid date format. Must be YYYY-MM-DD.{C_RESET}")

def cleanup_raw_data(asset):
    print(f"\n{C_BLUE}[Cleanup]{C_RESET} Removing raw data files for {asset.upper()}...")
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    deleted_count = 0
    for data_dir in [raw_dir, processed_dir]:
        if data_dir.exists():
            for file_path in data_dir.glob(f"*{asset}*"):
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
    print(f"{C_GREEN}[Success]{C_RESET} Cleaned up {deleted_count} files.\n")

# ==========================================
# ⭐️ เพิ่มระบบ Backtest & Save Results
# ==========================================
def run_backtest(asset):
    print_header(f"BACKTESTING MODEL: {asset.upper()}")
    
    # สร้างโฟลเดอร์สำหรับเก็บผลลัพธ์ Backtest
    results_dir = PROJECT_ROOT / "results" / "backtests"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    step_name = f"Running Out-of-Sample Backtest for {asset.upper()}"
    print(f"{C_BLUE}[Running]{C_RESET} {step_name}...")
    
    # ⭐️ ชี้เป้าไปที่ 03_run_backtest.py และส่ง --save_dir ไปด้วย
    cmd = [
        "python3", str(PROJECT_ROOT / "scripts" / "03_run_backtest.py"), 
        "--pair", asset,
        "--save_dir", str(results_dir) 
    ]
    
    try:
        # ⭐️ เปิดระบบให้ทำงานจริงๆ (เอาของ Mock ทิ้งไป)
        subprocess.run(cmd, check=True)
        print(f"{C_GREEN}[Success]{C_RESET} {step_name}")
        print(f"{C_GREEN}[Saved]{C_RESET} Backtest results saved to: {results_dir}/backtest_{asset}.png")
        time.sleep(1)
    except subprocess.CalledProcessError as e:
        print(f"{C_RED}[Failed]{C_RESET} Backtest script crashed (Exit code: {e.returncode})")
        sys.exit(1)
    except Exception as e:
        print(f"{C_RED}[Error]{C_RESET} System error during backtest: {e}")
        sys.exit(1)

def run_live_dashboard(asset):
    print(f"\n{C_HEADER}--- DEPLOYMENT MODE ---{C_RESET}")
    print("1. Sandbox (Demo/Testnet)")
    print(f"2. {C_RED}LIVE (Real Money){C_RESET}")
    
    mode_choice = get_user_input("Select Deployment Mode (1 or 2)", valid_options=['1', '2'])
    mode = "sandbox" if mode_choice == '1' else "live"
    
    port_input = input(f"{C_YELLOW}Enter Port for Dashboard (Press Enter to use 8125):{C_RESET} ").strip()
    port = port_input if port_input else "8125"

    mode_text = f"{C_GREEN}SANDBOX{C_RESET}" if mode == "sandbox" else f"{C_RED}LIVE REAL MONEY{C_RESET}"
    print(f"\n{C_BLUE}[Starting]{C_RESET} {mode_text} Dashboard for {asset.upper()} on port {port}...")
    
    # ส่งพารามิเตอร์ --mode เข้าไปด้วย
    deploy_cmd = [
        "python3", str(PROJECT_ROOT / "scripts" / "07_live_real_money.py"), 
        "--pair", asset, 
        "--port", port,
        "--mode", mode
    ]
    
    try:
        subprocess.run(deploy_cmd)
    except KeyboardInterrupt:
        print(f"\n{C_RED}[Stopped]{C_RESET} Dashboard system offline.")
    sys.exit(0)

def main():
    clear_screen()
    print_header("HFT QUANT PIPELINE MANAGER")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    
    model_files = list(models_dir.glob("ppo_*.zip"))
    available_models = []
    for f in model_files:
        parts = f.stem.split('_')
        if len(parts) >= 2:
            available_models.append(parts[1].lower())
            
    available_models = sorted(list(set(available_models)))
    
    print(f"{C_HEADER}--- AVAILABLE MODELS ---{C_RESET}")
    if available_models:
        for idx, m in enumerate(available_models, 1):
            print(f"  {idx}. {C_GREEN}{m.upper()}{C_RESET}")
    else:
        print(f"  {C_YELLOW}No compiled models found in 'models/' directory.{C_RESET}")
        
    print(f"\n{C_HEADER}--- MAIN MENU ---{C_RESET}")
    print("1. Deploy an Existing Model (Live Dashboard)")
    print("2. Train a New Model (Fetch -> Train -> Backtest -> Deploy)")
    print("3. Run Backtest on Existing Model") # ⭐️ เพิ่มเมนูที่ 3
    
    choice = get_user_input("Select Action (1, 2, or 3)", valid_options=['1', '2', '3'])
    
    # --- PATH 1: Deploy Existing ---
    if choice == '1':
        if not available_models:
            print(f"{C_RED}[!] You have no models to deploy.{C_RESET}")
            sys.exit(1)
        valid_selections = [str(i) for i in range(1, len(available_models) + 1)] + available_models
        user_sel = get_user_input(f"Select model to deploy", valid_options=valid_selections)
        deploy_target = available_models[int(user_sel) - 1] if user_sel.isdigit() else user_sel
        run_live_dashboard(deploy_target)

    # --- PATH 3: Backtest Only ---
    elif choice == '3':
        if not available_models:
            print(f"{C_RED}[!] You have no models to backtest.{C_RESET}")
            sys.exit(1)
        valid_selections = [str(i) for i in range(1, len(available_models) + 1)] + available_models
        user_sel = get_user_input(f"Select model to backtest", valid_options=valid_selections)
        backtest_target = available_models[int(user_sel) - 1] if user_sel.isdigit() else user_sel
        run_backtest(backtest_target)
        sys.exit(0)

    # --- PATH 2: Train New Model ---
    print_header("NEW MODEL PIPELINE")
    asset = get_user_input("Select Asset (e.g. btc, eth, bnb)", cast_type=str)
    if asset in available_models:
        print(f"{C_YELLOW}[!] Notice: A model for {asset.upper()} already exists. It will be overwritten.{C_RESET}")
        
    timeframe = get_user_input("Timeframe in days (15, 30, 45, 60)", valid_options=[15, 30, 45, 60], cast_type=int)
    
    print(f"\n{C_HEADER}--- 2. MARKET REGIMES CONFIGURATION ---{C_RESET}")
    regimes_config = {}
    for regime in ["Mean-Reverting", "Trending", "High Volatility"]:
        print(f"{C_BLUE}>> Regime: {regime}{C_RESET}")
        start, end = get_date_input(regime, timeframe)
        regimes_config[regime.lower().replace(" ", "_").replace("-", "_")] = {'start': start, 'end': end}
        print(f"   End Date: {C_GREEN}{end}{C_RESET}\n")
    
    confirm = get_user_input("Confirm setup and start training process? (y/n)", valid_options=['y', 'n'])
    if confirm == 'n': sys.exit(0)

    print_header("DATA PREPARATION & MODEL TRAINING")
    for regime, dates in regimes_config.items():
        step_name = f"Downloading {regime.upper()} data ({dates['start']} to {dates['end']})"
        # subprocess.run(["python3", "scripts/download_data.py", ...])
        print(f"{C_GREEN}[Mock Success]{C_RESET} {step_name}")
        time.sleep(0.5)

    step_name = f"Training PPO Model for {asset.upper()}"
    # subprocess.run(["python3", "scripts/train_model.py", "--pair", asset])
    print(f"{C_GREEN}[Mock Success]{C_RESET} {step_name}.")
    time.sleep(0.5)

    cleanup_raw_data(asset)

    # ⭐️ 1.5 BACKTESTING PHASE (แทรกก่อนขึ้น Live)
    run_backtest(asset)

    print_header("TRAINING & BACKTEST COMPLETE")
    start_live = get_user_input(f"Do you want to start the Live Dashboard for {asset.upper()} now? (y/n)", valid_options=['y', 'n'])
    if start_live == 'y':
        run_live_dashboard(asset)
    else:
        print(f"{C_GREEN}[Done]{C_RESET} Exiting pipeline.")

if __name__ == "__main__":
    main()