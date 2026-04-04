import os
import sys
import time
import subprocess
import yaml
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parent

# ==========================================
# 🎨 COLOR PALETTE (DARK GREEN THEME)
# ==========================================
C_DARK_GREEN = '\033[38;5;28m'   
C_GREEN = '\033[38;5;34m'        
C_LIME = '\033[38;5;46m'         
C_GRAY = '\033[38;5;240m'        
C_WHITE = '\033[97m'
C_YELLOW = '\033[38;5;214m'      
C_RED = '\033[38;5;160m'         
C_RESET = '\033[0m'

# ⭐️ เติมบรรทัดนี้ลงไปตรงนี้ครับ! ⭐️
DEFAULT_ASSETS = ["btc", "eth", "bnb", "xrp", "ltc", "ada"]

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_hft_logo():
    """พิมพ์โลโก้ HFT แบบ ASCII Art ด้วยโทนสีเขียว"""
    logo = f"""
{C_DARK_GREEN}██╗  ██╗{C_GREEN}███████╗{C_LIME}████████╗
{C_DARK_GREEN}██║  ██║{C_GREEN}██╔════╝{C_LIME}╚══██╔══╝
{C_DARK_GREEN}███████║{C_GREEN}█████╗  {C_LIME}   ██║   
{C_DARK_GREEN}██╔══██║{C_GREEN}██╔══╝  {C_LIME}   ██║   
{C_DARK_GREEN}██║  ██║{C_GREEN}██║     {C_LIME}   ██║   
{C_DARK_GREEN}╚═╝  ╚═╝{C_GREEN}╚═╝     {C_LIME}   ╚═╝   {C_RESET}
    """
    print(logo)
    print(f"{C_GRAY}======================================================{C_RESET}")
    print(f"{C_WHITE} QUANTITATIVE DEPLOYMENT & EVALUATION MANAGER (v3.0){C_RESET}")
    print(f"{C_GRAY}======================================================{C_RESET}\n")

def get_user_input(prompt, valid_options=None, cast_type=str):
    while True:
        try:
            user_input = input(f"{C_GREEN}▶ {C_WHITE}{prompt}: {C_RESET}").strip().lower()
            if not user_input: return ""
            value = cast_type(user_input)
            if valid_options and value not in valid_options:
                print(f"{C_RED}[!] Please choose from: {valid_options}{C_RESET}")
                continue
            return value
        except ValueError:
            print(f"{C_RED}[!] Invalid format.{C_RESET}")

def select_assets_menu(title="SELECT ASSETS"):
    """ระบบเลือกเหรียญแบบ Multiple Choice (Numbered List)"""
    print(f"\n{C_LIME}>>> {title}{C_RESET}")
    print(f"{C_GRAY}Enter numbers separated by spaces (e.g., 1 2 4), or 'all', or 'other'{C_RESET}")
    
    for i, asset in enumerate(DEFAULT_ASSETS, 1):
        print(f"  {C_GRAY}[{i}]{C_RESET} {C_WHITE}{asset.upper()}{C_RESET}")
    
    choice = input(f"\n{C_GREEN}▶ {C_WHITE}Your selection (Default: All): {C_RESET}").strip().lower()
    
    # 1. กรณีไม่เลือก (ค่าว่าง) -> ใช้ Default ทั้งหมด
    if not choice or choice == 'all':
        return DEFAULT_ASSETS

    # 2. กรณีเลือก 'other' -> ให้กรอกเอง
    if choice == 'other':
        custom = input(f"{C_GREEN}▶ {C_WHITE}Enter custom assets (comma separated): {C_RESET}").strip().lower()
        return [a.strip() for a in custom.split(',')]

    # 3. กรณีเลือกตามหมายเลข
    try:
        indices = [int(i) - 1 for i in choice.split() if i.isdigit()]
        selected = [DEFAULT_ASSETS[i] for i in indices if 0 <= i < len(DEFAULT_ASSETS)]
        return selected if selected else DEFAULT_ASSETS
    except Exception:
        return DEFAULT_ASSETS

def load_regime_config(asset):
    config_path = PROJECT_ROOT / f"downloadData_{asset}.yaml"
    if not config_path.exists(): config_path = PROJECT_ROOT / "configs" / f"downloadData_{asset}.yaml"
    return yaml.safe_load(open(config_path)) if config_path.exists() else None

def auto_create_config(asset, template_asset="btc"):
    template_path = PROJECT_ROOT / "configs" / f"downloadData_{template_asset}.yaml"
    new_config_path = PROJECT_ROOT / "configs" / f"downloadData_{asset}.yaml"
    if not template_path.exists(): return False
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
        new_content = template_content.replace(f'asset: "{template_asset.upper()}USDT"', f'asset: "{asset.upper()}USDT"')
        new_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(new_config_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        return True
    except Exception:
        return False

def cleanup_raw_data(asset):
    print(f"{C_GRAY}[Cleanup] Removing raw CSV for {asset.upper()}...{C_RESET}")
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if raw_dir.exists():
        for file_path in raw_dir.glob(f"*{asset.upper()}*"):
            if file_path.is_file() and file_path.suffix == '.csv':
                file_path.unlink()

def run_backtest(asset):
    print(f"\n{C_LIME}>>> BACKTESTING OUT-OF-SAMPLE: {asset.upper()}{C_RESET}")
    results_dir = PROJECT_ROOT / "results" / "backtests"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = ["python3", str(PROJECT_ROOT / "scripts" / "03_run_backtest.py"), "--pair", asset, "--save_dir", str(results_dir)]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"{C_RED}[Failed] Backtest script crashed.{C_RESET}")
    
    input(f"\n{C_GRAY}Press Enter to continue...{C_RESET}")

def run_live_dashboard(selected_assets):
    """รัน Dashboard สำหรับเหรียญที่เลือก พร้อมระบบ Fix Port"""
    print(f"\n{C_LIME}>>> DEPLOYMENT MODE{C_RESET}")
    print(f"  {C_WHITE}1. Sandbox (Demo/Testnet){C_RESET}")
    print(f"  {C_RED}2. LIVE (Real Money){C_RESET}")
    mode_choice = get_user_input("Select Mode (1 or 2)", valid_options=['1', '2'])
    mode = "sandbox" if mode_choice == '1' else "live"

    # สร้าง Dictionary สำหรับจับคู่ เหรียญ -> พอร์ต คงที่ (เริ่ม 8001)
    port_mapping = {}
    for idx, asset in enumerate(DEFAULT_ASSETS):
        port_mapping[asset] = 8001 + idx
    
    # สำหรับเหรียญแปลกๆ ที่ไม่อยู่ในลิสต์หลัก (เริ่ม 8090)
    custom_port_start = 8090

    print(f"\n{C_LIME}[Deploying Portfolio] Initializing {len(selected_assets)} assets...{C_RESET}")
    
    # 1. รัน Bot แต่ละตัว
    for asset in selected_assets:
        if asset in port_mapping:
            port = str(port_mapping[asset])
        else:
            port = str(custom_port_start)
            custom_port_start += 1
            
        print(f"  {C_GREEN}✓{C_RESET} {asset.upper():<5} -> http://localhost:{port}")
        subprocess.Popen(["python3", "scripts/07_live_real_money.py", "--pair", asset, "--port", port, "--mode", mode])
    
    # 2. รัน Master Portfolio Dashboard (ล็อกพอร์ต 8000)
    # เราสามารถสั่งให้ Streamlit รันบนพอร์ตที่กำหนดได้ด้วย --server.port
    print(f"\n{C_LIME}>>> Opening Master Portfolio Dashboard (http://localhost:8000)...{C_RESET}")
    subprocess.Popen([
        "python3", "-m", "streamlit", "run", "scripts/08_portfolio_dashboard.py", 
        "--server.port", "8000"
    ])

    print(f"\n{C_YELLOW}💡 Tip: You can open your browser and go to:{C_RESET}")
    print(f"   {C_WHITE}- Master Dashboard: {C_LIME}http://localhost:8000{C_RESET}")
    for asset in selected_assets:
        p = port_mapping.get(asset, "custom")
        if p != "custom":
            print(f"   {C_WHITE}- {asset.upper():<14}: {C_LIME}http://localhost:{p}{C_RESET}")

    input(f"\n{C_GRAY}Deployment started. Press Enter to return to main menu...{C_RESET}")
    
def train_universal_model():
    # 1. ใช้ระบบเลือกเมนูใหม่ที่คุณออกแบบไว้ (ดีมาก)
    selected_assets = select_assets_menu("TRAIN UNIVERSAL MODEL")
    
    valid_assets = []
    for asset in selected_assets:
        config = load_regime_config(asset)
        if config:
            valid_assets.append(asset)
        else:
            print(f"{C_YELLOW}⚠️  Creating config for {asset.upper()}...{C_RESET}")
            if auto_create_config(asset): 
                valid_assets.append(asset)
            else:
                print(f"{C_RED}❌ Skipped {asset.upper()}{C_RESET}")

    if not valid_assets: return

    print(f"\n{C_LIME}[POOL] {', '.join([a.upper() for a in valid_assets])}{C_RESET}")
    confirm = get_user_input("Start UNIVERSAL training? (y/n)", valid_options=['y', 'n'])
    if confirm == 'n': return
    
    # --- [ส่วนที่เหลือคือ Pipeline การโหลดและเทรน (ของเดิม) ซึ่งโอเคแล้วครับ] ---
    try:
        master_config = load_regime_config(valid_assets[0])
        regimes_data = master_config.get('regimes', {})

        print(f"\n{C_LIME}>>> [Step 1 & 2] Slicing Data & Features...{C_RESET}")
        for regime_key, regime_info in regimes_data.items():
            start_date = datetime.strptime(regime_info.get('start_date'), "%Y-%m-%d")
            end_date = datetime.strptime(regime_info.get('end_date'), "%Y-%m-%d")
            
            total_days = (end_date - start_date).days + 1
            days_per_asset = max(1, total_days // len(valid_assets)) 
            
            print(f"\n{C_DARK_GREEN}--- REGIME: {regime_key.upper()} ({total_days} days) ---{C_RESET}")
            current_start = start_date
            
            for idx, asset in enumerate(valid_assets):
                chunk_start = current_start
                chunk_end = end_date if idx == len(valid_assets) - 1 else chunk_start + timedelta(days=days_per_asset - 1)
                current_start = chunk_end + timedelta(days=1)
                start_str, end_str = chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
                
                # ⭐️ ระบบเช็คไฟล์ว่าเคยทำไปแล้วหรือยัง (Resume Feature)
                feature_file = PROJECT_ROOT / "data" / "processed" / f"{asset.upper()}USDT_features_{regime_key}.parquet"
                if feature_file.exists():
                    print(f"{C_GRAY} ⏩ Skipped {asset.upper():<5} | Already processed {start_str} to {end_str}{C_RESET}")
                    continue

                print(f"{C_GREEN} ✓ {asset.upper():<5} {C_GRAY}| {start_str} to {end_str}{C_RESET}")
                config_name = f"downloadData_{asset}.yaml"
                subprocess.run(["python3", str(PROJECT_ROOT / "scripts" / "00_download_data.py"), "--config", config_name, "--override_regime", regime_key, "--override_start", start_str, "--override_end", end_str], check=True)
                subprocess.run(["python3", str(PROJECT_ROOT / "scripts" / "01_build_dataset.py"), "--config", config_name, "--override_regime", regime_key], check=True)
                cleanup_raw_data(asset)

        print(f"\n{C_LIME}>>> [Step 3] Cross-Asset Pooling...{C_RESET}")
        subprocess.run(["python3", str(PROJECT_ROOT / "scripts" / "01b_prepare_chunks.py")], check=True)

        print(f"\n{C_LIME}>>> [Step 4] GA AS-Baseline Optimization...{C_RESET}")
        ga_file = PROJECT_ROOT / "configs" / "ga_optimized_baseline.yaml"
        if ga_file.exists():
            reuse_ga = get_user_input(f"Found existing GA Baseline. Reuse it to save time? (y/n)", valid_options=['y', 'n'])
            if reuse_ga == 'y':
                print(f"{C_GRAY} ⏩ Skipped GA Optimization.{C_RESET}")
            else:
                subprocess.run(["python3", str(PROJECT_ROOT / "scripts" / "03b_optimize_ga.py")], check=True)
        else:
            subprocess.run(["python3", str(PROJECT_ROOT / "scripts" / "03b_optimize_ga.py")], check=True)

        print(f"\n{C_LIME}>>> [Step 5] PPO Universal Training...{C_RESET}")
        ppo_cmd = ["python3", str(PROJECT_ROOT / "scripts" / "04_train_chunked.py")]
        latest_model = PROJECT_ROOT / "models" / "ppo_universal_hft_latest.zip"
        if latest_model.exists():
            resume_ppo = get_user_input(f"Found incomplete PPO Model. Resume training from checkpoint? (y/n)", valid_options=['y', 'n'])
            if resume_ppo == 'y':
                ppo_cmd.append("--resume")
                
        subprocess.run(ppo_cmd, check=True)

        print(f"\n{C_LIME}✅ UNIVERSAL PIPELINE COMPLETED!{C_RESET}")
        
        do_bt = get_user_input("Run Out-of-Sample Backtest? (y/n)", valid_options=['y', 'n'])
        if do_bt == 'y':
            test_asset = input(f"{C_GREEN}▶ {C_WHITE}Asset to backtest (e.g., btc): {C_RESET}").strip().lower()
            run_backtest(test_asset)

    except Exception as e:
        print(f"\n{C_RED}❌ Error: {e}{C_RESET}")
        input(f"{C_GRAY}Press Enter to return...{C_RESET}")

def main():
    while True:
        clear_screen()
        print_hft_logo()
        
        print(f"{C_LIME}SYSTEM COMMANDS:{C_RESET}")
        print(f"  {C_GRAY}[1]{C_RESET} {C_WHITE}Deploy Dashboard (Single / Portfolio Overview){C_RESET}")
        print(f"  {C_GRAY}[2]{C_RESET} {C_WHITE}Run Out-of-Sample Backtest{C_RESET}")
        print(f"  {C_GRAY}[3]{C_RESET} {C_WHITE}Train UNIVERSAL Model (Multi-Asset Selection){C_RESET}")
        print(f"  {C_GRAY}[0]{C_RESET} {C_GRAY}Exit System{C_RESET}\n")
        
        choice = get_user_input("Select Command (0-3)", valid_options=['1', '2', '3', '0'])
        
        if choice == '0': sys.exit(0)

        if choice == '1':
            # ⭐️ ถามก่อนว่าดูเดี่ยวหรือดูภาพรวม
            print(f"\n{C_WHITE}1. Single Asset View{C_RESET}")
            print(f"{C_WHITE}2. Multi-Asset Portfolio Overview{C_RESET}")
            sub_choice = get_user_input("Select View", valid_options=['1', '2'])
            
            if sub_choice == '1':
                asset = input(f"{C_GREEN}▶ {C_WHITE}Enter Asset Name: {C_RESET}").strip().lower()
                if asset: run_live_dashboard([asset])
            else:
                selected = select_assets_menu("PORTFOLIO OVERVIEW SETUP")
                run_live_dashboard(selected)

        elif choice == '2':
            asset = input(f"{C_GREEN}▶ {C_WHITE}Asset to Backtest: {C_RESET}").strip().lower()
            # run_backtest(asset)
        elif choice == '3':
            train_universal_model()

if __name__ == "__main__":
    main()