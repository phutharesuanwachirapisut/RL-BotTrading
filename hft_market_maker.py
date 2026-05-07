import os
import sys
import time
import subprocess
import json
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import webbrowser
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

# แก้ไขฟังก์ชันนี้
def get_user_input(prompt, valid_options=None, cast_type=str, default=None):
    while True:
        try:
            # เพิ่มการแสดงผล default ในวงเล็บถ้ามี
            display_prompt = f"{prompt} [{default}]" if default else prompt
            user_input = input(f"{C_GREEN}▶ {C_WHITE}{display_prompt}: {C_RESET}").strip().lower()
            
            # ถ้าผู้ใช้กด Enter เปล่าๆ และมีการตั้ง default ไว้
            if not user_input and default is not None:
                return default
                
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
    """Automatically generates ALL required config files for a new asset."""
    success = True
    config_types = [
        (f"downloadData_{template_asset}.yaml", f"downloadData_{asset}.yaml"),
        (f"{template_asset}_hyperparameters.yaml", f"{asset}_hyperparameters.yaml"),
        (f"{template_asset}_trading_env.yaml", f"{asset}_trading_env.yaml")
    ]
    
    for template_name, new_name in config_types:
        template_path = PROJECT_ROOT / "configs" / template_name
        
        # ⭐️ บางครั้งไฟล์ downloadData อาจจะเผลอไปอยู่ที่ Root Directory ให้ดักทางไว้ด้วย
        if not template_path.exists() and (PROJECT_ROOT / template_name).exists():
            template_path = PROJECT_ROOT / template_name
            
        new_config_path = PROJECT_ROOT / "configs" / new_name
        
        if not template_path.exists():
            print(f"{C_RED}❌ Error: Template file {template_name} missing.{C_RESET}")
            success = False
            continue
            
        if not new_config_path.exists():
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                
                # Replace the symbol naming conventions inside the file
                new_content = template_content.replace(f'"{template_asset.upper()}USDT"', f'"{asset.upper()}USDT"')
                new_content = new_content.replace(f"'{template_asset.upper()}/USDT'", f"'{asset.upper()}/USDT'")
                new_content = new_content.replace(f'symbol: "{template_asset.upper()}/USDT:USDT"', f'symbol: "{asset.upper()}/USDT:USDT"')
                
                new_config_path.parent.mkdir(parents=True, exist_ok=True)
                with open(new_config_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                print(f"{C_GRAY}  [+] Generated {new_name}{C_RESET}")
            except Exception as e:
                print(f"{C_RED}❌ Error generating {new_name}: {e}{C_RESET}")
                success = False
                
    return success


def get_single_asset_port(asset):
    """ระบบความจำ: ดึงหรือสร้าง Port เฉพาะสำหรับแต่ละเหรียญในโหมด Single"""
    registry_file = PROJECT_ROOT / "configs" / "single_port_registry.json"
    
    # ค่าเริ่มต้นตามที่คุณต้องการ
    registry = {
        "btc": 8001,
        "eth": 8002,
        "bnb": 8003,
        "xrp": 8004,
        "ltc": 8005,
        "ada": 8006
    }
    
    # โหลดไฟล์ความจำเก่ามาผสาน (ถ้ามี)
    if registry_file.exists():
        try:
            with open(registry_file, 'r') as f:
                saved_registry = json.load(f)
                registry.update(saved_registry)
        except Exception:
            pass
            
    # ถ้าเป็นเหรียญใหม่ที่ไม่เคยรู้จัก ให้หาพอร์ตที่ว่างถัดไป
    if asset not in registry:
        next_port = max(registry.values()) + 1
        registry[asset] = next_port
        
    # เซฟความจำกลับลงไฟล์
    registry_file.parent.mkdir(parents=True, exist_ok=True)
    with open(registry_file, 'w') as f:
        json.dump(registry, f, indent=4)
        
    return str(registry[asset])

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

def run_live_dashboard(selected_assets, view_choice):
    """รัน Dashboard สำหรับเหรียญที่เลือก พร้อมระบบ Fix Port"""
    print(f"\n{C_LIME}>>> DEPLOYMENT MODE{C_RESET}")
    print(f"  {C_WHITE}1. Sandbox (Demo/Testnet){C_RESET}")
    print(f"  {C_RED}2. LIVE (Real Money){C_RESET}")
    mode_choice = get_user_input("Select Mode (1 or 2)", valid_options=['1', '2'])
    mode = "sandbox" if mode_choice == '1' else "live"

    # ==========================================
    # กรณี 1: เปิดแบบ Single Asset (เปิด HTML Dashboard ล้ำๆ)
    # ==========================================
    if view_choice == '1':
        asset = selected_assets[0]
        port = get_single_asset_port(asset) # ⭐️ ดึงพอร์ตจากระบบความจำ
        
        print(f"\n{C_LIME}[Deploying Single Bot] Initializing {asset.upper()} on port {port}...{C_RESET}")
        
        # รันบอท (มันจะรัน FastAPI server ที่เสิร์ฟไฟล์ dashboard.html ด้วย)
        subprocess.Popen(["python3", "scripts/07_live_real_money.py", "--pair", asset, "--port", port, "--mode", mode])
        
        time.sleep(3) # ให้เวลาเซิร์ฟเวอร์เปิดแป๊บนึง
        url = f"http://localhost:{port}"
        print(f"\n{C_YELLOW}💡 Tip: Opening your browser directly to the High-Tech Dashboard...{C_RESET}")
        print(f"   {C_WHITE}-> {C_LIME}{url}{C_RESET}")
        
        # สั่งเปิดเบราว์เซอร์ให้อัตโนมัติ
        webbrowser.open(url)

    # ==========================================
    # กรณี 2: เปิดแบบ Multi-Asset (เปิด Streamlit Command Center)
    # ==========================================
    # ==========================================
    # กรณี 2: เปิดแบบ Multi-Asset (เปิด Streamlit Command Center)
    # ==========================================
    else:
        port_mapping = {}
        for idx, asset in enumerate(DEFAULT_ASSETS):
            port_mapping[asset] = 8001 + idx
        
        custom_port_start = 8090

        print(f"\n{C_LIME}[Deploying Portfolio] Initializing {len(selected_assets)} assets...{C_RESET}")
        
        # ⭐️ ADD THIS BLOCK: Verify BOTH configs exist before deploying
        valid_assets_to_deploy = []
        for asset in selected_assets:
            env_file = PROJECT_ROOT / "configs" / f"{asset}_trading_env.yaml"
            hyper_file = PROJECT_ROOT / "configs" / f"{asset}_hyperparameters.yaml"
            
            # ถ้าขาดไฟล์ใดไฟล์หนึ่งไป (เช่นมี env แต่ไม่มี hyper) ให้รัน Auto-generate ซ่อมแซมทันที
            if not env_file.exists() or not hyper_file.exists():
                print(f"{C_YELLOW}⚠️  Missing configs for {asset.upper()}. Auto-generating from BTC template...{C_RESET}")
                if auto_create_config(asset):
                    valid_assets_to_deploy.append(asset)
                else:
                    print(f"{C_RED}❌ Failed to configure {asset.upper()}. Skipping deployment.{C_RESET}")
            else:
                valid_assets_to_deploy.append(asset)
        
        # นำเหรียญที่ไฟล์ครบถ้วนแล้วมา Deploy
        for asset in valid_assets_to_deploy:
            if asset in port_mapping:
                port = str(port_mapping[asset])
            else:
                port = str(custom_port_start)
                custom_port_start += 1
                
            print(f"  {C_GREEN}✓{C_RESET} {asset.upper():<5} -> http://localhost:{port}")
            subprocess.Popen(["python3", "scripts/07_live_real_money.py", "--pair", asset, "--port", port, "--mode", mode])
            
        time.sleep(5) # ให้เวลาเซิร์ฟเวอร์เปิด
        webbrowser.open("http://localhost:8000")

    input(f"\n{C_GRAY}Deployment started. Bots are running in the background. Press Enter to return to main menu...{C_RESET}")

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
            reuse_ga = get_user_input(f"Found existing GA Baseline. Reuse it for this time? (y/n)", valid_options=['y', 'n'], default='y')
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
            # ⭐️ เพิ่ม default='y' ตรงนี้
            resume_ppo = get_user_input(f"Found incomplete PPO Model. Resume training from checkpoint? (y/n)", valid_options=['y', 'n'], default='y')
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
                # ❌ This is the line causing the error:
                # if asset: run_live_dashboard([asset])
                
                # ✅ Change it to include sub_choice:
                if asset: run_live_dashboard([asset], sub_choice)
            else:
                selected = select_assets_menu("PORTFOLIO OVERVIEW SETUP")
                # ❌ This is also missing the second argument:
                # run_live_dashboard(selected)
                
                # ✅ Change it to include sub_choice:
                run_live_dashboard(selected, sub_choice)
        elif choice == '2':
            asset = input(f"{C_GREEN}▶ {C_WHITE}Asset to Backtest: {C_RESET}").strip().lower()
            run_backtest(asset)
        elif choice == '3':
            train_universal_model()

if __name__ == "__main__":
    main()