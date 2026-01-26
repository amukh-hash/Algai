import time
import os
import subprocess
import sys

# Constants
DATA_FILE = r"backend\data\processed\orthogonal_features_final.parquet"
TRAIN_SCRIPT = r"backend\scripts\train_chronos_phase2.py"
PYTHON_EXE = r"venv_gpu\Scripts\python.exe"

def main():
    print("--- Overnight Orchestrator Started ---")
    print(f"Monitoring for: {DATA_FILE}")
    print("User can leave PC. Do not close this terminal.")
    
    # Wait loop
    start_time = time.time()
    while not os.path.exists(DATA_FILE):
        elapsed = int(time.time() - start_time) // 60
        sys.stdout.write(f"\rWaiting for Phase 1 Data... Elapsed: {elapsed} min")
        sys.stdout.flush()
        time.sleep(30)
        
    print(f"\n\nPhase 1 Data Detected! Starting Phase 2 Training in 30 seconds...")
    time.sleep(30) # Safety buffer for full write
    
    print(f"Launching: {TRAIN_SCRIPT}")
    cmd = [PYTHON_EXE, TRAIN_SCRIPT]
    
    try:
        subprocess.run(cmd, check=True)
        print("\n\n--- Overnight Training Complete! ---")
    except subprocess.CalledProcessError as e:
        print(f"Training Failed with error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
