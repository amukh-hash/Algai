import os
import sys
import uuid
import json
import random
import time
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.train_global import train_model

MANIFEST_PATH = "backend/models/ensemble_manifest.json"

def load_manifest():
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            return json.load(f)
    return []

def save_manifest(manifest):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)

def get_random_config():
    return {
        'lookback': random.choice([60, 180, 365]),
        'feature_set': random.choice(['raw', 'tech', 'sentiment']),
        'n_heads': random.choice([4, 8])
    }

def run_overlord():
    print("ALL HAIL THE OVERLORD. Training loop initialized.")
    print("Press Ctrl+C to stop training and yield to the Judge.")

    os.makedirs("backend/models", exist_ok=True)

    while True:
        try:
            # 1. Generate Config
            config = get_random_config()
            model_id = str(uuid.uuid4())[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 2. Define Save Path
            model_name = f"ensemble_{model_id}_lb{config['lookback']}_{config['feature_set']}_h{config['n_heads']}.pth"
            save_path = os.path.join("backend/models", model_name)
            config['save_path'] = save_path

            print(f"\n[Overlord] Spawning Model {model_id}...")
            print(f"Config: {config}")

            # 3. Train
            train_model(config)

            # 4. Update Manifest
            manifest = load_manifest()
            entry = {
                'id': model_id,
                'path': save_path,
                'config': config,
                'created_at': timestamp,
                'status': 'trained'
            }
            manifest.append(entry)
            save_manifest(manifest)

            print(f"[Overlord] Model {model_id} successfully stored in the vault.")
            print("-" * 50)

            # Tiny sleep to ensure file handles close?
            time.sleep(2)

        except KeyboardInterrupt:
            print("\n[Overlord] Training interrupted by user. Shutting down.")
            break
        except Exception as e:
            print(f"\n[Overlord] CRITICAL ERROR spawning model: {e}")
            print("Retrying next spawn in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    run_overlord()
