"""Fix checkpoint by copying config from base model."""

import shutil
from pathlib import Path

# Paths
base_model = Path("models/gemma-2-2b-it")
checkpoint = Path("experiments/checkpoints/phase1_232680/final")

# Copy config.json from base model to checkpoint
config_src = base_model / "config.json"
config_dst = checkpoint / "config.json"

if config_src.exists():
    shutil.copy(config_src, config_dst)
    print(f"✓ Copied config.json to {checkpoint}")
else:
    print(f"! Base model config not found at {config_src}")
    print("You'll need to rerun training with the fixed script")
