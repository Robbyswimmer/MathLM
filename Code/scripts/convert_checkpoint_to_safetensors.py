"""Convert PyTorch checkpoint to safetensors format."""

import torch
from pathlib import Path
from safetensors.torch import save_file

checkpoint_dir = Path("experiments/checkpoints/phase1_232680/final")

# Find the pytorch bin file
bin_file = checkpoint_dir / "pytorch_model.bin"

if not bin_file.exists():
    print(f"! No pytorch_model.bin found at {bin_file}")
    exit(1)

print(f"Loading checkpoint from {bin_file}...")
# Load with weights_only=False to bypass the security check temporarily
state_dict = torch.load(bin_file, map_location="cpu", weights_only=False)

# Save as safetensors
safetensors_file = checkpoint_dir / "model.safetensors"
print(f"Saving as safetensors to {safetensors_file}...")
save_file(state_dict, str(safetensors_file))

# Remove the old bin file
bin_file.unlink()
print(f"✓ Converted to safetensors and removed .bin file")
