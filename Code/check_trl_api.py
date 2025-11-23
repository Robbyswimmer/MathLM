#!/usr/bin/env python
"""Check TRL API to determine correct parameters."""

import inspect
import trl

print("TRL version:", trl.__version__)
print("\nAvailable TRL classes:")
for name in dir(trl):
    if 'PPO' in name or 'Trainer' in name:
        print(f"  - {name}")

print("\n" + "="*60)

try:
    from trl import PPOConfig, PPOTrainer
    print("PPOConfig signature:")
    print(inspect.signature(PPOConfig.__init__))

    print("\n" + "="*60)
    print("PPOTrainer signature:")
    print(inspect.signature(PPOTrainer.__init__))
except Exception as e:
    print(f"Error importing PPO classes: {e}")

print("\n" + "="*60)
print("\nLooking for alternative PPO implementations...")
try:
    # Check for OnlineDPOTrainer or other alternatives
    from trl import OnlineDPOTrainer
    print("✓ OnlineDPOTrainer available")
except:
    print("✗ OnlineDPOTrainer not available")

try:
    from trl import AutoModelForCausalLMWithValueHead
    print("✓ AutoModelForCausalLMWithValueHead available")
except:
    print("✗ AutoModelForCausalLMWithValueHead not available")
