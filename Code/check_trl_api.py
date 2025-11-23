#!/usr/bin/env python
"""Check TRL API to determine correct parameters."""

import inspect
from trl import PPOConfig, PPOTrainer

print("PPOConfig signature:")
print(inspect.signature(PPOConfig.__init__))
print("\nPPOConfig parameters:")
for param in inspect.signature(PPOConfig.__init__).parameters.values():
    print(f"  {param}")

print("\n" + "="*60)
print("PPOTrainer signature:")
print(inspect.signature(PPOTrainer.__init__))
print("\nPPOTrainer parameters:")
for param in inspect.signature(PPOTrainer.__init__).parameters.values():
    print(f"  {param}")
