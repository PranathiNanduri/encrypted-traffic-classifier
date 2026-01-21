import numpy as np
from collections import Counter

CLASSES = ["vpn", "nonvpn", "tor"]

tr = np.load("data/processed/train.npz")
y = tr["y"]

print("y counts:", Counter(y.tolist()))
print("label mapping:")
for i, name in enumerate(CLASSES):
    print(f"  {i} -> {name}")

print("\nfirst 30 labels:", y[:30].tolist())
