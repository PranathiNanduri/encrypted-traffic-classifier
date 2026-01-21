import numpy as np
from collections import Counter

for split in ["train", "test"]:
    data = np.load(f"data/processed/{split}.npz")
    y = data["y"].tolist()
    c = Counter(y)
    print(split, c, "total:", len(y))
