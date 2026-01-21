import numpy as np
from pathlib import Path

OUT = Path("models/deep_learning")
OUT.mkdir(parents=True, exist_ok=True)

train = np.load("data/processed/train.npz")
X_len = train["X_len"].astype("float32")
X_iat = train["X_iat"].astype("float32")

# Robust clipping to reduce outliers (VERY IMPORTANT)
len_clip = 1500.0
iat_clip = np.percentile(X_iat[X_iat > 0], 99) if np.any(X_iat > 0) else 1.0

X_len_c = np.clip(X_len, 0, len_clip)
X_iat_c = np.clip(X_iat, 0, iat_clip)

X_iat_log = np.log1p(X_iat_c)

len_mean, len_std = float(X_len_c.mean()), float(X_len_c.std() + 1e-9)
iat_mean, iat_std = float(X_iat_log.mean()), float(X_iat_log.std() + 1e-9)

np.savez(
    OUT / "norm_stats.npz",
    len_clip=len_clip,
    iat_clip=float(iat_clip),
    len_mean=len_mean,
    len_std=len_std,
    iat_mean=iat_mean,
    iat_std=iat_std,
)

print("✅ Saved:", OUT / "norm_stats.npz")
print("len_clip:", len_clip, "iat_clip:", float(iat_clip))
print("len_mean/std:", len_mean, len_std)
print("iat_mean/std:", iat_mean, iat_std)
