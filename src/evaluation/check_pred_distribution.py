import numpy as np
import tensorflow as tf
from collections import Counter

test = np.load("data/processed/test.npz")
Xlen, Xiat, y = test["X_len"], test["X_iat"], test["y"]

stats = np.load("models/deep_learning/norm_stats.npz")

Xlen = np.clip(Xlen, 0, stats["len_clip"])
Xiat = np.clip(Xiat, 0, stats["iat_clip"])

Xlen = (Xlen - stats["len_mean"]) / (stats["len_std"] + 1e-9)
Xiat = (Xiat - stats["iat_mean"]) / (stats["iat_std"] + 1e-9)

# ✅ STEP 4 FIX
X = np.stack([Xlen, Xiat], axis=-1)

model = tf.keras.models.load_model(
    "models/deep_learning/parallel_cnn_nin.keras",
    compile=False
)

preds = np.argmax(model.predict(X, verbose=0), axis=1)

print("✅ True distribution :", Counter(y))
print("✅ Pred distribution :", Counter(preds))
