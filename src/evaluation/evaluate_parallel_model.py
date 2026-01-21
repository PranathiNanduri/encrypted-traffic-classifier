import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["vpn", "nonvpn", "tor"]

test = np.load("data/processed/test.npz")
Xlen, Xiat, y_true = test["X_len"], test["X_iat"], test["y"]

stats = np.load("models/deep_learning/norm_stats.npz")

# SAME preprocessing as training
Xlen = np.clip(Xlen, 0, stats["len_clip"])
Xiat = np.clip(Xiat, 0, stats["iat_clip"])

Xlen = (Xlen - stats["len_mean"]) / (stats["len_std"] + 1e-9)
Xiat = (Xiat - stats["iat_mean"]) / (stats["iat_std"] + 1e-9)

# ✅ STEP 4 FIX: single 2-channel tensor
X = np.stack([Xlen, Xiat], axis=-1)  # (N, 50, 2)

model = tf.keras.models.load_model(
    "models/deep_learning/parallel_cnn_nin.keras",
    compile=False
)

probs = model.predict(X, verbose=0)
y_pred = np.argmax(probs, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
