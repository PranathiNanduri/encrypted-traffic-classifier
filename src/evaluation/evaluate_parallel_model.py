import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# Config
# -----------------------------
MODEL_PATH = "models/deep_learning/parallel_cnn_nin.keras"
DATA_PATH = "data/processed/test.npz"
STATS_PATH = "models/deep_learning/norm_stats.npz"
OUT_DIR = Path("docs")

CLASS_NAMES = ["vpn", "nonvpn", "tor"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_TXT = OUT_DIR / "parallel_cnn_nin_results.txt"
REPORT_TXT = OUT_DIR / "classification_report.txt"
CM_IMG = OUT_DIR / "confusion_matrix.png"
CSV_PATH = OUT_DIR / "true_vs_predicted.csv"

# -----------------------------
# Load model (no compile needed)
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -----------------------------
# Load data
# -----------------------------
data = np.load(DATA_PATH)
X_len = data["X_len"].astype("float32")
X_iat = data["X_iat"].astype("float32")
y_true = data["y"]

# -----------------------------
# Load normalization stats
# -----------------------------
stats = np.load(STATS_PATH)

def preprocess(X_len, X_iat):
    X_len = np.clip(X_len, 0, stats["len_clip"])
    X_iat = np.clip(X_iat, 0, stats["iat_clip"])

    X_len = (X_len - stats["len_mean"]) / (stats["len_std"] + 1e-9)
    X_iat = (X_iat - stats["iat_mean"]) / (stats["iat_std"] + 1e-9)

    # (N, 50, 2)
    return np.stack([X_len, X_iat], axis=-1).astype("float32")

X = preprocess(X_len, X_iat)

# -----------------------------
# Predict
# -----------------------------
probs = model.predict(X, batch_size=64, verbose=1)
y_pred = np.argmax(probs, axis=1)

# -----------------------------
# 1) Classification report (string)
# -----------------------------
report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

# save report alone
REPORT_TXT.write_text(report, encoding="utf-8")

# -----------------------------
# 2) Confusion matrix (text + image)
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

# Save confusion matrix image WITHOUT seaborn (clean + no extra deps)
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES)
plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)

# write numbers in cells
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")

plt.tight_layout()
plt.savefig(CM_IMG)
plt.close()

# -----------------------------
# 3) True vs Predicted CSV
# -----------------------------
df = pd.DataFrame({
    "true_label": [CLASS_NAMES[int(i)] for i in y_true],
    "predicted_label": [CLASS_NAMES[int(i)] for i in y_pred],
    "true_id": y_true,
    "predicted_id": y_pred
})
df.to_csv(CSV_PATH, index=False)

# -----------------------------
# 4) Save "parallel_cnn_nin_results.txt" (combined)
# -----------------------------
with open(RESULTS_TXT, "w", encoding="utf-8") as f:
    f.write("Parallel CNN+NIN Classification Report\n")
    f.write(report)
    f.write("\n\nConfusion Matrix (rows=true, cols=pred)\n")
    f.write(np.array2string(cm))

print("\n✅ Saved outputs to docs/:")
print(f" - {RESULTS_TXT}")
print(f" - {REPORT_TXT}")
print(f" - {CM_IMG}")
print(f" - {CSV_PATH}")
