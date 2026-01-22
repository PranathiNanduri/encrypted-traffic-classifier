import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# Config
# -----------------------------
TRAIN_PATH = "data/processed/train.npz"
TEST_PATH  = "data/processed/test.npz"
OUT_DIR = Path("docs/baseline_rf")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["vpn", "nonvpn", "tor"]

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(X_len, X_iat):
    features = []

    for l, i in zip(X_len, X_iat):
        feats = [
            np.mean(l), np.std(l), np.max(l),
            np.mean(i), np.std(i), np.max(i)
        ]
        features.append(feats)

    return np.array(features)

# -----------------------------
# Load data
# -----------------------------
train = np.load(TRAIN_PATH)
test  = np.load(TEST_PATH)

Xtr = extract_features(train["X_len"], train["X_iat"])
ytr = train["y"]

Xte = extract_features(test["X_len"], test["X_iat"])
yte = test["y"]

# -----------------------------
# Train Random Forest
# -----------------------------
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf.fit(Xtr, ytr)

# -----------------------------
# Evaluate
# -----------------------------
y_pred = rf.predict(Xte)

report = classification_report(
    yte, y_pred,
    target_names=CLASS_NAMES,
    digits=4
)

print("\nRandom Forest Classification Report:\n")
print(report)

with open(OUT_DIR / "classification_report.txt", "w") as f:
    f.write(report)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(yte, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix.png")
plt.close()

# -----------------------------
# Save predictions
# -----------------------------
df = pd.DataFrame({
    "true_label": [CLASS_NAMES[i] for i in yte],
    "predicted_label": [CLASS_NAMES[i] for i in y_pred]
})
df.to_csv(OUT_DIR / "true_vs_predicted.csv", index=False)

print("\n✅ Random Forest results saved in docs/baseline_rf/")
