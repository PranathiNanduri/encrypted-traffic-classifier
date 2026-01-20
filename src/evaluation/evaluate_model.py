import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

MODEL_PATH = "models/deep_learning/baseline_cnn.h5"
TEST_PATH = "data/processed/test.npz"

# IMPORTANT: label order must match build_dataset.py CLASSES
CLASS_NAMES = ["vpn", "nonvpn", "tor"]

def main():
    test = np.load(TEST_PATH)
    X_test = test["X_len"]
    y_test = test["y"]

    # same normalization used in training
    X_test = X_test / (X_test.max() + 1e-9)
    X_test = X_test[..., None]  # (N, 50, 1)

    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict(X_test, verbose=0)
    y_pred = probs.argmax(axis=1)

    print("\n✅ Classification Report:\n")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))

    print("\n✅ Confusion Matrix (rows=true, cols=pred):\n")
    print(confusion_matrix(y_test, y_pred))

    # save results for report
    Path("docs").mkdir(exist_ok=True)
    with open("docs/baseline_results.txt", "w") as f:
        f.write("Classification Report\n")
        f.write(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4))
        f.write("\n\nConfusion Matrix (rows=true, cols=pred)\n")
        f.write(str(confusion_matrix(y_test, y_pred)))

    print("\n✅ Saved: docs/baseline_results.txt")

if __name__ == "__main__":
    main()
