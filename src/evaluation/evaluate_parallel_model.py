import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

MODEL_PATH = "models/deep_learning/parallel_cnn_nin.keras"
TEST_PATH = "data/processed/test.npz"

# Must match label order used in build_dataset.py
CLASS_NAMES = ["vpn", "nonvpn", "tor"]

def preprocess_inputs(test_npz):
    X_len = test_npz["X_len"].astype("float32")
    X_iat = test_npz["X_iat"].astype("float32")
    y = test_npz["y"].astype("int64")

    # same preprocessing used in training
    X_len = X_len / (X_len.max() + 1e-9)

    X_iat = np.log1p(X_iat)
    X_iat = X_iat / (X_iat.max() + 1e-9)

    # add channel dim: (N, 50, 1)
    X_len = X_len[..., None]
    X_iat = X_iat[..., None]

    return X_len, X_iat, y

def main():
    test = np.load(TEST_PATH)
    X_len, X_iat, y_true = preprocess_inputs(test)

    model = tf.keras.models.load_model(MODEL_PATH)
    probs = model.predict({"pkt_len": X_len, "pkt_iat": X_iat}, verbose=0)
    y_pred = probs.argmax(axis=1)

    print("\n✅ Parallel CNN+NIN Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    print("\n✅ Confusion Matrix (rows=true, cols=pred):\n")
    print(cm)

    # save results for report
    Path("docs").mkdir(exist_ok=True)
    with open("docs/parallel_cnn_nin_results.txt", "w") as f:
        f.write("Parallel CNN+NIN Classification Report\n")
        f.write(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))
        f.write("\n\nConfusion Matrix (rows=true, cols=pred)\n")
        f.write(str(cm))

    print("\n✅ Saved: docs/parallel_cnn_nin_results.txt")

if __name__ == "__main__":
    main()
