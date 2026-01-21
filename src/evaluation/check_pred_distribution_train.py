import numpy as np
import tensorflow as tf
from collections import Counter

MODEL_PATH = "models/deep_learning/parallel_cnn_nin.keras"
STATS_PATH = "models/deep_learning/norm_stats.npz"

def preprocess_2ch(X_len, X_iat, stats):
    X_len = np.clip(X_len.astype("float32"), 0, float(stats["len_clip"]))
    X_iat = np.clip(X_iat.astype("float32"), 0, float(stats["iat_clip"]))

    X_len = (X_len - float(stats["len_mean"])) / (float(stats["len_std"]) + 1e-9)
    X_iat = (X_iat - float(stats["iat_mean"])) / (float(stats["iat_std"]) + 1e-9)

    return np.stack([X_len, X_iat], axis=-1)

def main():
    train = np.load("data/processed/train.npz")
    X_len, X_iat, y = train["X_len"], train["X_iat"], train["y"]

    stats = np.load(STATS_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    X = preprocess_2ch(X_len, X_iat, stats)
    probs = model.predict(X, verbose=0)
    pred = probs.argmax(axis=1)

    print("✅ Train true :", Counter(y.tolist()))
    print("✅ Train pred :", Counter(pred.tolist()))

if __name__ == "__main__":
    main()
