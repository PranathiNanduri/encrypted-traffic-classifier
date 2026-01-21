import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split

MAX_LEN = 50
NUM_CLASSES = 3


# ----------------------------
# Step 3: Focal Loss (multiclass)
# ----------------------------
def focal_loss(alpha=None, gamma=2.0, num_classes=3):
    """
    Multiclass focal loss.

    alpha: list/np.array length = num_classes
           example: [1.0, 2.0, 1.0] -> gives more importance to class 1
           If None -> all ones.
    gamma: focusing parameter (2.0 is common)
    """
    if alpha is None:
        alpha = np.ones((num_classes,), dtype=np.float32)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=num_classes)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)      # prob for true class
        alpha_t = tf.gather(alpha, y_true)                    # alpha for true class

        return tf.reduce_mean(-alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t))

    return loss


def compute_alpha_from_counts(y, cap=3.0):
    """
    Compute alpha weights inversely proportional to class frequency:
      alpha_c = total / (num_classes * count_c)
    then cap to avoid extreme weights.
    Finally normalize so average alpha ~ 1.0 (stable training).
    """
    counts = Counter(y.tolist())
    total = sum(counts.values())
    alpha = np.zeros((NUM_CLASSES,), dtype=np.float32)

    for c in range(NUM_CLASSES):
        cnt = counts.get(c, 1)
        alpha[c] = total / (NUM_CLASSES * cnt)

    # cap
    alpha = np.minimum(alpha, cap)

    # normalize to mean 1.0
    alpha = alpha / (alpha.mean() + 1e-9)
    return alpha


# ----------------------------
# Preprocessing (unchanged)
# ----------------------------
def compute_norm_stats(X_len, X_iat):
    # Clip at high percentile to reduce outliers
    len_clip = np.percentile(X_len, 99.5).astype(np.float32)
    iat_clip = np.percentile(X_iat, 99.5).astype(np.float32)

    Xl = np.clip(X_len, 0, len_clip)
    Xi = np.clip(X_iat, 0, iat_clip)

    len_mean = Xl.mean().astype(np.float32)
    len_std  = Xl.std().astype(np.float32) + 1e-9
    iat_mean = Xi.mean().astype(np.float32)
    iat_std  = Xi.std().astype(np.float32) + 1e-9

    return {
        "len_clip": len_clip,
        "iat_clip": iat_clip,
        "len_mean": len_mean,
        "len_std":  len_std,
        "iat_mean": iat_mean,
        "iat_std":  iat_std,
    }


def preprocess_2ch(X_len, X_iat, stats):
    X_len = X_len.astype("float32")
    X_iat = X_iat.astype("float32")

    X_len = np.clip(X_len, 0, stats["len_clip"])
    X_iat = np.clip(X_iat, 0, stats["iat_clip"])

    X_len = (X_len - stats["len_mean"]) / (stats["len_std"] + 1e-9)
    X_iat = (X_iat - stats["iat_mean"]) / (stats["iat_std"] + 1e-9)

    # (N, 50, 2)
    X = np.stack([X_len, X_iat], axis=-1)
    return X


# ----------------------------
# Model
# ----------------------------
def nin_block(x, filters, kernel_size):
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)
    x = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_model(alpha):
    inp = layers.Input(shape=(MAX_LEN, 2), name="flow_features")
    x = nin_block(inp, 64, 5)
    x = layers.MaxPooling1D(2)(x)
    x = nin_block(x, 128, 3)
    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inp, out, name="CNN_NIN_2CH")

    # ✅ Step 3: focal loss instead of cross-entropy
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=focal_loss(alpha=alpha, gamma=2.0, num_classes=NUM_CLASSES),
        metrics=["accuracy"]
    )
    return model


def main():
    train = np.load("data/processed/train.npz")
    test  = np.load("data/processed/test.npz")

    Xlen_tr, Xiat_tr, y_tr = train["X_len"], train["X_iat"], train["y"]
    Xlen_te, Xiat_te, y_te = test["X_len"], test["X_iat"], test["y"]

    print("✅ Train distribution:", Counter(y_tr.tolist()))
    print("✅ Test  distribution:", Counter(y_te.tolist()))

    # ✅ Stratified train/val split
    Xlen_train, Xlen_val, Xiat_train, Xiat_val, y_train, y_val = train_test_split(
        Xlen_tr, Xiat_tr, y_tr,
        test_size=0.2,
        random_state=42,
        stratify=y_tr
    )

    # ✅ Compute stats from TRAIN only
    stats = compute_norm_stats(Xlen_train, Xiat_train)

    Path("models/deep_learning").mkdir(parents=True, exist_ok=True)
    np.savez("models/deep_learning/norm_stats.npz", **stats)
    print("✅ Saved stats: models/deep_learning/norm_stats.npz")

    # ✅ Preprocess everything with SAME stats
    X_train = preprocess_2ch(Xlen_train, Xiat_train, stats)
    X_val   = preprocess_2ch(Xlen_val,   Xiat_val,   stats)
    X_test  = preprocess_2ch(Xlen_te,    Xiat_te,    stats)

    # ✅ Step 3: compute alpha from TRAIN SPLIT counts (not full train file)
    alpha = compute_alpha_from_counts(y_train, cap=3.0)
    print("✅ Focal alpha:", alpha.tolist())

    model = build_model(alpha)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-5),
    ]

    # ✅ IMPORTANT: no class_weight when using focal loss
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_te, verbose=0)
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    # ✅ Save WITHOUT compile info to avoid FastAPI load errors
    model.save("models/deep_learning/parallel_cnn_nin.keras", include_optimizer=False)
    print("✅ Saved model: models/deep_learning/parallel_cnn_nin.keras")


if __name__ == "__main__":
    main()
