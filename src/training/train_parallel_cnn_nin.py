import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from collections import Counter
from pathlib import Path
from sklearn.model_selection import train_test_split

MAX_LEN = 50
NUM_CLASSES = 3

# ============================================================
# FOCAL LOSS (MULTI-CLASS) — STABLE VERSION
# ============================================================
def focal_loss(alpha, gamma=2.0, num_classes=3):
    """
    alpha: array/list shape (num_classes,)
    gamma: 2.0 recommended (3.0 is often too harsh)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=num_classes)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1)  # prob of true class
        alpha_t = tf.gather(alpha, y_true)                # alpha for true class

        return tf.reduce_mean(-alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t))

    return loss


# ============================================================
# NORMALIZATION (train-only stats)
# ============================================================
def compute_norm_stats(X_len, X_iat):
    len_clip = np.percentile(X_len, 99.5).astype(np.float32)
    iat_clip = np.percentile(X_iat, 99.5).astype(np.float32)

    Xl = np.clip(X_len, 0, len_clip)
    Xi = np.clip(X_iat, 0, iat_clip)

    return {
        "len_clip": len_clip,
        "iat_clip": iat_clip,
        "len_mean": Xl.mean().astype(np.float32),
        "len_std":  Xl.std().astype(np.float32) + 1e-9,
        "iat_mean": Xi.mean().astype(np.float32),
        "iat_std":  Xi.std().astype(np.float32) + 1e-9,
    }


def preprocess_2ch(X_len, X_iat, stats):
    X_len = X_len.astype("float32")
    X_iat = X_iat.astype("float32")

    X_len = np.clip(X_len, 0, stats["len_clip"])
    X_iat = np.clip(X_iat, 0, stats["iat_clip"])

    X_len = (X_len - stats["len_mean"]) / stats["len_std"]
    X_iat = (X_iat - stats["iat_mean"]) / stats["iat_std"]

    return np.stack([X_len, X_iat], axis=-1).astype("float32")


# ============================================================
# MODEL
# ============================================================
def nin_block(x, filters, kernel):
    x = layers.Conv1D(filters, kernel, padding="same", activation="relu")(x)
    x = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)
    x = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x


def build_model(alpha):
    inp = layers.Input((MAX_LEN, 2), name="flow_features")

    x = nin_block(inp, 64, 5)
    x = layers.MaxPooling1D(2)(x)

    x = nin_block(x, 128, 3)
    x = layers.GlobalMaxPooling1D()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.25)(x)  # ✅ reduced dropout for stability
    x = layers.Dense(128, activation="relu")(x)

    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inp, out, name="CNN_NIN_2CH")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=focal_loss(alpha=alpha, gamma=2.0, num_classes=NUM_CLASSES),  # ✅ gamma lowered
        metrics=["accuracy"]
    )
    return model


# ============================================================
# TRAINING
# ============================================================
def main():
    train = np.load("data/processed/train.npz")
    test  = np.load("data/processed/test.npz")

    Xlen, Xiat, y = train["X_len"], train["X_iat"], train["y"]
    Xlen_te, Xiat_te, y_te = test["X_len"], test["X_iat"], test["y"]

    print("✅ Train distribution:", Counter(y.tolist()))
    print("✅ Test distribution :", Counter(y_te.tolist()))

    Xl_tr, Xl_val, Xi_tr, Xi_val, y_tr, y_val = train_test_split(
        Xlen, Xiat, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Train-only normalization stats
    stats = compute_norm_stats(Xl_tr, Xi_tr)
    Path("models/deep_learning").mkdir(parents=True, exist_ok=True)
    np.savez("models/deep_learning/norm_stats.npz", **stats)
    print("✅ Saved stats: models/deep_learning/norm_stats.npz")

    X_tr  = preprocess_2ch(Xl_tr,  Xi_tr,  stats)
    X_val = preprocess_2ch(Xl_val, Xi_val, stats)
    X_te  = preprocess_2ch(Xlen_te, Xiat_te, stats)

    # ========================================================
    # ✅ STEP-4 FIX: MANUAL ALPHA (stronger for vpn/tor)
    # ========================================================
    alpha = np.array([4.0, 0.5, 3.0], dtype=np.float32)
    print("✅ Using focal alpha:", alpha.tolist())

    model = build_model(alpha)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5),
    ]

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=80,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\n✅ Test Accuracy: {acc:.4f}")

    # Save without optimizer/compile info (prevents FastAPI load errors)
    model.save("models/deep_learning/parallel_cnn_nin.keras", include_optimizer=False)
    print("✅ Model saved: models/deep_learning/parallel_cnn_nin.keras")


if __name__ == "__main__":
    main()
