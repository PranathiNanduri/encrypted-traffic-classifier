import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

MAX_LEN = 50
NUM_CLASSES = 3

def load_data():
    train = np.load("data/processed/train.npz")
    test = np.load("data/processed/test.npz")

    Xlen_tr, Xiat_tr, y_tr = train["X_len"], train["X_iat"], train["y"]
    Xlen_te, Xiat_te, y_te = test["X_len"], test["X_iat"], test["y"]

    # normalize separately (safer)
    Xlen_tr = Xlen_tr / (Xlen_tr.max() + 1e-9)
    Xlen_te = Xlen_te / (Xlen_te.max() + 1e-9)

    # iat can have large values; use log transform + normalize
    Xiat_tr = np.log1p(Xiat_tr)
    Xiat_te = np.log1p(Xiat_te)
    Xiat_tr = Xiat_tr / (Xiat_tr.max() + 1e-9)
    Xiat_te = Xiat_te / (Xiat_te.max() + 1e-9)

    # add channels
    Xlen_tr = Xlen_tr[..., None]
    Xlen_te = Xlen_te[..., None]
    Xiat_tr = Xiat_tr[..., None]
    Xiat_te = Xiat_te[..., None]

    return (Xlen_tr, Xiat_tr, y_tr), (Xlen_te, Xiat_te, y_te)

def nin_block(x, filters, kernel_size):
    # "MLP conv": Conv -> 1x1 Conv -> 1x1 Conv
    x = layers.Conv1D(filters, kernel_size, padding="same", activation="relu")(x)
    x = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)
    x = layers.Conv1D(filters, 1, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    return x

def build_parallel_model():
    # Branch 1: packet length
    inp_len = layers.Input(shape=(MAX_LEN, 1), name="pkt_len")
    x1 = nin_block(inp_len, 64, 5)
    x1 = layers.MaxPooling1D(2)(x1)
    x1 = nin_block(x1, 128, 3)
    x1 = layers.GlobalMaxPooling1D()(x1)

    # Branch 2: inter-arrival times
    inp_iat = layers.Input(shape=(MAX_LEN, 1), name="pkt_iat")
    x2 = nin_block(inp_iat, 64, 5)
    x2 = layers.MaxPooling1D(2)(x2)
    x2 = nin_block(x2, 128, 3)
    x2 = layers.GlobalMaxPooling1D()(x2)

    # Fusion
    fused = layers.Concatenate()([x1, x2])
    fused = layers.Dense(256, activation="relu")(fused)
    fused = layers.Dropout(0.35)(fused)
    fused = layers.Dense(128, activation="relu")(fused)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(fused)

    model = Model(inputs=[inp_len, inp_iat], outputs=out, name="Parallel_CNN_NIN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    (Xlen_tr, Xiat_tr, y_tr), (Xlen_te, Xiat_te, y_te) = load_data()
    model = build_parallel_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5),
    ]

    model.fit(
        {"pkt_len": Xlen_tr, "pkt_iat": Xiat_tr},
        y_tr,
        validation_split=0.2,
        epochs=30,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate({"pkt_len": Xlen_te, "pkt_iat": Xiat_te}, y_te, verbose=0)
    print(f"\n✅ Parallel CNN+NIN Test Accuracy: {acc:.4f}")

    tf.keras.utils.get_file  # keep lint happy

    # Save in modern format
    Path = __import__("pathlib").Path
    Path("models/deep_learning").mkdir(parents=True, exist_ok=True)
    model.save("models/deep_learning/parallel_cnn_nin.keras")
    print("✅ Saved: models/deep_learning/parallel_cnn_nin.keras")

if __name__ == "__main__":
    main()
