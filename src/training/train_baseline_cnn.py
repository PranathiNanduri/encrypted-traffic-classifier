import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

MAX_LEN = 50
NUM_CLASSES = 3

def load_data():
    train = np.load("data/processed/train.npz")
    test = np.load("data/processed/test.npz")

    X_train = train["X_len"]
    y_train = train["y"]
    X_test = test["X_len"]
    y_test = test["y"]

    # normalize
    X_train = X_train / (X_train.max() + 1e-9)
    X_test = X_test / (X_test.max() + 1e-9)

    # add channel dim for CNN: (N, 50, 1)
    X_train = X_train[..., None]
    X_test = X_test[..., None]

    return X_train, y_train, X_test, y_test

def build_model():
    model = models.Sequential([
        layers.Input(shape=(MAX_LEN, 1)),
        layers.Conv1D(64, 5, activation="relu"),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation="relu"),
        layers.GlobalMaxPooling1D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    X_train, y_train, X_test, y_test = load_data()
    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    ]

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"✅ Baseline CNN Test Accuracy: {acc:.4f}")

    model.save("models/deep_learning/baseline_cnn.h5")
    print("✅ Saved: models/deep_learning/baseline_cnn.h5")

if __name__ == "__main__":
    main()
