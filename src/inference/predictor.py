import numpy as np
import tensorflow as tf

CLASS_NAMES = ["vpn", "nonvpn", "tor"]

class Predictor:
    def __init__(self, model_path="models/deep_learning/parallel_cnn_nin.keras"):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        stats = np.load("models/deep_learning/norm_stats.npz")

        self.len_clip = stats["len_clip"]
        self.iat_clip = stats["iat_clip"]
        self.len_mean = stats["len_mean"]
        self.len_std  = stats["len_std"]
        self.iat_mean = stats["iat_mean"]
        self.iat_std  = stats["iat_std"]

    def predict(self, x_len, x_iat):
        x_len = np.array(x_len, dtype=np.float32)
        x_iat = np.array(x_iat, dtype=np.float32)

        x_len = np.clip(x_len, 0, self.len_clip)
        x_iat = np.clip(x_iat, 0, self.iat_clip)

        x_len = (x_len - self.len_mean) / (self.len_std + 1e-9)
        x_iat = (x_iat - self.iat_mean) / (self.iat_std + 1e-9)

        # ✅ STEP 4 FIX
        X = np.stack([x_len, x_iat], axis=-1)[None, ...]

        probs = self.model.predict(X, verbose=0)[0]
        pred = int(np.argmax(probs))

        return {
            "predicted_class": CLASS_NAMES[pred],
            "predicted_id": pred,
            "confidence": float(probs[pred]),
            "probabilities": {
                "vpn": float(probs[0]),
                "nonvpn": float(probs[1]),
                "tor": float(probs[2]),
            }
        }
