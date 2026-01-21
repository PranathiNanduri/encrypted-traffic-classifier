import json
import numpy as np
import requests

API_URL = "http://127.0.0.1:8000/predict"

def main():
    test = np.load("data/processed/test.npz")

    # pick any sample index
    i = 0
    x_len = test["X_len"][i].tolist()
    x_iat = test["X_iat"][i].tolist()
    y_true = int(test["y"][i])

    payload = {"x_len": x_len, "x_iat": x_iat}

    r = requests.post(API_URL, json=payload, timeout=30)
    r.raise_for_status()

    print("True label index:", y_true)
    print("Response:\n", json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    main()
