import streamlit as st
import numpy as np
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"
CLASS_NAMES = ["vpn", "nonvpn", "tor"]

st.set_page_config(page_title="Encrypted Traffic Classifier", layout="centered")

st.title("🔐 Encrypted Network Traffic Classification")
st.write("This dashboard sends flow features to a FastAPI model endpoint and shows predictions.")

@st.cache_data
def load_test_data():
    test = np.load("data/processed/test.npz")
    return test["X_len"], test["X_iat"], test["y"]

X_len, X_iat, y = load_test_data()

st.subheader("1) Select a test sample")
idx = st.slider("Sample index", 0, len(y) - 1, 0)

true_label = int(y[idx])
st.write(f"**True label:** {CLASS_NAMES[true_label]} (class id: {true_label})")

with st.expander("View input sequences (first 10 values)"):
    st.write("Packet lengths (first 10):", X_len[idx][:10].tolist())
    st.write("Inter-arrival times (first 10):", X_iat[idx][:10].tolist())

st.subheader("2) Predict using API")

if st.button("🔎 Predict"):
    payload = {
        "x_len": X_len[idx].reshape(-1).tolist(),
        "x_iat": X_iat[idx].reshape(-1).tolist()
    }

    try:
        resp = requests.post(API_URL, json=payload, timeout=30)
        if resp.status_code != 200:
            st.error(f"API Error {resp.status_code}: {resp.text}")
        else:
            result = resp.json()

            # ✅ supports both API formats
            pred = result.get("predicted_class") or result.get("predicted_label") or result.get("predicted")
            if pred is None:
                st.error(f"Unexpected API response keys: {list(result.keys())}")
                st.json(result)
                st.stop()

            st.success(f"✅ Predicted: **{pred}**")

            # ✅ compute confidence if not provided
            probs = result.get("probabilities", {})
            if isinstance(probs, dict) and len(probs) > 0:
                conf = float(max(probs.values()))
                st.info(f"Confidence: **{conf*100:.2f}%**")
            else:
                st.warning("No probabilities returned from API.")

            # probabilities table
            if isinstance(probs, dict) and len(probs) > 0:
                df = pd.DataFrame({
                    "Class": list(probs.keys()),
                    "Probability": list(probs.values())
                }).sort_values("Probability", ascending=False)

                st.subheader("Class Probabilities")
                st.dataframe(df, use_container_width=True)

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API. Is FastAPI running?\n\nError: {e}")

st.markdown("---")
st.caption("Tip: Keep FastAPI running in one terminal and Streamlit in another terminal.")
