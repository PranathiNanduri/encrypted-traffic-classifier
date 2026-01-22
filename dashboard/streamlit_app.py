import streamlit as st
import numpy as np
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"
CLASS_NAMES = ["vpn", "nonvpn", "tor"]

st.set_page_config(page_title="Encrypted Traffic Classifier", layout="centered")

st.title("🔐 Encrypted Network Traffic Classification")
st.write("This dashboard sends flow features to a FastAPI model endpoint and shows predictions.")

# ----------------------------
# Load test dataset
# ----------------------------
@st.cache_data
def load_test_data():
    test = np.load("data/processed/test.npz")
    return test["X_len"], test["X_iat"], test["y"]

X_len, X_iat, y = load_test_data()

# ----------------------------
# STEP 4.1 Requirement: Show input flow + true label + predicted + confidence
# ----------------------------
st.subheader("1) Select a test sample")
idx = st.slider("Sample index", 0, len(y) - 1, 0)

true_id = int(y[idx])
true_name = CLASS_NAMES[true_id]
st.write(f"✅ **True label:** **{true_name}** (class id: {true_id})")

# Input flow preview (mandatory: show input flow)
st.subheader("2) Input flow (length = 50)")
st.caption("Showing first 10 values (expand to see full 50).")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Packet lengths**")
    st.write(X_len[idx][:10].reshape(-1).tolist())
    st.write(f"Length: **{len(X_len[idx].reshape(-1))}**")
with col2:
    st.markdown("**Inter-arrival times (IAT)**")
    st.write(X_iat[idx][:10].reshape(-1).tolist())
    st.write(f"Length: **{len(X_iat[idx].reshape(-1))}**")

with st.expander("View full sequences (all 50 values)"):
    st.write("Packet lengths (50):", X_len[idx].reshape(-1).tolist())
    st.write("IAT (50):", X_iat[idx].reshape(-1).tolist())

# ----------------------------
# Predict
# ----------------------------
st.subheader("3) Predict using API")

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

            # ✅ supports multiple API response formats
            pred_name = (
                result.get("predicted_label")
                or result.get("predicted_class")
                or result.get("predicted")
            )

            probs = result.get("probabilities", {})

            if pred_name is None:
                st.error(f"Unexpected API response keys: {list(result.keys())}")
                st.json(result)
                st.stop()

            # Confidence:
            # - if API provides 'confidence' use it
            # - else if probabilities dict exists, take prob of predicted class (best)
            # - else fallback to max prob if predicted key mismatch
            conf = None
            if "confidence" in result:
                try:
                    conf = float(result["confidence"])
                except Exception:
                    conf = None

            if conf is None and isinstance(probs, dict) and len(probs) > 0:
                if pred_name in probs:
                    conf = float(probs[pred_name])
                else:
                    conf = float(max(probs.values()))

            # ✅ Display mandatory fields
            st.success(f"✅ **Predicted label:** **{pred_name}**")

            if conf is not None:
                st.info(f"📌 **Confidence:** **{conf*100:.2f}%**")
            else:
                st.warning("⚠️ Confidence not available (API did not return probabilities/confidence).")

            # ✅ Show probabilities table (nice-to-have, but useful)
            if isinstance(probs, dict) and len(probs) > 0:
                df = (
                    pd.DataFrame({"Class": list(probs.keys()), "Probability": list(probs.values())})
                    .sort_values("Probability", ascending=False)
                    .reset_index(drop=True)
                )
                st.subheader("4) Class probabilities")
                st.dataframe(df, use_container_width=True)

            # ✅ Quick summary line (helps during demo)
            st.markdown("---")
            match = "✅ Correct" if pred_name == true_name else "❌ Wrong"
            st.write(f"**Summary:** True = **{true_name}** | Pred = **{pred_name}** | {match}")

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to connect to API. Is FastAPI running?\n\nError: {e}")

st.markdown("---")
st.caption("Tip: Keep FastAPI running in one terminal and Streamlit in another terminal.")
