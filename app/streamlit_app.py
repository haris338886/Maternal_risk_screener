"""Streamlit UI for the maternal risk screener."""

import os
import sys

import streamlit as st

# Ensure project root on path so `from src...` works under `streamlit run`
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import FEATURE_RANGES, FEATURE_ORDER
from src.data_loader import validate_input
from src.predict import predict_risk

st.set_page_config(page_title="Maternal Risk Screener", layout="centered")

st.title("🤰 Maternal Risk Screener")

# Disclaimer as markdown (testable + always visible)
st.markdown(
    "> ⚠️ **Disclaimer**: This is a screening aid only. "
    "It does not replace a qualified clinician's assessment. "
    "Always correlate with clinical findings and local protocols."
)

# --- Input form ---
with st.form("vitals_form"):
    st.subheader("Vital Signs")
    col1, col2 = st.columns(2)

    features = {}
    for i, name in enumerate(FEATURE_ORDER):
        lo, hi = FEATURE_RANGES[name]
        col = col1 if i % 2 == 0 else col2
        with col:
            step = 0.1 if name in ("BS", "BodyTemp") else 1.0
            features[name] = st.number_input(
                name,
                min_value=float(lo),
                max_value=float(hi),
                value=float(round((lo + hi) / 2, 1)),
                step=step,
                key=f"input_{name}",
            )

    submitted = st.form_submit_button("Assess Risk", type="primary")

# --- Result ---
if submitted:
    valid, msg = validate_input(features)
    if not valid:
        st.error(f"Invalid input: {msg}")
    else:
        result = predict_risk(features)

        emoji = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[result["traffic_light"]]
        label = {
            "green": "Low concern",
            "yellow": "Moderate concern",
            "red": "High concern",
        }[result["traffic_light"]]

        st.markdown("---")
        st.subheader("Screening Result")
        st.markdown(f"## {emoji} {label}")
        st.write(f"**Predicted risk category:** {result['risk_level']}")

        st.write("**Estimated probabilities:**")
        for lbl in ["low risk", "mid risk", "high risk"]:
            st.progress(
                result["probabilities"][lbl],
                text=f"{lbl}: {result['probabilities'][lbl]:.1%}",
            )

        st.info(
            "**Clinical note:** A yellow/red flag suggests the patient may need attention "
            "urgent review or referral. Use your clinical judgment."
        )