import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Baseline Comparison",
    layout="wide",
)

st.title("🧪 Dominant Color – Baseline Comparison")
st.caption("Compare different color extraction strategies on the same image")

uploaded_file = st.file_uploader(
    "Upload a single image",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)


def decode_image(b64):
    return Image.open(BytesIO(base64.b64decode(b64)))


def render_colors(hex_colors):
    for hex_code in hex_colors:
        st.markdown(
            f"""
            <div style="
                background:{hex_code};
                height:36px;
                border-radius:6px;
                margin-bottom:6px;
                border:1px solid #333;
            "></div>
            """,
            unsafe_allow_html=True,
        )


if uploaded_file:
    with st.spinner("Running baseline comparisons…"):
        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }

        response = requests.post(f"{API_URL}/compare", files=files)

    if response.status_code != 200:
        st.error("API error")
        st.stop()

    data = response.json()

    # -------------------------
    # Input image
    # -------------------------
    st.subheader("Input Image")
    st.image(decode_image(data["input_image"]), width=280)

    st.markdown("---")

    # -------------------------
    # Baseline results
    # -------------------------
    st.subheader("Baseline Methods")

    baselines = data["baselines"]

    cols = st.columns(len(baselines))

    for col, (method, result) in zip(cols, baselines.items()):
        with col:
            st.markdown(f"### {method.replace('_', ' ').title()}")

            render_colors(result["colors"]["hex"])
