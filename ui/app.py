import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Perceptual Color Extractor", layout="wide")

st.title("🎨 Perceptual Color Extractor")
st.caption("Depth + Saliency aware dominant color extraction")

uploaded_files = st.file_uploader(
    "Upload image(s)", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)


def decode_image(b64):
    return Image.open(BytesIO(base64.b64decode(b64)))


if uploaded_files:
    if len(uploaded_files) == 1:
        file = uploaded_files[0]
        files = {"file": (file.name, file.getvalue(), file.type)}

        response = requests.post(f"{API_URL}/extract", files=files)

        if response.status_code != 200:
            st.error("API error")
            st.stop()

        data = response.json()

        cols = st.columns(5)
        titles = ["Input", "Depth", "Saliency", "Importance", "Colors"]
        images = [
            data["input_image"],
            data["depth_map"],
            data["saliency_map"],
            data["importance_map"],
        ]

        for i in range(4):
            with cols[i]:
                st.subheader(titles[i])
                st.image(decode_image(images[i]), width="stretch")

        with cols[4]:
            st.subheader("Dominant Colors")
            for hex_code in data["dominant_colors"]["hex"]:
                st.markdown(
                    f"<div style='background:{hex_code};height:40px;border-radius:6px;margin-bottom:6px'></div>",
                    unsafe_allow_html=True,
                )

    else:
        files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]

        response = requests.post(f"{API_URL}/extract-batch", files=files)

        if response.status_code != 200:
            st.error("API error")
            st.stop()

        results = response.json()

        for item in results:
            st.markdown("---")
            st.header(item["filename"])

            if "error" in item:
                st.error(item["error"])
                continue

            data = item["result"]

            cols = st.columns(5)
            titles = ["Input", "Depth", "Saliency", "Importance", "Colors"]
            images = [
                data["input_image"],
                data["depth_map"],
                data["saliency_map"],
                data["importance_map"],
            ]

            for i in range(4):
                with cols[i]:
                    st.subheader(titles[i])
                    st.image(decode_image(images[i]), width="stretch")

            with cols[4]:
                st.subheader("Dominant Colors")
                for hex_code in data["dominant_colors"]["hex"]:
                    st.markdown(
                        f"<div style='background:{hex_code};height:40px;border-radius:6px;margin-bottom:6px'></div>",
                        unsafe_allow_html=True,
                    )
