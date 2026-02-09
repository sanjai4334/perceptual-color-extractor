import streamlit as st
import requests
import base64
from io import BytesIO
from PIL import Image

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Color Grouping", layout="wide")

st.title("🧩 Color-Based Image Grouping")
st.caption("Images grouped by primary perceptual color (LAB A,B)")

uploaded_files = st.file_uploader(
    "Upload multiple images (same batch)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)


def decode_image(b64):
    return Image.open(BytesIO(base64.b64decode(b64)))


if uploaded_files:
    if len(uploaded_files) < 2:
        st.warning("Upload at least 2 images for grouping.")
        st.stop()

    if st.button("Submit & Group Images"):
        with st.spinner("Processing and grouping images..."):
            files = [("files", (f.name, f.getvalue(), f.type)) for f in uploaded_files]

            response = requests.post(f"{API_URL}/group", files=files)

        if response.status_code != 200:
            st.error("API error during grouping")
            st.stop()

        groups = response.json()

        st.success(f"Grouped into {len(groups)} color groups")

        # -------- Render groups --------
        for group in groups:
            st.markdown("---")

            group_id = group["group_id"]
            images = group["images"]

            # Use first image’s primary color as group swatch
            primary_hex = images[0]["result"]["dominant_colors"]["hex"][0]

            st.markdown(
                f"""
                <div style="display:flex;align-items:center;gap:12px;">
                    <div style="width:28px;height:28px;
                                background:{primary_hex};
                                border-radius:6px;"></div>
                    <h3 style="margin:0;">Group {group_id}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # One row, equal-sized thumbnails
            cols = st.columns(len(images))

            for col, item in zip(cols, images):
                with col:
                    img = decode_image(item["result"]["input_image"])
                    st.image(
                        img,
                        width=160,  # fixed size for clean grid
                        caption=item["filename"],
                    )
