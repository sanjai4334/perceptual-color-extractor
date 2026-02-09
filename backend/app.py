from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from io import BytesIO
from PIL import Image

from backend.extractor.pipeline import ProductColorExtractor
from backend.utils.image_utils import encode_image_to_base64

# -----------------------------
# App initialization
# -----------------------------
app = FastAPI(
    title="Perceptual Color Extractor",
    description="Object-aware dominant color extraction using depth, saliency",
    version="1.0.0",
)

# -----------------------------
# Load pipeline once (CPU)
# -----------------------------
extractor = ProductColorExtractor(device="cpu", image_size=300, k_colors=5)


# -----------------------------
# Helper: safe image handling
# -----------------------------
def process_uploaded_file(file: UploadFile):
    try:
        # Read raw bytes
        contents = file.file.read()
        image_bytes = BytesIO(contents)

        # Validate by actually loading the image
        # (DO NOT use verify())
        Image.open(image_bytes).convert("RGB")

        # Reset pointer for downstream processing
        image_bytes.seek(0)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Run ML pipeline
    result = extractor.process_image(image_bytes)

    # Encode outputs
    return {
        "input_image": encode_image_to_base64(result["input"]),
        "depth_map": encode_image_to_base64(result["depth"], is_gray=True),
        "saliency_map": encode_image_to_base64(result["saliency"], is_gray=True),
        "importance_map": encode_image_to_base64(result["importance"], is_gray=True),
        "dominant_colors": result["dominant_colors"],
    }


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def root():
    return {"message": "Perceptual Color Extractor API is running", "docs": "/docs"}


@app.post("/extract")
def extract_single_image(file: UploadFile = File(...)):
    """
    Process a single image.
    """
    result = process_uploaded_file(file)
    return JSONResponse(content=result)


@app.post("/extract-batch")
def extract_multiple_images(files: List[UploadFile] = File(...)):
    """
    Process multiple images.
    """
    results = []

    for file in files:
        try:
            output = process_uploaded_file(file)
            results.append({"filename": file.filename, "result": output})
        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return JSONResponse(content=results)
