from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from io import BytesIO
from PIL import Image
import numpy as np

from backend.extractor.pipeline import ProductColorExtractor
from backend.utils.image_utils import encode_image_to_base64
from backend.extractor.grouping import group_by_primary_color
from backend.baselines.runner import run_all_baselines


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


@app.post("/group")
def group_images(files: List[UploadFile] = File(...)):
    """
    Process multiple images and group them based on
    primary dominant color (LAB A,B).
    """

    processed = []

    for file in files:
        try:
            # --- Validate & load ---
            contents = file.file.read()
            image_bytes = BytesIO(contents)
            Image.open(image_bytes).convert("RGB")
            image_bytes.seek(0)

            # --- Run pipeline ---
            result = extractor.process_image(image_bytes)

            processed.append(
                {
                    "filename": file.filename,
                    "ab": result["primary_ab"],
                    "result": {
                        "input_image": encode_image_to_base64(result["input"]),
                        "depth_map": encode_image_to_base64(
                            result["depth"], is_gray=True
                        ),
                        "saliency_map": encode_image_to_base64(
                            result["saliency"], is_gray=True
                        ),
                        "importance_map": encode_image_to_base64(
                            result["importance"], is_gray=True
                        ),
                        "dominant_colors": result["dominant_colors"],
                    },
                }
            )

        except Exception as e:
            processed.append({"filename": file.filename, "error": str(e)})

    # --- Group by perceptual color ---
    groups = group_by_primary_color(processed)

    # --- Prepare response ---
    response = []
    for idx, group in enumerate(groups):
        response.append(
            {
                "group_id": idx + 1,
                "group_ab": group["ab"],
                "images": [
                    {"filename": item["filename"], "result": item["result"]}
                    for item in group["items"]
                ],
            }
        )

    return JSONResponse(content=response)


@app.post("/compare")
def compare_baselines(file: UploadFile = File(...)):
    """
    Compare dominant color extraction baselines
    on a single image.
    """

    try:
        # ---- Read & validate image ----
        contents = file.file.read()
        image_bytes = BytesIO(contents)

        image = Image.open(image_bytes).convert("RGB")
        image = image.resize((extractor.image_size, extractor.image_size))
        img = np.array(image)

    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # ---- Run shared depth + saliency ONCE ----
    image_bytes.seek(0)
    pipeline_out = extractor.process_image(BytesIO(contents))

    depth = pipeline_out["depth"]
    saliency = pipeline_out["saliency"]

    # ---- Run baselines ----
    baselines = run_all_baselines(
        img=img,
        depth=depth,
        saliency=saliency,
        k=extractor.k_colors,
    )

    # ---- Response ----
    return JSONResponse(
        content={
            "input_image": encode_image_to_base64(img),
            "baselines": baselines,
        }
    )
