import cv2
import base64
import numpy as np


def encode_image_to_base64(image, is_gray=False):
    """
    Encode a numpy image array to base64 PNG.

    Args:
        image (np.ndarray):
            - RGB image: (H, W, 3), uint8
            - Grayscale image: (H, W), float or uint8
        is_gray (bool): set True for depth/saliency/texture/importance maps

    Returns:
        str: base64-encoded PNG image
    """

    if is_gray:
        # Normalize float maps to [0, 255]
        if image.dtype != np.uint8:
            img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
            img_uint8 = (img_norm * 255).astype(np.uint8)
        else:
            img_uint8 = image

        # Encode grayscale PNG
        success, buffer = cv2.imencode(".png", img_uint8)
    else:
        # Ensure uint8 RGB
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Convert RGB → BGR for OpenCV
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        success, buffer = cv2.imencode(".png", image_bgr)

    if not success:
        raise RuntimeError("Failed to encode image")

    # Convert to base64 string
    encoded = base64.b64encode(buffer).decode("utf-8")
    return encoded
