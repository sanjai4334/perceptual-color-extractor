import cv2
import numpy as np


def get_texture_map(img):
    """
    Compute texture richness map using Laplacian magnitude.

    Args:
        img (np.ndarray): RGB image (H, W, 3), uint8

    Returns:
        np.ndarray: texture map (H, W), float32 in [0, 1]
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Laplacian for texture / edge strength
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    texture = np.abs(laplacian)

    # Normalize to [0, 1]
    tex_min = texture.min()
    tex_max = texture.max()
    texture = (texture - tex_min) / (tex_max - tex_min + 1e-8)

    return texture
