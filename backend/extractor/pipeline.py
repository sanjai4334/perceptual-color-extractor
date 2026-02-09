import cv2
import numpy as np
from PIL import Image

from .depth import get_depth_map
from .saliency import get_saliency_map
from .dominant_colors import extract_dominant_colors


class ProductColorExtractor:
    """
    Composite inference pipeline:
    RGB -> Depth + Saliency -> Importance -> Dominant Colors
    """

    def __init__(self, device="cpu", image_size=300, k_colors=5):
        self.device = device
        self.image_size = image_size
        self.k_colors = k_colors

    def process_image(self, image_bytes):
        """
        Input:
            image_bytes (bytes): raw uploaded image bytes

        Output:
            dict with all intermediate maps + dominant colors + primary_ab
        """

        # ================= Load image =================
        image = Image.open(image_bytes).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        img = np.array(image)

        # ================= Depth =================
        depth = get_depth_map(img)

        # ================= Saliency =================
        saliency = get_saliency_map(img)

        # ================= Importance Fusion =================
        importance = (depth ** 2) * saliency
        importance = np.clip(importance, 0, 1)

        # ================= Dominant Colors =================
        dominant_colors = extract_dominant_colors(
            img,
            importance,
            k=self.k_colors
        )

        # ================= Primary AB extraction =================
        # Take the top dominant color
        primary_rgb = dominant_colors["rgb"][0]

        # Convert RGB → LAB
        lab = cv2.cvtColor(
            np.array(primary_rgb, dtype=np.uint8).reshape(1, 1, 3),
            cv2.COLOR_RGB2LAB
        )[0][0]

        primary_ab = [int(lab[1]), int(lab[2])]

        return {
            "input": img,
            "depth": depth,
            "saliency": saliency,
            "importance": importance,
            "dominant_colors": dominant_colors,
            "primary_ab": primary_ab
        }
