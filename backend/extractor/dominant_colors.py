import cv2
import numpy as np
from sklearn.cluster import KMeans


def extract_dominant_colors(img, importance_map, k=5):
    """
    Extract dominant colors using importance-weighted clustering.

    Args:
        img (np.ndarray): RGB image (H, W, 3), uint8
        importance_map (np.ndarray): spatial importance (H, W), float32
        k (int): number of dominant colors

    Returns:
        dict:
            {
              "rgb": List[List[int]],
              "hex": List[str]
            }
    """

    # Convert image to LAB for perceptual clustering
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape((-1, 3))
    importance = importance_map.reshape((-1, 1))

    # KMeans clustering (no weighting in fit; weight in scoring)
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    # Score clusters using spatial importance
    scores = []
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            scores.append(0.0)
            continue

        # Mean importance × cluster size (same as notebook)
        score = float(np.mean(importance[mask]) * np.sum(mask))
        scores.append(score)

    # Rank clusters by score (descending)
    order = np.argsort(scores)[::-1]
    dominant_lab = centers[order]

    # Convert LAB → RGB
    dominant_rgb = cv2.cvtColor(
        dominant_lab.astype("uint8").reshape(1, -1, 3),
        cv2.COLOR_LAB2RGB
    )[0]

    # Prepare outputs
    rgb_colors = dominant_rgb.astype(int).tolist()
    hex_colors = [_rgb_to_hex(c) for c in rgb_colors]

    return {
        "rgb": rgb_colors,
        "hex": hex_colors
    }


def _rgb_to_hex(rgb):
    """Convert [R, G, B] to hex string."""
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])
