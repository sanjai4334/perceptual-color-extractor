import cv2
import numpy as np
from sklearn.cluster import KMeans


def extract_dominant_colors(img, importance_map, k=5):
    """
    Dominant color extraction with:
    - AB-only clustering (hue preserved)
    - Median L (shadow robust)
    - Importance-based ranking
    """

    # RGB → LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3)

    L = pixels[:, 0]
    AB = pixels[:, 1:3]

    importance = importance_map.reshape(-1)

    # --- KMeans on AB only ---
    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(AB)
    centers_ab = kmeans.cluster_centers_

    colors_lab = []
    scores = []

    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            continue

        # Median L → resistant to dark/shadow pixels
        L_med = np.median(L[mask])

        a, b = centers_ab[i]
        colors_lab.append([L_med, a, b])

        # Importance-weighted score
        score = float(np.mean(importance[mask]) * np.sum(mask))
        scores.append(score)

    # Rank by importance
    order = np.argsort(scores)[::-1]
    dominant_lab = np.array(colors_lab)[order]

    # LAB → RGB
    dominant_rgb = cv2.cvtColor(
        dominant_lab.astype("uint8").reshape(1, -1, 3), cv2.COLOR_LAB2RGB
    )[0]

    rgb_colors = dominant_rgb.astype(int).tolist()
    hex_colors = [_rgb_to_hex(c) for c in rgb_colors]

    return {"rgb": rgb_colors, "hex": hex_colors}


def _rgb_to_hex(rgb):
    """Convert [R, G, B] to hex string."""
    return "#{:02X}{:02X}{:02X}".format(rgb[0], rgb[1], rgb[2])
