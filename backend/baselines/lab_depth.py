import cv2
import numpy as np
from sklearn.cluster import KMeans

from .utils import wrap_colors


def extract_lab_depth(img, depth, k=5):
    """
    Baseline 3:
    LAB KMeans + depth weighting (mean L)
    """

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3)
    depth_flat = depth.reshape(-1)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    scores = []
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            scores.append(0.0)
            continue

        score = float(np.mean(depth_flat[mask]) * np.sum(mask))
        scores.append(score)

    order = np.argsort(scores)[::-1]
    dominant_lab = centers[order]

    dominant_rgb = cv2.cvtColor(
        dominant_lab.astype("uint8").reshape(1, -1, 3), cv2.COLOR_LAB2RGB
    )[0]

    return wrap_colors(dominant_rgb.astype(int).tolist())
