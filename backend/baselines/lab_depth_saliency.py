import cv2
import numpy as np
from sklearn.cluster import KMeans

from .utils import wrap_colors


def extract_lab_depth_saliency(img, depth, saliency, k=5):
    """
    Baseline 4:
    LAB KMeans + depth² × saliency (mean L)
    """

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3)

    importance = (depth**2) * saliency
    importance = importance.reshape(-1)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_

    scores = []
    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            scores.append(0.0)
            continue

        score = float(np.mean(importance[mask]) * np.sum(mask))
        scores.append(score)

    order = np.argsort(scores)[::-1]
    dominant_lab = centers[order]

    dominant_rgb = cv2.cvtColor(
        dominant_lab.astype("uint8").reshape(1, -1, 3), cv2.COLOR_LAB2RGB
    )[0]

    return wrap_colors(dominant_rgb.astype(int).tolist())
