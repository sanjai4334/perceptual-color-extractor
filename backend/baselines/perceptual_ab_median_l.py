import cv2
import numpy as np
from sklearn.cluster import KMeans

from .utils import wrap_colors


def extract_perceptual_ab_median_l(img, depth, saliency, k=5):
    """
    FINAL METHOD:
    AB-only clustering + median L
    Importance = depth² × saliency
    """

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3)

    L = pixels[:, 0]
    AB = pixels[:, 1:3]

    importance = (depth**2) * saliency
    importance = importance.reshape(-1)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    labels = kmeans.fit_predict(AB)
    centers_ab = kmeans.cluster_centers_

    colors_lab = []
    scores = []

    for i in range(k):
        mask = labels == i
        if not np.any(mask):
            continue

        L_med = np.median(L[mask])
        a, b = centers_ab[i]

        colors_lab.append([L_med, a, b])

        score = float(np.mean(importance[mask]) * np.sum(mask))
        scores.append(score)

    order = np.argsort(scores)[::-1]
    dominant_lab = np.array(colors_lab)[order]

    dominant_rgb = cv2.cvtColor(
        dominant_lab.astype("uint8").reshape(1, -1, 3), cv2.COLOR_LAB2RGB
    )[0]

    return wrap_colors(dominant_rgb.astype(int).tolist())
