import cv2
import numpy as np
from sklearn.cluster import KMeans

from .utils import wrap_colors


def extract_lab_kmeans(img, k=5):
    """
    Baseline 2:
    Plain LAB KMeans (mean L, no depth, no saliency)
    """

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    pixels = lab.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(pixels)

    centers_lab = kmeans.cluster_centers_

    centers_rgb = cv2.cvtColor(
        centers_lab.astype("uint8").reshape(1, -1, 3), cv2.COLOR_LAB2RGB
    )[0]

    return wrap_colors(centers_rgb.astype(int).tolist())
