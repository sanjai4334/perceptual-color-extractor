import numpy as np
from sklearn.cluster import KMeans

from .utils import wrap_colors


def extract_rgb_kmeans(img, k=5):
    """
    Baseline 1:
    Plain RGB KMeans (no depth, no saliency, no LAB)
    """

    pixels = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=42)
    kmeans.fit(pixels)

    centers = kmeans.cluster_centers_.astype(int).tolist()

    return wrap_colors(centers)
