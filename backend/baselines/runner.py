from .rgb_kmeans import extract_rgb_kmeans
from .lab_kmeans import extract_lab_kmeans
from .lab_depth import extract_lab_depth
from .lab_depth_saliency import extract_lab_depth_saliency
from .perceptual_ab_median_l import extract_perceptual_ab_median_l


def run_all_baselines(img, depth, saliency, k=5):
    """
    Runs all baseline color extraction methods
    on the same preprocessed image.

    Returns a dict keyed by method name.
    """

    return {
        "rgb_kmeans": {
            "label": "RGB K-Means",
            "colors": extract_rgb_kmeans(img, k=k),
        },

        "lab_kmeans": {
            "label": "LAB K-Means",
            "colors": extract_lab_kmeans(img, k=k),
        },

        "lab_depth": {
            "label": "LAB + Depth",
            "colors": extract_lab_depth(img, depth, k=k),
        },

        "lab_depth_saliency": {
            "label": "LAB + Depth + Saliency",
            "colors": extract_lab_depth_saliency(
                img, depth, saliency, k=k
            ),
        },

        "perceptual_ab_median_l": {
            "label": "AB-only + Median L (Ours)",
            "colors": extract_perceptual_ab_median_l(
                img, depth, saliency, k=k
            ),
        },
    }
