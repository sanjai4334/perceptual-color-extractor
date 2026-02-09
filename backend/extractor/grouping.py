# backend/extractor/grouping.py

import numpy as np
from typing import List, Dict


def ab_distance(ab1, ab2):
    return np.linalg.norm(np.array(ab1) - np.array(ab2))


def group_by_primary_color(items: List[Dict], threshold: float = 20.0):
    """
    items: [
      {
        "filename": str,
        "ab": [a, b],
        "result": full_extraction_result
      }
    ]
    """

    groups = []

    for item in items:
        placed = False

        for group in groups:
            ref_ab = group["ab"]
            if ab_distance(item["ab"], ref_ab) < threshold:
                group["items"].append(item)
                placed = True
                break

        if not placed:
            groups.append({"ab": item["ab"], "items": [item]})

    return groups
