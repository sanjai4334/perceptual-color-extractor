import os
import numpy as np
import pandas as pd
from skimage.color import rgb2lab, deltaE_cie76
from PIL import Image

from backend.extractor.depth import get_depth_map
from backend.extractor.saliency import get_saliency_map

from backend.baselines.rgb_kmeans import extract_rgb_kmeans
from backend.baselines.lab_kmeans import extract_lab_kmeans
from backend.baselines.lab_depth import extract_lab_depth
from backend.baselines.lab_depth_saliency import extract_lab_depth_saliency
from backend.baselines.perceptual_ab_median_l import extract_perceptual_ab_median_l

# ---------------------------
# CONFIG
# ---------------------------
DATASET_DIR = "dataset"
CSV_PATH = "dataset/datasetColors.csv"
DELTA_E_THRESHOLD = 20

methods = [
    "rgb_kmeans",
    "lab_kmeans",
    "lab_depth",
    "lab_depth_saliency",
    "perceptual_final"
]

# ---------------------------
# helpers
# ---------------------------
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)])


def delta_e(c1, c2):
    lab1 = rgb2lab(np.uint8([[c1]]) / 255.0)
    lab2 = rgb2lab(np.uint8([[c2]]) / 255.0)
    return deltaE_cie76(lab1, lab2)[0][0]


def color_match(pred, gt):
    return delta_e(pred, gt) < DELTA_E_THRESHOLD


def evaluate(predicted, gt_colors):
    top1 = any(color_match(predicted[0], gt) for gt in gt_colors)
    top3 = any(color_match(p, gt) for p in predicted[:3] for gt in gt_colors)
    top5 = any(color_match(p, gt) for p in predicted[:5] for gt in gt_colors)

    per_gt_errors = []
    for gt in gt_colors:
        best = min(delta_e(p, gt) for p in predicted[:5])
        per_gt_errors.append(best)

    return top1, top3, top5, np.mean(per_gt_errors)


# ---------------------------
# load ground truth
# ---------------------------
df = pd.read_csv(CSV_PATH)

results = {
    m: {"top1": 0, "top3": 0, "top5": 0, "delta": []}
    for m in methods
}

total = len(df)

# ---------------------------
# evaluation loop
# ---------------------------
for _, row in df.iterrows():
    path = os.path.join(DATASET_DIR, row["image"])

    image = Image.open(path).convert("RGB")
    image = image.resize((300, 300))
    img = np.array(image)

    depth = get_depth_map(img)
    saliency = get_saliency_map(img)

    gt_colors = []
    for col in ["primary", "secondary", "third"]:
        if col in row and pd.notna(row[col]):
            gt_colors.append(hex_to_rgb(row[col]))

    if len(gt_colors) == 0:
        continue

    # ---- RGB ----
    colors = extract_rgb_kmeans(img, k=5)["rgb"]
    t1, t3, t5, de = evaluate(colors, gt_colors)
    results["rgb_kmeans"]["top1"] += t1
    results["rgb_kmeans"]["top3"] += t3
    results["rgb_kmeans"]["top5"] += t5
    results["rgb_kmeans"]["delta"].append(de)

    # ---- LAB ----
    colors = extract_lab_kmeans(img, k=5)["rgb"]
    t1, t3, t5, de = evaluate(colors, gt_colors)
    results["lab_kmeans"]["top1"] += t1
    results["lab_kmeans"]["top3"] += t3
    results["lab_kmeans"]["top5"] += t5
    results["lab_kmeans"]["delta"].append(de)

    # ---- LAB + DEPTH ----
    colors = extract_lab_depth(img, depth, k=5)["rgb"]
    t1, t3, t5, de = evaluate(colors, gt_colors)
    results["lab_depth"]["top1"] += t1
    results["lab_depth"]["top3"] += t3
    results["lab_depth"]["top5"] += t5
    results["lab_depth"]["delta"].append(de)

    # ---- LAB + DEPTH + SALIENCY ----
    colors = extract_lab_depth_saliency(img, depth, saliency, k=5)["rgb"]
    t1, t3, t5, de = evaluate(colors, gt_colors)
    results["lab_depth_saliency"]["top1"] += t1
    results["lab_depth_saliency"]["top3"] += t3
    results["lab_depth_saliency"]["top5"] += t5
    results["lab_depth_saliency"]["delta"].append(de)

    # ---- FINAL METHOD ----
    colors = extract_perceptual_ab_median_l(img, depth, saliency, k=5)["rgb"]
    t1, t3, t5, de = evaluate(colors, gt_colors)
    results["perceptual_final"]["top1"] += t1
    results["perceptual_final"]["top3"] += t3
    results["perceptual_final"]["top5"] += t5
    results["perceptual_final"]["delta"].append(de)

# ---------------------------
# print results
# ---------------------------
print("\n===== COMPARISON RESULTS =====\n")

for m in methods:
    print(m.upper())
    print(f" Top-1 Accuracy : {(results[m]['top1']/total)*100:.2f}%")
    print(f" Top-3 Recall   : {(results[m]['top3']/total)*100:.2f}%")
    print(f" Top-5 Recall   : {(results[m]['top5']/total)*100:.2f}%")
    print(f" Avg ΔE Error   : {np.mean(results[m]['delta']):.2f}\n")