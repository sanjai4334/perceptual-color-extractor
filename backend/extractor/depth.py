import torch
import cv2
import numpy as np
from pathlib import Path

_DEVICE = torch.device("cpu")
_MIDAS_MODEL = None
_MIDAS_TRANSFORM = None


def _load_midas():
    global _MIDAS_MODEL, _MIDAS_TRANSFORM

    if _MIDAS_MODEL is not None:
        return

    try:
        _MIDAS_MODEL = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
        _MIDAS_MODEL.to(_DEVICE).eval()

        _MIDAS_TRANSFORM = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        ).dpt_transform

    except Exception as e:
        raise RuntimeError(
            "❌ MiDaS model download/load failed.\n"
            "➡ Check internet OR run once with stable connection.\n"
            f"Original error: {e}"
        )


def get_depth_map(image_rgb: np.ndarray) -> np.ndarray:
    _load_midas()

    input_batch = _MIDAS_TRANSFORM(image_rgb).to(_DEVICE)

    with torch.no_grad():
        prediction = _MIDAS_MODEL(input_batch)
        depth = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth
