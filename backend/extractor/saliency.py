import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

from backend.models.u2net import U2NETP  # noqa: E402
_DEVICE = torch.device("cpu")

_saliency_net = U2NETP(3, 1)
_saliency_net.load_state_dict(
    torch.load("weights/u2netp.pth", map_location=_DEVICE)
)
_saliency_net.to(_DEVICE)
_saliency_net.eval()

_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),
])


def get_saliency_map(img):
    """
    Compute saliency map using U²Net Lite.

    Args:
        img (np.ndarray): RGB image (H, W, 3), uint8

    Returns:
        np.ndarray: saliency map (H, W), float32 in [0, 1]
    """

    # Convert to PIL
    img_pil = Image.fromarray(img)

    # Preprocess
    input_tensor = _transform(img_pil).unsqueeze(0).to(_DEVICE)

    # Inference
    with torch.no_grad():
        pred, *_ = _saliency_net(input_tensor)

    saliency = pred.squeeze().cpu().numpy()

    # Normalize
    saliency_min = saliency.min()
    saliency_max = saliency.max()
    saliency = (saliency - saliency_min) / (saliency_max - saliency_min + 1e-8)

    # Resize back to input resolution
    saliency = cv2.resize(
        saliency,
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )

    return saliency
