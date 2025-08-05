import os
import cv2
import torch
import lpips
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
import math

# === Initialize LPIPS === #
lpips_fn = lpips.LPIPS(net='vgg').cuda()

def calculate_cpsnr(img1, img2):
    """Compute C-PSNR between two RGB images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 10 * math.log10((PIXEL_MAX ** 2) / mse)

def calculate_ssim(img1, img2):
    """Compute SSIM between two RGB images."""
    ssim = 0
    for i in range(3):  # channel-wise SSIM
        ssim += compare_ssim(img1[..., i], img2[..., i], data_range=1.0)
    return ssim / 3

def calculate_ebcm(img1, img2):
    """Compute edge-based contrast metric (like EBCM)."""
    def edge_detect(x):
        return cv2.Sobel(x, cv2.CV_64F, 1, 1, ksize=3)
    edge1 = edge_detect(cv2.cvtColor((img1 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    edge2 = edge_detect(cv2.cvtColor((img2 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    return np.mean((edge1 - edge2) ** 2)

def evaluate_metrics_individual(high_dir, enhanced_dir):
    filenames = sorted(os.listdir(high_dir))
    for fname in tqdm(filenames, desc="🔍 Evaluating"):
        high_path = os.path.join(high_dir, fname)
        enh_path = os.path.join(enhanced_dir, fname)
        high_img = cv2.imread(high_path)
        enh_img = cv2.imread(enh_path)
        if high_img is None or enh_img is None:
            continue
        high_img = cv2.cvtColor(high_img, cv2.COLOR_BGR2RGB) / 255.0
        enh_img = cv2.cvtColor(enh_img, cv2.COLOR_BGR2RGB) / 255.0

        # Resize if needed
        if high_img.shape != enh_img.shape:
            enh_img = cv2.resize(enh_img, (high_img.shape[1], high_img.shape[0]))

        # Compute metrics
        cpsnr = calculate_cpsnr(enh_img, high_img)
        ssim = calculate_ssim(enh_img, high_img)
        ebcm = calculate_ebcm(enh_img, high_img)

        # LPIPS requires torch tensors
        with torch.no_grad():
            high_tensor = torch.tensor(high_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            enh_tensor = torch.tensor(enh_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
            lpips_val = lpips_fn(enh_tensor, high_tensor).item()

        # 🔍 Print per-image metrics
        print(f"{fname}:  C-PSNR: {cpsnr:.4f}  SSIM: {ssim:.4f}  LPIPS: {lpips_val:.4f}  EBCM: {ebcm:.4f}")

# === Example usage === #
high_dir = "/content/cvccolondbsplit/test/high"
enhanced_dir = "/content/outputs/test_enhanced"
evaluate_metrics_individual(high_dir, enhanced_dir)
