"""
fda_patch.py - Patch-Level Fourier Domain Adaptation
Novel contribution: Instead of global FDA which creates inter-patch artifacts
that break ViT's Self-Attention, this applies FDA within each patch independently.

Why this works better with ViT:
- Standard FDA: Operates on the full image → creates frequency artifacts that
  cross patch boundaries → breaks Self-Attention alignment between patches.
- Patch-Level FDA: Operates within each 16×16 patch independently → artifacts
  are contained inside each patch → patch boundary coherence is preserved.
"""

import numpy as np
from PIL import Image


def _fda_single_channel(src_ch: np.ndarray, tgt_ch: np.ndarray, beta: float, alpha: float) -> np.ndarray:
    """Apply FDA on a single 2D channel. alpha controls blending (1.0 = full swap)."""
    src_fft = np.fft.fftshift(np.fft.fft2(src_ch))
    tgt_fft = np.fft.fftshift(np.fft.fft2(tgt_ch))

    amp_src = np.abs(src_fft)
    pha_src = np.angle(src_fft)
    amp_tgt = np.abs(tgt_fft)

    h, w = src_ch.shape
    b_h = max(1, int(h * beta))
    b_w = max(1, int(w * beta))
    c_h, c_w = h // 2, w // 2

    # Phase-preserving amplitude blending
    amp_src[c_h - b_h:c_h + b_h, c_w - b_w:c_w + b_w] = (
        (1 - alpha) * amp_src[c_h - b_h:c_h + b_h, c_w - b_w:c_w + b_w]
        + alpha     * amp_tgt[c_h - b_h:c_h + b_h, c_w - b_w:c_w + b_w]
    )

    fft_swapped = np.fft.ifftshift(amp_src * np.exp(1j * pha_src))
    return np.real(np.fft.ifft2(fft_swapped))


class PatchFDA:
    """
    Patch-Level Fourier Domain Adaptation for ViT compatibility.

    Args:
        patch_size: Should match ViT patch size (16 for Scale-MAE).
        beta: Low-frequency radius ratio (same as standard FDA).
        alpha: Amplitude blending. 0.3-0.6 recommended for ViT (1.0 = standard FDA).
        target_images: List of numpy [H,W,3] style reference images.
    """
    def __init__(self, patch_size=16, beta=0.01, alpha=0.5, target_images=None):
        self.patch_size = patch_size
        self.beta = beta
        self.alpha = alpha
        self.target_images = target_images or []

    def __call__(self, src_img: np.ndarray, tgt_img: np.ndarray = None) -> np.ndarray:
        src = src_img.astype(np.float32)
        h, w = src.shape[:2]
        p = self.patch_size

        if tgt_img is None:
            if not self.target_images:
                return src_img
            tgt_img = self.target_images[np.random.randint(len(self.target_images))]

        tgt = np.array(
            Image.fromarray(tgt_img.astype(np.uint8)).resize((w, h), Image.BILINEAR),
            dtype=np.float32
        )

        # Pad so image is divisible by patch_size
        pad_h = (p - h % p) % p
        pad_w = (p - w % p) % p
        src_pad = np.pad(src, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
        tgt_pad = np.pad(tgt, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

        H_pad, W_pad = src_pad.shape[:2]
        out = np.zeros_like(src_pad, dtype=np.float32)

        for i in range(0, H_pad, p):
            for j in range(0, W_pad, p):
                src_p = src_pad[i:i+p, j:j+p]
                tgt_p = tgt_pad[i:i+p, j:j+p]
                result_p = np.zeros_like(src_p, dtype=np.float32)
                for c in range(3):
                    result_p[:, :, c] = _fda_single_channel(src_p[:, :, c], tgt_p[:, :, c], self.beta, self.alpha)
                out[i:i+p, j:j+p] = result_p

        out = out[:h, :w]
        return np.clip(out, 0, 255).astype(np.uint8)


def global_fda(src_img: np.ndarray, tgt_img: np.ndarray, beta=0.01, alpha=1.0) -> np.ndarray:
    """Standard global FDA for comparison baseline."""
    src = src_img.astype(np.float32)
    h, w = src.shape[:2]
    tgt = np.array(Image.fromarray(tgt_img.astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32)
    result = np.zeros_like(src, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = _fda_single_channel(src[:, :, c], tgt[:, :, c], beta, alpha)
    return np.clip(result, 0, 255).astype(np.uint8)
