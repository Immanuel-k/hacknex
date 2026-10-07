import cv2
import numpy as np
import os
import uuid
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

class HeatmapGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_saliency_overlay(
        self,
        frame_bgr: np.ndarray,
        box: list,
        fake_score: float,
        frame_idx: int
    ) -> str:
        """
        Generates a visual heatmap highlighting the suspicious face box region.
        Returns the absolute local file path to the saved heatmap image.
        """
        img = frame_bgr.copy()
        h, w, _ = img.shape
        x, y, bw, bh = box

        # Create a saliency mask
        mask = np.zeros((h, w), dtype=np.float32)

        # Highlight face region with Gaussian density based on fake_score
        cx, cy = x + bw // 2, y + bh // 2
        radius = max(bw, bh) // 2

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)

        # Heatmap intensity scales with fake_score
        sigma = radius * 0.75
        heatmap_region = np.exp(-(dist_from_center**2) / (2 * sigma**2)) * fake_score
        mask = np.maximum(mask, heatmap_region)

        # Normalize mask to 0-255
        mask_norm = np.uint8(255 * mask)
        heatmap_color = cv2.applyColorMap(mask_norm, cv2.COLORMAP_JET)

        # Blend heatmap with original frame
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

        # Draw bounding box and label
        color = (0, 0, 255) if fake_score > 0.5 else (0, 255, 0)
        label = f"FAKE: {fake_score*100:.1f}%" if fake_score > 0.5 else f"REAL: {(1-fake_score)*100:.1f}%"

        cv2.rectangle(overlay, (x, y), (x + bw, y + bh), color, 3)
        cv2.rectangle(overlay, (x, y - 30), (x + bw, y), color, -1)
        cv2.putText(
            overlay,
            label,
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        filename = f"heatmap_frame_{frame_idx}_{uuid.uuid4().hex[:6]}.jpg"
        save_path = self.output_dir / filename
        cv2.imwrite(str(save_path), overlay)

        return str(save_path)
