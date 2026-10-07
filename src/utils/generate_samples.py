import cv2
import numpy as np
from pathlib import Path

def create_sample_images():
    sample_dir = Path(__file__).resolve().parent.parent.parent / "data" / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create Sample Real Face Image
    img_real = np.ones((400, 400, 3), dtype=np.uint8) * 220
    # Head ellipse
    cv2.ellipse(img_real, (200, 200), (100, 130), 0, 0, 360, (180, 190, 220), -1)
    # Eyes
    cv2.circle(img_real, (160, 170), 12, (255, 255, 255), -1)
    cv2.circle(img_real, (240, 170), 12, (255, 255, 255), -1)
    cv2.circle(img_real, (160, 170), 5, (50, 50, 50), -1)
    cv2.circle(img_real, (240, 170), 5, (50, 50, 50), -1)
    # Natural Mouth
    cv2.ellipse(img_real, (200, 250), (35, 15), 0, 0, 180, (80, 80, 180), 3)
    cv2.imwrite(str(sample_dir / "sample_real.jpg"), img_real)

    # 2. Create Sample Fake Face Image with Artificial Blending Noise
    img_fake = img_real.copy()
    # Add severe artificial blur around mouth region (simulating face-swap seam)
    mouth_roi = img_fake[220:280, 140:260]
    mouth_blurred = cv2.GaussianBlur(mouth_roi, (25, 25), 10)
    img_fake[220:280, 140:260] = mouth_blurred
    # Add sharp boundary seam line
    cv2.rectangle(img_fake, (135, 215), (265, 285), (0, 0, 255), 2)
    cv2.imwrite(str(sample_dir / "sample_fake.jpg"), img_fake)

    print(f"Sample test images generated in: {sample_dir}")

if __name__ == "__main__":
    create_sample_images()
