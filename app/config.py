import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds for decision
AUTHENTIC_THRESHOLD = 0.35
UNCERTAIN_THRESHOLD = 0.65

# Model choice
# Default Hugging Face pretrained deepfake classifier model
HF_MODEL_NAME = "prithivMLmods/Deep-Fake-Detector-v2"  # or "dima806/deepfake_vs_real_image_detection"
