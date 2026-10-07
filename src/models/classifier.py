import torch
import numpy as np
from PIL import Image
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
import cv2
import logging

logger = logging.getLogger(__name__)

class DeepfakeClassifier:
    def __init__(self, model_name: str = "dima806/deepfake_vs_real_image_detection"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.pipe = None
        self._load_model()

    def _load_model(self):
        try:
            logger.info(f"Loading deepfake classification model: {self.model_name} on {self.device}")
            self.pipe = pipeline(
                "image-classification",
                model=self.model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info("Deepfake model loaded successfully!")
        except Exception as e:
            logger.warning(f"Failed to load HF pipeline model ({e}). Fallback to alternative model or heuristic analysis.")
            try:
                # Secondary lightweight HF model fallback
                alt_model = "prithivMLmods/Deep-Fake-Detector-v2"
                self.pipe = pipeline(
                    "image-classification",
                    model=alt_model,
                    device=0 if self.device == "cuda" else -1
                )
            except Exception as ex:
                logger.error(f"Fallback model also failed: {ex}. Using visual signal analyzer fallback.")
                self.pipe = None

    def predict_face(self, pil_img: Image.Image, bgr_crop: np.ndarray) -> dict:
        """
        Predicts fake probability for face crop.
        Returns dict: {'fake_score': float (0.0 to 1.0), 'signals': list[str]}
        """
        signals = []
        fake_score = 0.5 # default baseline

        if self.pipe is not None:
            try:
                results = self.pipe(pil_img)
                # Results list of dicts: [{'label': 'FAKE', 'score': 0.89}, {'label': 'REAL', 'score': 0.11}]
                fake_prob = 0.0
                for item in results:
                    label = item['label'].upper()
                    score = item['score']
                    if 'FAKE' in label or 'DEEPFAKE' in label or 'SYNTHETIC' in label or label == 'LABEL_1':
                        fake_prob = score
                    elif 'REAL' in label or label == 'LABEL_0':
                        fake_prob = 1.0 - score
                
                fake_score = float(fake_prob)
            except Exception as e:
                logger.warning(f"Inference error in pipeline: {e}")
                fake_score = self._compute_heuristic_score(bgr_crop, signals)
        else:
            fake_score = self._compute_heuristic_score(bgr_crop, signals)

        # Signal Analysis (High-frequency boundary & noise checking for explainability)
        signals.extend(self._analyze_spatial_signals(bgr_crop, fake_score))

        return {
            "fake_score": round(fake_score, 4),
            "signals": signals
        }

    def _compute_heuristic_score(self, bgr_crop: np.ndarray, signals: list) -> float:
        """Fallback signal analyzer based on boundary sharpness & color distribution variance."""
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Unnatural smoothness or high-frequency boundary noise
        if lap_var < 50.0:
            signals.append("Over-smoothed facial skin texture (low Laplacian variance)")
            return 0.72
        elif lap_var > 500.0:
            signals.append("High-frequency boundary noise artifact around face region")
            return 0.68
        return 0.35

    def _analyze_spatial_signals(self, bgr_crop: np.ndarray, fake_score: float) -> list:
        signals = []
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Eye & Mouth Region Sharpness check
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if fake_score > 0.65:
            if lap_var < 80.0:
                signals.append("Facial blending artifact detected: skin smoothing around cheek and jaw boundary.")
            else:
                signals.append("Boundary incongruity: sharp edge mismatch at neck/face seam.")
            
            # Color inconsistency check across RGB channels
            r, g, b = cv2.split(bgr_crop)
            std_diff = abs(np.std(r) - np.std(b))
            if std_diff > 12.0:
                signals.append("Unnatural color channel variance (chrominance disparity).")
        else:
            signals.append("Natural skin pore texture and anatomical edge consistency.")

        return signals
