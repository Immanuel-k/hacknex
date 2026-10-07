import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any

class FaceDetector:
    def __init__(self, padding: float = 0.2):
        self.padding = padding
        self.face_cascade = None
        try:
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if hasattr(cv2, 'CascadeClassifier'):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            self.face_cascade = None

    def detect_and_crop(self, frame_bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects faces in BGR frame.
        Returns list of dicts with keys: 'box' [x, y, w, h], 'cropped_bgr', 'cropped_pil'
        """
        h, w, _ = frame_bgr.shape
        faces = []

        if self.face_cascade is not None and not self.face_cascade.empty():
            try:
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                detected = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 60)
                )
                if len(detected) > 0:
                    faces = detected
            except Exception:
                faces = []

        results = []

        if len(faces) == 0:
            # Fallback: take center facial region crop if no face cascade detected
            crop_w = int(w * 0.7)
            crop_h = int(h * 0.7)
            x = (w - crop_w) // 2
            y = (h - crop_h) // 2
            crop_bgr = frame_bgr[y:y+crop_h, x:x+crop_w]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            results.append({
                "box": [x, y, crop_w, crop_h],
                "is_fallback": True,
                "cropped_bgr": crop_bgr,
                "cropped_pil": Image.fromarray(crop_rgb)
            })
        else:
            sorted_faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[:2]
            for (x, y, fw, fh) in sorted_faces:
                pad_x = int(fw * self.padding)
                pad_y = int(fh * self.padding)

                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w, x + fw + pad_x)
                y2 = min(h, y + fh + pad_y)

                crop_bgr = frame_bgr[y1:y2, x1:x2]
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                
                results.append({
                    "box": [x1, y1, x2 - x1, y2 - y1],
                    "is_fallback": False,
                    "cropped_bgr": crop_bgr,
                    "cropped_pil": Image.fromarray(crop_rgb)
                })

        return results
