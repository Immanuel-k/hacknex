import cv2
import os
import numpy as np
from PIL import Image
from typing import List, Tuple

class FrameExtractor:
    def __init__(self, sample_rate_fps: float = 2.0, max_frames: int = 40):
        self.sample_rate_fps = sample_rate_fps
        self.max_frames = max_frames

    def extract_frames_from_video(self, video_path: str) -> List[Tuple[int, float, np.ndarray]]:
        """
        Extracts frames from video path at specified sample rate.
        Returns list of tuples: (frame_index, timestamp_sec, frame_bgr)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 30.0

        step = max(1, int(video_fps / self.sample_rate_fps))
        
        frames = []
        frame_count = 0
        extracted_count = 0

        while cap.isOpened() and extracted_count < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % step == 0:
                timestamp = frame_count / video_fps
                frames.append((frame_count, round(timestamp, 2), frame))
                extracted_count += 1

            frame_count += 1

        cap.release()
        return frames

    def load_image(self, image_path: str) -> List[Tuple[int, float, np.ndarray]]:
        """
        Loads a single static image formatted as a 1-frame list.
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Unable to read image file: {image_path}")
        return [(0, 0.0, img)]
