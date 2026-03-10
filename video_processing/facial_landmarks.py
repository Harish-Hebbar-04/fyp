"""
video_processing/facial_landmarks.py
──────────────────────────────────────
Uses MediaPipe FaceMesh to extract facial landmark coordinates
from each video frame and converts them into speech-relevant
numeric features:

  • lip_aperture      – vertical mouth opening (normalised)
  • lip_width         – horizontal lip spread (normalised)
  • jaw_opening       – chin-to-nose distance (normalised)
  • mouth_tension     – ratio of lip corner distance to lip aperture
  • blink_left        – left-eye openness
  • blink_right       – right-eye openness
  • lip_area          – bounding-box area of mouth region
  • upper_lip_curl    – upper lip curvature proxy
  • lower_lip_curl    – lower lip curvature proxy
  • head_nod          – nose-tip vertical displacement between frames

These 10 raw scalars are computed per frame and the full sequence is
returned as a (T, 10) float32 array.

Dependencies: mediapipe, opencv-python
"""

import sys
import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from typing import List

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import MAX_FRAMES, FPS, NUM_FACE_FEATURES

# ─────────────────────────────────────────────────────────────────────────
# MediaPipe landmark indices (FaceMesh 468-point model)
# ─────────────────────────────────────────────────────────────────────────
# Outer lip ring
UPPER_LIP_TOP    = 13
LOWER_LIP_BOTTOM = 14
LIP_LEFT         = 61
LIP_RIGHT        = 291

# Inner lip corners (for curvature)
UPPER_LIP_CENTER = 0
LOWER_LIP_CENTER = 17

# Jaw / chin
CHIN_TIP         = 152
NOSE_TIP         = 1

# Eyes (EAR – eye-aspect-ratio landmarks)
LEFT_EYE_UPPER   = 159
LEFT_EYE_LOWER   = 145
LEFT_EYE_LEFT    = 33
LEFT_EYE_RIGHT   = 133
RIGHT_EYE_UPPER  = 386
RIGHT_EYE_LOWER  = 374
RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263

# ─────────────────────────────────────────────────────────────────────────

def _eye_aspect_ratio(lm, upper: int, lower: int, left: int, right: int) -> float:
    """Eye aspect ratio (EAR) → 0 = closed, ~0.3 = open."""
    vert   = abs(lm[upper].y - lm[lower].y)
    horiz  = abs(lm[left].x  - lm[right].x) + 1e-6
    return vert / horiz


def _frame_features(lm) -> np.ndarray:
    """
    Compute the 10 scalar features for a single frame given
    a MediaPipe NormalizedLandmarkList.
    """
    # 1. Lip aperture (vertical)
    lip_aperture = abs(lm[UPPER_LIP_TOP].y - lm[LOWER_LIP_BOTTOM].y)

    # 2. Lip width (horizontal)
    lip_width = abs(lm[LIP_RIGHT].x - lm[LIP_LEFT].x) + 1e-6

    # 3. Jaw opening (chin to nose, normalised by face height proxy)
    jaw_opening = abs(lm[CHIN_TIP].y - lm[NOSE_TIP].y)

    # 4. Mouth tension (corner width / aperture ratio)
    mouth_tension = lip_width / (lip_aperture + 1e-6)

    # 5 & 6. Eye blink (EAR for each eye)
    blink_l = _eye_aspect_ratio(lm, LEFT_EYE_UPPER,  LEFT_EYE_LOWER,
                                 LEFT_EYE_LEFT,  LEFT_EYE_RIGHT)
    blink_r = _eye_aspect_ratio(lm, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER,
                                 RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)

    # 7. Lip bounding-box area
    lip_area = lip_aperture * lip_width

    # 8. Upper lip curl (y offset of center vs top)
    upper_lip_curl = lm[UPPER_LIP_CENTER].y - lm[UPPER_LIP_TOP].y

    # 9. Lower lip curl
    lower_lip_curl = lm[LOWER_LIP_BOTTOM].y - lm[LOWER_LIP_CENTER].y

    # 10. Head-nod proxy (nose-tip y; computed as delta between frames externally)
    nose_y = lm[NOSE_TIP].y

    return np.array([
        lip_aperture, lip_width, jaw_opening, mouth_tension,
        blink_l, blink_r, lip_area, upper_lip_curl, lower_lip_curl,
        nose_y,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────

class FacialLandmarkExtractor:
    """
    Processes a video file frame-by-frame and returns a (T, 10) array
    of facial features.

    Usage:
        extractor = FacialLandmarkExtractor()
        features  = extractor("path/to/video.mp4")   # (T, 10)
    """

    def __init__(self, max_frames: int = MAX_FRAMES):
        self.max_frames = max_frames
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, video_path: str) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray of shape (max_frames, 10), zero-padded if video is shorter.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        frames_feat: List[np.ndarray] = []
        prev_nose_y = None

        while len(frames_feat) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._face_mesh.process(rgb)

            if results.multi_face_landmarks:
                lm   = results.multi_face_landmarks[0].landmark
                feat = _frame_features(lm)

                # Convert last channel (nose_y) to delta (head-nod)
                nose_y = feat[-1]
                feat[-1] = 0.0 if prev_nose_y is None else nose_y - prev_nose_y
                prev_nose_y = nose_y
            else:
                # No face detected – use zeros
                feat = np.zeros(NUM_FACE_FEATURES, dtype=np.float32)

            frames_feat.append(feat)

        cap.release()

        # ── Pad / truncate to max_frames ──────────────────────────────
        seq = np.array(frames_feat, dtype=np.float32)          # (T, 10)
        if len(seq) < self.max_frames:
            pad = np.zeros((self.max_frames - len(seq), NUM_FACE_FEATURES),
                           dtype=np.float32)
            seq = np.vstack([seq, pad])
        else:
            seq = seq[:self.max_frames]

        return seq   # (max_frames, 10)

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
