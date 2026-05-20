"""
Hand landmark detection using MediaPipe Hands.

Wraps Google's MediaPipe Hands solution to detect 21 3D hand
landmarks per frame in real time.

Part of the Virtual AI Drawing & Painting thesis project.
University of Azad Jammu and Kashmir, 2022.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Optional, Tuple


# MediaPipe landmark indices for fingertips and PIP joints
# (used for finger-up detection)
FINGER_TIPS = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20,
}

FINGER_PIPS = {
    'thumb': 3,
    'index': 6,
    'middle': 10,
    'ring': 14,
    'pinky': 18,
}


class HandTracker:
    """
    Real-time hand landmark detector using MediaPipe Hands.

    Detects up to N hands per frame and returns 21 3D landmarks
    per detected hand. Optimized for single-hand drawing applications.

    Args:
        max_num_hands: Maximum number of hands to detect (default: 1)
        min_detection_confidence: Minimum confidence for detection (default: 0.7)
        min_tracking_confidence: Minimum confidence for tracking (default: 0.5)
    """

    def __init__(
        self,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.last_landmarks = None

    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Optional[List]]:
        """
        Detect hand landmarks in a frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Tuple of:
                - annotated_frame: Frame with landmarks drawn on it
                - landmarks: List of 21 landmarks for the first detected hand,
                             or None if no hand detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.hands.process(rgb_frame)
        rgb_frame.flags.writeable = True

        annotated_frame = frame.copy()
        landmarks = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = hand_landmarks.landmark
            self.last_landmarks = landmarks

            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

        return annotated_frame, landmarks

    @staticmethod
    def get_landmark_position(
        landmarks: List, idx: int, frame_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Convert normalized landmark coordinates to pixel coordinates.

        Args:
            landmarks: List of 21 MediaPipe landmarks
            idx: Landmark index (0-20)
            frame_shape: Frame dimensions as (height, width)

        Returns:
            Pixel coordinates (x, y)
        """
        h, w = frame_shape[:2]
        x = int(landmarks[idx].x * w)
        y = int(landmarks[idx].y * h)
        return x, y

    def release(self):
        """Release MediaPipe resources."""
        self.hands.close()


if __name__ == "__main__":
    tracker = HandTracker(max_num_hands=1)
    print("HandTracker initialized")
    print(f"Detecting up to {1} hand(s)")
    print(f"Fingertip landmark indices: {FINGER_TIPS}")
    print(f"PIP joint landmark indices: {FINGER_PIPS}")
