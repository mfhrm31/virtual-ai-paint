"""
Gesture recognition from hand landmarks.

Converts MediaPipe hand landmarks into discrete gesture states
that map to drawing actions (draw, select, clear, idle).

Part of the Virtual AI Drawing & Painting thesis project.
University of Azad Jammu and Kashmir, 2022.
"""

import math
from typing import List, Tuple
from enum import Enum


class Gesture(Enum):
    """Discrete gestures recognized by the system."""
    IDLE = "idle"
    DRAW = "draw"
    SELECT = "select"
    CLEAR = "clear"
    PINCH = "pinch"
    UNKNOWN = "unknown"


# MediaPipe landmark indices
LANDMARK_THUMB_TIP = 4
LANDMARK_THUMB_IP = 3
LANDMARK_INDEX_TIP = 8
LANDMARK_INDEX_PIP = 6
LANDMARK_MIDDLE_TIP = 12
LANDMARK_MIDDLE_PIP = 10
LANDMARK_RING_TIP = 16
LANDMARK_RING_PIP = 14
LANDMARK_PINKY_TIP = 20
LANDMARK_PINKY_PIP = 18


class GestureDetector:
    """
    Detect discrete gestures from MediaPipe hand landmarks.

    Each frame, computes a finger-state vector [thumb, index, middle, ring, pinky]
    where each value is 1 if the finger is extended ("up") or 0 if curled ("down").
    Specific patterns map to specific actions:

        [_, 1, 0, 0, 0]  →  DRAW       (index finger only)
        [_, 1, 1, 0, 0]  →  SELECT     (index + middle)
        [0, 0, 0, 0, 0]  →  CLEAR      (closed fist)
        [_, 0, 0, 0, 0]  →  IDLE       (open palm but no draw signal)
        thumb-index dist →  PINCH      (brush size control)
    """

    def __init__(self, pinch_threshold: float = 0.05):
        """
        Args:
            pinch_threshold: Normalized distance threshold for pinch detection
        """
        self.pinch_threshold = pinch_threshold

    def _is_finger_up(self, landmarks: List, tip_idx: int, pip_idx: int) -> bool:
        """
        Check if a non-thumb finger is extended.

        A finger is "up" if its tip is above (smaller y) its PIP joint.
        Note: MediaPipe y-axis increases downward in image coordinates.
        """
        return landmarks[tip_idx].y < landmarks[pip_idx].y

    def _is_thumb_up(self, landmarks: List) -> bool:
        """
        Check if the thumb is extended.

        Thumb uses horizontal (x) comparison rather than vertical because
        the thumb moves laterally, not vertically.
        """
        return abs(landmarks[LANDMARK_THUMB_TIP].x - landmarks[LANDMARK_THUMB_IP].x) > 0.04

    def get_finger_states(self, landmarks: List) -> List[int]:
        """
        Compute [thumb, index, middle, ring, pinky] finger state vector.

        Args:
            landmarks: List of 21 MediaPipe landmarks

        Returns:
            List of 5 integers (0 or 1) indicating each finger's state
        """
        states = [
            int(self._is_thumb_up(landmarks)),
            int(self._is_finger_up(landmarks, LANDMARK_INDEX_TIP, LANDMARK_INDEX_PIP)),
            int(self._is_finger_up(landmarks, LANDMARK_MIDDLE_TIP, LANDMARK_MIDDLE_PIP)),
            int(self._is_finger_up(landmarks, LANDMARK_RING_TIP, LANDMARK_RING_PIP)),
            int(self._is_finger_up(landmarks, LANDMARK_PINKY_TIP, LANDMARK_PINKY_PIP)),
        ]
        return states

    def _thumb_index_distance(self, landmarks: List) -> float:
        """Compute normalized Euclidean distance between thumb tip and index tip."""
        dx = landmarks[LANDMARK_THUMB_TIP].x - landmarks[LANDMARK_INDEX_TIP].x
        dy = landmarks[LANDMARK_THUMB_TIP].y - landmarks[LANDMARK_INDEX_TIP].y
        return math.sqrt(dx * dx + dy * dy)

    def detect_gesture(self, landmarks: List) -> Tuple[Gesture, dict]:
        """
        Recognize the current gesture from hand landmarks.

        Args:
            landmarks: List of 21 MediaPipe landmarks

        Returns:
            Tuple of:
                - gesture: Gesture enum value
                - meta: Dictionary with additional info (finger_states, distance, etc.)
        """
        states = self.get_finger_states(landmarks)
        thumb, index, middle, ring, pinky = states

        pinch_distance = self._thumb_index_distance(landmarks)
        is_pinching = pinch_distance < self.pinch_threshold

        meta = {
            'finger_states': states,
            'pinch_distance': pinch_distance,
        }

        # Gesture rules (order matters — most specific first)
        if is_pinching and index == 1:
            return Gesture.PINCH, meta

        if index == 0 and middle == 0 and ring == 0 and pinky == 0:
            return Gesture.CLEAR, meta

        if index == 1 and middle == 1 and ring == 0 and pinky == 0:
            return Gesture.SELECT, meta

        if index == 1 and middle == 0 and ring == 0 and pinky == 0:
            return Gesture.DRAW, meta

        if index == 1 and middle == 1 and ring == 1 and pinky == 1:
            return Gesture.IDLE, meta

        return Gesture.UNKNOWN, meta


if __name__ == "__main__":
    detector = GestureDetector()
    print("GestureDetector initialized")
    print(f"\nSupported gestures:")
    for gesture in Gesture:
        print(f"  - {gesture.name}: {gesture.value}")
    print(f"\nPinch detection threshold: {detector.pinch_threshold}")
    print(f"\nFinger state vector format: [thumb, index, middle, ring, pinky]")
