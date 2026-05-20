"""
Virtual AI Drawing & Painting — main entry point.

Real-time gesture-based drawing application using webcam + MediaPipe.

Usage:
    python main.py

Controls:
    Index finger up           → Draw
    Index + middle up         → Select color (cycles palette)
    Closed fist               → Clear canvas
    Open palm (all fingers)   → Idle
    Thumb-index pinch         → Adjust brush thickness

Keyboard:
    s → Save canvas as PNG
    c → Clear canvas
    q → Quit

University of Azad Jammu and Kashmir, 2022.
Thesis grade: 4.0 / 4.0
"""

import cv2
import time

from src.hand_tracker import HandTracker, FINGER_TIPS
from src.gesture_detector import GestureDetector, Gesture, LANDMARK_INDEX_TIP
from src.canvas import DrawingCanvas


def draw_ui_overlay(frame, canvas, gesture, fps):
    """Draw status info on the frame."""
    h, w = frame.shape[:2]

    # Top bar background
    cv2.rectangle(frame, (0, 0), (w, 60), (0, 0, 0), -1)

    # Color indicator
    cv2.rectangle(frame, (10, 10), (60, 50), canvas.current_color, -1)
    cv2.rectangle(frame, (10, 10), (60, 50), (255, 255, 255), 2)

    # Status text
    color_name = canvas.get_current_color_name()
    cv2.putText(
        frame, f"Color: {color_name}", (75, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
    )
    cv2.putText(
        frame, f"Thickness: {canvas.thickness}", (75, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

    # Gesture indicator
    gesture_text = f"Gesture: {gesture.value.upper()}"
    cv2.putText(
        frame, gesture_text, (w - 280, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1,
    )

    # FPS
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (w - 280, 50),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )


def main():
    # Setup webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    # Get actual frame dimensions
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Could not read frame")
        cap.release()
        return
    h, w = frame.shape[:2]

    # Initialize components
    tracker = HandTracker(max_num_hands=1)
    detector = GestureDetector(pinch_threshold=0.05)
    canvas = DrawingCanvas(width=w, height=h)

    # FPS tracking
    prev_time = time.time()

    print("=" * 50)
    print("Virtual AI Drawing & Painting")
    print("=" * 50)
    print("Controls:")
    print("  Index finger up       → Draw")
    print("  Index + middle up     → Cycle color")
    print("  Closed fist           → Clear canvas")
    print("  Thumb-index pinch     → Adjust thickness")
    print("Keys:")
    print("  s → Save  |  c → Clear  |  q → Quit")
    print("=" * 50)

    last_select_time = 0
    select_cooldown = 0.8  # seconds between color cycles

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # Hand tracking
        annotated_frame, landmarks = tracker.process_frame(frame)
        gesture = Gesture.IDLE

        if landmarks is not None:
            gesture, meta = detector.detect_gesture(landmarks)

            # Get index fingertip position
            tip_x, tip_y = tracker.get_landmark_position(
                landmarks, LANDMARK_INDEX_TIP, frame.shape
            )

            # Handle gestures
            if gesture == Gesture.DRAW:
                canvas.draw(tip_x, tip_y)

            elif gesture == Gesture.SELECT:
                now = time.time()
                if now - last_select_time > select_cooldown:
                    canvas.cycle_color()
                    last_select_time = now
                canvas.lift_pen()

            elif gesture == Gesture.CLEAR:
                canvas.clear()

            elif gesture == Gesture.PINCH:
                canvas.adjust_thickness_from_pinch(meta['pinch_distance'])
                canvas.lift_pen()

            else:
                canvas.lift_pen()
        else:
            canvas.lift_pen()

        # Overlay canvas on frame
        composite = canvas.overlay_on_frame(annotated_frame)

        # FPS computation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time

        # UI overlay
        draw_ui_overlay(composite, canvas, gesture, fps)

        cv2.imshow("Virtual AI Drawing & Painting", composite)

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filepath = canvas.save()
            print(f"Saved: {filepath}")
        elif key == ord('c'):
            canvas.clear()
            print("Canvas cleared")

    # Cleanup
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed.")


if __name__ == "__main__":
    main()
