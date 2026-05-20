"""
Drawing canvas with color and brush management.

Maintains a persistent drawing layer that overlays the webcam feed.
Supports color switching, variable brush thickness, eraser mode, and
saving the canvas as a PNG.

Part of the Virtual AI Drawing & Painting thesis project.
University of Azad Jammu and Kashmir, 2022.
"""

import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional


# Default color palette (BGR format for OpenCV)
DEFAULT_COLORS = [
    ('Red', (0, 0, 255)),
    ('Green', (0, 255, 0)),
    ('Blue', (255, 0, 0)),
    ('Yellow', (0, 255, 255)),
    ('White', (255, 255, 255)),
    ('Eraser', (0, 0, 0)),
]


class DrawingCanvas:
    """
    Transparent drawing canvas blended on top of a webcam feed.

    Maintains:
        - A persistent canvas (np.ndarray) where strokes are drawn
        - Current color (BGR tuple)
        - Current brush thickness
        - Previous fingertip position (for connecting line segments)

    Args:
        width: Canvas width in pixels
        height: Canvas height in pixels
        default_thickness: Initial brush thickness (default: 8)
        min_thickness: Minimum brush thickness (default: 2)
        max_thickness: Maximum brush thickness (default: 50)
    """

    def __init__(
        self,
        width: int,
        height: int,
        default_thickness: int = 8,
        min_thickness: int = 2,
        max_thickness: int = 50,
    ):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)

        self.colors = DEFAULT_COLORS
        self.current_color_idx = 0
        self.current_color = self.colors[0][1]
        self.eraser_color = (0, 0, 0)

        self.thickness = default_thickness
        self.min_thickness = min_thickness
        self.max_thickness = max_thickness

        self.prev_x = None
        self.prev_y = None

    def draw(self, x: int, y: int) -> None:
        """
        Draw a stroke at (x, y), connecting to the previous point.

        Args:
            x, y: Current fingertip pixel coordinates
        """
        if self.prev_x is None or self.prev_y is None:
            self.prev_x, self.prev_y = x, y
            return

        cv2.line(
            self.canvas,
            (self.prev_x, self.prev_y),
            (x, y),
            self.current_color,
            self.thickness,
        )
        self.prev_x, self.prev_y = x, y

    def lift_pen(self) -> None:
        """Reset previous position. Call when not drawing."""
        self.prev_x = None
        self.prev_y = None

    def clear(self) -> None:
        """Wipe the canvas clean."""
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.lift_pen()

    def select_color(self, idx: int) -> None:
        """
        Switch to the color at the given palette index.

        Args:
            idx: Index into self.colors (wraps around)
        """
        self.current_color_idx = idx % len(self.colors)
        self.current_color = self.colors[self.current_color_idx][1]
        self.lift_pen()

    def cycle_color(self) -> None:
        """Move to the next color in the palette."""
        self.select_color(self.current_color_idx + 1)

    def set_thickness(self, thickness: int) -> None:
        """Set brush thickness, clamped to [min, max]."""
        self.thickness = max(self.min_thickness, min(self.max_thickness, thickness))

    def adjust_thickness_from_pinch(
        self, pinch_distance: float, max_pinch: float = 0.3
    ) -> None:
        """
        Map normalized pinch distance to brush thickness.

        Args:
            pinch_distance: Normalized distance between thumb and index tip
            max_pinch: Maximum expected pinch distance (default: 0.3)
        """
        ratio = min(pinch_distance / max_pinch, 1.0)
        new_thickness = int(
            self.min_thickness + ratio * (self.max_thickness - self.min_thickness)
        )
        self.set_thickness(new_thickness)

    def overlay_on_frame(self, frame: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """
        Blend the canvas with the webcam frame.

        Uses a gray-mask approach: any non-black pixel on the canvas
        replaces the corresponding pixel in the frame.

        Args:
            frame: BGR webcam frame
            alpha: Blending weight (unused in mask-based overlay)

        Returns:
            Frame with drawing overlaid
        """
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        return cv2.add(frame_bg, canvas_fg)

    def save(self, output_dir: str = "outputs") -> str:
        """
        Save the current canvas as a timestamped PNG.

        Args:
            output_dir: Directory to save into

        Returns:
            Path to the saved file
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f"{output_dir}/drawing_{timestamp}.png"
        cv2.imwrite(filepath, self.canvas)
        return filepath

    def get_current_color_name(self) -> str:
        """Return the name of the currently selected color."""
        return self.colors[self.current_color_idx][0]


if __name__ == "__main__":
    canvas = DrawingCanvas(width=1280, height=720)
    print(f"Canvas initialized: {canvas.width}x{canvas.height}")
    print(f"Default color: {canvas.get_current_color_name()}")
    print(f"Default thickness: {canvas.thickness}")
    print(f"\nAvailable colors:")
    for i, (name, bgr) in enumerate(canvas.colors):
        print(f"
