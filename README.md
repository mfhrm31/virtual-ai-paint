# virtual-ai-paint
Real-time gesture-based drawing interface using OpenCV and MediaPipe. 
# Virtual AI Drawing & Painting 

**Real-Time Touch-Free Drawing Interface Using Hand-Gesture Recognition**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-blue.svg)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-orange.svg)](https://mediapipe.dev)

---

##  Overview

A real-time computer vision application that lets users draw and paint on a virtual canvas using only hand gestures — no mouse, no stylus, no touchscreen. The system detects 21 hand landmarks via the webcam, recognizes finger configurations, and translates them into drawing actions in real time.

This work demonstrates accessible human–computer interaction (HCI) with applications in:
- **Touch-free interfaces** for sterile clinical environments (surgical rooms, examination rooms)
- **Accessibility tools** for users with limited mobility or motor impairments
- **Educational technology** for interactive classrooms
- **Foundation work** extended in my subsequent medical-AI research on sensor-based activity recognition

##  Features

- **Real-time hand tracking** via MediaPipe's 21-landmark hand model
- **Gesture-based mode switching**: index finger raised = draw, two fingers = select color, fist = clear canvas
- **Color palette** with 4+ colors selectable via gesture
- **Variable brush thickness** controlled by finger-pinch distance
- **Eraser mode** activated by specific finger configuration
- **Save canvas** as PNG with timestamped filename

##  How It Works

1. **Webcam capture** → OpenCV reads frames at ~30 FPS
2. **Hand landmark detection** → MediaPipe extracts 21 3D hand keypoints
3. **Gesture recognition** → Custom logic identifies finger states (up/down) from landmark positions
4. **Mode mapping** → Specific finger combinations map to drawing actions
5. **Canvas overlay** → Drawing rendered on a transparent canvas blended with webcam feed
6. **Output** → Real-time display of webcam + drawing canvas overlay

##  Quickstart

### Installation

```bash
git clone https://github.com/mfhrm31/virtual-ai-paint.git
cd virtual-ai-paint
pip install -r requirements.txt
```

### Run

```bash
python main.py
```

A window will open showing your webcam feed. Use the following gestures:

| Gesture | Action |
|---|---|
| ☝️ Index finger up | Draw |
| ✌️ Index + middle fingers up | Select color from top palette |
| ✊ Closed fist | Clear canvas |
| 🖐️ Open palm | Idle (no drawing) |
| Index-thumb pinch | Adjust brush size |

Press `s` to save the canvas. Press `q` to quit.

## 📂 Project Structure
