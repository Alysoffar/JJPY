# Jujutsu Vision: Real-Time Domain Expansion AR

> Computer vision meets cursed energy.

**Jujutsu Vision** is an experimental augmented reality project that uses a webcam to detect anime-inspired hand signs in real time. The goal is to recognize specific gestures inspired by *Jujutsu Kaisen*, hold them for a short activation window, and trigger a dramatic "Domain Expansion" visual overlay around the user.

The project blends **OpenCV** video processing, **MediaPipe** hand landmark detection, and planned **Scikit-learn** gesture classification to create an interactive AR experience powered by hand motion and visual effects.

## Features

- **Real-time hand tracking** using MediaPipe hand landmarks.
- **Webcam video processing** with OpenCV.
- **Gesture recognition pipeline** for classifying hand signs such as Unlimited Void or Malevolent Shrine.
- **Hold-to-activate logic** planned for preventing accidental domain triggers.
- **Dynamic AR overlays** planned for cursed-energy-style visual effects.
- **Extensible model workflow** for collecting gesture data and training classifiers.

## Tech Stack

- **Python**
- **OpenCV** for camera input and frame rendering
- **MediaPipe** for hand landmark detection
- **Scikit-learn** for planned gesture classification
- **Random Forest** model for planned hand-sign prediction

## Installation

Clone the repository and create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the core dependencies:

```powershell
pip install mediapipe opencv-python scikit-learn
```

If you are on Windows and MediaPipe installation fails on newer Python versions, use Python 3.10 or 3.11.

## Usage

Run the current hand tracking script:

```powershell
python Hand_Detection.py
```

The camera window will open and MediaPipe will draw hand landmarks when a hand is visible. Press `q` while the OpenCV window is focused to exit.


## Project Roadmap

- [x] Set up Python environment
- [x] Test webcam input with OpenCV
- [x] Implement MediaPipe hand landmark tracking
- [ ] Collect hand sign landmark data
- [ ] Train gesture classifier with Scikit-learn
- [ ] Add gesture hold-duration detection
- [ ] Trigger Domain Expansion states from recognized signs
- [ ] Design Unlimited Void visual overlay
- [ ] Design Malevolent Shrine visual overlay
- [ ] Add polished AR effect timing, transitions, and cooldowns

## Current Status

The base hand tracking system is functional. MediaPipe can detect hands from the webcam feed and draw landmark connections in real time.

The next development phase focuses on collecting landmark data for target hand signs, training a Random Forest classifier, and mapping classified gestures to visual Domain Expansion effects.

## Future Theory Section

A future version of this README will include a mathematical explanation of the gesture recognition pipeline, including landmark normalization, Euclidean distances between key hand points, and how those geometric features can be used to classify signs reliably.

## Vision

Jujutsu Vision aims to turn hand gestures into cinematic AR interactions. The long-term goal is simple: raise a hand sign, hold the pose, and watch the room become your domain.

