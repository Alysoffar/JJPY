# Jujutsu Vision: Real-Time Domain Expansion AR

Jujutsu Vision is a real-time computer vision project that recognizes custom hand signs from a webcam and triggers anime-inspired domain expansion effects. The system combines MediaPipe hand landmark tracking, a trained Random Forest classifier, and OpenCV-based rendering to create a lightweight AR-style experience on a standard laptop.

This project is designed as a full end-to-end pipeline: data collection, model training, live inference, timed effect activation, and one-shot visual playback. It demonstrates practical work in computer vision, interactive systems, and applied machine learning.

## Project Highlights

- Real-time hand landmark extraction with MediaPipe.
- Custom gesture classification using a Random Forest model trained on collected landmark data.
- Stable activation logic that requires a sign to be held briefly before triggering.
- One-shot video playback so the effect completes once and then returns to normal.
- Lightweight webcam processing tuned to avoid unnecessary CPU load.
- Separate pipelines for data collection, training, and live inference.

## Technical Summary

The live application works as follows:

1. The webcam feed is captured and resized for lower processing overhead.
2. MediaPipe extracts 21 hand landmarks per detected hand.
3. Landmark coordinates are passed into a Random Forest classifier.
4. A sign must be detected consistently for several frames before activation.
5. The matching domain video and optional sound effect play once.
6. When the animation finishes, the system returns to the normal camera view.

This structure keeps the interaction responsive while avoiding the constant re-triggering that can happen in frame-by-frame vision systems.

## Repository Structure

- [Domain_Expansion.py](Domain_Expansion.py) - live webcam inference and visual effects.
- [Data_collector.py](Data_collector.py) - records hand landmark samples into CSV format.
- [train_model.py](train_model.py) - trains the gesture classifier and saves `model.pkl`.
- [Hand_Detection.py](Hand_Detection.py) - standalone MediaPipe webcam test.
- [hand_data.csv](hand_data.csv) - collected landmark dataset.
- [Domain_vids/](Domain_vids/) - domain expansion video assets.
- `model.pkl` - generated classifier artifact created by `train_model.py`.

## Requirements

- Python 3.10 or 3.11 is recommended on Windows.
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install mediapipe opencv-python numpy pandas scikit-learn
```

If MediaPipe fails to install on a newer Python version, use Python 3.10 or 3.11.

## Usage

### 1. Collect training data

Run the data collection script and record samples for each sign:

```powershell
python Data_collector.py
```

Press `s` to start or stop recording, and `q` to exit.

### 2. Train the model

Train the classifier from the collected CSV data:

```powershell
python train_model.py
```

This creates `model.pkl`, which is used by the live application.

### 3. Run the live demo

Start the real-time gesture application:

```powershell
python Domain_Expansion.py
```

Raise one of the supported hand signs and hold it briefly. The matching domain video will play once, then stop and return to the normal camera view.

### 4. Quick camera test

If you only want to verify the webcam and MediaPipe pipeline:

```powershell
python Hand_Detection.py
```

## Implementation Notes

- The live script uses a reduced camera resolution to keep the application responsive.
- Hand-sign activation is debounced across multiple frames to reduce false positives.
- The effect is intentionally one-shot instead of looping, which gives the interaction a more polished feel.
- The project is structured so the model, data collection, and rendering logic are separated cleanly.
- Optional sound hooks are present in the live script; matching `.wav` files can be added beside the video assets if audio playback is desired.

## Current Status

The core pipeline is functional:

- webcam capture works,
- hand landmarks are detected in real time,
- trained gestures can trigger matching domain sequences,
- and each sequence ends cleanly before the system returns to the neutral state.

## Future Work

- Improve gesture robustness across lighting and camera angles.
- Add more signs and larger training coverage.
- Refine sound design and synchronization.
- Add better transitions and timed cooldown behavior.
- Package the project into a more polished demo format for presentation.

## Why This Project Matters

This project shows more than a visual demo. It demonstrates data collection, model training, real-time inference, event-driven UI behavior, and performance-aware implementation. Those are the same kinds of engineering decisions that matter in research environments, internships, and product-facing software work.

