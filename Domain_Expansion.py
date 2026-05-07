import cv2
import mediapipe as mp
import pickle
import numpy as np
import os
import winsound

# --- 1. SETUP & MUTE WARNINGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- 2. LOAD AI MODELS ---
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, model_complexity=0, 
                       min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Selfie Segmentation for the Background Swap
mp_selfie = mp.solutions.selfie_segmentation
segmenter = mp_selfie.SelfieSegmentation(model_selection=1) 

# --- 3. LOAD VIDEOS ---
# Replace these filenames with your actual downloaded files
vfx_files = {
    "GOJO": cv2.VideoCapture('Domain_vids\gojo_void.mp4'),
    "SUKUNA": cv2.VideoCapture('Domain_vids\sukuna_shrine.mp4'),
    "MEGUMI": cv2.VideoCapture('Domain_vids\megumi_shadow.mp4')
}

sound_files = {
    "GOJO": r"Domain_vids\gojo_void.wav",
    "SUKUNA": r"Domain_vids\sukuna_shrine.wav",
    "MEGUMI": r"Domain_vids\megumi_shadow.wav"
}

active_vfx = None # Keeps track of which video is playing
completed_vfx = None
last_prediction = "NEUTRAL"
prediction_streak = 0
ACTIVATION_STREAK = 5

# --- 4. MAIN LOOP ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

while cap.isOpened():
    success, img = cap.read()
    if not success: break
    img = cv2.resize(img, (640, 480))
    h, w, _ = img.shape

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process Hand for Detection
    hand_results = hands.process(img_rgb)
    
    # Only run segmentation when a VFX is active.
    # This is the most expensive part of the loop, so skipping it when idle
    # keeps the script much lighter on CPU.
    mask = None
    if active_vfx:
        seg_results = segmenter.process(img_rgb)
        mask = seg_results.segmentation_mask > 0.2 # Adjust 0.2 to 0.5 for "tightness"

    detected_vfx = None

    if hand_results.multi_hand_landmarks:
        for hand_lms in hand_results.multi_hand_landmarks:
            data_row = []
            for lm in hand_lms.landmark:
                data_row.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([data_row])[0]
            probability = np.max(model.predict_proba([data_row]))

            if probability > 0.70 and prediction != "NEUTRAL":
                detected_vfx = prediction
                break

    if detected_vfx:
        if detected_vfx != completed_vfx:
            if detected_vfx == last_prediction:
                prediction_streak += 1
            else:
                last_prediction = detected_vfx
                prediction_streak = 1

            # Only activate after the same sign has been seen consistently.
            if prediction_streak >= ACTIVATION_STREAK and active_vfx != detected_vfx:
                active_vfx = detected_vfx
                completed_vfx = None
                vfx_files[active_vfx].set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset to start
                sound_path = sound_files.get(active_vfx)
                if sound_path and os.path.exists(sound_path):
                    winsound.PlaySound(sound_path, winsound.SND_FILENAME | winsound.SND_ASYNC)
    else:
        prediction_streak = 0
        last_prediction = "NEUTRAL"
        if completed_vfx is not None:
            completed_vfx = None

    # --- 5. THE BACKGROUND SWAP LOGIC ---
    if active_vfx:
        ret, vfx_frame = vfx_files[active_vfx].read()
        
        if ret:
            vfx_frame = cv2.resize(vfx_frame, (w, h))

            # STAGE 2 MASKING:
            # If mask is True (your body), keep 'img'. Else, use 'vfx_frame'.
            if mask is not None:
                img = np.where(mask[:, :, None], img, vfx_frame)
        else:
            # Video finished, go back to normal
            completed_vfx = active_vfx
            active_vfx = None
            prediction_streak = 0
            last_prediction = "NEUTRAL"

    cv2.imshow("Jujutsu Vision", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
for v in vfx_files.values():
    v.release()
cv2.destroyAllWindows()