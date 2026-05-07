import cv2
import mediapipe as mp
import csv

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# The name of the sign you are recording
LABEL = "MEGUMI" 
recording = False  # This is our "Switch"

with open('hand_data.csv', mode='a', newline='') as f:
    writer = csv.writer(f)

    while cap.isOpened():
        success, img = cap.read()
        if not success: break
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        key = cv2.waitKey(1) & 0xFF
        
        # Toggle recording on/off when 's' is pressed
        if key == ord('s'):
            recording = not recording
            state = "STARTED" if recording else "STOPPED"
            print(f"Recording {state} for {LABEL}")

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                # If the switch is ON, save the data every single frame
                if recording:
                    landmarks = []
                    for lm in hand_lms.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    landmarks.append(LABEL)
                    writer.writerow(landmarks)

        # Update the UI to show if we are recording or not
        color = (0, 0, 255) if recording else (0, 255, 0)
        status_text = f"RECORDING: {LABEL}" if recording else "IDLE (Press 'S' to start)"
        
        cv2.putText(img, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        cv2.imshow("Data Collector", img)
        
        if key == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()