import cv2
import mediapipe as mp
import pickle
import numpy as np

# 1. Load your trained "Brain"
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 2. Setup Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    model_complexity=0, # Use 0 for a lighter model that might bypass XNNPACK issues
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Draw tracking dots
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Get current landmarks for prediction
            data_row = []
            for lm in hand_lms.landmark:
                data_row.extend([lm.x, lm.y, lm.z])

            # 3. Make the Prediction
            prediction = model.predict([data_row])[0]
            probability = np.max(model.predict_proba([data_row])) # Confidence level

            # 4. Trigger Visuals (The Domain Expansion)
            if probability > 0.60: # Only trigger if 60% sure
                text_color = (0, 255, 0)
                if prediction == "GOJO":
                    cv2.putText(img, "DOMAIN EXPANSION: UNLIMITED VOID", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    # TODO: Add your blue overlay/VFX here
                elif prediction == "SUKUNA":
                    cv2.putText(img, "DOMAIN EXPANSION: MALEVOLENT SHRINE", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                
                    
                    overlay = img.copy()
                    h, w = img.shape[:2]
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
                    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
                    center = (w // 2, h // 2)
                    radius = min(w, h) // 3
                    cv2.circle(img, center, radius, (0, 0, 255), 3)
                    cv2.circle(img, center, radius // 2, (0, 0, 180), 2)
                    for angle in range(0, 360, 45):
                        rad = np.deg2rad(angle)
                        end = (int(center[0] + np.cos(rad) * radius),
                               int(center[1] + np.sin(rad) * radius))
                        cv2.line(img, center, end, (0, 0, 255), 2)

                elif prediction == "MEGUMI":
                    cv2.putText(img, "DOMAIN EXPANSION: Chimera Shadow Garden", (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                prediction = "NEUTRAL"

    cv2.imshow("Jujutsu Vision", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
