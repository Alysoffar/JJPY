import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

cap = None
backends = [
    ("default", cv2.CAP_ANY),
    ("DirectShow", cv2.CAP_DSHOW),
    ("Media Foundation", cv2.CAP_MSMF),
]

for backend_name, backend in backends:
    for camera_index in range(10):
        test_cap = cv2.VideoCapture(camera_index, backend)
        success, frame = test_cap.read()
        if test_cap.isOpened() and success and frame is not None:
            print(f"Using camera index {camera_index} with {backend_name}")
            cap = test_cap
            break
        test_cap.release()
    if cap is not None:
        break

if cap is None:
    print("Could not open camera. Try the Windows Camera app, then check camera permissions and drivers.")
    exit()


while cap.isOpened():
    success, img = cap.read()
    if not success: break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Mediapipe Test", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
