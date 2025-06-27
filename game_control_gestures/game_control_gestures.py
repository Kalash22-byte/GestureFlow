import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe and drawing utilityetgt
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Open the webcam
cap = cv2.VideoCapture(0)

# Function to detect which fingers are up
def count_fingers(hand_landmarks):
    tips = [8, 12, 16, 20]  # Index to pinky tips
    fingers = []

    # Thumb: compare x position (left or right)
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Other fingers: compare tip.y < pip.y
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

last_action = ""

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    action = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            fingers = count_fingers(hand_landmarks)

            # GESTURE → ACTIONS:
            if fingers == [0, 1, 0, 0, 0]:   # Index finger up
                action = "RIGHT"
                pyautogui.press("right")
            elif fingers == [1, 0, 0, 0, 0]: # Thumb out (left)
                action = "LEFT"
                pyautogui.press("left")
            elif sum(fingers) == 5:         # All fingers up
                action = "JUMP"
                pyautogui.press("up")
            elif sum(fingers) == 0:         # Fist
                action = "ROLL"
                pyautogui.press("down")

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if action and action != last_action:
        print(f"Gesture detected: {action}")
        last_action = action

    cv2.putText(frame, f"Gesture: {action}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow("Hand Gesture Game Controller", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
