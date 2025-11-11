import cv2
import mediapipe as mp
import math


cap = cv2.VideoCapture(0)
if not cap.isOpened:
    print("Cannot open camera")
    exit()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

def hand_inclination(hand_landmarks):
    degrees = None

    midfinger_base_id = 9
    wrist_id = 0

    midfinger_base = hand_landmarks.landmark[midfinger_base_id]
    wrist = hand_landmarks.landmark[wrist_id]

    y_len = wrist.y - midfinger_base.y #y-axis is inverted
    x_len = midfinger_base.x - wrist.x

    rad = math.atan2(x_len, y_len)
    degrees = math.degrees(rad)

    return degrees

def rotate(x, y, angle):
    x_rot = x * math.cos(angle) - y * math.sin(angle)
    y_rot = x * math.sin(angle) + y * math.cos(angle)
    return x_rot, y_rot

def finger_states(hand_landmarks, inclination):
    finger_states = []
    fingertips_ids = [8, 12, 16, 20] #excluding thumb
    bases_ids = [6, 10, 14, 18] #excluding thumb
    thumb_tip_id = 4
    thumb_base_id = 2

    rad = math.radians(inclination)

    thumb_tip = hand_landmarks.landmark[thumb_tip_id]
    thumb_base = hand_landmarks.landmark[thumb_base_id]

    thumb_tip_x, thumb_tip_y = rotate(thumb_tip.x, thumb_tip.y, -rad)
    thumb_base_x, thumb_base_y = rotate(thumb_base.x, thumb_base.y, -rad)

    if thumb_tip_x < thumb_base_x:
        finger_states.append(1)
    else:
        finger_states.append(0)

    for k in range(len(fingertips_ids)):
        tip = hand_landmarks.landmark[fingertips_ids[k]]
        base = hand_landmarks.landmark[bases_ids[k]]

        tip_x, tip_y = rotate(tip.x, tip.y, -rad)
        base_x, base_y = rotate(base.x, base.y, -rad)

        if tip_y < base_y:
            finger_states.append(1)
        else:
            finger_states.append(0)

    return finger_states


while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            inclination = hand_inclination(hand_landmarks)

            print(finger_states(hand_landmarks, inclination), "{:.2f}".format(inclination))
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()