import cv2
import mediapipe as mp

CALIBRATION_FRAMES = 30
FRAMES_ELAPSED = 0

class HandData:
    isWaving = False
    fingers = None
    left_hand = False
    right_hand = False
    prev_hand_x = None
    movement = 0

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

def writeOnImage(frame):
    global isWaving, fingers
    if FRAMES_ELAPSED < CALIBRATION_FRAMES:
        text = "Calibrating..."
    elif HandData.isWaving:
        text = "Waving"
    elif HandData.fingers is not None:
        if HandData.fingers > 0:
            text = str(HandData.fingers)
        else:
            text = "None"
    else:
        text = "None"

    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.8,( 0 , 0 , 0 ),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),1,cv2.LINE_AA)

def checkWaving(movement):
    # global HandData.isWaving

    if abs(movement) > 10:
        HandData.isWaving = True
    else:
        HandData.isWaving = False

def isFingerExtended(hand_landmarks, finger_tip_id, finger_dip_id):
    # If top part of finger is higher then "middle" part
    return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_dip_id].y

def countFingers(hand_landmarks):
    # global fingers

    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    index = isFingerExtended(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP)
    middle = isFingerExtended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP)
    ring = isFingerExtended(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP)
    pinky = isFingerExtended(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP)

    # Counts how many fingers are extended
    HandData.fingers = 0
    count = [thumb, index, middle, ring, pinky]
    for finger in count:
        if finger:
            HandData.fingers += 1

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width, _ = frame.shape
    
    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    
    # Check if hands are detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            # Hand labels are mirrored so mirror it again
            if handedness.classification[0].label == "Right":
                HandData.left_hand = True
            if handedness.classification[0].label == "Left":
                HandData.right_hand = True

            hand_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            hand_x_pixels = int(hand_x * frame_width)

            # If there is previous pixel calculate movement else initialize previous pixel
            if HandData.prev_hand_x is not None:
                movement = hand_x_pixels - HandData.prev_hand_x

            HandData.prev_hand_x = hand_x_pixels

            # Check only every 8 frames
            if FRAMES_ELAPSED % 8 == 0:
                checkWaving(movement)
                countFingers(hand_landmarks)

            # Draw landmarks(lines) on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        HandData.fingers = None
        HandData.isWaving = None
        HandData.left_hand = False
        HandData.right_hand = False

    # Mirror the image camera displays
    frame = cv2.flip(frame, 1)
    # Display the frame with hand landmarks and text
    writeOnImage(frame)

    if HandData.left_hand and not HandData.right_hand:
        cv2.putText(frame, "Left hand", (10,40), cv2.FONT_HERSHEY_COMPLEX, 0.8,( 0 , 0 , 0 ),2,cv2.LINE_AA)
        cv2.putText(frame, "Left hand", (10,40), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),1,cv2.LINE_AA)
    if not HandData.left_hand and HandData.right_hand:
        cv2.putText(frame, "Right hand", (10,40), cv2.FONT_HERSHEY_COMPLEX, 0.8,( 0 , 0 , 0 ),2,cv2.LINE_AA)
        cv2.putText(frame, "Right hand", (10,40), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),1,cv2.LINE_AA)

    cv2.imshow('Hand Recognition', frame)
    FRAMES_ELAPSED += 1
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()