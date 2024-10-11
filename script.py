import cv2
import mediapipe as mp

CALIBRATION_FRAMES = 30

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

frames_elapsed = 0
isWaving = False
fingers = None
prev_hand_x = None
movement = 0

def writeOnImage(frame):
    global isWaving, fingers
    if frames_elapsed < CALIBRATION_FRAMES:
        text = "Calibrating..."
    elif isWaving:
        text = "Waving"
    elif fingers and not isWaving:
        text = str(fingers)
    else:
        text = "None"

    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.8,( 0 , 0 , 0 ),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.8,(255,255,255),1,cv2.LINE_AA)

def checkWaving(movement):
    global isWaving

    if abs(movement) > 10:
        isWaving = True
    else:
        isWaving = False

def isFingerExtended(hand_landmarks, finger_tip_id, finger_dip_id):
    # If top part of finger is higher then "middle" part
    return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_dip_id].y

def countFingers(hand_landmarks):
    global fingers

    thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x
    index = isFingerExtended(hand_landmarks, mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_DIP)
    middle = isFingerExtended(hand_landmarks, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_DIP)
    ring = isFingerExtended(hand_landmarks, mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_DIP)
    pinky = isFingerExtended(hand_landmarks, mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_DIP)

    fingers = 0
    count = [thumb, index, middle, ring, pinky]
    for finger in count:
        if finger:
            fingers += 1
    print(count)

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
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Gets x coordinates of hand every 8 frames
            hand_x = hand_landmarks.landmark[i].x
            # Turn coordinates into pixels
            hand_x_pixels = int(hand_x * frame_width)

            # If there is previous pixel calculate movement else initialize previous pixel
            if prev_hand_x is not None:
                movement = hand_x_pixels - prev_hand_x

            prev_hand_x = hand_x_pixels

            if frames_elapsed % 8 == 0:
                checkWaving(movement)
                countFingers(hand_landmarks)

            # Draw landmarks(lines) on the hand
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    #flip the camera to be like mirror
    frame = cv2.flip(frame, 1)
    # Display the frame with hand landmarks and text
    writeOnImage(frame)
    cv2.imshow('Hand Recognition', frame)
    frames_elapsed += 1
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()