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
prev_wrist_x = None
movement = 0

def writeOnImage(frame):
    global isWaving
    if frames_elapsed < CALIBRATION_FRAMES:
        text = "Calibrating"
    elif isWaving:
        text = "Waving"
    else:
        text = "None"

    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,( 0 , 0 , 0 ),2,cv2.LINE_AA)
    cv2.putText(frame, text, (10,20), cv2.FONT_HERSHEY_COMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)

def checkWaving(movement):
    global isWaving

    if abs(movement) > 20:
        isWaving = True
    else:
        isWaving = False

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
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x
            wrist_x_pixels = int(wrist_x * frame_width)

            if prev_wrist_x is not None:
                movement = wrist_x_pixels - prev_wrist_x

            prev_wrist_x = wrist_x_pixels

            if frames_elapsed % 8 == 0:
                checkWaving(movement)

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # for i, landmark in enumerate(hand_landmarks.landmark):
            #     if i > 0 and frames_elapsed % 8 == 0:
            #         prevCenterX = hand_landmarks.landmark[i-1].x
            #         checkWaving(landmark.x)
            #         # print(prevCenterX, landmark.x)
    
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