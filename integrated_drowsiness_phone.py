import cv2
import dlib
import numpy as np
import tensorflow as tf
import mediapipe as mp
from ultralytics import YOLO
import pygame
from scipy.spatial import distance as dist

# Load YOLOv8 model for phone detection
yolo_model = YOLO("yolov8n.pt")

# Load MobileNetV2 model for drowsiness detection
drowsiness_model = tf.keras.models.load_model("drowsiness_mobilenetv2.h5")

# Load face detector and landmark predictor for EAR calculation
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize pygame for sound alerts
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("drowsiness.wav")  # Use pygame Sound

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Eye landmark indices
LEFT_EYE = list(range(42, 48))
RIGHT_EYE = list(range(36, 42))

# Drowsiness detection thresholds
EAR_THRESHOLD = 0.3  # Adjusted for better sensitivity
FRAME_THRESHOLD = 10  # Reduce frames to trigger alert faster

# Initialize counters
frame_counter = 0
alert_active = False

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # **Drowsiness Detection**
    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)

        # Get eye landmarks
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE])

        # Compute EAR
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        print(f"EAR: {avg_EAR}")  # Debugging EAR values

        # **Check if drowsy**
        if avg_EAR < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= FRAME_THRESHOLD:
                label = "DROWSY! WAKE UP!"
                color = (0, 0, 255)

                # Extract face for MobileNetV2 verification
                face_roi = frame[max(face.top(), 0):min(face.bottom(), h),
                                 max(face.left(), 0):min(face.right(), w)]

                if face_roi.size == 0:
                    print("Face ROI is empty, skipping prediction")
                    continue  # Skip invalid regions

                face_roi = cv2.resize(face_roi, (224, 224))
                face_roi = face_roi.astype("float32") / 255.0
                face_roi = np.expand_dims(face_roi, axis=0)

                prediction = drowsiness_model.predict(face_roi)[0][0]
                print(f"MobileNetV2 Prediction: {prediction}")  # Debugging

                if prediction < 0.5:
                    if not alert_active:
                        print("Drowsiness detected! Playing alert...")
                        alarm_sound.play()
                        alert_active = True
        else:
            frame_counter = 0

            label = "Awake"

            color = (0, 255, 0)
            if alert_active:
                print("Drowsiness cleared, stopping alert...")
                alarm_sound.stop()
                alert_active = False

        # Display EAR and status
        label = "Awake"
        color = (0, 255, 0)
        cv2.putText(frame, label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # **Phone Usage Detection with YOLOv8**
    results = yolo_model(frame)
    phone_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 67 and conf > 0.5:
                phone_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Phone Detected", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # **Hand Tracking with MediaPipe**
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hand_results = hands.process(rgb_frame)
    hand_moving = False

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

            # Detect movement of index finger tip
            index_finger_tip = landmarks[8]
            if 'prev_index_finger_pos' in locals():
                movement = np.linalg.norm(np.array(index_finger_tip) - np.array(prev_index_finger_pos))
                if movement > 15:
                    hand_moving = True
            prev_index_finger_pos = index_finger_tip

    # **Check if using phone while driving**
    if phone_detected and hand_moving:
        cv2.putText(frame, "WARNING: Phone Usage While Driving!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # **Show Frame**
    cv2.imshow("Driver Safety System (Drowsiness + Phone Usage)", frame)

    # **Exit on 'q' key**
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
