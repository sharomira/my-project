import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
import mediapipe as mp
import pygame

# **Load Pretrained Model**
model_path = "D:/desktop/learning 2/SCS 422/yawning_detection_mobilenetv2.h5"
model = tf.keras.models.load_model(model_path)

# **Class Labels**
class_labels = ["No Yawn", "Yawn"]

# **Initialize Pygame for Sound Alerts**
pygame.mixer.init()
alert_sound_path = "D:/desktop/learning 2/SCS 422/drowsiness.wav"
alert_sound = pygame.mixer.Sound(alert_sound_path)

# **Initialize MediaPipe Face Mesh**
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# **Webcam Setup**
cap = cv2.VideoCapture(0)
alert_playing = False  # Track alert state

# **Thresholds for Yawn Detection**
YAWN_THRESHOLD = 0.08  # Adjust if needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # **Preprocess the Frame for the Model**
    img = cv2.resize(frame, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize

    # **Predict Using MobileNetV2**
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    model_label = class_labels[class_index]

    # **Face Landmark Detection**
    yawn_detected_landmarks = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for upper and lower lip
            upper_lip = np.array([face_landmarks.landmark[13].x, face_landmarks.landmark[13].y])
            lower_lip = np.array([face_landmarks.landmark[14].x, face_landmarks.landmark[14].y])

            # Calculate mouth opening distance
            mouth_opening = np.linalg.norm(upper_lip - lower_lip)

            # If mouth opening is greater than threshold, mark as yawn
            if mouth_opening > YAWN_THRESHOLD:
                yawn_detected_landmarks = True

            # Draw landmark points for debugging
            h, w, _ = frame.shape
            cv2.circle(frame, (int(upper_lip[0] * w), int(upper_lip[1] * h)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(lower_lip[0] * w), int(lower_lip[1] * h)), 5, (0, 0, 255), -1)

    # **Final Decision: Combine Both Methods**
    final_label = "No Yawn"
    if model_label == "Yawn" or yawn_detected_landmarks:
        final_label = "Yawn"

    # **Display Prediction & Confidence**
    cv2.putText(frame, f"Status: {final_label} ({confidence:.2f})", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # **Play or Stop Sound Based on Yawning Detection**
    if final_label == "Yawn":
        if not alert_playing:
            alert_sound.play(-1)  # **(-1) makes the sound loop until stopped**
            alert_playing = True
    else:
        if alert_playing:
            alert_sound.stop()  # **STOP the sound immediately**
            alert_playing = False

    # **Show the Frame**
    cv2.imshow("Yawning Detection", frame)

    # **Exit When 'q' Is Pressed**
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# **Release Resources**
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
