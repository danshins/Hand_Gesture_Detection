import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\Igor\Desktop\Crypto_Tracker\Hand_Gesture_Detection\training\asl_alphabet_model.h5')

# Define class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'del', 'space', 'nothing']

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Load the video file
video_path = r'C:\Users\Igor\Desktop\Crypto_Tracker\Hand_Gesture_Detection\utils\downloaded_video.mp4'  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit the loop if no more frames are available

    # Convert the frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform hand detection
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the frame (optional, for visualization)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract the bounding box around the detected hand
            h, w, c = frame.shape
            x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
            x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
            y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
            y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)

            # Ensure bounding box is within frame bounds
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Crop the hand region
            cropped_hand = frame[y_min:y_max, x_min:x_max]

            # Ensure the cropped hand is not empty
            if cropped_hand.size == 0:
                continue  # Skip to the next frame if the hand region is empty

            # Preprocess the cropped hand for prediction
            gray_hand = cv2.cvtColor(cropped_hand, cv2.COLOR_BGR2GRAY)
            resized_hand = cv2.resize(gray_hand, (64, 64))
            normalized_hand = resized_hand / 255.0
            reshaped_hand = normalized_hand.reshape(1, 64, 64, 1)

            # Perform prediction
            predictions = model.predict(reshaped_hand)
            predicted_class = np.argmax(predictions)
            predicted_label = class_labels[predicted_class]

            # Print the detected letter
            if predicted_label in class_labels[:26]:  # Only print letters (A-Z)
                print(f"Detected letter: {predicted_label}")

            # Display the prediction on the video frame
            cv2.putText(frame, predicted_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('ASL Gesture Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
