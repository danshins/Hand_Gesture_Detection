import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model(r'C:\Users\Igor\Desktop\Crypto_Tracker\Hand_Gesture_Detection\training\asl_alphabet_model.h5')

# Define class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'del', 'space', 'nothing']

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

    # Preprocess the frame (resize, grayscale, normalize)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = normalized_frame.reshape(1, 64, 64, 1)

    # Perform prediction
    predictions = model.predict(reshaped_frame)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]

    # Print the detected letter
    if predicted_label in class_labels[:26]:  # Only print letters (A-Z)
        print(f"Detected letter: {predicted_label}")

    # Display the prediction on the video frame
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('ASL Gesture Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
