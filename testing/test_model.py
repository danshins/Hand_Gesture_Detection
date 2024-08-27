import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model('asl_alphabet_model.h5')

# Define class labels
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# Load the downloaded video
video_path = r'C:\Users\Igor\Desktop\Crypto_Tracker\Hand_Gesture_Detection\downloaded_video.mp4'
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for gesture recognition
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))  # Resize to match model input
    normalized_frame = resized_frame / 255.0
    reshaped_frame = normalized_frame.reshape(1, 64, 64, 1)

    # Perform gesture prediction
    predictions = model.predict(reshaped_frame)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]

    # Display the predicted label on the video
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the video with the predicted gesture
    cv2.imshow('Gesture Recognition', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
