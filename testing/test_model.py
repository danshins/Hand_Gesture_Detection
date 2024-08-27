import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Initial load of the trained model
model = load_model('asl_alphabet_model.h5')

# Define class labels (A-Z + additional labels)
class_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
                'del', 'space', 'nothing']

# Start video capture (webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame (resize, grayscale, normalize)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (64, 64))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = normalized_frame.reshape(1, 64, 64, 1)

    # Perform prediction
    predictions = model.predict(reshaped_frame)
    predicted_class = np.argmax(predictions)
    predicted_label = class_labels[predicted_class]

    # Display the prediction on the video feed
    cv2.putText(frame, predicted_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow('ASL Gesture Recognition', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
