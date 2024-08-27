import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define the image size and dataset path
img_size = 64
dataset_path = r'C:\Users\Igor\Downloads\archive\asl_alphabet_train\asl_alphabet_train'  # Replace with your dataset path

# Define the valid labels including 'del', 'space', and 'nothing'
valid_labels = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                'V', 'W', 'X', 'Y', 'Z', 'del', 'space', 'nothing'}

data = []
labels = []

for label in os.listdir(dataset_path):
    if label not in valid_labels:
        continue  # Skip any unknown labels

    folder_path = os.path.join(dataset_path, label)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        data.append(img)
        labels.append(label)

data = np.array(data).reshape(-1, img_size, img_size, 1)
data = data / 255.0

# Update the label map to include 'del', 'space', and 'nothing'
label_map = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9,
    'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18,
    'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, 'Y': 24, 'Z': 25, 'del': 26, 'space': 27, 'nothing': 28
}

labels = np.array([label_map[label] for label in labels])

# Convert labels to one-hot encoding
labels = to_categorical(labels, num_classes=len(label_map))

X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Print the shapes of the datasets
print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(29, activation='softmax')  # Updated to 29 classes (A-Z + del + space + nothing)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

# Save the model
model.save('asl_alphabet_model.h5')
