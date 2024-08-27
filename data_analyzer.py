import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def load_data(dataset_path, img_size=64):
    """
    Load and preprocess the dataset.

    Args:
        dataset_path (str): Path to the dataset directory.
        img_size (int): Size to which images should be resized.

    Returns:
        tuple: Tuple containing preprocessed images and their labels.
    """
    data = []
    labels = []

    for label in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, label)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            data.append(img)
            labels.append(label)

    data = np.array(data).reshape(-1, img_size, img_size, 1)  # Reshape to include the channel dimension
    data = data / 255.0  # Normalize the data

    # Encode labels as integers (A=0, B=1, ..., Z=25)
    label_map = {chr(65 + i): i for i in range(26)}
    labels = np.array([label_map[label] for label in labels])

    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=26)

    return data, labels


def split_data(data, labels, test_size=0.2):
    """
    Split the dataset into training and validation sets.

    Args:
        data (numpy.ndarray): Array of images.
        labels (numpy.ndarray): Array of labels.

    Returns:
        tuple: Training and validation data and labels.
    """
    return train_test_split(data, labels, test_size=test_size, random_state=42)


def visualize_samples(data, labels, label_map, num_samples=5):
    """
    Display sample images and their labels.

    Args:
        data (numpy.ndarray): Array of images.
        labels (numpy.ndarray): Array of labels (one-hot encoded).
        label_map (dict): Dictionary mapping label indices to characters.
        num_samples (int): Number of samples to display.
    """
    plt.figure(figsize=(10, 10))
    for i in range(num_samples):
        index = np.random.randint(0, len(data))
        img = data[index].reshape(64, 64)
        label = np.argmax(labels[index])
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {list(label_map.keys())[list(label_map.values()).index(label)]}")
        plt.axis('off')
    plt.show()


def main():
    dataset_path = r'C:\Users\Igor\Downloads\archive\asl_alphabet_train\asl_alphabet_train'  # Replace with your dataset path
    img_size = 64

    # Load and preprocess the data
    data, labels = load_data(dataset_path, img_size)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = split_data(data, labels)

    # Print the shapes of the datasets
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")

    # Visualize some sample images
    label_map = {chr(65 + i): i for i in range(26)}
    visualize_samples(X_train, y_train, label_map)


if __name__ == "__main__":
    main()
