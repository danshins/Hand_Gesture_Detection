
# Hand Gesture Recognition using ASL Alphabet

This project is a **hand gesture recognition system** that uses computer vision and deep learning to translate American Sign Language (ASL) gestures into text. The model is trained to recognize 26 gestures, each representing a letter of the alphabet (A-Z).

## Features
- **Real-time Hand Gesture Recognition:** Detect and classify hand gestures captured from a webcam.
- **ASL Alphabet Support:** Recognizes all 26 letters of the ASL alphabet.
- **Integration with YouTube Videos:** Test the model on pre-recorded YouTube videos.
- **Simple and Efficient CNN Model:** The model is designed for quick training and accurate recognition.

## Project Structure
```
Hand_Gesture_Detection/
│
├── training/
│   └── cnn_training.py          # Script to train the CNN model
├── testing/
│   └── test_model.py            # Script to test the trained model
├── utils/
│   └── yt_dl.py                 # Script to download and test YouTube videos
├── data_analyzer.py             # Script for data analysis and visualization
├── downloaded_video.mp4         # Example video for testing
├── init.py                      # Placeholder script (consider renaming or removing)
└── README.md                    # Project documentation
```

## Getting Started

### Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.x
- OpenCV
- TensorFlow / Keras
- MediaPipe
- Pytube (for YouTube video testing)

You can install the dependencies using:
```bash
pip install -r requirements.txt
```

### Dataset
The project uses the **ASL Alphabet Dataset**, which contains thousands of images for each letter (A-Z). You can download it from [Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

Place the dataset in the `dataset/` directory and ensure the folder structure is correct:
```
dataset/
│
├── A/
├── B/
├── C/
└── ...
```

### Running the Project

1. **Train the Model:**
   If you want to train the model from scratch, run:
   ```bash
   python training/cnn_training.py
   ```

2. **Real-Time Gesture Recognition:**
   To test the model using your webcam:
   ```bash
   python testing/test_model.py
   ```

3. **Test on YouTube Videos:**
   You can download and test the model on a YouTube video:
   ```bash
   python utils/yt_dl.py
   ```

### How it Works
The project uses a convolutional neural network (CNN) to classify hand gestures. The real-time recognition is powered by OpenCV for video capture and MediaPipe for hand landmark detection.

## Future Improvements
- Expand the dataset to include more complex gestures and words.
- Implement dynamic gesture recognition (e.g., full words or phrases).
- Add a GUI for easier interaction.

## Contributing
Feel free to open issues or submit pull requests if you have suggestions or improvements!

## License
This project is licensed under the MIT License.
