import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm  # For progress bars

def load_dataset(dataset_path):
    print(f"Loading dataset from {dataset_path}...")

    data = []
    labels = []

    # Iterate through subfolders
    subjects = sorted(os.listdir(dataset_path))
    for subject in tqdm(subjects, desc="Subjects"):
        subject_path = os.path.join(dataset_path, subject)
        if os.path.isdir(subject_path):
            gestures = sorted(os.listdir(subject_path))
            for gesture in tqdm(gestures, desc=f"Gestures in {subject}", leave=False):
                gesture_path = os.path.join(subject_path, gesture)
                if os.path.isdir(gesture_path):
                    for file in os.listdir(gesture_path):
                        file_path = os.path.join(gesture_path, file)
                        try:
                            # Load raw image without preprocessing
                            image = tf.io.read_file(file_path)
                            image = tf.image.decode_jpeg(image, channels=1)  # Load grayscale
                            data.append(image.numpy())  # Keep raw image as numpy array
                            labels.append(int(gesture.split('_')[0]) - 1)  # Zero-based labels
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")

    print("Dataset loaded successfully.")
    return np.array(data, dtype=np.uint8), np.array(labels, dtype=np.int32)