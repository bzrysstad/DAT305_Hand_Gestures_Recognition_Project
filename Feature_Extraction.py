import tensorflow as tf
import numpy as np

def extract_features(data, labels):
    try:
        print("Extracting features...")

        # Flatten features for SVM/KNN
        flattened_features = tf.reshape(data, [data.shape[0], -1]).numpy()

        # Standardize flattened features for SVM/KNN
        flattened_features = (flattened_features - np.mean(flattened_features, axis=0)) / (
            np.std(flattened_features, axis=0) + 1e-8
        )

        # Convert TensorFlow tensor to NumPy array for CNN features
        cnn_features = data.numpy()  # Ensure compatibility with train_test_split

        # Ensure both are numpy arrays for compatibility with train_test_split
        flattened_features = np.array(flattened_features)
        cnn_features = np.array(cnn_features)

        print(f"Flattened Features Shape: {flattened_features.shape}")
        print(f"CNN Features Shape: {cnn_features.shape}")

        return flattened_features, cnn_features, labels

    except Exception as e:
        print(f"Error in extract_features: {e}")
        raise e


def extract_features_Old(data, labels):
    try:
        print("Extracting features...")

        # Flatten features for SVM/KNN
        flattened_features = tf.reshape(data, [data.shape[0], -1]).numpy()

        # Standardize flattened features for SVM/KNN
        flattened_features = (flattened_features - np.mean(flattened_features, axis=0)) / (
            np.std(flattened_features, axis=0) + 1e-8
        )

        # CNN features remain as-is
        cnn_features = data  # Already in correct shape

        print(f"Flattened Features Shape: {flattened_features.shape}")
        print(f"CNN Features Shape: {cnn_features.shape}")

        return flattened_features, cnn_features, labels

    except Exception as e:
        print(f"Error in extract_features: {e}")
        raise e