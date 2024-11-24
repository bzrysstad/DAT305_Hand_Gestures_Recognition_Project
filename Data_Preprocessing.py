import tensorflow as tf
import numpy as np
import cv2

def preprocess_data(data, labels):
    try:
        processed_data = []

        # Iterate through each image in the dataset
        for image in data:
            # Convert image to a TensorFlow tensor if it's not already
            image = tf.convert_to_tensor(image, dtype=tf.float32)

            # Apply Gaussian blur (smoothen the image)
            image_blurred = tf.image.resize(image, (64, 64))  # Resize image to 64x64
            image_blurred = tf.image.adjust_brightness(image_blurred, delta=0.1)  # Adjust brightness

            # Edge detection using Canny (since the image is already grayscale)
            image_edges = image_blurred.numpy()  # Convert to numpy for use with OpenCV
            image_edges = cv2.Canny(image_edges.astype(np.uint8), 100, 200)

            # Convert back to TensorFlow tensor after applying Canny edge detection
            image_edges = tf.convert_to_tensor(image_edges, dtype=tf.float32)

            # Normalize pixel values to the range [0, 1]
            image = image_edges / 255.0  # Normalize image values to [0, 1]

            # Add channel dimension if missing (for CNN compatibility)
            if len(image.shape) == 2:  # If image is [height, width], add channel dimension
                image = tf.expand_dims(image, axis=-1)

            processed_data.append(image)

        processed_data = tf.stack(processed_data)  # Convert list of tensors to a single tensor

        # Standardize images: mean = 0, std = 1 across each image's pixel values
        mean = tf.reduce_mean(processed_data, axis=(1, 2, 3), keepdims=True)
        std = tf.math.reduce_std(processed_data, axis=(1, 2, 3), keepdims=True)
        processed_data = (processed_data - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero

        # Ensure labels are numpy arrays for consistency
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)

        return processed_data, labels

    except Exception as e:
        print(f"Error in preprocess_data: {e}")
        raise e