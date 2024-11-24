import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import History
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
import tensorflow as tf
import json

from Data_Visualization import plot_training_history, plot_confusion_matrix


# Split the data into training, validation, and testing sets
def split_data(cnn_features, flattened_features, labels, metrics_folder="metrics"):
    try:
        print("Splitting data...")

        # Calculate split sizes
        test_ratio = 0.1  # 10% for testing
        val_ratio = 0.1  # 10% for validation
        train_ratio = 1 - test_ratio - val_ratio  # 80% for training

        # Convert CNN features and labels to numpy arrays to ensure proper indexing
        cnn_features = np.array(cnn_features)
        flattened_features = np.array(flattened_features)
        labels = np.array(labels)

        # Split CNN features (4D data for CNN model)
        X_train_cnn, X_temp_cnn, y_train, y_temp = train_test_split(
            cnn_features, labels, test_size=(1 - train_ratio), random_state=42
        )
        X_val_cnn, X_test_cnn, y_val, y_test = train_test_split(
            X_temp_cnn, y_temp, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
        )

        # Split flattened features (for SVM/KNN model)
        X_train_knn, X_temp_knn, y_train_knn, y_temp_knn = train_test_split(
            flattened_features, labels, test_size=(1 - train_ratio), random_state=42
        )
        X_val_knn, X_test_knn, y_val_knn, y_test_knn = train_test_split(
            X_temp_knn, y_temp_knn, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42
        )

        print("Data split successfully.")

        # Calculate and display percentage breakdown for CNN
        total_samples_cnn = len(cnn_features)
        train_percentage_cnn = (len(X_train_cnn) / total_samples_cnn) * 100
        val_percentage_cnn = (len(X_val_cnn) / total_samples_cnn) * 100
        test_percentage_cnn = (len(X_test_cnn) / total_samples_cnn) * 100

        print(f"CNN - Training set: {len(X_train_cnn)} samples ({train_percentage_cnn:.2f}%)")
        print(f"CNN - Validation set: {len(X_val_cnn)} samples ({val_percentage_cnn:.2f}%)")
        print(f"CNN - Testing set: {len(X_test_cnn)} samples ({test_percentage_cnn:.2f}%)")

        # Calculate and display percentage breakdown for KNN
        total_samples_knn = len(flattened_features)
        train_percentage_knn = (len(X_train_knn) / total_samples_knn) * 100
        val_percentage_knn = (len(X_val_knn) / total_samples_knn) * 100
        test_percentage_knn = (len(X_test_knn) / total_samples_knn) * 100

        print(f"SVM/KNN - Training set: {len(X_train_knn)} samples ({train_percentage_knn:.2f}%)")
        print(f"SVM/KNN - Validation set: {len(X_val_knn)} samples ({val_percentage_knn:.2f}%)")
        print(f"SVM/KNN - Testing set: {len(X_test_knn)} samples ({test_percentage_knn:.2f}%)")

        # Create 'cnn_data' and 'svm_knn_data' folders inside 'metrics/data' if they don't exist
        cnn_data_folder = os.path.join(metrics_folder, 'data', 'cnn_data')
        svm_knn_data_folder = os.path.join(metrics_folder, 'data', 'svm_knn_data')
        os.makedirs(cnn_data_folder, exist_ok=True)
        os.makedirs(svm_knn_data_folder, exist_ok=True)

        # Save the CNN data (without flattening)
        np.save(os.path.join(cnn_data_folder, "X_train.npy"), X_train_cnn)
        np.save(os.path.join(cnn_data_folder, "X_val.npy"), X_val_cnn)
        np.save(os.path.join(cnn_data_folder, "X_test.npy"), X_test_cnn)
        np.save(os.path.join(cnn_data_folder, "y_train.npy"), y_train)
        np.save(os.path.join(cnn_data_folder, "y_val.npy"), y_val)
        np.save(os.path.join(cnn_data_folder, "y_test.npy"), y_test)

        # Save the flattened data for KNN/SVM
        np.save(os.path.join(svm_knn_data_folder, "X_train.npy"), X_train_knn)
        np.save(os.path.join(svm_knn_data_folder, "X_val.npy"), X_val_knn)
        np.save(os.path.join(svm_knn_data_folder, "X_test.npy"), X_test_knn)
        np.save(os.path.join(svm_knn_data_folder, "y_train.npy"), y_train_knn)
        np.save(os.path.join(svm_knn_data_folder, "y_val.npy"), y_val_knn)
        np.save(os.path.join(svm_knn_data_folder, "y_test.npy"), y_test_knn)

        # Verify that the files have been saved
        for folder in [cnn_data_folder, svm_knn_data_folder]:
            for file_name in ["X_train.npy", "X_val.npy", "X_test.npy", "y_train.npy", "y_val.npy", "y_test.npy"]:
                file_path = os.path.join(folder, file_name)
                if os.path.exists(file_path):
                    print(f"Successfully saved {file_name} in {folder}")
                else:
                    print(f"Error: {file_name} was not saved in {folder}")

        return X_train_cnn, X_val_cnn, X_test_cnn, y_train, y_val, y_test

    except Exception as e:
        print(f"Error in split_data: {e}")
        raise e

def compute_metrics(y_true, y_pred, model_name, save_dir="metrics"):
    # Compute metrics
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    class_report = classification_report(y_true, y_pred, output_dict=True)  # dict format for JSON

    # Print metrics
    print(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
    print(f"\n{model_name} - Classification Report:")
    print(classification_report(y_true, y_pred))

    # Create subfolder for model_name
    model_save_path = os.path.join(save_dir, model_name)
    os.makedirs(model_save_path, exist_ok=True)

    # Save metrics to a .txt file
    metrics_txt_file = os.path.join(model_save_path, f"{model_name}_metrics.txt")
    with open(metrics_txt_file, "w") as file:
        file.write(f"{model_name} - Metrics\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1-score: {f1:.4f}\n")

    # Save classification report to a .txt file
    class_report_txt_file = os.path.join(model_save_path, f"{model_name}_classification_report.txt")
    with open(class_report_txt_file, "w") as file:
        file.write(f"{model_name} - Classification Report\n")
        file.write(classification_report(y_true, y_pred))

    # Save metrics and classification report to a .json file
    metrics_json_file = os.path.join(model_save_path, f"{model_name}_metrics.json")
    with open(metrics_json_file, "w") as file:
        json.dump(
            {
                "model_name": model_name,
                "metrics": {
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1,
                },
                "classification_report": class_report,
            },
            file,
            indent=4
        )

    print(f"Metrics and classification reports saved in '{model_save_path}'")


def save_labels(y_true, y_pred, model_name):
    folder_path = os.path.join('metrics', model_name)
    os.makedirs(folder_path, exist_ok=True)

    np.save(os.path.join(folder_path, 'true_labels.npy'), y_true)
    np.save(os.path.join(folder_path, 'predicted_labels.npy'), y_pred)
    print(f"Saved true and predicted labels for {model_name} in {folder_path}")


# Save training history
def save_training_history(history, model_name, folder_path):
    history_path = os.path.join(folder_path, f"{model_name}_training_history.json")
    with open(history_path, "w") as file:
        json.dump(history, file)
    print(f"Saved {model_name} training history to {history_path}")


def train_cnn_model():
    data_folder = os.path.join('metrics', 'data', 'cnn_data')
    try:
        X_train = np.load(os.path.join(data_folder, 'X_train.npy'))
        X_val = np.load(os.path.join(data_folder, 'X_val.npy'))
        X_test = np.load(os.path.join(data_folder, 'X_test.npy'))
        y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
        y_val = np.load(os.path.join(data_folder, 'y_val.npy'))
        y_test = np.load(os.path.join(data_folder, 'y_test.npy'))
        print(f"Loaded CNN data from {data_folder}")
    except Exception as e:
        print(f"Error loading CNN data: {e}")
        return

    if X_train.shape[-1] == 1:
        X_train = np.repeat(X_train, 3, axis=-1)
        X_val = np.repeat(X_val, 3, axis=-1)
        X_test = np.repeat(X_test, 3, axis=-1)

    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])

    cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

    # Save training history
    save_training_history(history.history, "CNN", data_folder)

    # Evaluate on the test set
    y_pred = cnn_model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Save true and predicted labels
    save_labels(y_test, y_pred, "CNN")

    # Evaluate metrics
    compute_metrics(y_test, y_pred, "CNN")
    plot_confusion_matrix(y_test, y_pred, "CNN")

    return cnn_model


def train_svm_model():
    data_folder = os.path.join('metrics', 'data', 'svm_knn_data')
    try:
        X_train = np.load(os.path.join(data_folder, 'X_train.npy'))
        X_val = np.load(os.path.join(data_folder, 'X_val.npy'))
        X_test = np.load(os.path.join(data_folder, 'X_test.npy'))
        y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
        y_val = np.load(os.path.join(data_folder, 'y_val.npy'))
        y_test = np.load(os.path.join(data_folder, 'y_test.npy'))
        print(f"Loaded SVM data from {data_folder}")
    except Exception as e:
        print(f"Error loading SVM data: {e}")
        return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_model = SVC(kernel='linear', verbose=True, probability=True)
    svm_model.fit(X_train_scaled, y_train)

    # Save simulated training history
    history = {"accuracy": [svm_model.score(X_train_scaled, y_train)], "val_accuracy": [svm_model.score(X_test_scaled, y_test)]}
    save_training_history(history, "SVM", data_folder)

    y_pred = svm_model.predict(X_test_scaled)

    # Save true and predicted labels
    save_labels(y_test, y_pred, "SVM")

    compute_metrics(y_test, y_pred, "SVM")
    plot_confusion_matrix(y_test, y_pred, "SVM")

    return svm_model


def train_knn_model():
    data_folder = os.path.join('metrics', 'data', 'svm_knn_data')
    try:
        X_train = np.load(os.path.join(data_folder, 'X_train.npy'))
        X_val = np.load(os.path.join(data_folder, 'X_val.npy'))
        X_test = np.load(os.path.join(data_folder, 'X_test.npy'))
        y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
        y_val = np.load(os.path.join(data_folder, 'y_val.npy'))
        y_test = np.load(os.path.join(data_folder, 'y_test.npy'))
        print(f"Loaded KNN data from {data_folder}")
    except Exception as e:
        print(f"Error loading KNN data: {e}")
        return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn_model = KNeighborsClassifier(n_neighbors=3)
    knn_model.fit(X_train_scaled, y_train)

    # Save simulated training history
    history = {"accuracy": [knn_model.score(X_train_scaled, y_train)], "val_accuracy": [knn_model.score(X_test_scaled, y_test)]}
    save_training_history(history, "KNN", data_folder)

    y_pred = knn_model.predict(X_test_scaled)

    # Save true and predicted labels
    save_labels(y_test, y_pred, "KNN")

    compute_metrics(y_test, y_pred, "KNN")
    plot_confusion_matrix(y_test, y_pred, "KNN")

    return knn_model