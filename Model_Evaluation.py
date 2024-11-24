
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, classification_report

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, auc
from keras.models import load_model
import joblib
import pandas as pd
from Model_Training import plot_confusion_matrix, compute_metrics

def load_and_compare_metrics_with_visualizations_v7(
    models_folder="saved_models", metrics_folder="metrics", data_folder="metrics/data"
):
    model_names = ["CNN", "KNN", "SVM"]
    metrics_data = {}
    test_data = {}
    models = {}
    training_histories = {}
    auc_scores = {}

    # Ensure the figures/Evaluation directory exists
    figures_dir = "figures/Evaluation"
    os.makedirs(figures_dir, exist_ok=True)

    for model_name in model_names:
        model_metrics_folder = os.path.join(metrics_folder, model_name)

        if model_name == "CNN":
            data_folder_path = os.path.join(data_folder, "cnn_data")
            history_file = "CNN_training_history.json"  # Correct file for CNN
        else:
            data_folder_path = os.path.join(data_folder, "svm_knn_data")
            history_file = f"{model_name}_training_history.json"  # Correct files for KNN and SVM

        metrics_data[model_name] = {}

        try:
            # Load true and predicted labels
            y_true = np.load(os.path.join(model_metrics_folder, "true_labels.npy"))
            y_pred = np.load(os.path.join(model_metrics_folder, "predicted_labels.npy"))
            metrics_data[model_name]["y_true"] = y_true
            metrics_data[model_name]["y_pred"] = y_pred

            # Load the model
            if model_name == "CNN":
                model_path = os.path.join(models_folder, "cnn_model.keras")
                model = load_model(model_path)
            else:
                model_path = os.path.join(models_folder, f"{model_name.lower()}_model.pkl")
                model = joblib.load(model_path)

            # Load test data
            X_test = np.load(os.path.join(data_folder_path, "X_test.npy"))

            # Adjust X_test shape for CNN if grayscale images are used
            if model_name == "CNN" and X_test.shape[-1] == 1:
                X_test = np.repeat(X_test, 3, axis=-1)

            # Flatten test data for KNN and SVM
            if model_name in ["SVM", "KNN"]:
                X_test = X_test.reshape(X_test.shape[0], -1)

            test_data[model_name] = {"X_test": X_test}
            models[model_name] = model

            # Load training history if available
            history_path = os.path.join(data_folder_path, history_file)
            if os.path.exists(history_path):
                with open(history_path, "r") as f:
                    training_histories[model_name] = json.load(f)
            else:
                print(f"No training history found for {model_name}.")

        except Exception as e:
            print(f"Error loading data for {model_name}: {e}")

    # Plot and save training and validation accuracy/loss for CNN only
    if "CNN" in training_histories:
        try:
            history = training_histories["CNN"]
            epochs = range(1, len(history["accuracy"]) + 1)

            # Plot accuracy
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, history["accuracy"], label="Training Accuracy")
            plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
            plt.title("CNN Training and Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            accuracy_file = os.path.join(figures_dir, "CNN_training_validation_accuracy.png")
            plt.savefig(accuracy_file)
            plt.close()

            # Plot loss
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, history["loss"], label="Training Loss")
            plt.plot(epochs, history["val_loss"], label="Validation Loss")
            plt.title("CNN Training and Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            loss_file = os.path.join(figures_dir, "CNN_training_validation_loss.png")
            plt.savefig(loss_file)
            plt.close()

        except KeyError as e:
            print(f"Key error in training history for CNN: {e}")
        except Exception as e:
            print(f"Error plotting training history for CNN: {e}")

    # Visualization and analysis
    for model_name in model_names:
        try:
            y_true = metrics_data[model_name]["y_true"]
            y_pred = metrics_data[model_name]["y_pred"]
            X_test = test_data[model_name]["X_test"]
            model = models[model_name]

            # Confusion Matrix
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true),
            )
            plt.title(f"{model_name} Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            cm_file = os.path.join(figures_dir, f"{model_name}_confusion_matrix.png")
            plt.savefig(cm_file)
            plt.close()

            # Classification Report
            print(f"\n{model_name} Classification Report:")
            print(classification_report(y_true, y_pred))

            # Precision-Recall Curve & ROC Curve
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            n_classes = y_true_bin.shape[1]

            try:
                if model_name == "CNN":
                    y_pred_proba = model.predict(X_test)
                else:
                    if hasattr(model, "predict_proba"):
                        y_pred_proba = model.predict_proba(X_test)
                    else:
                        raise AttributeError("Model does not support predict_proba.")

                # ROC Curve
                plt.figure(figsize=(6, 5))
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
                plt.title(f"{model_name} ROC Curve")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend(loc="lower right")
                roc_file = os.path.join(figures_dir, f"{model_name}_roc_curve.png")
                plt.savefig(roc_file)
                plt.close()

                # Compute AUC Score and save it
                auc_score = roc_auc_score(y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
                auc_scores[model_name] = auc_score
                print(f"{model_name} AUC Score: {auc_score:.4f}")

                # Plot AUC Score as a separate plot
                plt.figure(figsize=(6, 5))
                plt.bar(model_name, auc_score, color='skyblue')
                plt.ylim(0, 1)
                plt.title(f"{model_name} AUC Score")
                plt.ylabel("AUC Score")
                auc_file = os.path.join(figures_dir, f"{model_name}_auc_score.png")
                plt.savefig(auc_file)
                plt.close()

            except Exception as e:
                print(f"Error generating curves for {model_name}: {e}")

        except KeyError:
            print(f"Skipping visualization for {model_name}. Data not loaded correctly.")

    # Convert metrics data into a DataFrame for comparison
    model_metrics = {}
    for model_name in model_names:
        report = classification_report(
            metrics_data[model_name]["y_true"],
            metrics_data[model_name]["y_pred"],
            output_dict=True,
        )
        model_metrics[model_name] = {
            "accuracy": report["accuracy"],
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1": report["macro avg"]["f1-score"],
            "AUC": auc_scores.get(model_name, np.nan),  # AUC is not included here
        }

    # Create a DataFrame for the metrics
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df.index.name = 'Model Name'

    # Display the metrics comparison table
    print("\nModel Metrics Comparison:\n")
    print(metrics_df)

    # Plot comparison of metrics excluding AUC score (as it's separate now)
    plt.figure(figsize=(10, 6))
    metrics_melted = metrics_df.reset_index().melt(id_vars="Model Name", var_name="Metric", value_name="Score")
    ax = sns.barplot(x="Model Name", y="Score", hue="Metric", data=metrics_melted)
    ax.set_title("Model Comparison - Accuracy, Precision, Recall, and F1")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.legend(title="Metrics", bbox_to_anchor=(1.05, 1), loc='best')

    metrics_comparison_file = os.path.join(figures_dir, "model_comparison_metrics_without_auc.png")
    plt.tight_layout()
    plt.savefig(metrics_comparison_file)
    plt.close()