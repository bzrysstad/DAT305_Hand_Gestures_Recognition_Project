import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import cv2


def perform_analysis(data, labels):
    # Ensure the data is in the correct shape (flattened for PCA/SVM)
    print("Flattening data for analysis...")
    data_flattened = tf.reshape(data, [data.shape[0], -1])  # Flatten data for PCA/SVM
    data_numpy = data_flattened.numpy()  # Convert to numpy array for compatibility with sklearn

    # Standardize the data before fitting models
    print("Standardizing data...")
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_numpy)

    # Ensure the folder for saving figures exists
    output_folder = "figures/Analysis"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. Class Distribution (Countplot)
    print("Visualizing class distribution...")
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels, hue=labels, legend=False, palette="Set2")  # No palette needed
    plt.title("Class Distribution of Gestures")
    plt.xlabel("Gesture Label")
    plt.ylabel("Count")
    class_distribution_path = os.path.join(output_folder, "class_distribution.png")
    plt.savefig(class_distribution_path)
    print(f"Figure saved to {class_distribution_path}")
    plt.close()

    # 2. Correlation Matrix (Heatmap)
    print("Displaying correlation matrix...")

    # Reduce the data size for large datasets by limiting the number of features to visualize
    max_features = 50  # Limit to the first 50 features or adjust this threshold
    subset_data = data_scaled[:, :max_features]

    # Calculate the correlation matrix on a subset
    correlation_matrix = pd.DataFrame(subset_data).corr()

    # Plotting the correlation matrix (for a subset of features)
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title("Correlation Matrix of Features (Subset)")
    correlation_matrix_path = os.path.join(output_folder, "correlation_matrix.png")
    plt.savefig(correlation_matrix_path)
    print(f"Figure saved to {correlation_matrix_path}")
    plt.close()

    # 3. PCA (Principal Component Analysis) for dimensionality reduction
    print("Applying PCA for 2D visualization...")
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=labels, palette="Set2", s=100, edgecolor='k')
    plt.title("PCA of Hand Gesture Data (2D Projection)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    pca_path = os.path.join(output_folder, "pca_projection.png")
    plt.savefig(pca_path)
    print(f"Figure saved to {pca_path}")
    plt.close()

    # 4. t-SNE (t-Distributed Stochastic Neighbor Embedding) for non-linear dimensionality reduction
    print("Applying t-SNE for 2D visualization...")

    data_sampled = data_scaled
    labels_sampled = labels

    pca_tsne = PCA(n_components=50)  # Reduce to 50 components before applying t-SNE
    data_pca_tsne = pca_tsne.fit_transform(data_sampled)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    data_tsne = tsne.fit_transform(data_pca_tsne)

    # Plot the t-SNE result
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data_tsne[:, 0], y=data_tsne[:, 1], hue=labels_sampled, palette="Set2", s=100, edgecolor='k')
    plt.title("t-SNE of Hand Gesture Data (2D Projection)")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    tsne_path = os.path.join(output_folder, "tsne_projection.png")
    plt.savefig(tsne_path)
    print(f"Figure saved to {tsne_path}")
    plt.close()

    # 5. Pairwise Feature Distributions (e.g., using seaborn pairplot)
    print("Displaying pairwise feature distributions...")
    if data_numpy.shape[0] < 1000:  # Limiting this to smaller datasets for performance reasons
        plt.figure(figsize=(10, 6))
        sns.pairplot(pd.DataFrame(data_scaled[:1000]), hue=labels[:1000], palette="Set2", diag_kind="kde")
        pairwise_path = os.path.join(output_folder, "pairwise_distribution.png")
        plt.savefig(pairwise_path)
        print(f"Figure saved to {pairwise_path}")
        plt.close()

    # 6. Image size scatterplots for each label
    print("Visualizing and saving edge maps for gesture labels...")
    unique_labels = np.unique(labels)
    saved_figures = []  # Track saved filenames for later display

    # Iterate through the labels in chunks of 4 (for 4 subplots per row)
    for i in range(0, len(unique_labels), 4):
        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))  # Adjust figure size for 4 subplots

        # Loop over the 4 subplots in the current figure
        for j in range(4):
            if i + j < len(unique_labels):  # Ensure we don't exceed the number of labels
                label = unique_labels[i + j]
                ax = axes[j]
                label_data = data[labels == label]

                # Display edges for the first image in the label's data
                image = label_data[0].numpy().squeeze()  # Show one image per label
                ax.imshow(image, cmap='gray')
                ax.set_title(f'Edges for Gesture Label {label}', y=1.1)  # Adjust label position
                ax.axis('off')  # Hide axes for better image display
            else:
                axes[j].axis('off')  # Hide the subplot if there is no corresponding label

        # Add whitespace at the top of the figure
        fig.subplots_adjust(top=0.85)
        plt.tight_layout()

        # Save each figure with a unique name
        figure_filename = f"edges_labels_{i}-{min(i + 3, len(unique_labels) - 1)}.png"
        figure_path = os.path.join(output_folder, figure_filename)
        plt.savefig(figure_path)
        print(f"Figure saved to {figure_path}")
        saved_figures.append(figure_path)  # Track this figure

        plt.close(fig)  # Close the figure to free memory

    # Display saved figures from this session
    for figure_path in saved_figures:
        img = plt.imread(figure_path)
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Displaying {os.path.basename(figure_path)}", y=1.05)
        plt.show()

    print("All figures saved and selected ones displayed successfully.")