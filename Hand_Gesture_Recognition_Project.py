try:
    import Setup_and_Data_Loading as setup
    import Data_Preprocessing as dp
    import Exploratory_Analysis as ea
    import Feature_Extraction as fe
    import Model_Training as mt
    import Model_Evaluation as me
    from Program_Monitor import progress_bar
    import joblib
    import os
    from joblib import parallel_backend
except Exception as e:
    print(f"ERROR: {e}")

def process_data(models_folder):
    print("\nStep 1/4: Setting up and loading data...")
    data, labels = setup.load_dataset("leapGestRecog")
    progress_bar("Data Loading", 1, 1)

    print("\nStep 2/4: Preprocessing data...")
    processed_data, processed_labels = dp.preprocess_data(data, labels)
    progress_bar("Data Preprocessing", 1, 1)

    print("\nStep 3/4: Performing exploratory analysis...")
    ea.perform_analysis(processed_data, processed_labels)

    print("\nStep 4/4: Extracting features...")
    flattened_features, cnn_features, labels = fe.extract_features(processed_data, processed_labels)
    progress_bar("Feature Extraction", 1, 1)

    # Splitting data into 80% training, 10% validation and 10% testing sets
    mt.split_data(cnn_features, flattened_features, labels)

def train_models(models_folder):
    selector = False
    print("Select model training option")
    select_model_training = input("(1: Train CNN, 2: Train SVM, 3: Train KNN, 4: Train all, press any other key to exit): ")
    if select_model_training == '1':
        # Train CNN model
        print("Training CNN model...")
        cnn_model = mt.train_cnn_model()
        # Save the trained models
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        temp_dir = os.path.join(models_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        with parallel_backend('loky', temp_folder=temp_dir):
            cnn_model.save(os.path.join(models_folder, "cnn_model.keras"))
    elif select_model_training == '2':
        # Train SVM model
        print("Training SVM model...")
        svm_model = mt.train_svm_model()
        # Save the trained models
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        temp_dir = os.path.join(models_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        with parallel_backend('loky', temp_folder=temp_dir):
            joblib.dump(svm_model, os.path.join(models_folder, "svm_model.pkl"))
    elif select_model_training == '3':
        # Train KNN model
        print("Training KNN model...")
        knn_model = mt.train_knn_model()
        # Save the trained models
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        temp_dir = os.path.join(models_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        with parallel_backend('loky', temp_folder=temp_dir):
            joblib.dump(knn_model, os.path.join(models_folder, "knn_model.pkl"))
    elif select_model_training == '4':
        print("\nTraining models...")
        cnn_model = mt.train_cnn_model()
        svm_model = mt.train_svm_model()
        knn_model = mt.train_knn_model()
        # Save the trained models
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)

        temp_dir = os.path.join(models_folder, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        with parallel_backend('loky', temp_folder=temp_dir):
            cnn_model.save(os.path.join(models_folder, "cnn_model.keras"))
            joblib.dump(svm_model, os.path.join(models_folder, "svm_model.pkl"))
            joblib.dump(knn_model, os.path.join(models_folder, "knn_model.pkl"))

    else:
        print("Invalid selection, no models trained.")

def evaluate_models():
    me.load_and_compare_metrics_with_visualizations_v7()

def Hand_Gesture_Recognition_Project(models_folder):
    print("===============================================")
    print("==      Hand Gesture Recognition Project     ==")
    print("===============================================")
    print("  Program by Bjørge Zagros Rysstad for DAT305  ")
    print()
    print("Starting main routine...")
    print("Ensure data folder ""leapGestRecog"" is located in same folder as this script.")
    while True:
        print("Select program.")
        user_selection = input("(1: Process data, 2: Train models, 3: Evaluate results, 4: Run full program): ")
        if user_selection == '1':
            process_data(models_folder)
            print("\nData processing complete!")
        elif user_selection == '2':
            train_models(models_folder)
            print("\nModel training complete!")
        elif user_selection == '3':
            evaluate_models()
            print("\nEvaluation complete!")
        elif user_selection == '4':
            process_data(models_folder)
            cnn_model = mt.train_cnn_model()
            svm_model = mt.train_svm_model()
            knn_model = mt.train_knn_model()
            # Save the trained models
            if not os.path.exists(models_folder):
                os.makedirs(models_folder)

            temp_dir = os.path.join(models_folder, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            with parallel_backend('loky', temp_folder=temp_dir):
                cnn_model.save(os.path.join(models_folder, "cnn_model.keras"))
                joblib.dump(svm_model, os.path.join(models_folder, "svm_model.pkl"))
                joblib.dump(knn_model, os.path.join(models_folder, "knn_model.pkl"))
            evaluate_models()
            print("\nProject complete!")
        else:
            print("Invalid input. Please select between 1-4.")


if __name__ == "__main__":
    models_folder = "saved_models"
    metrics_folder = "metrics"

    Hand_Gesture_Recognition_Project(models_folder)