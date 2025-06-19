import xgboost as xgb
import pandas as pd
from pathlib import Path
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():
    start_time = time.time()
    DATA_PATH = Path('./intermediate_datafiles_bouldering/')
    
    # Ensure only the combined file is used
    combined_file = DATA_PATH / 'chapter5_result_combined.csv'
    if combined_file.exists():
        input_files = [combined_file]
        print("Mode: Processing only the combined chapter 5 final result file...")
        print(f"Combined file found: {combined_file.name}")
    else:
        print(f"Combined file not found at '{combined_file}'. Please run bouldering_ch5.py with '--mode final' and '--source combined'.")
        return

    for input_file_path in input_files:
        print(f"\n\n======================================================")
        print(f"--- Processing file: {input_file_path.name} ---")
        print(f"======================================================\n")

        # Load the dataset
        data = pd.read_csv(input_file_path)

        # Preprocess the dataset
        # Drop unnecessary columns
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])

        # Define target columns and expected classes
        target_columns = ['labelEasy', 'labelMedium', 'labelHard']
        expected_classes = ['labelEasy', 'labelMedium', 'labelHard']

        # Combine target columns into a single target variable
        y = data[target_columns].idxmax(axis=1)  # Assumes one-hot encoding for target columns
        print(f"Unique classes in y before filtering: {y.unique()}")
        problematic_columns = ['temp_pattern_labelEasy', 'temp_pattern_labelMedium', 'temp_pattern_labelHard',\
                                'temp_pattern_labelEasy(b)labelEasy', 'temp_pattern_labelMedium(b)labelMedium', 'temp_pattern_labelHard(b)labelHard']
        # Filter valid classes
        y = y[y.isin(expected_classes)]
        X = data.drop(columns=target_columns+problematic_columns)  # Drop target columns from features
        X = X.loc[y.index]  # Ensure X matches the filtered y

        # Select valid feature columns (numeric and categorical)
        X = X.select_dtypes(include=['int', 'float', 'bool', 'category'])

        print(f"Unique classes in y after filtering: {y.unique()}")
        print(f"Number of samples after filtering: {len(y)}")

        # Encode target labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        print(f"Label encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

        # Train an XGBoost classifier
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=5, max_depth=3, learning_rate=0.1, random_state=12)
        model.fit(X_train, y_train)

        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        print(f"Accuracy for file {input_file_path.name}: {accuracy:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, model.predict(X_test))
        report = classification_report(y_test, model.predict(X_test), target_names=label_encoder.classes_)
        print(f'Confusion Matrix:\n{cm}')
        print(f'Classification Report:\n{report}')

        # Plot and save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=expected_classes, yticklabels=expected_classes)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'{input_file_path.stem}_confusion_matrix.png')
        plt.close()

        # Set a threshold for feature importance
        threshold = 0.02  # Adjust this value as needed

        # Filter features based on the threshold
        importance = model.feature_importances_
        features = pd.Series(importance, index=X.columns)
        filtered_features = features[features >= threshold]

        # Plot the filtered features
        plt.figure(figsize=(12, 8))
        filtered_features.sort_values(ascending=False).plot(kind='bar', color='skyblue')
        plt.title('Feature Importance (Filtered)', fontsize=14)
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.tight_layout()
        plt.savefig(f'{input_file_path.stem}_filtered_feature_importance.png')
        plt.close()

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()