import xgboost as xgb
import pandas as pd
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def main():
    start_time = time.time()
    DATA_PATH = Path('./intermediate_datafiles_bouldering/')
    
    combined_file = DATA_PATH / 'chapter5_result_combined.csv'
    if not combined_file.exists():
        print(f"Combined file not found at '{combined_file}'. Please run bouldering_ch5.py with '--mode final' and '--source combined'.")
        return

    print(f"Processing file: {combined_file.name}")
    data = pd.read_csv(combined_file)

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    target_columns = ['labelEasy', 'labelMedium', 'labelHard']
    expected_classes = ['labelEasy', 'labelMedium', 'labelHard']

    y = data[target_columns].idxmax(axis=1)
    y = y[y.isin(expected_classes)]
    X = data.drop(columns=target_columns + [
        'temp_pattern_labelEasy', 'temp_pattern_labelMedium', 'temp_pattern_labelHard',
        'temp_pattern_labelEasy(b)labelEasy', 'temp_pattern_labelMedium(b)labelMedium', 'temp_pattern_labelHard(b)labelHard'
    ])
    X = X.loc[y.index].select_dtypes(include=['int', 'float', 'bool', 'category'])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, model.predict(X_test))
    report = classification_report(y_test, model.predict(X_test), target_names=label_encoder.classes_)
    print(f'Confusion Matrix:\n{cm}')
    print(f'Classification Report:\n{report}')

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=expected_classes, yticklabels=expected_classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{combined_file.stem}_confusion_matrix.png')
    plt.close()

    threshold = 0.02
    importance = model.feature_importances_
    features = pd.Series(importance, index=X.columns)
    filtered_features = features[features >= threshold]

    plt.figure(figsize=(12, 8))
    filtered_features.sort_values(ascending=False).plot(kind='bar', color='skyblue')
    plt.title('Feature Importance (Filtered)', fontsize=14)
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{combined_file.stem}_filtered_feature_importance.png')
    plt.close()

    print(f"Total processing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()