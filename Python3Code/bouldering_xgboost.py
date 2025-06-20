import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from datetime import datetime

def grid_search_and_save_outputs():
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
    
    param_grid = {
        'n_estimators': [25, 50, 100],
        'max_depth': [1, 2, 3],
        'learning_rate': [0.01, 0.05, 0.1],
        'random_state': [1234]
    }
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Score: {grid_search.best_score_:.2f}")
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    print(f'Confusion Matrix:\n{cm}')
    print(f'Classification Report:\n{report}')
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=expected_classes, yticklabels=expected_classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'{combined_file.stem}_confusion_matrix_{timestamp}.png')
    plt.close()
    
    # Feature importance threshold
    importance_threshold = 0.0001  # Only show features with importance above this threshold

    # Extract feature importance and filter by threshold
    importance = best_model.feature_importances_
    feature_names = X.columns
    important_features = [(feature, score) for feature, score in zip(feature_names, importance) if score > importance_threshold]

    # Sort features by importance
    important_features = sorted(important_features, key=lambda x: x[1], reverse=True)

    # Prepare data for plotting
    features = [f[0] for f in important_features]
    scores = [f[1] for f in important_features]

    # Plot filtered feature importance
    plt.figure(figsize=(12, 8))  # Increase figure size for better readability
    plt.barh(features, scores, color='skyblue')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Feature Importance (Threshold > {importance_threshold})', fontsize=14)
    plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature at the top
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()  # Adjust layout to prevent cutting off labels
    plt.savefig(f'{combined_file.stem}_filtered_feature_importance_{timestamp}.png')
    plt.close()
    
    # Save classification report, accuracy, and F1 score to a text file
    report_file = f'{combined_file.stem}_classification_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")
        f.write("Classification Report:\n")
        f.write(report)

if __name__ == "__main__":
    grid_search_and_save_outputs()