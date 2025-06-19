import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

# Load the dataset
dataset_path = "./intermediate_datafiles_bouldering/chapter5_result_combined.csv"
data = pd.read_csv(dataset_path)

# Drop invalid columns (e.g., 'Unnamed: 0')
data = data.drop(columns=["Unnamed: 0"], errors="ignore")

# Define features and labels
features = data.drop(columns=["labelEasy", "labelMedium", "labelHard"])
labels = data[["labelEasy", "labelMedium", "labelHard"]]

# Combine labels into a single column for stratified sampling
labels_combined = labels.idxmax(axis=1)  # Assumes one-hot encoding

# Map string labels to numeric values
label_mapping = {"labelEasy": 0, "labelMedium": 1, "labelHard": 2}
labels_combined = labels_combined.map(label_mapping)

# Ensure all feature columns are numeric
features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

# Split the dataset into balanced train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    features, labels_combined, test_size=0.2, stratify=labels_combined, random_state=42
)

# Train an XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Evaluate the model
y_pred = xgb_model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print metrics
print(f"Model Accuracy: {accuracy}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)