import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import xgboost as xgb

# Load the dataset
dataset_path = "./intermediate_datafiles_bouldering/chapter5_result_combined.csv"
data = pd.read_csv(dataset_path)

# Drop invalid columns (e.g., 'Unnamed: 0')
data = data.drop(columns=["Unnamed: 0"], errors="ignore")

# Assign a name to the first column (timestamp) for clarity
data.rename(columns={data.columns[0]: "timestamp"}, inplace=True)

# Separate instances based on consecutive one-hot labels and timestamps
data["instance_id"] = (data[["labelEasy", "labelMedium", "labelHard"]]
                       .idxmax(axis=1)  # Get the label column name
                       .ne(data[["labelEasy", "labelMedium", "labelHard"]]
                           .idxmax(axis=1).shift())  # Compare with previous row
                       .cumsum())  # Create unique instance IDs

# Print the instance IDs for verification
print("Instance IDs:")
print(data["instance_id"].unique())  # Print unique instance IDs

# Verify instance separation using the renamed timestamp column
data = data.sort_values(by=["instance_id", "timestamp"])

# Aggregate features for each instance (e.g., calculate mean, max, min, etc.)
grouped_data = data.groupby("instance_id")
aggregated_features = grouped_data.agg(["mean", "max", "min"]).reset_index()

# Flatten multi-level columns created by aggregation
aggregated_features.columns = [
    f"{col[0]}_{col[1]}" if col[1] else col[0] for col in aggregated_features.columns
]

# Extract labels for each instance (assuming labels are consistent within each group)
aggregated_labels = grouped_data[["labelEasy", "labelMedium", "labelHard"]].first()

# Combine labels into a single column for stratified sampling
labels_combined = aggregated_labels.idxmax(axis=1)  # Assumes one-hot encoding

# Map string labels to numeric values
label_mapping = {"labelEasy": 0, "labelMedium": 1, "labelHard": 2}
labels_combined = labels_combined.map(label_mapping)

# Ensure all feature columns are numeric
aggregated_features = aggregated_features.apply(pd.to_numeric, errors="coerce").fillna(0)

# Verify the number of instances
print("Total number of instances:", aggregated_features.shape[0])

# Add instance IDs back to the features for tracking
aggregated_features["instance_id"] = grouped_data["instance_id"].first().values

# Split the dataset into balanced train-test sets
X_train, X_test, y_train, y_test = train_test_split(
    aggregated_features, labels_combined, test_size=0.2, stratify=labels_combined, random_state=123
)

# Print instance IDs and labels used for testing
print("Instance IDs and labels used for testing:")
test_instances = X_test[["instance_id"]].copy()
test_instances["label"] = y_test.values
print(test_instances)

# Train an XGBoost model
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train.drop(columns=["instance_id"]), y_train)  # Drop instance_id for training

# Evaluate the model
y_pred = xgb_model.predict(X_test.drop(columns=["instance_id"]))  # Drop instance_id for testing

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