import os
import re
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Path to the folder containing CSV files
folder_path = './intermediate_datafiles_bouldering/xgboost'
figure_folder_path = './intermediate_datafiles_bouldering/figures_xgboost'  #TODO change to correct path

if not os.path.exists(figure_folder_path):
    os.makedirs(figure_folder_path)

# Initialize lists to store dataframes by label
easy_instances = []
medium_instances = []
hard_instances = []

# Iterate through each file in the folder
for file_name in os.listdir(folder_path):
    # Use regex to check and extract the label from the filename
    match = re.search(r'(Easy|Medium|Hard)', file_name)
    if match:
        # Extract the label from the regex match
        label = match.group(1)
        
        # Load the CSV file
        file_path = os.path.join(folder_path, file_name)
        print(f"Processing file: {file_name}")  # Debugging step
        df = pd.read_csv(file_path)

        # Combine one-hot encoded columns into a single target column
        df['target'] = df[['labelEasy', 'labelMedium', 'labelHard']].idxmax(axis=1)

        # Map string labels to numeric values
        label_mapping = {'labelEasy': 0, 'labelMedium': 1, 'labelHard': 2}
        df['target'] = df['target'].map(label_mapping)

        # Append the dataframe to the correct list based on the label
        if label == 'Easy':
            easy_instances.append(df)
        elif label == 'Medium':
            medium_instances.append(df)
        elif label == 'Hard':
            hard_instances.append(df)

# Debugging: Print the number of instances in each list
print(f"Easy instances: {len(easy_instances)}")
print(f"Medium instances: {len(medium_instances)}")
print(f"Hard instances: {len(hard_instances)}")

# Ensure there are enough instances for each label
if not easy_instances or not medium_instances or not hard_instances:
    raise ValueError("Not enough instances for one or more labels. Please check the folder contents.")

# Randomly select one instance from each label for the test set --> Now hard coded, need to make this dynamic!!
test_set = [
    easy_instances.pop(0),  # Select the first instance (or use random.choice for randomness)
    medium_instances.pop(0),
    hard_instances.pop(0)
]

# Combine remaining instances into the training set
train_set = easy_instances + medium_instances + hard_instances

# Combine all dataframes into single DataFrames for train and test sets
train_data = pd.concat(train_set, ignore_index=True)
test_data = pd.concat(test_set, ignore_index=True)

# Drop unnecessary columns like 'Unnamed: 0' and one-hot encoded label columns
train_data = train_data.drop(columns=['Unnamed: 0', 'labelEasy', 'labelMedium', 'labelHard'], errors='ignore')
test_data = test_data.drop(columns=['Unnamed: 0', 'labelEasy', 'labelMedium', 'labelHard'], errors='ignore')

# Ensure all columns in X are numeric or categorical
X_train = train_data.drop(columns=['target'], errors='ignore')  # Features
X_train = X_train.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric columns to NaN and then numeric
X_train = X_train.dropna()  # Drop rows with NaN values (optional)

X_test = test_data.drop(columns=['target'], errors='ignore')  # Features
X_test = X_test.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric columns to NaN and then numeric
X_test = X_test.dropna()  # Drop rows with NaN values (optional)

# Target variable
y_train = train_data['target']
y_test = test_data['target']

# Train the XGBoost model
model = xgb.XGBClassifier(enable_categorical=False)  # Set enable_categorical=True if using categorical columns
model.fit(X_train, y_train)

class_names = ["Easy", "Medium", "Hard"]

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Print metrics in a more interpretable format
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")

# Save the figure to a file
plt.savefig(f"{figure_folder_path}/confusion_matrix_heatmap.png", dpi=300, bbox_inches='tight')