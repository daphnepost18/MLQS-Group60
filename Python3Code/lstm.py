import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_curve, average_precision_score, classification_report
from datetime import datetime
import seaborn as sns

def lstm_classification():
    DATA_PATH = Path('./intermediate_datafiles_bouldering/')
    combined_file = DATA_PATH / 'chapter5_result_combined.csv'
    
    if not combined_file.exists():
        print(f"Combined file not found at '{combined_file}'. Please run bouldering_ch5.py with '--mode final' and '--source combined'.")
        return
    
    print(f"Processing file: {combined_file.name}")
    data = pd.read_csv(combined_file)

    # Preprocessing: Handle missing values
    data.fillna(0, inplace=True)  # Replace NaN values with 0 (or use mean/median if appropriate)

    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])
    
    target_columns = ['labelEasy', 'labelMedium', 'labelHard']
    expected_classes = ['labelEasy', 'labelMedium', 'labelHard']
    
    # Extract labels and features
    print("Extracting labels and features...")
    y = data[target_columns].idxmax(axis=1)
    y = y[y.isin(expected_classes)]
    
    # Use tqdm to show progress during feature selection
    print("Selecting features...")
    X = data.drop(columns=target_columns + [
        'temp_pattern_labelEasy', 'temp_pattern_labelMedium', 'temp_pattern_labelHard',
        'temp_pattern_labelEasy(b)labelEasy', 'temp_pattern_labelMedium(b)labelMedium', 'temp_pattern_labelHard(b)labelHard'
    ])
    X = X.loc[y.index]
    
    # Apply tqdm for progress during dtype filtering
    X_filtered = []
    for column in tqdm(X.columns, desc="Filtering columns by dtype"):
        if X[column].dtype in ['int64', 'float64', 'bool', 'category']:
            X_filtered.append(column)
    X = X[X_filtered]
    
    # Encode labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y = np.eye(len(label_encoder.classes_))[y]  # Convert to one-hot encoding

    # Reshape data for LSTM (assuming timesteps = 1 for simplicity)
    X = X.values
    X = X.reshape((X.shape[0], 1, X.shape[1]))  # Reshape to (samples, timesteps, features)

    # Random split into training and test sets
    print("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create DataLoaders
    batch_size = 16
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for time series

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.13363596286792953)
            self.fc1 = nn.Linear(hidden_size, 32)
            self.fc2 = nn.Linear(32, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden_state = lstm_out[:, -1, :]
            out = self.dropout(last_hidden_state)
            out = torch.relu(self.fc1(out))
            # out = torch.softmax(self.fc2(out), dim=1)
            return out

    input_size = X_train.shape[2]
    hidden_size = 109
    output_size = y_train.shape[1]
    model = LSTMModel(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005519741145744467)

    # Initialize lists to store training and validation metrics
    loss_history = []
    val_loss_history = []
    train_accuracy_history = []
    val_accuracy_history = []

    # Training loop
    epochs = 100
    previous_loss = float('inf')  # Initialize previous loss to infinity
    print("Training the model...")

    for epoch in tqdm(range(1, epochs + 1), desc="Epoch Progress"):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

        # Training phase
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, torch.argmax(y_batch, dim=1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_accuracy += (outputs.argmax(dim=1) == y_batch.argmax(dim=1)).sum().item()

        train_loss /= len(train_loader)
        train_accuracy /= len(train_dataset)
        loss_history.append(train_loss)  # Store training loss
        train_accuracy_history.append(train_accuracy)  # Store training accuracy

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)

    test_accuracy = (outputs.argmax(dim=1) == y_test_tensor.argmax(dim=1)).sum().item() / len(y_test_tensor)
    print(f"Test Accuracy: {test_accuracy:.2f}")

    # Plot training loss and accuracy
    plt.figure(figsize=(8, 6))
    epochs = range(1, len(loss_history) + 1)

    # Plot training loss
    plt.plot(epochs, loss_history, label="Train Loss", color="blue")
    plt.xlabel("Epochs", color="black")
    plt.ylabel("Loss", color="black")
    plt.tick_params(axis="y", labelcolor="black")
    plt.tick_params(axis="x", labelcolor="black")

    # Create a second y-axis for accuracy
    ax2 = plt.gca().twinx()
    ax2.plot(epochs, train_accuracy_history, label="Train Accuracy", color="green")
    ax2.set_ylabel("Accuracy", color="black")
    ax2.tick_params(axis="y", labelcolor="black")

    # Add title and legend
    plt.title("Training Loss and Accuracy", color="black")
    loss_legend = plt.Line2D([], [], color="blue", label="Train Loss")
    accuracy_legend = plt.Line2D([], [], color="green", label="Train Accuracy")
    plt.legend(handles=[loss_legend, accuracy_legend], loc="center right", facecolor="white", edgecolor="black", fontsize=10)

    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"train_loss_accuracy_{timestamp}.png")
    plt.close()

    # Generate confusion matrix
    y_pred = outputs.argmax(dim=1).cpu().numpy()
    y_true = y_test_tensor.argmax(dim=1).cpu().numpy()

    cm = confusion_matrix(y_true, y_pred)

    # Save confusion matrix plot with timestamp
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=expected_classes, yticklabels=expected_classes)
    plt.title(f"Confusion Matrix LSTM")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(f"confusion_matrix_{timestamp}.png")
    plt.close()

    # Binarize y_true for multi-class handling
    n_classes = outputs.shape[1]  # Number of classes (3 in this case)
    y_true_binarized = label_binarize(y_true, classes=range(n_classes))

    # Compute PR curve for each class
    precision = {}
    recall = {}
    average_precision = {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_binarized[:, i], outputs[:, i].cpu().numpy())
        average_precision[i] = average_precision_score(y_true_binarized[:, i], outputs[:, i].cpu().numpy())

    # Plot PR curves for all classes
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], label=f"Class {i} (AP = {average_precision[i]:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve for all classes")
    plt.legend(labels=expected_classes)
    plt.savefig(f"pr_curve_{timestamp}.png")
    plt.close()

    print(f"Plots saved with timestamp {timestamp}.")

    print(f'Classification report:\n{classification_report(y_true, y_pred, target_names=expected_classes)}')


if __name__ == "__main__":
    print("Starting LSTM classification...")
    lstm_classification()