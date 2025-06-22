import pandas as pd
import numpy as np
import optuna
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
import time
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

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Define the Optuna objective function
    def objective(trial):
        hidden_size = trial.suggest_int("hidden_size", 16, 128)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.dropout = nn.Dropout(dropout_rate)
                self.fc1 = nn.Linear(hidden_size, 32)
                self.fc2 = nn.Linear(32, output_size)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])  # Use the last hidden state
                out = torch.relu(self.fc1(out))
                out = self.fc2(out)
                return out

        # Correctly set the input size
        input_size = X_train_tensor.shape[2]  # Number of features in the input data
        output_size = y_train_tensor.shape[1]  # Number of target classes
        model = LSTMModel(input_size, hidden_size, output_size)


        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(10):  # Fixed number of epochs
            for X_batch, y_batch in train_loader:
                # Ensure X_batch has the correct shape (batch_size, sequence_length, input_size)
                if len(X_batch.shape) == 4:  # If X_batch has an extra dimension
                    X_batch = X_batch.squeeze(1)  # Remove the extra dimension
                elif len(X_batch.shape) == 2:  # If X_batch is missing sequence dimension
                    X_batch = X_batch.unsqueeze(1)  # Add sequence length dimension

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, torch.argmax(y_batch, axis=1))
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            # Initialize X_test_tensor_seq with X_test_tensor as a fallback
            X_test_tensor_seq = X_test_tensor

            if len(X_test_tensor.shape) == 4:  # If X_test_tensor has an extra dimension
                X_test_tensor_seq = X_test_tensor.squeeze(1)  # Remove the extra dimension
            elif len(X_test_tensor.shape) == 2:  # If missing sequence dimension
                X_test_tensor_seq = X_test_tensor.unsqueeze(1)  # Add sequence length dimension

            y_pred = model(X_test_tensor_seq)
            y_pred_classes = torch.argmax(y_pred, axis=1)
            y_true_classes = torch.argmax(y_test_tensor, axis=1)
            accuracy = accuracy_score(y_true_classes.numpy(), y_pred_classes.numpy())
            print(f'Trial accuracy: {accuracy:.4f}')

        return accuracy
    
    # Run Optuna optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5)

    print("Best hyperparameters:", study.best_params)



    #     # Generate confusion matrix
    #     y_pred = outputs.argmax(dim=1).cpu().numpy()
    #     y_true = y_test_tensor.argmax(dim=1).cpu().numpy()
    #     cm = confusion_matrix(y_true, y_pred)

    #     # Save confusion matrix plot with timestamp
    #     timestamp = time.strftime("%Y%m%d-%H%M%S")
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=expected_classes, yticklabels=expected_classes)
    #     plt.title(f"Confusion Matrix - {timestamp}")
    #     plt.xlabel("Predicted Labels")
    #     plt.ylabel("True Labels")
    #     plt.savefig(f"confusion_matrix_{timestamp}.png")
    #     plt.close()

    # # Plot training loss
    # #TODO decide if we want to have accuracy in same plot?
    # plt.plot(range(1, len(loss_history) + 1), loss_history, label="Train Loss")
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title(f"Training Loss - {timestamp}")
    # plt.legend()
    # plt.savefig(f"training_loss_{timestamp}.png")
    # plt.close()

    # print(f"Plots saved with timestamp {timestamp}.")


if __name__ == "__main__":
    print("Starting LSTM classification with Optuna...")
    lstm_classification()
