import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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

    # Create DataLoader
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.dropout = nn.Dropout(0.2)
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
    hidden_size = 64
    output_size = y_train.shape[1]
    model = LSTMModel(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 50
    print("Training the model...")
    #TODO stop training after delta-loss < ..
    for epoch in tqdm(range(1, epochs + 1), desc="Epoch Progress"):
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0

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

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        test_accuracy = (outputs.argmax(dim=1) == y_test_tensor.argmax(dim=1)).sum().item() / len(y_test_tensor)
        print(f"Test Accuracy: {test_accuracy:.2f}")

    #TODO plots of train loss, predictions, etc.

if __name__ == "__main__":
    print("Starting LSTM classification...")
    lstm_classification()