import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor

from sklearn.ensemble import GradientBoostingRegressor  # Changed
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import datetime

#using gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


filename = "motor_data11-14lats.csv"
filename2 = "motor_data14-2018.csv"

#Reading the csv
data1 = pd.read_csv(filename)
data2 = pd.read_csv(filename2)

data = pd.concat([data1, data2])



# Convert "INSR_BEGIN" column to datetime format
data['INSR_BEGIN'] =  pd.to_datetime(data['INSR_BEGIN'], format='%d-%b-%y')
# Extract year from "INSR_BEGIN" column
data['INSR_BEGIN_YEAR'] = data['INSR_BEGIN'].dt.year

#new code
data['INSR_END'] = pd.to_datetime(data['INSR_END'], format='%d-%b-%y')
data['INSR_DURATION'] = data['INSR_END'] - data['INSR_BEGIN']
data['INSR_DURATION_MONTHS'] = data['INSR_DURATION'] / pd.Timedelta(days=30.436875)  # Approximate average number of days in a month

# Round the duration to the nearest whole number of months
data['INSR_DURATION_MONTHS'] = data['INSR_DURATION_MONTHS'].round().astype(int)
current_year = datetime.datetime.now().year
data['AGE_VEHICLE'] = current_year - data['PROD_YEAR']

data['Month'] = data['INSR_BEGIN'].dt.month

# Define a function to map month to season
def get_season(month):
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter' 

# Map month to season
data['SEASON'] = data['Month'].apply(get_season)

columnsToUse = ['SEX','INSURED_VALUE', 'USAGE', 'AGE_VEHICLE', 'TYPE_VEHICLE', 'PREMIUM','MAKE', 'INSR_BEGIN_YEAR', 'SEASON', 'SEATS_NUM','INSR_DURATION_MONTHS']

data = data[columnsToUse]

#Remove rows with empty values. There aren't many of them so this doesn't affect the data
data = data.dropna()

categoricalColumns = ['SEX','USAGE', 'TYPE_VEHICLE', 'MAKE', 'SEASON']
numericalColumns = ['INSURED_VALUE', 'AGE_VEHICLE', 'INSR_BEGIN_YEAR', 'SEATS_NUM', 'INSR_DURATION_MONTHS']



preprocessor = ColumnTransformer(
transformers=[
	('num', StandardScaler(), numericalColumns),
	('cat', OneHotEncoder(), categoricalColumns)
])


data.info()


features = data.drop(columns=['PREMIUM'])
labels = data['PREMIUM']

features_preprocessed = preprocessor.fit_transform(features)


X_train, X_test, y_train, y_test = train_test_split(features_preprocessed, labels, test_size=0.2, random_state=42)


# Convert features and labels to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device).unsqueeze(1)

# Create a TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Define batch size for the DataLoader
batch_size = 64
epochs = 10

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class NeuralNet(nn.Module):
    def __init__(self, input_size, num_numerical_cols, embedding_sizes, output_size):
        super(NeuralNet, self).__init__()
        self.num_numerical_cols = num_numerical_cols
        self.embeddings = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in embedding_sizes])
        self.num_categorical_cols = sum((nf for ni, nf in embedding_sizes))
        self.num_total_cols = self.num_categorical_cols + self.num_numerical_cols

        self.fc1 = nn.Linear(self.num_total_cols, 256)  
        self.fc2 = nn.Linear(256, 128)  
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_numerical, x_categorical):
        embedded = []
        for i, e in enumerate(self.embeddings):
            embedded.append(e(x_categorical[:, i].long()))
        x_categorical = torch.cat(embedded, 1)
        x = torch.cat([x_numerical, x_categorical], 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = F.relu(self.fc5(x))
        x = self.dropout(x)
        x = self.fc6(x)
        return x


# Define the sizes for embedding layers
embedding_sizes = [(len(data[column].unique()), min(50, (len(data[column].unique()) + 1) // 2)) for column in categoricalColumns]

# Initialize the neural network
input_size = X_train_tensor.shape[1]  # Number of features after preprocessing
output_size = 1
model = NeuralNet(input_size, len(numericalColumns), embedding_sizes, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer

from sklearn.metrics import mean_absolute_error, r2_score

# Define training loop
# Define training loop
# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define training loop
def train_model(model, criterion, optimizer, train_loader, epochs):
    model.train()  # Set the model to training mode
    num_samples = len(train_loader.dataset)  # Get the number of samples
    for epoch in range(epochs):
        running_loss = 0.0
        running_mae = 0.0
        predictions = []
        targets = []
        for inputs, labels in train_loader:  # Iterate over batches from DataLoader
            inputs_numerical = inputs[:, :len(numericalColumns)].to(device)
            inputs_categorical = inputs[:, len(categoricalColumns):].to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Zero the parameter gradients

            # Forward pass
            outputs = model(inputs_numerical, inputs_categorical)  # Pass both numerical and categorical inputs
            loss = criterion(outputs, labels)  # Calculate the loss

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate MAE
            predictions.extend(outputs.cpu().detach().numpy())
            targets.extend(labels.cpu().detach().numpy())

        # Calculate metrics
        running_mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        # Print average metrics for this epoch
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss / num_samples}, MAE: {running_mae}, R-squared: {r2}")




from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define evaluation function
# Define evaluation function
def evaluate_model(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Split the features into numerical and categorical components
        numerical_features = X_test[:, :len(numericalColumns)]
        categorical_features = X_test[:, len(numericalColumns):]

        # Convert to torch tensors
        numerical_tensor = torch.tensor(numerical_features.toarray(), dtype=torch.float32, device=device)
        categorical_tensor = torch.tensor(categorical_features.toarray(), dtype=torch.float32, device=device)

        # Pass both numerical and categorical inputs to the model
        outputs = model(numerical_tensor, categorical_tensor)

        # Convert y_test to a PyTorch tensor
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32, device=device)

        # Calculate Mean Squared Error
        mse = criterion(outputs, y_test_tensor.unsqueeze(1))
        print(f"Mean Squared Error on Test Set: {mse.item()}")

        # Calculate Mean Absolute Error
        mae = mean_absolute_error(y_test, outputs.cpu().numpy())
        print(f"Mean Absolute Error on Test Set: {mae}")




# Train the model
train_model(model, criterion, optimizer, train_loader, epochs)

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Define the file path for saving the model
model_path = 'trained_model.pth'

# Save the model's state dictionary
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")




