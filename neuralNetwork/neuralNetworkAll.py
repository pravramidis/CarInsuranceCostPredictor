import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler 

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Choose your device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filename = "Motor_vehicle_insurance_data.csv"

# Reading the CSV file
data = pd.read_csv(filename, sep=';')

# Feature engineering
last_renewal_day = pd.to_datetime(data['Date_last_renewal'], format='%d/%m/%Y')
last_renewal_year = last_renewal_day.dt.year
data['Last_renewal_year'] = last_renewal_year

birthday = pd.to_datetime(data['Date_birth'], format='%d/%m/%Y')
birthyear = birthday.dt.year

contract_day = pd.to_datetime(data['Date_start_contract'], format='%d/%m/%Y')
contract_year = contract_day.dt.year
data['Contract_year'] = contract_year

avg_premium_per_id = data.groupby('ID')['Premium'].mean().reset_index()
data = data.merge(avg_premium_per_id, on='ID', suffixes=('', '_avg'))
data['Premium'] = data['Premium_avg']
data.drop('Premium_avg', axis=1, inplace=True)

data.sort_values(by=['ID', 'Last_renewal_year'], ascending=[True, False], inplace=True)
data.drop_duplicates(subset='ID', keep='first', inplace=True)

day_start_driving = pd.to_datetime(data['Date_driving_licence'], format='%d/%m/%Y')
year_start = day_start_driving.dt.year

data['Years_driving'] = last_renewal_year - year_start
data['Age'] = last_renewal_year - birthyear

data['Distribution_channel'] = data['Distribution_channel'].astype(str)
registration_year = data['Year_matriculation']
data['Years_on_road'] = last_renewal_year - registration_year

next_renewal_day = pd.to_datetime(data['Date_next_renewal'], format='%d/%m/%Y')
next_renewal_year = next_renewal_day.dt.year
data['Next_renewal_year'] = next_renewal_year

policy_duration = next_renewal_year - last_renewal_year
data['Policy_duration'] = policy_duration

years_on_policy = last_renewal_year - contract_year
data['Years_on_policy'] = years_on_policy

data['accidents'] = data['N_claims_history'] / (data['Years_on_policy'] + 1)
data['Age_Years_Driving_Interaction'] = data['Age'] * data['Years_driving']


columnsToUse = ['Seniority', 'Premium', 'Type_risk', 'Area', 'Second_driver', 'Years_on_road', 'R_Claims_history', 'Years_on_policy', 'accidents', 'Value_vehicle', 'Age', 'Years_driving', 'Distribution_channel', 'N_claims_history', 'Power', 'Cylinder_capacity', 'Weight', 'Length', 'Type_fuel', 'Payment', 'Contract_year', 'Policies_in_force', 'Lapse']

data = data[columnsToUse]

data['Length'] = data['Length'].fillna(data['Length'].mean())
data.loc[data['Type_fuel'].isnull(), 'Type_fuel'] = 'Unknown'

categoricalColumns = ['Area', 'Second_driver', 'Distribution_channel', 'Type_fuel', 'Payment', 'Type_risk']
numericalColumns = ['Seniority', 'Years_on_road', 'Value_vehicle', 'Age', 'Years_driving', 'N_claims_history', 'Power', 'Cylinder_capacity', 'Weight', 'Length', 'Contract_year', 'R_Claims_history', 'Years_on_policy', 'accidents', 'Policies_in_force', 'Lapse']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numericalColumns),
        ('cat', OneHotEncoder(), categoricalColumns)
    ]
)

data.info()

features = data.drop(columns=['Premium'])
labels = data['Premium']

features_preprocessed = preprocessor.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_preprocessed, labels, test_size=0.2, random_state=42)

# Convert features and labels to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32, device=device).unsqueeze(1)

# Create a TensorDataset
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Define batch size for the DataLoader
batch_size = 64
epochs = 10

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class NeuralNet(nn.Module):
    def __init__(self, input_size, num_numerical_cols, categorical_columns, data, output_size):
        super(NeuralNet, self).__init__()
        self.num_numerical_cols = num_numerical_cols
        
        # Initialize the embeddings
        self.embeddings = nn.ModuleList()
        for column in categorical_columns:
            num_unique_values = len(data[column].unique()) + 1  # Adding 1 for unknown category
            embedding_size = min(50, (num_unique_values + 1) // 2)  
            self.embeddings.append(nn.Embedding(num_unique_values, embedding_size))
        
        self.num_categorical_cols = sum(e.embedding_dim for e in self.embeddings)
        
        # Total number of input features
        total_input_size = self.num_numerical_cols + self.num_categorical_cols
        
        # Define fully connected layers
        self.fc1 = nn.Linear(total_input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, output_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(0.2)

    def forward(self, x_numerical, x_categorical):
        # Convert categorical data to LongTensor
        x_categorical = x_categorical.long()
        
        # Process categorical input through the embedding layers
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            # Process each categorical column through the corresponding embedding layer
            cat_col = x_categorical[:, i]  # Select the i-th categorical column
            embedded.append(embedding(cat_col))
        
        # Concatenate all embedded categorical features
        x_categorical = torch.cat(embedded, dim=1)
        
        # Concatenate numerical and categorical features
        x = torch.cat([x_numerical, x_categorical], dim=1)
        
        # Pass through the fully connected layers
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
embedding_sizes = [(len(data[column].unique()) + 1, min(50, (len(data[column].unique()) + 1) // 2)) for column in categoricalColumns]

# Initialize the neural network
input_size = len(numericalColumns) + sum(embedding_size for _, embedding_size in embedding_sizes)
output_size = 1  # For regression
model = NeuralNet(input_size, len(numericalColumns), categoricalColumns, data, output_size).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define training loop
def train_model(model, criterion, optimizer, train_loader, epochs):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        predictions = []
        targets = []
        for inputs, labels in train_loader:
            # Split inputs into numerical and categorical parts
            inputs_numerical = inputs[:, :len(numericalColumns)].to(device)
            inputs_categorical = inputs[:, len(numericalColumns):].to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs_numerical, inputs_categorical)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Accumulate running loss
            running_loss += loss.item()
            
            # Collect predictions and targets for metrics
            predictions.extend(outputs.cpu().detach().numpy())
            targets.extend(labels.cpu().detach().numpy())

        # Calculate and print metrics
        mean_mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, MAE: {mean_mae:.4f}, R-squared: {r2:.4f}")

# Define evaluation function
def evaluate_model(model, X_test, y_test):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        # Split the features into numerical and categorical components
        numerical_features = X_test[:, :len(numericalColumns)]
        categorical_features = X_test[:, len(numericalColumns):]

        # Convert the numerical and categorical features directly to PyTorch tensors
        numerical_tensor = torch.tensor(numerical_features, dtype=torch.float32, device=device)
        categorical_tensor = torch.tensor(categorical_features, dtype=torch.long, device=device)

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

        r2 = r2_score(y_test, outputs.cpu().numpy())
        print(f"R² Score:", r2)
        print()

        threshold_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Calculate percentage within threshold
        percentages_within_threshold = []
        for threshold in threshold_values:
            absolute_errors = np.abs(outputs.cpu().numpy() - y_test.values)
            within_threshold = np.mean(absolute_errors <= threshold / 100 * y_test.values) * 100
            percentages_within_threshold.append(within_threshold)
            print(f"Percentage of predictions within {threshold}% of the actual value: {within_threshold}")

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.plot(threshold_values, percentages_within_threshold)
        plt.title('Percentage of Predictions within Threshold')
        plt.xlabel('Threshold (%)')
        plt.ylabel('Percentage within the Threshold')
        plt.grid(True)
        plt.savefig(f'report\\images\\neural_network_all_thresholds.png')
        plt.show()



        
# Train the model
train_model(model, criterion, optimizer, train_loader, epochs)

# Evaluate the model
evaluate_model(model, X_test, y_test)
