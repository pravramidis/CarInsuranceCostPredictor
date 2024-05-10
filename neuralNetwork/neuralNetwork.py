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

def definition_of_type_risk(number):
    if number == 1:
        return "Motorbike"
    elif number == 2:
        return "Van"
    elif number == 3:
        return "Passenger Car"
    elif number == 4:
        return "Agricultural Vehicle"
    else:
        return "All"
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

features_preprocessed = preprocessor.fit(features)

batch_size = 64
epochs = 10


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
criterion = nn.MSELoss()  # Mean Squared Error loss for regression

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
def evaluate_model(model, X_test, y_test, risk_type):
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
        type_defined = definition_of_type_risk(risk_type)

        threshold_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Calculate percentage within threshold
        percentages_within_threshold = []
        for threshold in threshold_values:
            absolute_errors = np.abs(outputs.cpu().numpy() - y_test.values)
            within_threshold = np.mean(absolute_errors <= threshold / 100 * y_test.values) * 100
            percentages_within_threshold.append(within_threshold)
            print(f"Percentage of predictions within {threshold}% of the actual value: {within_threshold}")

        return percentages_within_threshold


# Get unique values of 'Type_risk' column
risk_types = data['Type_risk'].unique()

# Dictionary to store models and corresponding datasets
models = {}
datasets = {}

# Loop through each risk type
for risk_type in risk_types:
    # Filter data based on risk type
    risk_data = data[data['Type_risk'] == risk_type]
    
    # Separate features and labels
    risk_features = risk_data.drop(columns=['Premium'])
    risk_labels = risk_data['Premium']
    
    # Preprocess features
    risk_features_preprocessed = preprocessor.transform(risk_features)
    
    # Split into train and test sets
    X_train_risk, X_test_risk, y_train_risk, y_test_risk = train_test_split(
        risk_features_preprocessed, risk_labels, test_size=0.2, random_state=42)
    
    # Create TensorDatasets
    train_dataset_risk = TensorDataset(
        torch.tensor(X_train_risk, dtype=torch.float32, device=device),
        torch.tensor(y_train_risk.values, dtype=torch.float32, device=device).unsqueeze(1)
    )
    
    # Save dataset
    datasets[risk_type] = (train_dataset_risk, X_test_risk, y_test_risk)
    
    # Initialize and train model
    model_risk = NeuralNet(input_size, len(numericalColumns), categoricalColumns, risk_data, output_size).to(device)
    optimizer_risk = optim.Adam(model_risk.parameters(), lr=0.001)
    train_model(model_risk, criterion, optimizer_risk, DataLoader(train_dataset_risk, batch_size=batch_size, shuffle=True), epochs)
    
    # Save trained model
    models[risk_type] = model_risk


# Initialize a list to store risk_type-specific data for plotting
plots_data = []

# Evaluate each model and collect data for plotting
for risk_type, (train_dataset_risk, X_test_risk, y_test_risk) in datasets.items():
    print(f"Evaluating model for risk type: {risk_type}")
    percentages_within_threshold = evaluate_model(models[risk_type], X_test_risk, y_test_risk, risk_type)
    plots_data.append((risk_type,percentages_within_threshold))
    
threshold_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# Plotting all graphs together
plt.figure(figsize=(12, 8))
for risk_type, percentages_within_threshold in plots_data:
    type_defined = definition_of_type_risk(risk_type)
    plt.plot(threshold_values, percentages_within_threshold, label=f'{type_defined}')

plt.title('Percentage of Predictions within Threshold for All Risk Types')
plt.xlabel('Threshold (%)')
plt.ylabel('Percentage within the Threshold')
plt.legend(loc='lower right') 
plt.grid(True)
plt.savefig('report\\images\\individual_thresholds_neural.png')
plt.show()