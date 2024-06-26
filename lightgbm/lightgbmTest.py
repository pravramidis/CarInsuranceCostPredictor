import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor  # Changed
from sklearn.metrics import mean_squared_error, mean_absolute_error

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.metrics import r2_score
import joblib

#using gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


filename = "Motor_vehicle_insurance_data.csv"

#Reading the csv
data = pd.read_csv(filename, sep=';')


#ID;Date_start_contract;Date_last_renewal;Date_next_renewal;Date_birth;Date_driving_licence;
#Distribution_channel;Seniority;Policies_in_force;Max_policies;Max_products;Lapse;Date_lapse;
#Payment;Premium;Cost_claims_year;N_claims_year;N_claims_history;R_Claims_history;Type_risk;Area;
#Second_driver;Year_matriculation;Power;Cylinder_capacity;Value_vehicle;N_doors;Type_fuel;Length;Weight

#Feature engineering

last_renewal_day = pd.to_datetime(data['Date_last_renewal'], format='%d/%m/%Y')
last_renewal_year = last_renewal_day.dt.year
data['Last_renewal_year'] = last_renewal_year

birthday =  pd.to_datetime(data['Date_birth'], format='%d/%m/%Y')
birthyear = birthday.dt.year

contract_day = pd.to_datetime(data['Date_start_contract'], format='%d/%m/%Y')
contract_year = contract_day.dt.year
data['Contract_year'] = contract_year

# Step 1: Group by ID and calculate the average premium
avg_premium_per_id = data.groupby('ID')['Premium'].mean().reset_index()

# Step 2: Merge the averages back into the original DataFrame
data = data.merge(avg_premium_per_id, on='ID', suffixes=('', '_avg'))

# Step 3: Replace the original premium values with the averages
data['Premium'] = data['Premium_avg']

# Step 4: Drop the auxiliary column if needed
data.drop('Premium_avg', axis=1, inplace=True)


data = data.sort_values(by=['ID', 'Last_renewal_year'], ascending=[True, False])

# Drop duplicates keeping only the first occurrence (which will have the highest 'Contract_year' for each 'ID')
data = data.drop_duplicates(subset='ID', keep='first')

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

# temp test
# data['Years_driving'] = contract_year - year_start
# data['Age'] = contract_year - birthyear
# data['Years_on_road'] = contract_year - registration_year

data['accidents'] = data['N_claims_history'] / (data['Years_on_policy'] + 1)

data['Age_Years_Driving_Interaction'] = data['Age'] * data['Years_driving']


#We combine the doors and the category into one column
combined_values = data['Type_risk'].astype(str) + '_' + data['N_doors'].astype(str)
data['Combined_doors_type'] = combined_values


columnsToUse = ['Seniority', 'Premium', 'Type_risk', 'Area', 'Second_driver', 'Years_on_road', 'R_Claims_history', 'Years_on_policy', 'accidents', 
			'Value_vehicle', 'Age', 'Years_driving', 'Distribution_channel', 'N_claims_history', 'Power', 'Cylinder_capacity',
			'Weight', 'Length', 'Type_fuel', 'Payment', 'Contract_year', 'Policies_in_force', 'Lapse']

data = data[columnsToUse]



#We fill the missing length value with the mean so we can use the rest of the row
data['Length'] = data['Length'].fillna(data['Length'].mean())

#Replace the null values of fuel types the a third value
null_indices = data['Type_fuel'].isnull()
data.loc[null_indices, 'Type_fuel'] = 'Unknown'

# Remove rows with empty values. There aren't many of them so this doesn't affect the data
# data = data.dropna()

categoricalColumns = [ 'Area', 'Second_driver', 'Distribution_channel', 'Type_fuel', 'Payment']
numericalColumns = ['Seniority', 'Years_on_road','Value_vehicle','Age', 'Years_driving', 'N_claims_history', 'Power', 'Cylinder_capacity', 
					'Weight', 'Length', 'Contract_year', 'R_Claims_history', 'Years_on_policy', 'accidents', 'Policies_in_force', 'Lapse']


preprocessor = ColumnTransformer(
transformers=[
	('num', StandardScaler(), numericalColumns),
	('cat', OneHotEncoder(), categoricalColumns)
])


data.info()

params = {
    'objective': 'regression',  # or 'binary' for binary classification
    'metric': 'rmse',           # or other evaluation metrics
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

# Get unique values of 'Type_risk'
type_risk_values = data['Type_risk'].unique()

# Dictionary to store models
models = {}
import matplotlib.pyplot as plt

# Dictionary to store percentage of predictions within different thresholds for each Type_risk value
percentage_within_thresholds = {type_risk_value: {} for type_risk_value in type_risk_values}

all_absolute_errors = []
all_actual_values = []

# Iterate over unique 'Type_risk' values
for type_risk_value in type_risk_values:
    # Filter data for the current 'Type_risk' value
    subset_data = data[data['Type_risk'] == type_risk_value]
    
    # Split the subset data into features and labels
    subset_features = subset_data.drop(columns=['Premium', 'Type_risk'])
    subset_labels = subset_data['Premium']
    # Preprocess features
    subset_features_preprocessed = preprocessor.fit_transform(subset_features)

    joblib.dump(preprocessor, f"lightgbmModels\\preprocessor_{type_risk_value}.pkl")
    
    # Split data into train and test sets
    subset_X_train, subset_X_test, subset_y_train, subset_y_test = train_test_split(
        subset_features_preprocessed, subset_labels, test_size=0.2, random_state=42)
    
    # Create a LightGBM dataset
    subset_train_data = lgb.Dataset(subset_X_train, label=subset_y_train)
    subset_test_data = lgb.Dataset(subset_X_test, label=subset_y_test)
    
    # Train the model
    model = lgb.train(params, subset_train_data, num_boost_round=1000, valid_sets=[subset_test_data])
    
    # Store the model
    models[type_risk_value] = model
    
    # Make predictions
    y_pred = model.predict(subset_X_test, num_iteration=model.best_iteration)

    model.save_model(f"lightgbmModels\\model_{type_risk_value}.txt")
    
    # Evaluate the model
    mse = mean_squared_error(subset_y_test, y_pred)
    print(f"Mean Squared Error for Type_risk {type_risk_value}:", mse)
    
    absolute_error = np.abs(subset_y_test - y_pred)
    average_absolute_error = np.mean(absolute_error)
    print(f"Average Absolute Error for Type_risk {type_risk_value}:", average_absolute_error)
    
    # Calculate and print percentage of predictions within different percentage thresholds
    for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        percentage_within_threshold = np.mean(absolute_error / subset_y_test <= threshold / 100) * 100
        print(f"Percentage of predictions within {threshold}% of the actual value for Type_risk {type_risk_value}: {percentage_within_threshold}")
    
    r2 = r2_score(subset_y_test, y_pred)
    print(f"R² Score for Type_risk {type_risk_value}:", r2)
    print()
    
    # Store percentage of predictions within different thresholds
    for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        percentage_within_threshold = np.mean(absolute_error / subset_y_test <= threshold / 100) * 100
        percentage_within_thresholds[type_risk_value][threshold] = percentage_within_threshold

    # Print variable importance
    print("Feature importance for Type_risk", type_risk_value)
    for feature, importance in zip(subset_features.columns, model.feature_importance()):
        print(f"{feature}: {importance}")

    all_absolute_errors.extend(absolute_error)
    all_actual_values.extend(subset_y_test)

    
# Convert lists to numpy arrays for easier calculation
all_absolute_errors = np.array(all_absolute_errors)
all_actual_values = np.array(all_actual_values)

# Calculate the percentage within threshold for the whole dataset
for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    percentage_within_threshold = np.mean(all_absolute_errors / all_actual_values <= threshold / 100) * 100
    print(f"Percentage of predictions within {threshold}% of the actual value for the whole dataset: {percentage_within_threshold}")

# Plotting
for type_risk_value, thresholds in percentage_within_thresholds.items():
    plt.plot(thresholds.keys(), thresholds.values(), label=f'Type_risk {type_risk_value}')

plt.xlabel('Threshold (%)')
plt.ylabel('Percentage of Predictions within Threshold')
plt.title('Percentage of Predictions within Different Thresholds for Each Type_risk')
plt.legend()
plt.grid(True)
plt.show()

# Dictionary to store feature importance for each category
feature_importance_per_category = {}

# Iterate over unique 'Type_risk' values
for type_risk_value in type_risk_values:
    # Get the model for the current 'Type_risk' value
    model = models[type_risk_value]
    
    # Get feature names
    features = subset_features.columns
    
    # Get feature importance values
    importance_values = model.feature_importance()
    
    # Store feature importance in the dictionary
    feature_importance_per_category[type_risk_value] = dict(zip(features, importance_values))

    # Sort features based on importance in ascending order
    sorted_features = sorted(features, key=lambda x: feature_importance_per_category[type_risk_value][x])
    sorted_importance = [feature_importance_per_category[type_risk_value][feature] for feature in sorted_features]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features, sorted_importance)
    plt.xlabel('Number of Uses')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for Type_risk {type_risk_value}')
    plt.tight_layout()
    plt.show()

