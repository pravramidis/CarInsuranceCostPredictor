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


filename = "Motor_vehicle_insurance_data.csv"

#Reading the csv
data = pd.read_csv(filename, sep=';')


#ID;Date_start_contract;Date_last_renewal;Date_next_renewal;Date_birth;Date_driving_licence;
#Distribution_channel;Seniority;Policies_in_force;Max_policies;Max_products;Lapse;Date_lapse;
#Payment;Premium;Cost_claims_year;N_claims_year;N_claims_history;R_Claims_history;Type_risk;Area;
#Second_driver;Year_matriculation;Power;Cylinder_capacity;Value_vehicle;N_doors;Type_fuel;Length;Weight

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


#We combine the doors and the category into one column
combined_values = data['Type_risk'].astype(str) + '_' + data['N_doors'].astype(str)
data['Combined_doors_type'] = combined_values

columnsToUse = ['Seniority', 'Premium', 'Type_risk', 'Area', 'Second_driver', 'Years_on_road', 'R_Claims_history', 'Years_on_policy', 'accidents',
			'Value_vehicle', 'Age', 'Years_driving', 'Distribution_channel', 'N_claims_history', 'Power', 'Cylinder_capacity',
			'Weight', 'Length', 'Type_fuel', 'Payment', 'Contract_year', 'Policies_in_force', 'Lapse']

data = data[columnsToUse]

# data = data[(data['Type_risk'] == 3)]

#We fill the missing length value with the mean so we can use the rest of the row
# data['Length'].fillna(data['Length'].mean(), inplace=True)
#We fill the missing fuel type with the most common type
# data['Type_fuel'].fillna(data['Type_fuel'].mode()[0], inplace=True)

# Remove rows with empty values. There aren't many of them so this doesn't affect the data
# data = data.dropna()

categoricalColumns = ['Type_risk', 'Area', 'Second_driver', 'Distribution_channel', 'Type_fuel', 'Payment']
numericalColumns = ['Seniority', 'Years_on_road','Value_vehicle','Age', 'Years_driving', 'N_claims_history', 'Power', 'Cylinder_capacity', 
					'Weight', 'Length', 'Contract_year', 'R_Claims_history', 'Years_on_policy', 'accidents', 'Policies_in_force', 'Lapse']





preprocessor = ColumnTransformer(
transformers=[
	('num', StandardScaler(), numericalColumns),
	('cat', OneHotEncoder(), categoricalColumns)
])


data.info()

features = data.drop(columns=['Premium'])
labels = data['Premium']

features_preprocessed = preprocessor.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(features_preprocessed, labels, test_size=0.2, random_state=42)

# Initialize Gradient Boosting Regressor
xgb_regressor = XGBRegressor(n_estimators=1000, random_state=42, n_jobs=-1)

# Train Gradient Boosting Regressor
xgb_regressor.fit(X_train, y_train)

# Save the trained model to a file
model_filename = "xgboost_model.pkl"
joblib.dump(xgb_regressor, model_filename)

# Predict on test set
y_pred = xgb_regressor.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate absolute error
absolute_error = np.abs(y_test - y_pred)

average_absolute_error = np.mean(absolute_error)
print("Average Absolute Error:", average_absolute_error)

# Calculate the percentage of predictions within 10% of the actual value
percentage_within_set_percent = np.mean(absolute_error / y_test <= 0.1) * 100
print("Percentage of predictions within 10% of the actual value:", percentage_within_set_percent)

# Get feature importances from the trained model
feature_importances = xgb_regressor.feature_importances_

# Get feature names after one-hot encoding
encoded_categorical_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categoricalColumns))

# Combine numerical and encoded categorical feature names
all_feature_names = numericalColumns + encoded_categorical_features

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})

# Combine importance of one-hot encoded features
combined_importance = {}
for orig_feature in categoricalColumns:
    indices = [i for i, name in enumerate(all_feature_names) if orig_feature + '_' in name]
    combined_importance[orig_feature] = feature_importance_df.iloc[indices]['Importance'].sum()

# Remove one-hot encoded features from DataFrame
feature_importance_df = feature_importance_df[~feature_importance_df['Feature'].isin(encoded_categorical_features)]

# Create a DataFrame for combined importance
combined_importance_df = pd.DataFrame({'Feature': list(combined_importance.keys()), 'Importance': list(combined_importance.values())})

# Concatenate both DataFrames
feature_importance_df = pd.concat([feature_importance_df, combined_importance_df], ignore_index=True)

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(feature_importance_df)

# # Define a function to evaluate model performance for each risk type
# def evaluate_model_by_feature(model, X_test, y_test, risk_types):
#     for risk_type in risk_types:
#         # Filter test data for the current risk type
#         risk_type_indices = X_test[:, all_feature_names.index('Type_risk_' + str(risk_type))] == 1
#         X_test_risk_type = X_test[risk_type_indices]
#         y_test_risk_type = y_test[risk_type_indices]

#         # Predict on the filtered test set
#         y_pred_risk_type = model.predict(X_test_risk_type)

#         # Calculate metrics for the current risk type
#         mse = mean_squared_error(y_test_risk_type, y_pred_risk_type)
#         mae = mean_absolute_error(y_test_risk_type, y_pred_risk_type)
#         absolute_error_risk_type = np.abs(y_test_risk_type - y_pred_risk_type)

#         # Calculate the percentage of predictions within 10% of the actual premium
#         percentage_within_10_percent = np.mean(absolute_error_risk_type / y_test_risk_type <= 0.1) * 100

#         # Print metrics for the current risk type
#         print(f"Metrics for {risk_type} Risk Type:")
#         print(f"Mean Squared Error: {mse}")
#         print(f"Mean Absolute Error: {mae}")
#         print(f"Percentage of predictions within 10% of the actual premium: {percentage_within_10_percent}%")
#         print()  # Add an empty line for better readability
        

# risk_types = data['Type_risk'].unique()
# evaluate_model_by_feature(xgb_regressor, X_test, y_test, risk_types)