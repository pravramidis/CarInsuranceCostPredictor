import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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

# Concatenate numerical and encoded categorical feature names
all_feature_names = numericalColumns + encoded_categorical_features

# Create a DataFrame to display feature importances
feature_importance_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importances
print(feature_importance_df)


# Function to add statistics to plots
def add_stats_to_plot(ax, data):
    # Calculate statistics
    min_val = np.min(data)
    max_val = np.max(data)
    median_val = np.median(data)
    mean_val = np.mean(data)
    record_count = len(data)  # Calculate the number of records in the data
    
    # Create statistics text, including the number of records
    stats_text = f'Min: {min_val:.2f}\nMax: {max_val:.2f}\nMedian: {median_val:.2f}\nMean: {mean_val:.2f}\nRecords: {record_count}'
    
    # Add the statistics to the plot as text
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# # # Label: Premium
# plt.figure(figsize=(12, 8))
# ax = sns.histplot(data['Premium'], kde=True, bins=20)
# plt.title('Distribution of Premium')
# plt.xlabel('Premium')
# plt.ylabel('Frequency')
# add_stats_to_plot(ax, data['Premium'])
# plt.show()


# def plot_premium_by_type_risk(data, type_risk_value):
#     # Filter the data for the given Type_risk category
#     subset_data = data[data['Type_risk'] == type_risk_value]
    
#     # Create a plot for the Premium distribution of the subset data
#     plt.figure(figsize=(12, 8))
#     ax = sns.histplot(subset_data['Premium'], kde=True, bins=20)
#     plt.title(f'Distribution of Premium for Type_risk = {type_risk_value}')
#     plt.xlabel('Premium')
#     plt.ylabel('Frequency')
    
#     # Add statistics to the plot
#     add_stats_to_plot(ax, subset_data['Premium'])
    
#     # Show the plot
#     plt.show()

# # Create plots for each Type_risk category
# plot_premium_by_type_risk(data, 1)  # For Type_risk = 1
# plot_premium_by_type_risk(data, 2)  # For Type_risk = 2
# plot_premium_by_type_risk(data, 3)  # For Type_risk = 3
# plot_premium_by_type_risk(data, 4)  # For Type_risk = 4
# List of categorical features
# List of categorical features
categorical_features = ['Payment', 'Distribution_channel', 'Second_driver']

# # Numerical feature to visualize
numerical_feature = 'Premium'

def map_x_labels(x):
    if x == 0:
        return 'one driver'
    elif x == 1:
        return 'multiple drivers'
    else:
        return x
    
# Mapping function for x-axis labels
def map_distribution_channel_labels(x):
    if x == 0:
        return 'Agent'
    elif x == 1:
        return 'Insurance brokers'
    elif x ==x:
        return 'Other'
# Mapping function for x-axis labels
def map_payment_labels(x):
    if x == 0:
        return 'Annual'
    elif x == 1:
        return 'Half-yearly administrative'
    

# Loop through each categorical feature and create separate plots for each Type_risk category
for feature in categorical_features:
    for type_risk_value in data['Type_risk'].unique():
        # Filter the data for the given Type_risk category
        subset_data = data[data['Type_risk'] == type_risk_value]

        # Create a new plot
        plt.figure(figsize=(12, 8))

        # Create a swarm plot
        sns.stripplot(
            x=feature,
            y=numerical_feature,
            data=subset_data,
            palette='viridis',
            jitter=True,  # Add jitter for better separation of points
            label=f'Type_risk = {type_risk_value}'
        )

        # Map x-axis labels for each feature
        if feature == 'Second_driver':
            plt.xticks(ticks=[0, 1], labels=[map_x_labels(0), map_x_labels(1)])
        elif feature == 'Distribution_channel':
            plt.xticks(ticks=[0, 1], labels=[map_distribution_channel_labels(0), map_distribution_channel_labels(1)])
        elif feature == 'Payment':
            plt.xticks(ticks=subset_data['Payment'].unique(), labels=[map_payment_labels(x) for x in subset_data['Payment'].unique()])

        # Add title and labels
        if (type_risk_value == 1):
            plt.title(f'Distribution of {numerical_feature} by {feature} for Motorcycles', fontsize=14, pad=20)
        elif (type_risk_value == 2):
             plt.title(f'Distribution of {numerical_feature} by {feature} for Vans', fontsize=14, pad=20)
        elif (type_risk_value == 3):
             plt.title(f'Distribution of {numerical_feature} by {feature} for Passenger Cars', fontsize=14, pad=20)
        elif (type_risk_value == 4):
             plt.title(f'Distribution of {numerical_feature} by {feature} for Argicultular Vehicles', fontsize=14, pad=20)

        plt.xlabel(feature, fontsize=12)

        plt.ylabel(numerical_feature, fontsize=12)

        # # Add legend
        # plt.legend(title='Type_risk')

        # Optional: Add grid lines for better readability
        plt.grid(True, linestyle='--', alpha=0.6)

        # Show the plot
        plt.show()
# Loop through each categorical feature and create a swarm plot
# for feature in categorical_features:
#     plt.figure(figsize=(12, 8))  # Adjust figure size as needed

#     # Create a swarm plot
#     sns.stripplot(
#         x=feature,
#         y=numerical_feature,
#         data=data,
#         palette='viridis',
#         jitter=True  # Add jitter for better separation of points
#     )
    
#     # Map x-axis labels for 'Second_driver' feature
#     if feature == 'Second_driver':
#         plt.xticks(ticks=[0, 1], labels=[map_x_labels(0), map_x_labels(1)])
#     if feature == 'Distribution_channel':
#         plt.xticks(ticks=[0, 1], labels=[map_distribution_channel_labels(0), map_distribution_channel_labels(1)])
#     # Map x-axis labels for 'Payment' feature
#     if feature == 'Payment':
#         plt.xticks(ticks=data['Payment'].unique(), labels=[map_payment_labels(x) for x in data['Payment'].unique()])
#     # Add title and labels
#     plt.title(f'Distribution of {numerical_feature} by {feature}', fontsize=14, pad=20)
#     plt.xlabel(feature, fontsize=12)
#     plt.ylabel(numerical_feature, fontsize=12)

#     # Optional: Add grid lines for better readability
#     plt.grid(True, linestyle='--', alpha=0.6)

#     # Show the plot
#     plt.show()
# # Plotting the distribution of the 'Length' feature
# plt.figure(figsize=(10, 6))
# ax_length = sns.histplot(data['Length'], kde=True, bins=20, color='blue')
# plt.title('Distribution of Length')
# plt.xlabel('Length')
# plt.ylabel('Frequency')
# # Add statistics to the plot
# add_stats_to_plot(ax_length, data['Length'])
# # Show the plot
# plt.show()

# # Plotting the distribution of the 'Age' feature
# plt.figure(figsize=(10, 6))
# ax_age = sns.histplot(data['Age'], kde=True, bins=20, color='orange')
# plt.title('Distribution of Age')
# plt.xlabel('Age')
# plt.ylabel('Frequency')
# # Add statistics to the plot
# add_stats_to_plot(ax_age, data['Age'])
# # Show the plot
# plt.show()

# Plotting distributions for categorical columns
# for column in categoricalColumns:
#     plt.figure(figsize=(10, 6))
#     sns.countplot(data=data, x=column, palette='coolwarm')
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()

# # Plotting distributions for numerical columns
# for column in numericalColumns:
#     plt.figure(figsize=(10, 6))
#     sns.histplot(data[column], kde=True, bins=20, color='blue')
#     plt.title(f'Distribution of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()