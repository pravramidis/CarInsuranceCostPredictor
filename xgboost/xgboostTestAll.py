import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np # linear algebra

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor  # Changed
from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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

# data = data[(data['Type_risk'] == 4)]


#We fill the missing length value with the mean so we can use the rest of the row
data['Length'] = data['Length'].fillna(data['Length'].mean())

#Replace the null values of fuel types the a third value
null_indices = data['Type_fuel'].isnull()
data.loc[null_indices, 'Type_fuel'] = 'Unknown'

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

params = {
    'objective': 'reg:squarederror',  # or 'binary:logistic' for binary classification
    'eval_metric': 'rmse', 
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'seed': 42
}

# XGBoost model training
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)


# Predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate absolute error
absolute_error = np.abs(y_test - y_pred)

average_absolute_error = np.mean(absolute_error)
print("Average Absolute Error:", average_absolute_error)

# Calculate the percentage of predictions within %
for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        percentage_within_threshold = np.mean(absolute_error / y_test <= threshold / 100) * 100
        print(f"Percentage of predictions within {threshold}% of the actual value: {percentage_within_threshold}")
    

r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

encoded_categorical_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categoricalColumns))

# Combine numerical and encoded categorical feature names
all_feature_names = numericalColumns + encoded_categorical_features
percentage_within_thresholds = {type_risk_value: {} for type_risk_value in data['Type_risk'].unique()}

# Define a function to evaluate model performance for each risk type
def evaluate_model_by_feature(model, X_test, y_test, risk_types):
    for risk_type in risk_types:
        # Filter test data for the current risk type
        risk_type_indices = X_test[:, all_feature_names.index('Type_risk_' + str(risk_type))] == 1
        X_test_risk_type = X_test[risk_type_indices]
        y_test_risk_type = y_test[risk_type_indices]

        # Predict on the filtered test set
        y_pred_risk_type = model.predict(X_test_risk_type)

        # Calculate metrics for the current risk type
        mse = mean_squared_error(y_test_risk_type, y_pred_risk_type)
        mae = mean_absolute_error(y_test_risk_type, y_pred_risk_type)
        absolute_error_risk_type = np.abs(y_test_risk_type - y_pred_risk_type)

        # Print metrics for the current risk type
        print(f"Metrics for {risk_type} Risk Type:")
        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        thresholds_percentages = {}
        for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            percentage_within_threshold = np.mean(absolute_error_risk_type / y_test_risk_type <= threshold / 100) * 100
            thresholds_percentages[threshold] = percentage_within_threshold
            print(f"Percentage of predictions within {threshold}% of the actual value for Type_risk {risk_type}: {percentage_within_threshold}")
        percentage_within_thresholds[risk_type] = thresholds_percentages

    return percentage_within_thresholds
        

risk_types = data['Type_risk'].unique()
# Call the function and store the returned values
percentage_within_thresholds_by_risk = evaluate_model_by_feature(model, X_test, y_test, risk_types)

# Calculate the percentage within threshold for the whole dataset
all_thresholds = {}
for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    percentage_within_threshold = np.mean(absolute_error / y_test <= threshold / 100) * 100
    all_thresholds[threshold] = percentage_within_threshold

# Store the thresholds for the whole dataset
percentage_within_thresholds_by_risk['All'] = all_thresholds

# Plotting
for risk_type, thresholds in percentage_within_thresholds_by_risk.items():
    plt.plot(thresholds.keys(), thresholds.values(), label=f'Type_risk {risk_type}')



plt.xlabel('Threshold (%)')
plt.ylabel('Percentage of Predictions within Threshold')
plt.title('Percentage of Predictions within Different Thresholds for Each Type_risk')
plt.legend()
plt.grid(True)
plt.show()