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


filename = "Motor vehicle insurance data.csv"

#Reading the csv
data = pd.read_csv(filename, sep=';')


#ID;Date_start_contract;Date_last_renewal;Date_next_renewal;Date_birth;Date_driving_licence;
#Distribution_channel;Seniority;Policies_in_force;Max_policies;Max_products;Lapse;Date_lapse;
#Payment;Premium;Cost_claims_year;N_claims_year;N_claims_history;R_Claims_history;Type_risk;Area;
#Second_driver;Year_matriculation;Power;Cylinder_capacity;Value_vehicle;N_doors;Type_fuel;Length;Weight

birthday =  pd.to_datetime(data['Date_birth'], format='%d/%m/%Y')
birthyear = birthday.dt.year

contract_day = pd.to_datetime(data['Date_start_contract'], format='%d/%m/%Y')
contract_year = contract_day.dt.year
data['Contract_year'] = contract_year

day_start_driving = pd.to_datetime(data['Date_driving_licence'], format='%d/%m/%Y')
year_start = day_start_driving.dt.year

data['Years_driving'] = contract_year - year_start
data['Age'] = contract_year - birthyear

data['Distribution_channel'] = data['Distribution_channel'].astype(str)



print(data.columns)
columnsToUse = ['Seniority', 'Premium', 'Type_risk', 'Area', 'Second_driver', 'Year_matriculation', 
			'Value_vehicle', 'Age', 'Years_driving', 'Distribution_channel', 'N_claims_history', 'Power', 'Cylinder_capacity','Weight', 'Length', 'Type_fuel', 'Payment', 'Contract_year']

data = data[columnsToUse]

#Remove rows with empty values. There aren't many of them so this doesn't affect the data
data = data.dropna()

categoricalColumns = ['Type_risk', 'Area', 'Second_driver', 'Distribution_channel', 'Type_fuel', 'Payment']
numericalColumns = ['Seniority', 'Year_matriculation','Value_vehicle','Age', 'Years_driving', 'N_claims_history', 'Power', 'Cylinder_capacity', 'Weight', 'Length', 'Contract_year']



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
