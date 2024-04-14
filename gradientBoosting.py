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
