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

#using gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


filename = "motor_data11-14lats.csv"
filename2 = "motor_data14-2018.csv"

#Reading the csv
data1 = pd.read_csv(filename)
data2 = pd.read_csv(filename2)

data = pd.concat([data1, data2])

columnsToUse = ['SEX','INSURED_VALUE', 'USAGE', 'PROD_YEAR', 'TYPE_VEHICLE', 'PREMIUM']

data = data[columnsToUse]

#Remove rows with empty values. There aren't many of them so this doesn't affect the data
data = data.dropna()

categoricalColumns = ['SEX','USAGE', 'TYPE_VEHICLE'] 
numericalColumns = ['INSURED_VALUE', 'PROD_YEAR']



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

from sklearn.ensemble import RandomForestRegressor

# Initialize Random Forest regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Train Random Forest regressor
rf_regressor.fit(X_train, y_train)

from joblib import dump, load

# Save the trained model to a file
model_filename = "random_forest_model_3.joblib"
dump(rf_regressor, model_filename)

# Predict on test set
y_pred = rf_regressor.predict(X_test)

# Calculate Mean Squared Error
from sklearn.metrics import mean_squared_error  # Import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
# Calculate absolute error
absolute_error = np.abs(y_test - y_pred)

# Print absolute error
print("Absolute Error:", absolute_error)
