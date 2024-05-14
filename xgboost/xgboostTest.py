import pandas as pd
import torch

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.metrics import r2_score
import joblib
import os

#using gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    
    
filename = "Motor_vehicle_insurance_data.csv"

data = pd.read_csv(filename, sep=';')

directory = "xgbModels"
if not os.path.exists(directory):
    os.makedirs(directory) 


#Feature engineering

last_renewal_day = pd.to_datetime(data['Date_last_renewal'], format='%d/%m/%Y')
last_renewal_year = last_renewal_day.dt.year
data['Last_renewal_year'] = last_renewal_year

birthday =  pd.to_datetime(data['Date_birth'], format='%d/%m/%Y')
birthyear = birthday.dt.year

contract_day = pd.to_datetime(data['Date_start_contract'], format='%d/%m/%Y')
contract_year = contract_day.dt.year
data['Contract_year'] = contract_year

#Removes the multiple entries for the same contract by keeping only the most recent
avg_premium_per_id = data.groupby('ID')['Premium'].mean().reset_index()
data = data.merge(avg_premium_per_id, on='ID', suffixes=('', '_avg'))
data['Premium'] = data['Premium_avg']
data.drop('Premium_avg', axis=1, inplace=True)
data = data.sort_values(by=['ID', 'Last_renewal_year'], ascending=[True, False])
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

years_on_policy = last_renewal_year - contract_year
data['Years_on_policy'] = years_on_policy

data['accidents'] = data['N_claims_history'] / (data['Years_on_policy'] + 1)

data['Age_Years_Driving_Interaction'] = data['Age'] * data['Years_driving']

columnsToUse = ['Seniority', 'Premium', 'Type_risk', 'Area', 'Second_driver', 'Years_on_road', 'R_Claims_history', 'Years_on_policy', 'accidents', 
			'Value_vehicle', 'Age', 'Years_driving', 'Distribution_channel', 'N_claims_history', 'Power', 'Cylinder_capacity',
			'Weight', 'Length', 'Type_fuel', 'Payment', 'Contract_year', 'Policies_in_force', 'Lapse']

data = data[columnsToUse]



#We fill the missing length value with the mean so we can use the rest of the row
data['Length'] = data['Length'].fillna(data['Length'].mean())

#Replace the null values of fuel types the a third value
null_indices = data['Type_fuel'].isnull()
data.loc[null_indices, 'Type_fuel'] = 'Unknown'

categoricalColumns = [ 'Area', 'Second_driver', 'Distribution_channel', 'Type_fuel', 'Payment']
numericalColumns = ['Seniority', 'Years_on_road','Value_vehicle','Age', 'Years_driving', 'N_claims_history', 'Power', 'Cylinder_capacity', 
					'Weight', 'Length', 'Contract_year', 'R_Claims_history', 'Years_on_policy', 'accidents', 'Policies_in_force', 'Lapse']


preprocessor = ColumnTransformer(
transformers=[
	('num', StandardScaler(), numericalColumns),
	('cat', OneHotEncoder(), categoricalColumns)
])


data.info()

# XGBoost parameters
params = {
    'objective': 'reg:squarederror',  
    'eval_metric': 'rmse', 
    'learning_rate': 0.01,
    'max_depth': 6,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 1000,
    'seed': 42
}

type_risk_values = data['Type_risk'].unique()

models = {}
import matplotlib.pyplot as plt

percentage_within_thresholds = {type_risk_value: {} for type_risk_value in type_risk_values}

all_absolute_errors = []
all_actual_values = []

for type_risk_value in type_risk_values:
    subset_data = data[data['Type_risk'] == type_risk_value]
    
    subset_features = subset_data.drop(columns=['Premium', 'Type_risk'])
    subset_labels = subset_data['Premium']
    
    subset_features_preprocessed = preprocessor.fit_transform(subset_features)

    joblib.dump(preprocessor, f"{directory}\\preprocessor_{type_risk_value}.pkl")   
    
    subset_X_train, subset_X_test, subset_y_train, subset_y_test = train_test_split(
        subset_features_preprocessed, subset_labels, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(**params)
    model.fit(subset_X_train, subset_y_train)
    
    models[type_risk_value] = model
    
    y_pred = model.predict(subset_X_test)
    
    model.save_model(f"xgbModels\\model_{type_risk_value}.json")

    mse = mean_squared_error(subset_y_test, y_pred)
    print(f"Mean Squared Error for Type_risk {type_risk_value}:", mse)
    
    absolute_error = np.abs(subset_y_test - y_pred)
    average_absolute_error = np.mean(absolute_error)
    print(f"Average Absolute Error for Type_risk {type_risk_value}:", average_absolute_error)
    
    thresholds = {}
    for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        percentage_within_threshold = np.mean(absolute_error / subset_y_test <= threshold / 100) * 100
        thresholds[threshold] = percentage_within_threshold
        print(f"Percentage of predictions within {threshold}% of the actual value for Type_risk {type_risk_value}: {percentage_within_threshold}")
    
    percentage_within_thresholds[type_risk_value] = thresholds
    r2 = r2_score(subset_y_test, y_pred)
    print(f"R² Score for Type_risk {type_risk_value}:", r2)
    print()
    
    all_absolute_errors.extend(absolute_error)
    all_actual_values.extend(subset_y_test)


    
all_absolute_errors = np.array(all_absolute_errors)
all_actual_values = np.array(all_actual_values)

all_thresholds = {}
for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    percentage_within_threshold = np.mean(all_absolute_errors / all_actual_values <= threshold / 100) * 100
    all_thresholds[threshold] = percentage_within_threshold
    print(f"Percentage of predictions within {threshold}% of the actual value for the whole dataset: {percentage_within_threshold}")

percentage_within_thresholds['All'] = all_thresholds

for type_risk_value, thresholds in percentage_within_thresholds.items():
    type_defined = definition_of_type_risk(type_risk_value)
    if type_risk_value != 'All':
        plt.plot(thresholds.keys(), thresholds.values(), label=f'{type_defined}')
percentage_within_thresholds['All'] = all_thresholds

plt.xlabel('Threshold (%)')
plt.ylabel('Percentage of Predictions within Threshold')
plt.title('Percentage of Predictions within Different Thresholds for Each Type_risk')
plt.legend()
plt.grid(True)
plt.savefig('report\\images\\individual_thresholds_xgb.png')
plt.show()

combined_thresholds = percentage_within_thresholds['All']

plt.plot(combined_thresholds.keys(), combined_thresholds.values(), label='All risk types')
plt.xlabel('Threshold (%)')
plt.ylabel('Percentage of Predictions within Threshold')
plt.title('Percentage of Predictions within Different Thresholds for Combined Data')
plt.legend()
plt.grid(True)
plt.savefig('report\\images\\combined_thresholds_xgbt.png')
plt.show()

encoded_categorical_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categoricalColumns))

all_feature_names = numericalColumns + encoded_categorical_features

for type_risk_value, model in models.items():
    feature_importance = model.feature_importances_

    all_feature_names = numericalColumns + encoded_categorical_features

    feature_importance_dict = dict(zip(all_feature_names, feature_importance))

    aggregated_feature_importance = {}
    for cat_col in categoricalColumns:
        related_features = [col for col in all_feature_names if col.startswith(cat_col)]
        importance_sum = sum(feature_importance_dict[feature] for feature in related_features)
        for feature in related_features:
            if feature in feature_importance_dict:
                del feature_importance_dict[feature]
        aggregated_feature_importance[cat_col] = importance_sum


    combined_importances = {**feature_importance_dict, **aggregated_feature_importance}

    combined_importance_df = pd.DataFrame({'Feature': list(combined_importances.keys()), 'Importance': list(combined_importances.values())})

    combined_importance_df = combined_importance_df.sort_values(by='Importance', ascending=True)

    type_defined = definition_of_type_risk(type_risk_value)
    plt.figure(figsize=(10, 6))
    plt.barh(combined_importance_df['Feature'], combined_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Feature Importance for {type_defined}')
    plt.savefig(f'report\\images\\{type_defined}_feature_importance_xgb.png', bbox_inches='tight')
    plt.show()

