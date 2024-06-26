import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

import numpy as np # linear algebra
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor  # Changed
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.metrics import r2_score

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
    'n_estimators': 700,  
    'max_depth': 15,  
    'min_samples_split': 4,  
    'min_samples_leaf': 2, 
    'max_features': 'sqrt',  
    'bootstrap': True, 
    'n_jobs': -1, 
    'random_state': 42
}


model = RandomForestRegressor(**params)
    
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

absolute_error = np.abs(y_test - y_pred)

average_absolute_error = np.mean(absolute_error)
print("Average Absolute Error:", average_absolute_error)

for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        percentage_within_threshold = np.mean(absolute_error / y_test <= threshold / 100) * 100
        print(f"Percentage of predictions within {threshold}% of the actual value: {percentage_within_threshold}")
    

r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

encoded_categorical_features = list(preprocessor.named_transformers_['cat'].get_feature_names_out(categoricalColumns))

all_feature_names = numericalColumns + encoded_categorical_features
percentage_within_thresholds = {type_risk_value: {} for type_risk_value in data['Type_risk'].unique()}

def evaluate_model_by_feature(model, X_test, y_test, risk_types):
    for risk_type in risk_types:
        risk_type_indices = X_test[:, all_feature_names.index('Type_risk_' + str(risk_type))] == 1
        X_test_risk_type = X_test[risk_type_indices]
        y_test_risk_type = y_test[risk_type_indices]

        y_pred_risk_type = model.predict(X_test_risk_type)

        mse = mean_squared_error(y_test_risk_type, y_pred_risk_type)
        mae = mean_absolute_error(y_test_risk_type, y_pred_risk_type)
        absolute_error_risk_type = np.abs(y_test_risk_type - y_pred_risk_type)

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
percentage_within_thresholds_by_risk = evaluate_model_by_feature(model, X_test, y_test, risk_types)

all_thresholds = {}
for threshold in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    percentage_within_threshold = np.mean(absolute_error / y_test <= threshold / 100) * 100
    all_thresholds[threshold] = percentage_within_threshold

percentage_within_thresholds_by_risk['All'] = all_thresholds


for type_risk_value, thresholds in percentage_within_thresholds.items():
    type_defined = definition_of_type_risk(type_risk_value)
    if type_risk_value != 'All':
        plt.plot(thresholds.keys(), thresholds.values(), label=f'{type_defined}')

plt.xlabel('Threshold (%)')
plt.ylabel('Percentage of Predictions within Threshold')
plt.title('Percentage of Predictions within Different Thresholds for Each Type_risk')
plt.legend()
plt.grid(True)
plt.savefig('report\\images\\combined_trained_individual_thresholds_random_forest.png')
plt.show()

combined_thresholds = percentage_within_thresholds['All']

plt.plot(combined_thresholds.keys(), combined_thresholds.values(), label='All risk types')
plt.xlabel('Threshold (%)')
plt.ylabel('Percentage of Predictions within Threshold')
plt.title('Percentage of Predictions within Different Thresholds for Combined Data')
plt.legend()
plt.grid(True)
plt.savefig('report\\images\\combined_trained_all_thresholds_random_forest.png')
plt.show()

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

plt.figure(figsize=(10, 6))
plt.barh(combined_importance_df['Feature'], combined_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.savefig(f'report\\images\\feature_importance_combined_random_forest.png', bbox_inches='tight')
plt.show()