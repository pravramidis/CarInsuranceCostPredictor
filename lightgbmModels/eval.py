import lightgbm as lgb
import pandas as pd
import joblib

# Load the saved model
loaded_model = lgb.Booster(model_file='lightgbmModels/model_1.txt')

# Define the example input manually
example_input = {
    'Seniority': 4,
    'Area': 1,
    'Second_driver': 0,
    'Years_on_road': 3,
    'R_Claims_history': 0,
    'Years_on_policy': 1,
    'accidents': 0,
    'Value_vehicle': 10000,
    'Age': 30,
    'Years_driving': 10,
    'Distribution_channel': 0,
    'N_claims_history': 0,
    'Power': 80,
    'Cylinder_capacity': 599,
    'Weight': 1200,
    'Length': 4.5,
    'Type_fuel': 'P',
    'Payment': 0,
    'Contract_year': 2023,
    'Policies_in_force': 1,
    'Lapse': 0
}

# Convert the example input to a DataFrame
example_input_df = pd.DataFrame([example_input])

print(example_input_df)

# Load the preprocessor corresponding to your model
loaded_preprocessor = joblib.load('lightgbmModels/preprocessor_1.pkl')

# Define categorical and numerical columns
categoricalColumns = ['Area', 'Second_driver', 'Distribution_channel', 'Type_fuel', 'Payment']
numericalColumns = ['Seniority', 'Years_on_road', 'Value_vehicle', 'Age', 'Years_driving', 'N_claims_history', 
                    'Power', 'Cylinder_capacity', 'Weight', 'Length', 'Contract_year', 'R_Claims_history', 
                    'Years_on_policy', 'accidents', 'Policies_in_force', 'Lapse']

# Transform new data using the loaded preprocessor
new_data_preprocessed = loaded_preprocessor.transform(example_input_df)

# Make prediction using the loaded model
prediction = loaded_model.predict(new_data_preprocessed)

# Print the prediction
print("Predicted Premium:", prediction)
