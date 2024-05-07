import lightgbm as lgb
import pandas as pd
import joblib

def makePrediction(input):

    print(input)
    return
    # Load the saved model
    loaded_model = lgb.Booster(model_file='lightgbmModels/model_3.txt')


    # print(example_input_df)
    filename = "lightgbmModels\\test.csv"

    #Reading the csv
    testinput = pd.read_csv(filename, sep=';')

    data = pd.DataFrame()

    data['Seniority'] = testinput['Seniority']
    data['Area'] = testinput['Area']
    data['Second_driver'] = testinput['Second_driver']
    data['Date_last_renewal'] = testinput['Date_last_renewal']
    data['Date_next_renewal'] = testinput['Date_next_renewal']
    data['R_Claims_history'] = testinput['R_Claims_history']
    data['Date_birth'] = testinput['Date_birth']
    data['Value_vehicle'] = testinput['Value_vehicle']
    data['Date_start_contract'] = testinput['Date_start_contract']
    data['Date_driving_licence'] = testinput['Date_driving_licence']
    data['Distribution_channel'] = testinput['Distribution_channel']
    data['N_claims_history'] = testinput['N_claims_history']
    data['Power'] = testinput['Power']
    data['Cylinder_capacity'] = testinput['Cylinder_capacity']
    data['Weight'] = testinput['Weight']
    data['Length'] = testinput['Length']
    data['Type_fuel'] = testinput['Type_fuel']
    data['Payment'] = testinput['Payment']
    data['Year_matriculation'] = testinput['Year_matriculation']
    data['Policies_in_force'] = testinput['Policies_in_force']
    data['Seniority'] = testinput['Seniority']
    data['Lapse'] = testinput['Lapse']



    #Feature engineering

    last_renewal_day = pd.to_datetime(data['Date_last_renewal'], format='%d/%m/%Y')
    last_renewal_year = last_renewal_day.dt.year
    data['Last_renewal_year'] = last_renewal_year

    birthday =  pd.to_datetime(data['Date_birth'], format='%d/%m/%Y')
    birthyear = birthday.dt.year

    contract_day = pd.to_datetime(data['Date_start_contract'], format='%d/%m/%Y')
    contract_year = contract_day.dt.year
    data['Contract_year'] = contract_year

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

    data['accidents'] = data['N_claims_history'] / (data['Years_on_policy'] + 1)

    data['Age_Years_Driving_Interaction'] = data['Age'] * data['Years_driving']


    columnsToUse = ['Seniority', 'Area', 'Second_driver', 'Years_on_road', 'R_Claims_history', 'Years_on_policy', 'accidents', 
                'Value_vehicle', 'Age', 'Years_driving', 'Distribution_channel', 'N_claims_history', 'Power', 'Cylinder_capacity',
                'Weight', 'Length', 'Type_fuel', 'Payment', 'Contract_year', 'Policies_in_force', 'Lapse']

    data = data[columnsToUse]

    # Load the preprocessor corresponding to your model
    loaded_preprocessor = joblib.load('lightgbmModels/preprocessor_3.pkl')

    # Define categorical and numerical columns
    categoricalColumns = ['Area', 'Second_driver', 'Distribution_channel', 'Type_fuel', 'Payment']
    numericalColumns = ['Seniority', 'Years_on_road', 'Value_vehicle', 'Age', 'Years_driving', 'N_claims_history', 
                        'Power', 'Cylinder_capacity', 'Weight', 'Length', 'Contract_year', 'R_Claims_history', 
                        'Years_on_policy', 'accidents', 'Policies_in_force', 'Lapse']

    # Transform new data using the loaded preprocessor
    new_data_preprocessed = loaded_preprocessor.transform(data)

    # Make prediction using the loaded model
    prediction = loaded_model.predict(new_data_preprocessed)

    # Print the prediction
    print("Predicted Premium:", prediction)
