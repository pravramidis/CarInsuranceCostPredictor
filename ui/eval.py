import lightgbm as lgb
import pandas as pd
import joblib
from datetime import datetime

def makePrediction(input):

    type_risk_value = input['vehicle_screen']['vehicle_type']
    # print(input)

    data = pd.DataFrame(columns=['Seniority', 'Area', 'Second_driver', 'Years_on_road', 'R_Claims_history', 
                              'Years_on_policy', 'accidents', 'Value_vehicle', 'Age', 'Years_driving', 
                              'Distribution_channel', 'N_claims_history', 'Power', 'Cylinder_capacity',
                              'Weight', 'Length', 'Type_fuel', 'Payment', 'Contract_year', 
                              'Policies_in_force', 'Lapse'])



    #If the user left fields empty then for the numerical values the median will be substitued 
    #and for the categorical columns the most common value

    #changing the format
    if (type_risk_value == "Motorbike"):
        type_risk_value = 1
    elif (type_risk_value == "Van"):
        type_risk_value = 2
    elif (type_risk_value == "Passenger Car"):
        type_risk_value = 3
    elif (type_risk_value == "Agricultural Vehicle"):
        type_risk_value = 4
    else: 
        type_risk_value = 3

    if ( input['main_screen']['area'] == "Urban"):
        data.at[0,'Area'] = 1
    else:
        data.at[0,'Area'] = 0


    if ( input['vehicle_screen']['fuel_type'] == "Petrol"):
        data.at[0,'Type_fuel'] = 'P'
    else:
        data.at[0,'Type_fuel'] = 'D'


    if ( input['vehicle_screen']['drivers'] == "Yes"):
        data.at[0,'Second_driver'] = 1
    else:
        data.at[0,'Second_driver'] = 0


    if ( input['insurance_screen']['channel'] == "Insurance Brokers"):
        data.at[0,'Distribution_channel'] = '1'
    else:
        data.at[0,'Distribution_channel'] = '0'


    if ( input['insurance_screen']['payment'] == "Half-yearly"):
        data.at[0,'Payment'] = 1
    else:
        data.at[0,'Payment'] = 0



    data.at[0,'Seniority'] = input['main_screen']['seniority']
    data.at[0,'Date_last_renewal'] = input['insurance_screen']['last_renewal']
    data.at[0,'Date_next_renewal'] = input['insurance_screen']['next_renewal']
    data.at[0,'R_Claims_history'] = input['insurance_screen']['ratio_claims']
    data.at[0,'Date_birth'] = input['main_screen']['date_of_birth']
    data.at[0,'Value_vehicle'] = input['vehicle_screen']['value']
    data.at[0,'Date_start_contract'] = input['insurance_screen']['start_contract']
    data.at[0,'Date_driving_licence'] = input['main_screen']['licence_issue_date']
    data.at[0,'N_claims_history'] = input['insurance_screen']['claims']
    data.at[0,'Power'] = input['vehicle_screen']['horse_power']
    data.at[0,'Cylinder_capacity'] = input['vehicle_screen']['cylinder_capacity']
    data.at[0,'Weight'] = input['vehicle_screen']['weight']
    data.at[0,'Length'] = input['vehicle_screen']['length']
    data.at[0,'Year_matriculation'] = input['vehicle_screen']['registration_year']
    data.at[0,'Policies_in_force'] = input['insurance_screen']['policies']
    data.at[0,'Lapse'] = input['insurance_screen']['lapse']

    if (data.at[0,'Seniority'] == ''):
        data.at[0,'Seniority'] = 4 

    if (data.at[0,'Weight'] == ''):
        data.at[0,'Weight'] =  1211 

    if (data.at[0,'Length'] == ''):
        data.at[0,'Length'] =  4.239000 

    if (data.at[0,'R_Claims_history'] == ''):
        data.at[0,'R_Claims_history'] = 0    

    if (data.at[0,'Value_vehicle'] == ''):
        data.at[0,'Value_vehicle'] = 17770 

    if (data.at[0,'N_claims_history'] == ''):
        data.at[0,'N_claims_history'] = 1

    if (data.at[0,'Power'] == ''):
        data.at[0,'Power'] = 90

    if (data.at[0,'Cylinder_capacity'] == ''):
        data.at[0,'Cylinder_capacity'] = 1598


    if (data.at[0,'Policies_in_force'] == ''):
        data.at[0,'Policies_in_force'] = 1

    if (data.at[0,'Lapse'] == ''):
        data.at[0,'Lapse'] = 0
    
    if (data.at[0,'Year_matriculation'] == ''):
        data.at[0,'Year_matriculation'] = 0

    current_date = datetime.now()
    if (data.at[0,'Date_last_renewal'] == ''):
        data.at[0,'Date_last_renewal'] = current_date.strftime('%m/%d/%Y')
        

    if (data.at[0,'Date_next_renewal'] == ''):
        data.at[0, 'Date_next_renewal'] = (current_date + pd.DateOffset(years=1)).strftime('%m/%d/%Y')

    if (data.at[0,'Date_start_contract'] == ''):
        data.at[0,'Date_start_contract'] = current_date.strftime('%m/%d/%Y') 

    if (data.at[0,'Date_birth'] == ''):
        data.at[0,'Date_birth'] = current_date.strftime('%m/%d/%Y')    

    #Feature engineering
    last_renewal_day = pd.to_datetime(data['Date_last_renewal'], format='%m/%d/%Y')
    last_renewal_year = last_renewal_day.dt.year
    data['Last_renewal_year'] = last_renewal_year

    contract_day = pd.to_datetime(data['Date_start_contract'], format='%m/%d/%Y')
    contract_year = contract_day.dt.year
    data['Contract_year'] = contract_year

    if (data.at[0,'Date_driving_licence'] == ''):
       data['Years_driving'] =  24
    else:
        day_start_driving = pd.to_datetime(data['Date_driving_licence'], format='%m/%d/%Y')
        year_start = day_start_driving.dt.year
        data['Years_driving'] = last_renewal_year - year_start

    if (data.at[0,'Date_birth'] == ''): 
       data['Age'] = 47 
    else: 
        birthday =  pd.to_datetime(data['Date_birth'], format='%m/%d/%Y')
        birthyear = birthday.dt.year
        data['Age'] = last_renewal_year - birthyear

    registration_year = data['Year_matriculation'].astype(int)
    
    data['Years_on_road'] = last_renewal_year - registration_year

    next_renewal_day = pd.to_datetime(data['Date_next_renewal'], format='%m/%d/%Y')
    next_renewal_year = next_renewal_day.dt.year
    data['Next_renewal_year'] = next_renewal_year

    policy_duration = next_renewal_year - last_renewal_year
    data['Policy_duration'] = policy_duration

    years_on_policy = last_renewal_year - contract_year
    data['Years_on_policy'] = years_on_policy

    data['accidents'] = data['N_claims_history'].astype(int) / (data['Years_on_policy'] + 1)

    data['Age_Years_Driving_Interaction'] = data['Age'] * data['Years_driving']

    columnsToUse = ['Seniority', 'Area', 'Second_driver', 'Years_on_road', 'R_Claims_history', 'Years_on_policy', 'accidents', 
                'Value_vehicle', 'Age', 'Years_driving', 'Distribution_channel', 'N_claims_history', 'Power', 'Cylinder_capacity',
                'Weight', 'Length', 'Type_fuel', 'Payment', 'Contract_year', 'Policies_in_force', 'Lapse']

    data = data[columnsToUse]

    print(data)

    loaded_model = lgb.Booster(model_file=f"lightgbmModels\\model_{type_risk_value}.txt")

    # Load the preprocessor corresponding to your model
    loaded_preprocessor = joblib.load(f"lightgbmModels\\preprocessor_{type_risk_value}.pkl")


    # Transform new data using the loaded preprocessor
    new_data_preprocessed = loaded_preprocessor.transform(data)

    # Make prediction using the loaded model
    prediction = loaded_model.predict(new_data_preprocessed)

    # Print the prediction
    print("Predicted Premium:", prediction)

    return prediction
