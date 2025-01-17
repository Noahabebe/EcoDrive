import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import requests

class SurgePriceClassifier:
    def __init__(self, data_frame):
        '''
        Loading the dataframe and converting the commonly used
        surge price multipliers into categorical variables.
        params: input dataframe
        return: none
        '''
        self.data_frame = data_frame
        self.predictive_surge_mapping = {1: 1, 2: 1.25, 3: 1.5, 4: 1.75, 5: 2}
        self.model_features = [
            'location_latitude',
            'location_longitude',
            'temperature',
            'pressure',
            'humidity',
            'wind_speed',
            'wind_deg',
            'rain_1h',
            'snow_1h'
        ]

    def get_rush_hour(self):
        '''
        Based on the time of the day, a flag is assigned indicating
        if the particular hour classifies as being a rush hour or not.
        This is used as a parameter in deducing the surge price
        multiplier.
        params, return: none
        '''
        var_hour = datetime.now().hour

        if (var_hour >= 6 and var_hour < 10) or (var_hour >= 15 and var_hour < 19):
            self.data_frame['rush_hour'] = 1
        else:
            self.data_frame['rush_hour'] = 0

    def surge_prediction_model(self):
        '''
        Loading the surge price classification model using Python
        pickle load.
        params: none
        return: surge multiplier
        '''
        self.get_rush_hour()
        filename = "model_weights/surge_classification_rf_model.sav"
        
        # Loading the model
        try:
            with open(filename, 'rb') as f:
                loaded_model = pickle.load(f)
                print("Model loaded successfully. loaded model is: ", loaded_model)
        except FileNotFoundError:
            print(f"Model file not found at {filename}")
            return None
        
        self.df_append()

        # Dropping 'id' and 'surge_mult' columns if they exist
        if 'id' in self.data_frame.columns:
            self.data_frame = self.data_frame.drop(columns=['id'])
        if 'surge_mult' in self.data_frame.columns:
            self.data_frame = self.data_frame.drop(columns=['surge_mult'])

        # Convert non-numeric columns to NaN and handle them
        self.data_frame = self.data_frame.apply(pd.to_numeric, errors='coerce')

        # Replace NaN with 0 or another value (e.g., median or mean)
        self.data_frame = self.data_frame.fillna(0)

        # Check for infinity values and replace them with a large number or NaN
        self.data_frame.replace([np.inf, -np.inf], 9999, inplace=True)

        # Ensure all columns are in proper numeric format (float32 or float64)
        self.data_frame = self.data_frame.astype(np.float32)

        # Select only the features that the model was trained on
        self.data_frame = self.data_frame[self.model_features]

        # Debugging: Print columns and head of the DataFrame
        print('columns:', self.data_frame.columns.tolist())
        print('head:', self.data_frame.head())

        # Making a prediction
        try:
            result = loaded_model.predict(self.data_frame)
            return self.predictive_surge_mapping.get(int(result), "Unknown surge multiplier")
        except Exception as e:
            print("Error during prediction:", e)
            return "Prediction error"

    def df_append(self):
        '''
        Appending the existing training datasets with the current record.
        Implementation in progress.
        params, return: none
        '''
        return True