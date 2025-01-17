import pandas as pd
import pickle


class CabPricePredictor:
    def __init__(self, data_frame):
        '''
        Loading the passed dataframe and creating empty
        data frames for Uber and Lyft.
        params: dataframe for prediction
        return: none
        '''
        self.data_frame = data_frame
        self.uber = pd.DataFrame()
        self.lyft = pd.DataFrame()

    def df_modification(self):
        '''
        Splitting the columns in the dataframe into
        Uber and Lyft cab types and modifying the dataframe.
        params, return: none
        '''
        uber_list = ['Black', 'Black SUV', 'UberPool', 'UberX',
                     'UberXL', 'WAV']
        lyft_list = ['Lux', 'Lux Black', 'Lux Black XL', 'Lyft',
                     'Lyft XL', 'Shared']

        # Create columns for Uber and Lyft prices
        self.uber = self.data_frame[['source_lat', 'source_long',
                                     'dest_lat', 'dest_long', 'distance',
                                     'surge_multiplier', 'uber_price']]
        self.lyft = self.data_frame[['source_lat', 'source_long', 'dest_lat',
                                     'dest_long', 'distance', 
                                     'surge_multiplier', 'lyft_price']]

        # Add additional columns for cab types with initial value 0
        self.uber[uber_list] = 0
        self.lyft[lyft_list] = 0

        # Get the cab types for Uber and Lyft from the first row
        uber_type = self.data_frame["uber_cab_type"].iloc[0]
        lyft_type = self.data_frame["lyft_cab_type"].iloc[0]

        # Set the correct value for the selected Uber and Lyft cab types
        if uber_type in uber_list:
            self.uber[uber_type] = 1
        if lyft_type in lyft_list:
            self.lyft[lyft_type] = 1

    def get_uber_price(self):
        """
        Loading multi linear regression model for Uber to get the price.
        params: none
        return: uber price
        """
        filename = "model_weights/uber_mlr_model.sav"
        try:
            # Load the pre-trained model for Uber
            uber_mlr_model = pickle.load(open(filename, 'rb'))
            # Prepare the data for prediction (excluding 'uber_price' column)
            uber_features = self.uber.drop(columns=['uber_price'])
            # Make the prediction
            uber_price = uber_mlr_model.predict(uber_features)
            return uber_price
        except Exception as e:
            print(f"Error loading Uber model: {e}")
            return None

    def get_lyft_price(self):
        """
        Loading multi linear regression model for Lyft to get the price.
        params: none
        return: lyft price
        """
        filename = "model_weights/lyft_mlr_model.sav"
        try:
            # Load the pre-trained model for Lyft
            lyft_mlr_model = pickle.load(open(filename, 'rb'))
            # Prepare the data for prediction (excluding 'lyft_price' column)
            lyft_features = self.lyft.drop(columns=['lyft_price'])
            # Make the prediction
            lyft_price = lyft_mlr_model.predict(lyft_features)
            return lyft_price
        except Exception as e:
            print(f"Error loading Lyft model: {e}")
            return None

    def cab_price_prediction(self):
        """
        Price prediction function called by the aggregator
        params: none
        return: uber_price, lyft_price
        """
        self.df_modification()
        uber_price = self.get_uber_price()
        lyft_price = self.get_lyft_price()

        # Return prices if predictions are valid
        if uber_price is not None and lyft_price is not None:
            return round(uber_price[0], 2), round(lyft_price[0], 2)
        else:
            print("Error in price prediction")
            return None, None

    def df_append(self):
        """
        Appending the training datasets with the current record.
        Implementation in progress.
        params, return: none
        """
        return True
