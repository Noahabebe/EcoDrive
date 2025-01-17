
import pickle
import requests
import pandas as pd
def weather_information(latitude, longitude, api_key):
    """
    Get weather information for a specific location.
    
    Parameters:
    - latitude (float): Latitude of the location.
    - longitude (float): Longitude of the location.
    - api_key (str): Your OpenWeatherMap API key.
    
    Returns:
    - pd.DataFrame: DataFrame containing weather information.
    """
   
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}&units=metric"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error
    
    weather_data = {
        'location_latitude': latitude,
        'location_longitude': longitude,
        'temperature': data.get('main', {}).get('temp'),
        'pressure': data.get('main', {}).get('pressure'),
        'humidity': data.get('main', {}).get('humidity'),
        'weather_condition': data.get('weather', [{}])[0].get('description'),
        'wind_speed': data.get('wind', {}).get('speed'),
        'wind_deg': data.get('wind', {}).get('deg'),
        'rain_1h': data.get('rain', {}).get('1h', 0),
        'snow_1h': data.get('snow', {}).get('1h', 0)
    }
    
    return pd.DataFrame([weather_data])

print('passed')
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
        self.uber_train = pd.DataFrame()
        self.lyft_train = pd.DataFrame()

    def df_modification(self):
        '''
        Splitting the columns in the dataframe into
        Uber and Lyft cab types.
        params, return: none
        '''
        uber_list = ['Black', 'Black SUV', 'UberPool', 'UberX',
                     'UberXL', 'WAV']
        lyft_list = ['Lux', 'Lux Black', 'Lux Black XL', 'Lyft',
                     'Lyft XL', 'Shared']
        self.uber = self.data_frame[['source_lat', 'source_long',
                                     'dest_lat', 'dest_long', 'distance',
                                     'surge_multiplier', 'uber_price']]
        self.lyft = self.data_frame[['source_lat', 'source_long', 'dest_lat',
                                     'dest_long', 'distance',
                                     'surge_multiplier', 'lyft_price']]
        self.uber[['Black', 'Black SUV', 'UberPool', 'UberX', 'UberXL',
                   'WAV']] = 0
        self.lyft[['Lux', 'Lux Black', 'Lux Black XL', 'Lyft', 'Lyft XL',
                   'Shared']] = 0
        uber_type = self.data_frame["uber_cab_type"].iloc[0]
        lyft_type = self.data_frame["lyft_cab_type"].iloc[0]

        for i in uber_list:
            if uber_type == i:
                self.uber[i] = self.uber[i].replace(0, 1)

        for i in lyft_list:
            if lyft_type == i:
                self.lyft[i] = self.lyft[i].replace(0, 1)

    def get_uber_price(self):
        """
        loading multi linear regression model for Uber to get the price.
        params: none
        return: uber price
        """
        filename = "model_weights/uber_mlr_model.sav"
        try:
            uber_mlr_model = pickle.load(open(filename, 'rb'))
            uber_price = \
                uber_mlr_model.predict(self.uber.drop(columns=['uber_price']))
            return uber_price
        except FileNotFoundError:
            print(f"Model file not found at {filename}")
            return [0]  # Return a default value

    def get_lyft_price(self):
        """
        loading multi linear regression model for Lyft to get the price.
        params: none
        return: lyft price
        """
        filename = "model_weights/lyft_mlr_model.sav"
        try:
            lyft_mlr_model = pickle.load(open(filename, 'rb'))
            lyft_price = \
                lyft_mlr_model.predict(self.lyft.drop(columns=['lyft_price']))
            return lyft_price
        except FileNotFoundError:
            print(f"Model file not found at {filename}")
            return [0]  # Return a default value

    def df_append(self):
        """
        Appending the training datasets with the current record.
        Implementation in progress.
        params, return: none
        """
        return True