import os
import requests
from datetime import timedelta
from flask import Flask, render_template, request, redirect, session, jsonify, url_for
from urllib.parse import quote_plus
import pandas as pd
import openrouteservice
from model_scripts.surge_inference import SurgePriceClassifier
from model_scripts.dynamic_pricing_inference import CabPricePredictor
from utils.weather_information import weather_information
import re
app = Flask(__name__) 
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecretkey")


# In-memory user data (for simplicity)
user_data = {}
@app.route('/search_car', methods=['POST'])
def search_car():
    try:
        query = request.json.get('query', '').strip()
        if not query:
            return jsonify({"cars": []}), 200

        api_url = f"https://vpic.nhtsa.dot.gov/api/vehicles/getmodelsformake/{query}?format=json"
         
        response = requests.get(api_url)
        response.raise_for_status()

        car_data = response.json()
        cars = [f"{result['Make_Name']} {result['Model_Name']}" for result in car_data.get("Results", [])]



        return jsonify({"cars": cars}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# Route for initial setup
@app.route('/setup', methods=['GET', 'POST'])
def setup():
    if request.method == 'POST':
        try:
            car_details = request.form.get('selected_car')
            email = request.form.get('email')
            fuel_type = request.form.get('fuel_type')
            year = request.form.get('year')

            # Validate input fields
            if not car_details or not email or not year or not fuel_type:
                return render_template('setup.html', error="Please fill in all required fields.")

            

            # Store user data in session
            session['user_id'] = email
            user_data[session['user_id']] = {
                'car_details': car_details,
                'fuel_type': fuel_type,
                'year': year
            }

            return redirect(url_for('home'))
        except Exception as e:
            return render_template('setup.html', error=f"An error occurred: {e}")

    return render_template('setup.html')





@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('setup'))
    return render_template('dashboard.html')

# Route for logout
@app.route("/logout")
def logout():
    session.pop('user_id', None)
    return redirect(url_for("home"))

@app.route("/calculate_trip", methods=["POST"])
def calculate_trip():
    try:
        # Retrieve input from the user
        start_location = request.form.get('start_location')
        end_location = request.form.get('end_location')
        uber_cab_type = request.form.get('uber_cab_type')  # e.g., UberX
        lyft_cab_type = request.form.get('lyft_cab_type')  # e.g., Lyft XL

        # Validate input
        if not all([start_location, end_location, uber_cab_type, lyft_cab_type]):
            return jsonify({"error": "Start and end locations, along with cab types, are required."}), 400

        # Validate user session
        if 'user_id' not in session:
            return jsonify({"error": "User not logged in."}), 403

        user_car = user_data.get(session['user_id'], {}).get('car_details')
        fuel_type = user_data.get(session['user_id'], {}).get('fuel_type')
        car_year = user_data.get(session['user_id'], {}).get('year')

        # Check if user_car is valid
        if not user_car:
            return jsonify({"error": "Car details are missing. Please set them up again in your profile."}), 400
        
        if user_car:
                car_parts = user_car.split()
                make = car_parts[0]  # First word is make
                model = " ".join(car_parts[1:])   # Last part for model
        else:
                make, model = None, None
        

        
        car_data = {"car_model": model, "car_make": make, "car_year": car_year}

        start_coords = get_coordinates_osm(start_location)
        end_coords = get_coordinates_osm(end_location)

        # Fetch route details
        route = get_route_ors(start_coords, end_coords)
        distance_km = route['distance'] / 1000  # Convert meters to kilometers
        estimated_time = route['duration'] / 60  # Convert seconds to minutes

        # Fetch cab prices
        cab_price_data = getCabPrice(
            start_location, end_location, start_coords, end_coords, uber_cab_type, lyft_cab_type
        )
        uber_fare = cab_price_data.get("uber_price")
        lyft_fare = cab_price_data.get("lyft_price")

        print("passed2")
        # Calculate fuel cost and consumption
        fuel_cost_data = calculate_fuel_cost_and_consumption(car_data, distance_km, fuel_type)

        print(fuel_cost_data['total_fuel_cost'], fuel_cost_data['estimated_fuel_consumption_gallons'])
        print(uber_fare)
        print(lyft_fare)
        # Calculate profit margins
        profit_margin_uber = uber_fare - fuel_cost_data['total_fuel_cost'] if uber_fare else None
        profit_margin_lyft = lyft_fare - fuel_cost_data['total_fuel_cost'] if lyft_fare else None

        
        
        # Build the response
        result = {
            "start": start_location,
            "end": end_location,
            "distance_km": round(distance_km, 2),
            "estimated_time_min": round(estimated_time, 2),
            "fuel_cost": round(calculate_fuel_cost_and_consumption(car_data, distance_km, fuel_type)['total_fuel_cost'], 2),
            "fuel_consumption_gallons": round(calculate_fuel_cost_and_consumption(car_data, distance_km, fuel_type)['estimated_fuel_consumption_gallons'], 2),
            "uber_fare": round(uber_fare, 2) if uber_fare else None,
            "lyft_fare": round(lyft_fare, 2) if lyft_fare else None,
            "profit_margin_uber": round(profit_margin_uber, 2) if profit_margin_uber else None,
            "profit_margin_lyft": round(profit_margin_lyft, 2) if profit_margin_lyft else None,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500





def getCabPrice(start_location, end_location, start_coords, end_coords, uber_cab_type, lyft_cab_type):
    '''
    This function calculates the cab price for Uber and Lyft based on the input parameters.
    It handles the route calculation, surge pricing, and dynamic pricing for both Uber and Lyft.
    
    params: start_location, end_location, start_coords, end_coords, uber_cab_type, lyft_cab_type
    return: json object with Uber and Lyft prices, distance, and estimated time
    '''
    try:
        # Calculate route using OSRM or other routing service
        routes = get_route_ors(start_coords, end_coords)

        # Get the distance and estimated time from the route
        distance_km = routes['distance'] / 1000  # Convert meters to kilometers
        estimated_time = routes['duration'] / 60  # Convert seconds to minutes

        # Default surge values (can be adjusted based on actual conditions)
        uber_original_surge = 1.0
        lyft_original_surge = 1.0

        # Calculate surge price and dynamic pricing using the weather and surge models
        surge_inference_df = weather_information(start_coords['lat'], start_coords['lon'], api_key="bc896a839f2a3d71c59b3c3f9ae2b3b2")
        
        surge_inference_df['surge_mult'] = [(uber_original_surge + lyft_original_surge) / 2]
        surge_calculator = SurgePriceClassifier(surge_inference_df)
        surge_multiplier = surge_calculator.surge_prediction_model()


        uber_price = 20  # Dummy dynamic price for feedback
        lyft_price = 20  
        # Cab price prediction using your model, with the ride types included
        cab_price_inference_df = pd.DataFrame({
            'source_lat': [start_coords['lat']],
            'source_long': [start_coords['lon']],
            'dest_lat': [end_coords['lat']],
            'dest_long': [end_coords['lon']],
            'distance': [distance_km],
            'surge_multiplier': [surge_multiplier],
            'uber_cab_type': [uber_cab_type],
            'lyft_cab_type': [lyft_cab_type],
            'uber_price': [uber_price],  # Placeholder for Uber price (use model to calculate)
            'lyft_price': [lyft_price]   # Placeholder for Lyft price (use model to calculate)
        })
        
        # Use your existing model to calculate the final Uber and Lyft prices
        cab_price_object = CabPricePredictor(cab_price_inference_df)
        uber_predicted_price, lyft_predicted_price = cab_price_object.cab_price_prediction()

        
        print(uber_predicted_price, lyft_predicted_price)

        # Return the results
        kilometers_to_miles = 0.621371
        result = {
            'uber_price': round(uber_predicted_price, 2),
            'lyft_price': round(lyft_predicted_price, 2),
            'estimated_time': round(estimated_time, 2),
            'distance': round(distance_km * kilometers_to_miles, 2)
        }
       
        return result

    except Exception as e:
        raise ValueError(f"Error in calculating cab prices: {str(e)}")
    



def get_route_ors(start_coords, end_coords):
    try:
        # Initialize OpenRouteService client with your API key
        client = openrouteservice.Client(key='5b3ce3597851110001cf62485c31526b66624496bfcd3ef851865b89')  # Replace with your OpenRouteService API key
        
        # Coordinates for start and end (longitude, latitude)
        start_lon, start_lat = start_coords['lon'], start_coords['lat']
        end_lon, end_lat = end_coords['lon'], end_coords['lat']
        
        try:
            # Request route using OpenRouteService
            route = client.directions(
                coordinates=[(start_lon, start_lat), (end_lon, end_lat)],
                profile='driving-car',
                format='geojson'
            )

            # Validate response structure
            if not route.get('features'):
                raise ValueError("No route found. Check the coordinates or API response.")

            # Extract distance and duration
            distance = route['features'][0]['properties']['segments'][0]['distance']
            duration = route['features'][0]['properties']['segments'][0]['duration']

            return {
                'distance': distance,
                'duration': duration
            }
        except Exception as e:
            raise ValueError(f"Error fetching routes from OpenRouteService: {e}")

    except Exception as e:
        raise ValueError(f"Error fetching route from OpenRouteService: {e}")


# Helper function to get coordinates from OpenStreetMap (OSRM)
def get_coordinates_osm(location):
    place = str(location)
    try:
        encoded_location = quote_plus(place)
        api_url = f"https://nominatim.openstreetmap.org/search.php?q={encoded_location}&format=json"
        response = requests.get(api_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code != 200:
            raise ValueError(f"Error fetching coordinates: {response.status_code}")
        
        data = response.json()
        if not data:
            raise ValueError(f"No coordinates found for '{location}'")

        return {
            "lat": float(data[0]['lat']),
            "lon": float(data[0]['lon'])
        }
    except Exception as e:
        raise ValueError(f"Error fetching coordinates for '{location}': {e}")



def calculate_fuel_cost_and_consumption(car_details, distance_km, fuel_type):
    """
    Calculates the total fuel cost and estimated fuel consumption for a trip.
    
    Parameters:
        car_details (dict): A dictionary containing 'year', 'make', and 'model' of the car.
        distance_km (float): The distance of the trip in kilometers.
        fuel_type (str): The type of fuel used (e.g., 'regular', 'premium', 'diesel', etc.).
    
    Returns:
        dict: A dictionary containing estimated fuel consumption (gallons) and total fuel cost.
    """
    # Validate inputs
    if not car_details or not isinstance(car_details, dict):
        raise ValueError("Invalid car details.")
    if not distance_km or distance_km <= 0:
        raise ValueError("Distance must be a positive number.")
    if fuel_type not in ['regular', 'midgrade', 'premium', 'diesel', 'e85', 'cng', 'electric', 'lpg']:
        raise ValueError("Invalid fuel type. Choose from 'regular', 'midgrade', 'premium', 'diesel', 'e85', 'cng', 'electric', 'lpg'.")

    # Fetch the vehicle ID
    vehicle_menu_api = "https://www.fueleconomy.gov/ws/rest/vehicle/menu/options"
    

    params = {
        "year": car_details["car_year"],
        "make": car_details["car_make"],
        "model": car_details["car_model"]
    }

    response = requests.get(vehicle_menu_api, params=params, headers={"Accept": "application/json"})
    
    if response.status_code != 200:
        raise ValueError(f"Error fetching vehicle menu options: HTTP {response.status_code}")
    
    vehicle_data = response.json()
    if not vehicle_data or "menuItem" not in vehicle_data:
        raise ValueError("No vehicle options found for the provided car details.")
    
    vehicle_id = vehicle_data["menuItem"][0]["value"]

    print(f"Vehicle ID: {vehicle_id}")
    
    # Fetch average MPG for the vehicle
    vehicle_api_url = f"https://www.fueleconomy.gov/ws/rest/vehicle/{vehicle_id}"
    response = requests.get(vehicle_api_url, headers={"Accept": "application/json"})
    
    if response.status_code != 200:
        raise ValueError(f"Error fetching vehicle details: HTTP {response.status_code}")
    
    vehicle_info = response.json()
    avg_mpg = vehicle_info.get("comb08", None)  # Combined MPG (city/highway)
    
    if not avg_mpg:
        raise ValueError("Unable to retrieve average MPG for the vehicle.")

    # Fetch fuel prices
    fuel_price_api = "https://www.fueleconomy.gov/ws/rest/fuelprices"
    response = requests.get(fuel_price_api, headers={"Accept": "application/json"})
    
    if response.status_code != 200:
        raise ValueError(f"Error fetching fuel prices: HTTP {response.status_code}")
    
    fuel_prices = response.json()
    fuel_price_per_unit = fuel_prices.get(fuel_type, 3.5)  # Default to $3.50 per gallon if not available

    print("fuel price per unit",fuel_price_per_unit)
    # Convert distance from kilometers to miles
    distance_miles = distance_km * 0.621371

    avg_mpg = re.search(r'\d+(\.\d+)?', avg_mpg).group(0)  # Matches integers or decimals
    avg_mpg = float(avg_mpg)

    # Calculate gallons of fuel needed
    gallons_needed = distance_miles / avg_mpg

    # Calculate total fuel cost
    total_cost = int(gallons_needed) * float(fuel_price_per_unit)
    cost = {
        "estimated_fuel_consumption_gallons": gallons_needed,
        "total_fuel_cost": int(total_cost),
        "vehicle_id": vehicle_id,
        "average_mpg": avg_mpg,
        "fuel_price_per_unit": fuel_price_per_unit
    }
    print("cost",cost)
    return cost


# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
