# states.py

import requests
from bs4 import BeautifulSoup
import json
import re
import logging
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_state_data() -> Dict[str, float]:
    """
    Fetches and processes state data from multiple sources to calculate state rankings.
    Returns data in a format compatible with the existing analysis.py file.

    Returns:
        Dict[str, float]: A dictionary mapping state abbreviations to their calculated ranks.
    """
    mapurl = 'https://www.270towin.com/'
    forecast_url = 'https://projects.fivethirtyeight.com/2024-election-forecast/priors.json'
    electoral_total = 538

    pro_values = {
        'T': 0.8,  # Swing State
        'D1': 0.6, 'D2': 0.4, 'D3': 0.2, 'D4': 0.1,  # Democratic leaning
        'R1': 0.6, 'R2': 0.4, 'R3': 0.2, 'R4': 0.1   # Republican leaning
    }

    try:
        state_forecasts = fetch_forecast_data(forecast_url)
        seats_data = fetch_270towin_data(mapurl)
        
        if not state_forecasts or not seats_data:
            logging.error("Failed to fetch required data")
            return {}  # Return empty dict instead of None to maintain compatibility

        processed_data = process_state_data(seats_data, state_forecasts, pro_values, electoral_total)
        
        # Convert full state names to abbreviations and create the final dictionary
        state_data = {
            get_state_abbreviation(state_name): details[0]['rank']
            for state_name, details in processed_data.items()
        }

        logging.info(f"Successfully processed data for {len(state_data)} states")
        return state_data

    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        return {}  # Return empty dict in case of error

def fetch_forecast_data(url: str) -> Optional[Dict[str, float]]:
    """Fetches the FiveThirtyEight forecast data from the provided URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        forecast_data = response.json()
        
        state_forecasts = {
            state['state']: next((abs(metric['median']) for metric in state['metrics'] if metric['metric'] == 'Full forecast'), 50)
            for state in forecast_data
        }
        logging.info(f"Successfully fetched forecast data for {len(state_forecasts)} states")
        return state_forecasts
    except requests.RequestException as e:
        logging.error(f"Error fetching forecast data: {str(e)}")
        return None

def fetch_270towin_data(url: str) -> Optional[Dict[str, Any]]:
    """Fetches the 270towin electoral data from the provided URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        script_text = next((script.text for script in soup.find_all("script") if 'map_d3.seats' in script.text), None)
        
        if not script_text:
            logging.error("Failed to find the required script in the page")
            return None
        
        matches = re.search(r'map_d3.seats = (\{.*?\});', script_text, re.DOTALL)
        if not matches:
            logging.error("Failed to extract JSON data from script")
            return None
        
        seats_data = json.loads(matches.group(1))
        logging.info(f"Successfully fetched 270towin data for {len(seats_data)} states")
        return seats_data
    except requests.RequestException as e:
        logging.error(f"Error fetching 270towin data: {str(e)}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON data: {str(e)}")
        return None

def process_state_data(seats_data: Dict[str, Any], state_forecasts: Dict[str, float], 
                       pro_values: Dict[str, float], electoral_total: int) -> Dict[str, list]:
    """Processes the state data and calculates rankings."""
    processed_data = {}
    for state_fips, seats in seats_data.items():
        for seat in seats:
            state_name = seat['state_name']
            e_votes = seat['e_votes']
            pro_status_code = seat['pro_status']
            pro_status_value = pro_values.get(pro_status_code, 0.1)
            normalized_e_votes = e_votes / electoral_total
            
            forecast_value = state_forecasts.get(state_name, 50)
            forecast_weight = 1 - (forecast_value / 100)
            
            state_rank = (pro_status_value * 0.4) + (normalized_e_votes * 0.3) + (forecast_weight * 0.3)
            
            seat_details = {
                'e': e_votes,
                'code': pro_status_code,
                'bias': pro_status_value,
                'electoral': normalized_e_votes,
                'forecast': forecast_value,
                'rank': state_rank
            }
            processed_data.setdefault(state_name, []).append(seat_details)
    
    logging.info(f"Processed data for {len(processed_data)} states")
    return processed_data

def get_state_abbreviation(state_name: str) -> str:
    """Converts full state name to its abbreviation."""
    # This is a simple mapping. You might want to expand this for all states.
    state_abbr = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
        "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
        "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
        "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
        "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO",
        "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
        "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH",
        "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
        "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
        "District of Columbia": "DC"
    }
    return state_abbr.get(state_name, state_name)  # Return the original name if not found

if __name__ == "__main__":
    state_data = get_state_data()
    for state, rank in state_data.items():
        print(f"{state}: {rank:.4f}")