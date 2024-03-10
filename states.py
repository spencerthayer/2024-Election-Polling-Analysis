import requests
from bs4 import BeautifulSoup
import json
import re

def get_state_data():
    # URL of the page to scrape
    url = 'https://www.270towin.com/'
    electoral_total = 538

    # Send a GET request to the page
    response = requests.get(url)

    pro_values = {
        'T' : 0.8,
        'D1': 0.6,
        'D2': 0.4,
        'D3': 0.2,
        'D4': 0,
        'R1': 0.6,
        'R2': 0.4,
        'R3': 0.2,
        'R4': 0
    }

    # Initialize variables to hold the max and min electoral vote counts
    min_votes = float('inf')  # Initialize min_votes to infinity
    max_votes = 0  # Initialize max_votes to zero

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        script_text = None
        for script in soup.find_all("script"):
            if 'map_d3.seats' in script.text:
                script_text = script.text
                break

        if script_text:
            matches = re.search(r'map_d3.seats = (\{.*?\});', script_text, re.DOTALL)
            if matches:
                json_data = matches.group(1)
                seats_data = json.loads(json_data)
                
                # Find min and max electoral votes for normalization
                for state_fips, seats in seats_data.items():
                    for seat in seats:
                        e_votes = seat['e_votes']
                        min_votes = min(min_votes, e_votes)
                        max_votes = max(max_votes, e_votes)

                processed_data = {}
                for state_fips, seats in seats_data.items():
                    for seat in seats:
                        state_name = seat['state_name']
                        e_votes = seat['e_votes']
                        pro_status_code = seat['pro_status']
                        # Look up the integer value for pro_status
                        pro_status_value = pro_values.get(pro_status_code, None)
                        # Normalize the electoral votes
                        normalized_e_votes = e_votes / electoral_total
                        state_rank = pro_status_value + normalized_e_votes
                        seat_details = {
                            'e': e_votes,
                            'code': pro_status_code,
                            'bias': pro_status_value,
                            'electoral': normalized_e_votes,
                            'rank': state_rank
                        }
                        processed_data.setdefault(state_name,[]).append(seat_details)

                state_data = {}
                for state_name in sorted(processed_data.keys()):
                    seat_details = processed_data[state_name]
                    state_data[state_name] = seat_details[0]['rank']  # Assuming only one seat per state

                return state_data
    else:
        print("Failed to retrieve the page")
        return None