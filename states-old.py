import requests
from bs4 import BeautifulSoup
import json
import re

# URL of the page to scrape
url = 'https://www.270towin.com/'
electoral_total = 538

# Send a GET request to the page
response = requests.get(url)

pro_values = {
    'T' : 0.9,  # Swing State
    'D1': 0.6,  # Tilts Democrat
    'D2': 0.3,  # Leans Democrat
    'D3': 0.1,  # Likely Democrat
    'D4': 0,    # Safe Democrat
    'R1': 0.6,  # Tilts Republican
    'R2': 0.3,  # Leans Republican
    'R3': 0.1,  # Likely Republican
    'R4': 0     # Safe Republican
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
                    # normalized_e_votes = (e_votes - min_votes) / (max_votes - min_votes)
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

            for state_name in sorted(processed_data.keys()):
                seat_details = processed_data[state_name]
                print(f"'{state_name}':{seat_details}\n")
else:
    print("Failed to retrieve the page")