import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import numpy as np
from states import get_state_data

# print("Polling data...")

csv_url = 'https://projects.fivethirtyeight.com/polls/data/president_polls.csv'

# Coloring
start_color = 164
skip_color = 3

# Define the time decay weighting
decay_rate = 2
half_life_days = 28

# Constants for the weighting calculations
grade_weights = {
    'A+': 1.0,
    'A': 0.9,
    'A-': 0.8,
    'A/B': 0.75,
    'B+': 0.7,
    'B': 0.6,
    'B-': 0.5,
    'B/C': 0.45,
    'C+': 0.4,
    'C': 0.3,
    'C-': 0.2,
    'C/D': 0.15,
    'D+': 0.1,
    'D': 0.05,
    'D-': 0.025
    }

# Define partisan weights
partisan_weight = {
    True:0.1,
    False:1
    }

# Normalized population weights
population_weights = {
    'lv': 1.0,
    'rv': 0.6666666666666666,
    'v': 0.5,
    'a': 0.3333333333333333,
    'all': 0.3333333333333333
    }

# Function to download and return a pandas DataFrame from a CSV URL
def download_csv_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        csv_data = StringIO(response.content.decode('utf-8'))
        return pd.read_csv(csv_data)
    else:
        raise Exception("Failed to download CSV data")

# Define a function to calculate time decay weight
def time_decay_weight(dates, decay_rate, half_life_days):
    reference_date = pd.Timestamp.now()
    days_old = (reference_date - dates).dt.days
    return np.exp(-np.log(decay_rate) * days_old / half_life_days)

def format_percentage(value):
    # Remove unnecessary trailing zeros and avoid leading zero for numbers between -1 and 1
    formatted_str = f"{value:.2f}".rstrip('0').rstrip('.')
    if formatted_str.startswith('0.'):
        return formatted_str[1:] + "%"
    elif formatted_str.startswith('-0.'):
        return '-' + formatted_str[2:] + "%"
    else:
        return formatted_str + "%"

def format_differential(value):
    # Always format the differential with a "+" sign
    formatted_str = f"{value:.2f}".rstrip('0').rstrip('.')
    if formatted_str == "0":  # In case the differential rounds to zero
        return "+0"
    else:
        return f"+{formatted_str}"

def get_color_code(period_index, total_periods, skip_color):
    return start_color + (period_index * skip_color)

def calculate_and_print_differential(df, period_value, period_type='months', period_index=0, total_periods=1):
    df['created_at'] = pd.to_datetime(
        polls_df['created_at'], format='%m/%d/%y %H:%M', errors='coerce')
    filtered_df = df.dropna(subset=['created_at']).copy()
    if period_type == 'months':
        filtered_df = filtered_df[(filtered_df['created_at'] > (pd.Timestamp.now() - pd.DateOffset(months=period_value))) &
                                  (filtered_df['candidate_name'].isin(['Joe Biden', 'Donald Trump']))]
    elif period_type == 'days':
        filtered_df = filtered_df[(filtered_df['created_at'] > (pd.Timestamp.now() - pd.Timedelta(days=period_value))) &
                                  (filtered_df['candidate_name'].isin(['Joe Biden', 'Donald Trump']))]

    if not filtered_df.empty:
        filtered_df['time_decay_weight'] = time_decay_weight(filtered_df['created_at'], decay_rate, half_life_days)
        filtered_df['grade_weight'] = filtered_df['fte_grade'].map(grade_weights).fillna(0.0125)
        filtered_df['transparency_score'] = pd.to_numeric(filtered_df['transparency_score'], errors='coerce').fillna(0)
        max_transparency_score = filtered_df['transparency_score'].max()
        filtered_df['transparency_weight'] = filtered_df['transparency_score'] / max_transparency_score
        min_sample_size, max_sample_size = filtered_df['sample_size'].min(), filtered_df['sample_size'].max()
        filtered_df['sample_size_weight'] = (filtered_df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)
        filtered_df['population'] = filtered_df['population'].str.lower()
        filtered_df['population_weight'] = filtered_df['population'].map(lambda x: population_weights.get(x, 1))
        filtered_df['is_partisan'] = filtered_df['partisan'].notna() & filtered_df['partisan'].ne('')
        filtered_df['partisan_weight'] = filtered_df['is_partisan'].map(partisan_weight)
        
        # Fetch the latest state data
        state_data = get_state_data()
        
        # Match the state field with the state names in the state data dictionary
        def get_state_rank(state):
            return state_data.get(state, 1)  # Default to 1 if state is not found
        
        filtered_df['state_rank'] = filtered_df['state'].apply(get_state_rank)
        
        list_weights = np.array([
            filtered_df['time_decay_weight'],
            filtered_df['sample_size_weight'],
            filtered_df['grade_weight'],
            filtered_df['transparency_weight'],
            filtered_df['population_weight'],
            filtered_df['partisan_weight'],
            filtered_df['state_rank'],
        ])
        # Calculate the final combined weight
        # filtered_df['combined_weight'] = (sum(list_weights) / len(list_weights))
        filtered_df['combined_weight'] = np.prod(list_weights, axis=0)
        # print ( list_weights )
        # print ( filtered_df['time_decay_weight'] )
        # print ( filtered_df['combined_weight'] )
        
        weighted_sums = filtered_df.groupby('candidate_name')['combined_weight'].apply(lambda x: (x * filtered_df.loc[x.index, 'pct']).sum())
        total_weights = filtered_df.groupby('candidate_name')['combined_weight'].sum()
        weighted_averages = weighted_sums / total_weights

        biden_average = weighted_averages.get('Joe Biden', 0)
        trump_average = weighted_averages.get('Donald Trump', 0)
        differential = (biden_average) - (trump_average)

        favored_candidate = "Biden" if differential > 0 else "Trump"

        combined_period = f"{period_value}{period_type[0]}"

        color_code = get_color_code(period_index, total_periods, skip_color)
        
        # Modify your print statement to include the ANSI escape code for color
        print(f"\033[38;5;{color_code}m{combined_period:<4} B:{abs(biden_average):5.2f}% T:{abs(trump_average):5.2f}% {abs(differential):+5.2f} {favored_candidate}\033[0m")

    else:
        print(
            f"{period_value}{period_type[0]}: No data available for the specified period")

if __name__ == "__main__":
    polls_df = download_csv_data(csv_url)
    # print("Decay: " + str(decay_rate) + " over " + str(half_life_days) + " days.")
    periods = [
        # (24, 'months'),
        # (18, 'months'),
        (12, 'months'),
        (6, 'months'),
        (3, 'months'),
        (1, 'months'),
        (21, 'days'),
        (14, 'days'),
        (7, 'days'),
        (3, 'days'),
        (1, 'days')
    ]
    total_periods = len(periods)
    for index, (period_value, period_type) in enumerate(periods):
        calculate_and_print_differential(polls_df, period_value, period_type, index, total_periods)