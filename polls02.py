import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import numpy as np

csv_url = 'https://projects.fivethirtyeight.com/polls/data/president_polls.csv'

# Define the time decay weighting
half_life_days = 30
decay_rate = 0.75

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
partisan_weight = {True: 0.25, False: 1}

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
def time_decay_weight(dates):
    reference_date = pd.Timestamp.now()
    days_old = (reference_date - dates).dt.days
    days_old = np.where(days_old < 0, 0, days_old)
    # return np.exp(-np.log(2) * days_old / (half_life_days))
    return np.exp(-np.log(1 / decay_rate) * days_old / half_life_days)

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

def calculate_and_print_differential(df, period_value, period_type='months'):
    df['created_at'] = pd.to_datetime(polls_df['created_at'], format='%m/%d/%y %H:%M', errors='coerce')
    filtered_df = df.dropna(subset=['created_at']).copy()
    if period_type == 'months':
        filtered_df = filtered_df[(filtered_df['created_at'] > (pd.Timestamp.now() - pd.DateOffset(months=period_value))) & 
                                  (filtered_df['candidate_name'].isin(['Joe Biden', 'Donald Trump']))]
    elif period_type == 'days':
        filtered_df = filtered_df[(filtered_df['created_at'] > (pd.Timestamp.now() - pd.Timedelta(days=period_value))) & 
                                  (filtered_df['candidate_name'].isin(['Joe Biden', 'Donald Trump']))]
    
    if not filtered_df.empty:
        filtered_df['time_decay_weight'] = time_decay_weight(filtered_df['created_at'])
        filtered_df['adjusted_combined_weight'] = filtered_df['combined_weight'] * filtered_df['time_decay_weight']
        
        weighted_sums = filtered_df.groupby('candidate_name')['pct'].apply(
            lambda x: (x * filtered_df.loc[x.index, 'adjusted_combined_weight']).sum())
        total_weights = filtered_df.groupby('candidate_name')['adjusted_combined_weight'].sum()
        weighted_averages = weighted_sums / total_weights
        
        biden_average = weighted_averages.get('Joe Biden', 0)
        trump_average = weighted_averages.get('Donald Trump', 0)
        differential = biden_average - trump_average
        
        favored_candidate = "Biden" if differential > 0 else "Trump"
        
        # Print statement using the simplified differential formatting
        print(f"{period_value}{period_type[0]} B {format_percentage(biden_average)} | T {format_percentage(trump_average)} {format_differential(abs(differential))} {favored_candidate}")
    else:
        print(f"{period_value}{period_type[0]}: No data available for the specified period")

if __name__ == "__main__":
    polls_df = download_csv_data(csv_url)
    polls_df['is_partisan'] = polls_df['partisan'].notna() & polls_df['partisan'].ne('')
    polls_df['partisan_weight'] = polls_df['is_partisan'].map(partisan_weight)
    polls_df['grade_weight'] = polls_df['fte_grade'].map(grade_weights).fillna(0.0125)
    polls_df['transparency_score'] = pd.to_numeric(polls_df['transparency_score'], errors='coerce').fillna(0)
    max_transparency_score = polls_df['transparency_score'].max()
    polls_df['transparency_weight'] = polls_df['transparency_score'] / max_transparency_score

    min_sample_size, max_sample_size = polls_df['sample_size'].min(), polls_df['sample_size'].max()
    polls_df['sample_size_weight'] = (polls_df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)
    polls_df['population'] = polls_df['population'].str.lower()  
    polls_df['population_weight'] = polls_df['population'].map(lambda x: population_weights.get(x, 1))

    # Combine the weights and include time decay
    polls_df['combined_weight'] = polls_df['grade_weight'] * polls_df['transparency_weight'] * polls_df['sample_size_weight'] * polls_df['population_weight'] * polls_df['partisan_weight']

    print("Polling Over Time:")
    # Calculate and print the differentials for specified periods
    periods = [
        (12, 'months'),
        (6, 'months'),
        (3, 'months'),
        (1, 'months'),
        (14, 'days'),
        (7, 'days'),
        (3, 'days'),
        (1, 'days')
        ]
    for period_value, period_type in periods:
        calculate_and_print_differential(polls_df, period_value, period_type)