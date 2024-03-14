import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import numpy as np

# print("Likability data...")

csv_url = 'https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv'

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

def get_color_code(period_index, total_periods, skip_color):
    return start_color + (period_index * skip_color)

def calculate_and_print_favorability(df, period_value, period_type='months', period_index=0, total_periods=1):
    df['created_at'] = pd.to_datetime(
        df['created_at'], format='%m/%d/%y %H:%M', errors='coerce')
    filtered_df = df.dropna(subset=['created_at']).copy()
    if period_type == 'months':
        filtered_df = filtered_df[(filtered_df['created_at'] > (pd.Timestamp.now() - pd.DateOffset(months=period_value))) &
                                  (filtered_df['politician'].isin(['Joe Biden', 'Donald Trump']))]
    elif period_type == 'days':
        filtered_df = filtered_df[(filtered_df['created_at'] > (pd.Timestamp.now() - pd.Timedelta(days=period_value))) &
                                  (filtered_df['politician'].isin(['Joe Biden', 'Donald Trump']))]

    if not filtered_df.empty:
        filtered_df['time_decay_weight'] = time_decay_weight(
            filtered_df['created_at'], decay_rate, half_life_days)
        filtered_df['grade_weight'] = filtered_df['fte_grade'].map(grade_weights).fillna(0.0125)
        filtered_df['population'] = filtered_df['population'].str.lower()
        filtered_df['population_weight'] = filtered_df['population'].map(lambda x: population_weights.get(x, 1))

        list_weights = np.array([
            filtered_df['grade_weight'],
            filtered_df['population_weight'],
            filtered_df['time_decay_weight']
        ])
        filtered_df['combined_weight'] = np.prod(list_weights, axis=0)

        weighted_sums = filtered_df.groupby('politician')['combined_weight'].apply(lambda x: (x * filtered_df.loc[x.index, 'favorable']/100).sum())
        total_weights = filtered_df.groupby('politician')['combined_weight'].sum()
        weighted_averages = weighted_sums / total_weights

        biden_average = weighted_averages.get('Joe Biden', 0)
        trump_average = weighted_averages.get('Donald Trump', 0)
        differential = (biden_average) - (trump_average)

        favored_candidate = "Biden" if differential > 0 else "Trump"

        combined_period = f"{period_value}{period_type[0]}"

        color_code = get_color_code(period_index, total_periods, skip_color)

        print(f"\033[38;5;{color_code}m{combined_period:<4} B:{abs(biden_average):5.2%} T:{abs(trump_average):5.2%} {differential:+5.2%} {favored_candidate}\033[0m")

    else:
        print(f"{period_value}{period_type[0]}: No data available for the specified period")

if __name__ == "__main__":
    favorability_df = download_csv_data(csv_url)
    periods = [
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
        calculate_and_print_favorability(favorability_df, period_value, period_type, index, total_periods)