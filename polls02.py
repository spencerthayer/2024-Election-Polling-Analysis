import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import numpy as np

# Constants for the weighting calculations
grade_weights = {
    'A+': 1.0, 'A': 0.9, 'A-': 0.8, 'B+': 0.7, 'B': 0.6, 'B-': 0.5,
    'C+': 0.4, 'C': 0.3, 'C-': 0.2, 'D+': 0.1, 'D': 0.05, 'D-': 0.025,
}

# Normalized population weights
population_weights = {
    'lv': 1.0, 'rv': 0.6666666666666666, 'a': 0.3333333333333333, 
    'v': 0.5, 'all': 0.3333333333333333
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
def time_decay_weight(dates, half_life_days=30):
    reference_date = pd.Timestamp.now()
    days_old = (reference_date - dates).dt.days
    decay_weights = np.exp(-np.log(2) * days_old / half_life_days)
    return decay_weights

def calculate_and_print_differential(df, months_ago, half_life_days=30):
    df['end_date'] = pd.to_datetime(df['end_date'], format='%m/%d/%y', errors='coerce')
    filtered_df = df.dropna(subset=['end_date']).copy()
    filtered_df = filtered_df[(filtered_df['end_date'] > (pd.Timestamp.now() - pd.DateOffset(months=months_ago))) & 
                              (filtered_df['candidate_name'].isin(['Joe Biden', 'Donald Trump']))]
    
    if not filtered_df.empty:
        filtered_df['time_decay_weight'] = time_decay_weight(filtered_df['end_date'], half_life_days)
        filtered_df['adjusted_combined_weight'] = filtered_df['combined_weight'] * filtered_df['time_decay_weight']
        
        weighted_sums = filtered_df.groupby('candidate_name')['pct'].apply(
            lambda x: (x * filtered_df.loc[x.index, 'adjusted_combined_weight']).sum())
        total_weights = filtered_df.groupby('candidate_name')['adjusted_combined_weight'].sum()
        weighted_averages = weighted_sums / total_weights
        
        biden_average = weighted_averages.get('Joe Biden', 0)
        trump_average = weighted_averages.get('Donald Trump', 0)
        differential = biden_average - trump_average
        
        favored_candidate = "Biden" if differential > 0 else "Trump"
        
        print(f"{months_ago}m {abs(differential):.2f}% {favored_candidate}")
    else:
        print(f"{months_ago}m: No data available for the specified period")

if __name__ == "__main__":
    csv_url = 'https://projects.fivethirtyeight.com/polls/data/president_polls.csv'
    polls_df = download_csv_data(csv_url)

    polls_df['grade_weight'] = polls_df['fte_grade'].map(grade_weights).fillna(0)
    polls_df['transparency_score'] = pd.to_numeric(polls_df['transparency_score'], errors='coerce').fillna(0)
    max_transparency_score = polls_df['transparency_score'].max()
    polls_df['transparency_weight'] = polls_df['transparency_score'] / max_transparency_score

    min_sample_size, max_sample_size = polls_df['sample_size'].min(), polls_df['sample_size'].max()
    polls_df['sample_size_weight'] = (polls_df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)
    polls_df['population'] = polls_df['population'].str.lower()  
    polls_df['population_weight'] = polls_df['population'].map(lambda x: population_weights.get(x, 1))

    # Combine the weights and include time decay
    polls_df['combined_weight'] = polls_df['grade_weight'] * polls_df['transparency_weight'] * polls_df['sample_size_weight'] * polls_df['population_weight']

    for months in [12, 6, 3, 1]:
        calculate_and_print_differential(polls_df, months)