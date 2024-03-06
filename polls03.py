import pandas as pd
import numpy as np
import requests
from io import StringIO

# Constants for weighting calculations
grade_weights = {
    'A+': 1.0, 'A': 0.9, 'A-': 0.8, 'B+': 0.7, 'B': 0.6, 'B-': 0.5,
    'C+': 0.4, 'C': 0.3, 'C-': 0.2, 'D+': 0.1, 'D': 0.05, 'D-': 0.025,
}

population_weights = {
    'lv': 1.0, 'rv': 0.6666666666666666, 'a': 0.3333333333333333,
    'v': 0.5, 'all': 0.3333333333333333,
}

def time_decay_weight(dates, half_life_days=30):
    """Calculate time decay weights based on the half-life."""
    reference_date = pd.Timestamp.now()
    days_old = (reference_date - dates).dt.days
    return np.exp(-np.log(2) * days_old / half_life_days)

def download_csv_data(url):
    """Download CSV data from a specified URL."""
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.content.decode('utf-8')))
    raise Exception("Failed to download CSV data")

def process_data(data, is_favorability=False):
    """Process electoral or favorability polling data."""
    data['end_date'] = pd.to_datetime(data['end_date'], format='%m/%d/%y', errors='coerce')
    if is_favorability:
        data = data.rename(columns={'politician': 'candidate_name', 'favorable': 'pct'})
        data['pct'] = data['pct'] / 100  # Convert percentage to a proportion
    data['grade_weight'] = data['fte_grade'].map(grade_weights).fillna(0)
    data['population_weight'] = data['population'].map(lambda x: population_weights.get(x, 0.333))
    data['sample_size_weight'] = (data['sample_size'] - data['sample_size'].min()) / (data['sample_size'].max() - data['sample_size'].min()) if 'sample_size' in data.columns else 1
    data['transparency_weight'] = data['transparency_score'] / data['transparency_score'].max() if 'transparency_score' in data.columns else 1
    data['combined_weight'] = data['grade_weight'] * data['population_weight'] * data['sample_size_weight'] * data['transparency_weight']
    data['time_decay_weight'] = time_decay_weight(data['end_date'])
    return data

def calculate_and_print_favorability(df, months_ago):
    """Calculate and print favorability ratings for specified months ago."""
    df = df.copy()  # Work on a copy to prevent SettingWithCopyWarning
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_ago)
    df_filtered = df[df['end_date'] > cutoff_date]
    
    for candidate in ['Joe Biden', 'Donald Trump']:
        df_cand = df_filtered[df_filtered['candidate_name'] == candidate]
        if not df_cand.empty:
            weighted_avg = np.average(df_cand['pct'], weights=df_cand['combined_weight'] * df_cand['time_decay_weight'])
            print(f"{months_ago}m {candidate}: {weighted_avg:.2%}")
        else:
            print(f"{months_ago}m {candidate}: No data available")

if __name__ == "__main__":
    favorability_csv_url = 'https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv'
    favorability_polls_df = process_data(download_csv_data(favorability_csv_url), is_favorability=True)
    
    print("Favorability Ratings Over Time:")
    for months in [12, 6, 3, 1]:
        print(f"\n{months} months ago:")
        calculate_and_print_favorability(favorability_polls_df, months)