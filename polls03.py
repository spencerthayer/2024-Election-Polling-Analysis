import pandas as pd
import numpy as np
import requests
from io import StringIO

# Constants
grade_weights = {
    'A+': 1.0, 'A': 0.9, 'A-': 0.8, 'B+': 0.7, 'B': 0.6, 'B-': 0.5,
    'C+': 0.4, 'C': 0.3, 'C-': 0.2, 'D+': 0.1, 'D': 0.05, 'D-': 0.025
}
population_weights = {
    'lv': 1.0, 'rv': 0.6666666666666666, 'a': 0.3333333333333333,
    'v': 0.5, 'all': 0.3333333333333333
}

def time_decay_weight(dates, half_life_days=30):
    reference_date = pd.Timestamp.now()
    days_old = (reference_date - dates).dt.days
    return np.exp(-np.log(2) * days_old / half_life_days)

def download_csv_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.content.decode('utf-8')))
    raise Exception("Failed to download CSV data")

def process_data(data, is_favorability=False):
    # Specify the date format directly if you know it; adjust the format as needed
    data['end_date'] = pd.to_datetime(data['end_date'], format='%Y-%m-%d', errors='coerce')
    
    if is_favorability:
        data = data.rename(columns={'politician': 'candidate_name', 'favorable': 'pct'})
        # Convert favorable percentages to proportions
        data['pct'] = data['pct'] / 100

    # Apply weightings
    data['grade_weight'] = data['fte_grade'].map(grade_weights).fillna(0) if 'fte_grade' in data.columns else 1
    data['population_weight'] = data['population'].map(lambda x: population_weights.get(x, 0.333))
    data['sample_size_weight'] = (data['sample_size'] - data['sample_size'].min()) / (data['sample_size'].max() - data['sample_size'].min()) if 'sample_size' in data.columns else 1
    data['transparency_weight'] = data['transparency_score'] / data['transparency_score'].max() if 'transparency_score' in data.columns else 1
    data['combined_weight'] = data['grade_weight'] * data['population_weight'] * data['sample_size_weight'] * data['transparency_weight']
    data['time_decay_weight'] = time_decay_weight(data['end_date'])

    return data

def calculate_and_print_favorability(df, months_ago):
    # Correct filtering and weighted calculation
    df_filtered = df[(df['end_date'] > (pd.Timestamp.now() - pd.DateOffset(months=months_ago)))].copy()
    df_filtered['adjusted_weight'] = df_filtered['combined_weight'] * df_filtered['time_decay_weight']

    for candidate in ['Joe Biden', 'Donald Trump']:
        df_cand = df_filtered[df_filtered['candidate_name'] == candidate]
        if not df_cand.empty:
            weighted_avg = np.average(df_cand['pct'], weights=df_cand['adjusted_weight'])
            print(f"{months_ago}m {candidate}: {weighted_avg:.2%}")
        else:
            print(f"{months_ago}m {candidate}: No data available")

if __name__ == "__main__":
    electoral_csv_url = 'https://projects.fivethirtyeight.com/polls/data/president_polls.csv'
    favorability_csv_url = 'https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv'
    
    electoral_polls_df = process_data(download_csv_data(electoral_csv_url))
    favorability_polls_df = process_data(download_csv_data(favorability_csv_url), is_favorability=True)
    
    print("Favorability Ratings Over Time:")
    for months in [12, 6, 3, 1]:
        print(f"\n{months} months ago:")
        calculate_and_print_favorability(favorability_polls_df, months)
import pandas as pd
import numpy as np
import requests
from io import StringIO

# Constants remain the same

def download_csv_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.content.decode('utf-8')))
    raise Exception("Failed to download CSV data")

def process_data(data, is_favorability=False):
    data['end_date'] = pd.to_datetime(data['end_date'], errors='coerce')
    
    if is_favorability:
        data = data.rename(columns={'politician': 'candidate_name', 'favorable': 'pct'})
        data['pct'] = data['pct'] / 100

    # Apply weightings; this section remains unchanged

    return data

def calculate_and_print_favorability(df, months_ago):
    df['adjusted_weight'] = df['combined_weight'] * df['time_decay_weight']
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(months=months_ago)
    
    print(f"Looking for data after: {cutoff_date}")  # Diagnostic print

    df_filtered = df[df['end_date'] > cutoff_date].copy()

    for candidate in ['Joe Biden', 'Donald Trump']:
        df_cand = df_filtered[df_filtered['candidate_name'] == candidate]
        if not df_cand.empty:
            weighted_avg = np.average(df_cand['pct'], weights=df_cand['adjusted_weight'])
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
