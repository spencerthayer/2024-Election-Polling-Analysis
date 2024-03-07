import pandas as pd
import requests
from datetime import datetime
from io import StringIO

# Function to download and return a pandas DataFrame from a CSV URL
def download_csv_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = StringIO(response.content.decode('utf-8'))
        return pd.read_csv(data)
    else:
        raise Exception("Failed to download CSV data")

# Constants for the weighting calculations
grade_weights = {
    'A+': 1.0, 'A': 0.9, 'A-': 0.8, 'B+': 0.7, 'B': 0.6, 'B-': 0.5,
    'C+': 0.4, 'C': 0.3, 'C-': 0.2, 'D+': 0.1, 'D': 0.05, 'D-': 0.025,
}
population_weights = {
    'lv': 3, 'rv': 2, 'a': 1, 'v': 1.5, 'all': 1  # Example weights for population types
}

# Main script execution
if __name__ == "__main__":
    csv_url = 'https://projects.fivethirtyeight.com/polls/data/president_polls.csv'
    polls_df = download_csv_data(csv_url)

    # Convert 'end_date' to datetime format
    # Convert 'end_date' to datetime with the correct format
    polls_df['end_date'] = pd.to_datetime(polls_df['end_date'], format='%m/%d/%y', errors='coerce')

    # Apply the weights
    polls_df['grade_weight'] = polls_df['fte_grade'].map(grade_weights).fillna(0)
    polls_df['transparency_score'] = pd.to_numeric(polls_df['transparency_score'], errors='coerce').fillna(0)
    max_transparency_score = polls_df['transparency_score'].max()
    polls_df['transparency_weight'] = polls_df['transparency_score'] / max_transparency_score
    min_sample_size, max_sample_size = polls_df['sample_size'].min(), polls_df['sample_size'].max()
    polls_df['sample_size_weight'] = (polls_df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)
    polls_df['population_weight'] = polls_df['population'].map(lambda x: population_weights.get(x.lower(), 1))
    polls_df['combined_weight'] = polls_df['grade_weight'] * polls_df['transparency_weight'] * polls_df['sample_size_weight'] * polls_df['population_weight']

    # Function to calculate weighted polling differential
    def calculate_weighted_polling_differential(df, months_ago):
        date_threshold = pd.Timestamp.now() - pd.DateOffset(months=months_ago)
        filtered_df = df[df['end_date'] > date_threshold]
        weighted_sums = filtered_df.groupby('candidate_name').apply(lambda x: (x['pct'] * x['combined_weight']).sum())
        total_weights = filtered_df.groupby('candidate_name')['combined_weight'].sum()
        weighted_averages = weighted_sums / total_weights
        differential = weighted_averages.get('Joe Biden', 0) - weighted_averages.get('Donald Trump', 0)
        return differential

    # Calculate and print the differentials for specified periods
    for months in [12, 6, 3, 1]:
        differential = calculate_weighted_polling_differential(polls_df, months)
        print(f"Differential for the last {months} months: {differential:.2f} percentage points")
