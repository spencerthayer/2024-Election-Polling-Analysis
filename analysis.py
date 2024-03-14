import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import numpy as np
from typing import Dict, List
from states import get_state_data

# Download the Data
polling_url = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
favorability_url = "https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv"

#Data Parsing
candidate_names = ['Joe Biden', 'Donald Trump']
favorability_weight = 0.05
"""
When heavy_weight is set to True, the weights are multiplied together using np.prod(), giving more importance to the combined effect of all weights.

When heavy_weight is set to False, the weights are averaged by taking the sum of weights divided by the number of weights.
"""
heavy_weight = True

# Coloring
start_color = 164
skip_color = 3

# Define the time decay weighting
decay_rate = 2
half_life_days = 28

# Constants for the weighting calculations
grade_weights = {
    'A+': 1.0, 'A': 0.9, 'A-': 0.8, 'A/B': 0.75, 'B+': 0.7,
    'B': 0.6, 'B-': 0.5, 'B/C': 0.45, 'C+': 0.4, 'C': 0.3,
    'C-': 0.2, 'C/D': 0.15, 'D+': 0.1, 'D': 0.05, 'D-': 0.025
}
partisan_weight = {True: 0.1, False: 1}
population_weights = {
    'lv': 1.0, 'rv': 0.6666666666666666, 'v': 0.5,
    'a': 0.3333333333333333, 'all': 0.3333333333333333
}

def download_csv_data(url: str) -> pd.DataFrame:
    """
    Download CSV data from the specified URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = StringIO(response.content.decode('utf-8'))
        return pd.read_csv(csv_data)
    except (requests.RequestException, pd.errors.EmptyDataError, ValueError) as e:
        print(f"Error downloading data from {url}: {e}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data by converting date columns, handling missing values, and filtering irrelevant data.
    """
    # print("Original DataFrame shape:", df.shape)
    
    df['created_at'] = pd.to_datetime(df['created_at'], format='%m/%d/%y %H:%M', errors='coerce')
    df = df.dropna(subset=['created_at'])
    # print("DataFrame shape after handling missing values:", df.shape)
    
    # Calculate transparency_weight and sample_size_weight
    df['transparency_score'] = pd.to_numeric(df['transparency_score'], errors='coerce').fillna(0)
    max_transparency_score = df['transparency_score'].max()
    df['transparency_weight'] = df['transparency_score'] / max_transparency_score
    min_sample_size, max_sample_size = df['sample_size'].min(), df['sample_size'].max()
    df['sample_size_weight'] = (df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)
    # print("DataFrame shape after calculating weights:", df.shape)
    
    # Fetch state data and apply to calculate state_rank
    state_data = get_state_data()
    df['state_rank'] = df['state'].apply(lambda x: state_data.get(x, 1))
    # print("DataFrame shape after calculating state rank:", df.shape)
    
    # Ensure 'fte_grade' is processed to calculate 'grade_weight'
    df['grade_weight'] = df['fte_grade'].map(grade_weights).fillna(0.0125)
    # print("DataFrame shape after calculating grade weight:", df.shape)
    
    return df

def apply_time_decay_weight(df: pd.DataFrame, decay_rate: float, half_life_days: int) -> pd.DataFrame:
    """
    Apply time decay weighting to the data based on the specified decay rate and half-life.
    """
    reference_date = pd.Timestamp.now()
    days_old = (reference_date - df['created_at']).dt.days
    df['time_decay_weight'] = np.exp(-np.log(decay_rate) * days_old / half_life_days)
    return df

def calculate_polling_metrics(df: pd.DataFrame, candidate_names: List[str]) -> Dict[str, float]:
    """
    Calculate polling metrics for the specified candidate names.
    Ensure percentages are handled correctly.
    """
    df = df.copy()
    # Ensure 'pct' column values are between 0 and 100; adjust if they're between 0 and 1
    df['pct'] = df['pct'].apply(lambda x: x if x > 1 else x * 100)
    
    # Check if 'grade_weight' needs to be calculated
    if 'grade_weight' not in df.columns:
        df['grade_weight'] = df['fte_grade'].map(grade_weights).fillna(0.0125)
    
    # Calculate 'transparency_weight' using the filtered DataFrame
    df['transparency_score'] = pd.to_numeric(df['transparency_score'], errors='coerce').fillna(0)
    max_transparency_score = df['transparency_score'].max()
    df['transparency_weight'] = df['transparency_score'] / max_transparency_score
    
    # Calculate 'sample_size_weight' using the filtered DataFrame
    min_sample_size, max_sample_size = df['sample_size'].min(), df['sample_size'].max()
    df['sample_size_weight'] = (df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)
    
    df.loc[:, 'is_partisan'] = df['partisan'].notna() & df['partisan'].ne('')
    df.loc[:, 'partisan_weight'] = df['is_partisan'].map(partisan_weight)
    df.loc[:, 'population'] = df['population'].str.lower()
    df.loc[:, 'population_weight'] = df['population'].map(lambda x: population_weights.get(x, 1))
    
    # Fetch state data and apply to calculate state_rank
    state_data = get_state_data()
    df['state_rank'] = df['state'].apply(lambda x: state_data.get(x, 1))

    list_weights = np.array([
        df['time_decay_weight'],
        df['sample_size_weight'],
        df['grade_weight'],
        df['transparency_weight'],
        df['population_weight'],
        df['partisan_weight'],
        df['state_rank'],
    ])
    if heavy_weight == True:
        df['combined_weight'] = np.prod(list_weights, axis=0)
    elif heavy_weight == False:
        df['combined_weight'] = sum(list_weights) / len(list_weights)
    
    weighted_sums = df.groupby('candidate_name')['combined_weight'].apply(lambda x: (x * df.loc[x.index, 'pct']).sum())
    total_weights = df.groupby('candidate_name')['combined_weight'].sum()
    weighted_averages = (weighted_sums / total_weights)
    return {candidate: weighted_averages.get(candidate, 0) for candidate in candidate_names}

def calculate_favorability_differential(df: pd.DataFrame, candidate_names: List[str]) -> Dict[str, float]:
    """
    Calculate favorability differentials for the specified candidate names.
    Ensure percentages are handled correctly.
    """
    df = df.copy()
    # Assume 'favorable' column values are between 0 and 100; adjust if they're between 0 and 1
    df['favorable'] = df['favorable'].apply(lambda x: x if x > 1 else x * 100)
    
    # Check if 'grade_weight' needs to be calculated
    if 'grade_weight' not in df.columns:
        df['grade_weight'] = df['fte_grade'].map(grade_weights).fillna(0.0125)
    
    df.loc[:, 'population'] = df['population'].str.lower()
    df.loc[:, 'population_weight'] = df['population'].map(lambda x: population_weights.get(x, 1))

    list_weights = np.array([
        df['grade_weight'],
        df['population_weight'],
        df['time_decay_weight']
    ])
    df['combined_weight'] = np.prod(list_weights, axis=0)
    
    weighted_sums = df.groupby('politician')['combined_weight'].apply(lambda x: (x * df.loc[x.index, 'favorable']).sum())
    total_weights = df.groupby('politician')['combined_weight'].sum()
    weighted_averages = (weighted_sums / total_weights)
    return {candidate: weighted_averages.get(candidate, 0) for candidate in candidate_names}

def combine_analysis(polling_metrics: Dict[str, float], favorability_differential: Dict[str, float], favorability_weight: float) -> Dict[str, float]:
    """
    Combine polling metrics and favorability differentials into a unified analysis.
    """
    combined_metrics = {}
    for candidate in polling_metrics.keys():
        combined_metrics[candidate] = (
            polling_metrics[candidate] * (1 - favorability_weight) +
            favorability_differential[candidate] * favorability_weight
        )
    return combined_metrics

def print_with_color(text: str, color_code: int):
    """
    Print text with the specified color code using ANSI escape sequences.
    """
    print(f"\033[38;5;{color_code}m{text}\033[0m")

def output_results(combined_results: Dict[str, float], color_index: int, period_value: int, period_type: str):
    """
    Corrected output formatting to display percentages properly.
    """
    biden_score = combined_results['Joe Biden']
    trump_score = combined_results['Donald Trump']
    differential = trump_score - biden_score
    favored_candidate = "Biden" if differential < 0 else "Trump"
    color_code = start_color + (color_index * skip_color)
    # Ensure percentages are displayed correctly
    print(f"\033[38;5;{color_code}m{period_value:2d}{period_type[0]:<4} B:{biden_score:5.2f}% T:{trump_score:5.2f}% {abs(differential):+5.2f} {favored_candidate}\033[0m")

def main():
    
    polling_df = download_csv_data(polling_url)
    favorability_df = download_csv_data(favorability_url)
    
    polling_df = preprocess_data(polling_df)
    favorability_df = preprocess_data(favorability_df)
    
    polling_df = apply_time_decay_weight(polling_df, decay_rate, half_life_days)
    favorability_df = apply_time_decay_weight(favorability_df, decay_rate, half_life_days)
    
    color_index = 0
    for period in [(12, 'months'), (6, 'months'), (3, 'months'), (1, 'months'), (21, 'days'), (14, 'days'), (7, 'days'), (3, 'days'), (1, 'days')]:
        period_value, period_type = period
        filtered_polling_df = polling_df[(polling_df['created_at'] > (pd.Timestamp.now() - pd.DateOffset(**{period_type: period_value}))) &
                                         (polling_df['candidate_name'].isin(candidate_names))].copy()  # Create a copy of the filtered DataFrame
        filtered_favorability_df = favorability_df[(favorability_df['created_at'] > (pd.Timestamp.now() - pd.DateOffset(**{period_type: period_value}))) &
                                                   (favorability_df['politician'].isin(candidate_names))].copy()  # Create a copy of the filtered DataFrame
        
        polling_metrics = calculate_polling_metrics(filtered_polling_df, candidate_names)
        favorability_differential = calculate_favorability_differential(filtered_favorability_df, candidate_names)
        
        combined_results = combine_analysis(polling_metrics, favorability_differential, favorability_weight)
        
        output_results(combined_results, color_index, period_value, period_type)
        color_index += 1

if __name__ == "__main__":
    main()