import pandas as pd
import requests
from datetime import datetime
from io import StringIO
import numpy as np
from typing import Dict, List
from states import get_state_data
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Download the Data
polling_url = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
favorability_url = "https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv"

# Data Parsing
candidate_names = ['Kamala Harris', 'Donald Trump']
favorability_weight = 0.1  # Global value, but will be overridden when needed
heavy_weight = True

# Coloring
start_color = 164
skip_color = 3

# Define the time decay weighting
decay_rate = 2
half_life_days = 28

# Constants for the weighting calculations
partisan_weight = {True: 0.1, False: 1}
population_weights = {
    'lv': 1.0, 'rv': 0.6666666666666666, 'v': 0.5,
    'a': 0.3333333333333333, 'all': 0.3333333333333333
}

def margin_of_error(n, p=0.5, confidence_level=0.95):
    z = norm.ppf((1 + confidence_level) / 2)
    moe = z * np.sqrt((p * (1 - p)) / n)
    return moe * 100  # Convert to percentage

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

def preprocess_data(df: pd.DataFrame, start_period: pd.Timestamp = None) -> pd.DataFrame:
    """
    Preprocess the data by converting date columns, handling missing values, filtering irrelevant data,
    and normalizing numeric_grade, pollscore, and transparency_score.
    """
    df['created_at'] = pd.to_datetime(df['created_at'], format='%m/%d/%y %H:%M', errors='coerce')
    df = df.dropna(subset=['created_at'])
    if start_period is not None:
        df = df[df['created_at'] >= start_period]

    # Normalizing numeric_grade
    df['numeric_grade'] = pd.to_numeric(df['numeric_grade'], errors='coerce').fillna(0)
    max_numeric_grade = df['numeric_grade'].max()
    df['normalized_numeric_grade'] = df['numeric_grade'] / max_numeric_grade

    # Inverting and normalizing pollscore
    df['pollscore'] = pd.to_numeric(df['pollscore'], errors='coerce')  # Ensure pollscore is float
    min_pollscore = df['pollscore'].min()
    max_pollscore = df['pollscore'].max()
    df['normalized_pollscore'] = 1 - (df['pollscore'] - min_pollscore) / (max_pollscore - min_pollscore)

    # Normalize transparency_score
    df['transparency_score'] = pd.to_numeric(df['transparency_score'], errors='coerce').fillna(0)
    max_transparency_score = df['transparency_score'].max()
    df['normalized_transparency_score'] = df['transparency_score'] / max_transparency_score

    # Clip the normalized values to ensure they are within [0, 1] range
    df['normalized_numeric_grade'] = df['normalized_numeric_grade'].clip(0, 1)
    df['normalized_pollscore'] = df['normalized_pollscore'].clip(0, 1)
    df['normalized_transparency_score'] = df['normalized_transparency_score'].clip(0, 1)

    # Combining weights with the new scores
    df['combined_weight'] = df['normalized_numeric_grade'] * df['normalized_pollscore'] * df['normalized_transparency_score']

    min_sample_size, max_sample_size = df['sample_size'].min(), df['sample_size'].max()
    df['sample_size_weight'] = (df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)

    state_data = get_state_data()
    df['state_rank'] = df['state'].apply(lambda x: state_data.get(x, 1))

    if 'population_weight' not in df.columns:
        if 'population' in df.columns:
            df.loc[:, 'population'] = df['population'].str.lower()
            df.loc[:, 'population_weight'] = df['population'].map(lambda x: population_weights.get(x, 1))
        else:
            print("Warning: 'population' column is missing. Setting 'population_weight' to 1 for all rows.")
            df.loc[:, 'population_weight'] = 1

    return df

def apply_time_decay_weight(df: pd.DataFrame, decay_rate: float, half_life_days: int) -> pd.DataFrame:
    """
    Apply time decay weighting to the data based on the specified decay rate and half-life.
    """
    reference_date = pd.Timestamp.now()
    days_old = (reference_date - df['created_at']).dt.days
    df['time_decay_weight'] = np.exp(-np.log(decay_rate) * days_old / half_life_days)
    return df

def calculate_timeframe_specific_moe(df, candidate_names):
    moes = []
    for candidate in candidate_names:
        candidate_df = df[df['candidate_name'] == candidate]
        if candidate_df.empty:
            continue
        for _, poll in candidate_df.iterrows():
            if poll['sample_size'] > 0 and 0 <= poll['pct'] <= 100:
                moe = margin_of_error(n=poll['sample_size'], p=poll['pct']/100)
                moes.append(moe)
    return np.mean(moes) if moes else np.nan

def calculate_polling_metrics(df: pd.DataFrame, candidate_names: List[str]) -> Dict[str, float]:
    """
    Calculate polling metrics for the specified candidate names.
    Ensure percentages are handled correctly.
    """
    df = df.copy()
    df['pct'] = df['pct'].apply(lambda x: x if x > 1 else x * 100)

    df['transparency_score'] = pd.to_numeric(df['transparency_score'], errors='coerce').fillna(0)
    max_transparency_score = df['transparency_score'].max()
    df['transparency_weight'] = df['transparency_score'] / max_transparency_score

    min_sample_size, max_sample_size = df['sample_size'].min(), df['sample_size'].max()
    df['sample_size_weight'] = (df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)

    df.loc[:, 'is_partisan'] = df['partisan'].notna() & df['partisan'].ne('')
    df.loc[:, 'partisan_weight'] = df['is_partisan'].map(partisan_weight)
    df.loc[:, 'population'] = df['population'].str.lower()
    df.loc[:, 'population_weight'] = df['population'].map(lambda x: population_weights.get(x, 1))

    state_data = get_state_data()
    df['state_rank'] = df['state'].apply(lambda x: state_data.get(x, 1))

    list_weights = np.array([
        df['time_decay_weight'],
        df['sample_size_weight'],
        df['normalized_numeric_grade'],
        df['transparency_weight'],
        df['population_weight'],
        df['partisan_weight'],
        df['state_rank'],
    ])
    if heavy_weight:
        df['combined_weight'] = np.prod(list_weights, axis=0)
    else:
        df['combined_weight'] = sum(list_weights) / len(list_weights)

    weighted_sums = df.groupby('candidate_name')['combined_weight'].apply(lambda x: (x * df.loc[x.index, 'pct']).sum()).fillna(0)
    total_weights = df.groupby('candidate_name')['combined_weight'].sum().fillna(0)

    weighted_averages = (weighted_sums / total_weights).fillna(0)  # Handle NaN

    weighted_margins = {candidate: calculate_timeframe_specific_moe(df, [candidate]) for candidate in candidate_names}

    return {candidate: (weighted_averages.get(candidate, 0), weighted_margins.get(candidate, 0)) for candidate in candidate_names}

def calculate_favorability_differential(df: pd.DataFrame, candidate_names: List[str]) -> Dict[str, float]:
    """
    Calculate favorability differentials for the specified candidate names.
    Ensure percentages are handled correctly.
    """
    df = df.copy()
    df['favorable'] = df['favorable'].apply(lambda x: x if x > 1 else x * 100)

    df['numeric_grade'] = pd.to_numeric(df['numeric_grade'], errors='coerce').fillna(0)
    max_numeric_grade = df['numeric_grade'].max()
    df['normalized_numeric_grade'] = df['numeric_grade'] / max_numeric_grade

    df['pollscore'] = pd.to_numeric(df['pollscore'], errors='coerce')
    min_pollscore = df['pollscore'].min()
    max_pollscore = df['pollscore'].max()
    df['normalized_pollscore'] = 1 - (df['pollscore'] - min_pollscore) / (max_pollscore - min_pollscore)

    df['transparency_score'] = pd.to_numeric(df['transparency_score'], errors='coerce').fillna(0)
    max_transparency_score = df['transparency_score'].max()
    df['normalized_transparency_score'] = df['transparency_score'] / max_transparency_score

    df['normalized_numeric_grade'] = df['normalized_numeric_grade'].clip(0, 1)
    df['normalized_pollscore'] = df['normalized_pollscore'].clip(0, 1)
    df['normalized_transparency_score'] = df['normalized_transparency_score'].clip(0, 1)

    list_weights = np.array([
        df['normalized_numeric_grade'],
        df['normalized_pollscore'],
        df['normalized_transparency_score']
    ])
    df['combined_weight'] = np.prod(list_weights, axis=0)

    weighted_sums = df.groupby('politician')['combined_weight'].apply(lambda x: (x * df.loc[x.index, 'favorable']).sum()).fillna(0)
    total_weights = df.groupby('politician')['combined_weight'].sum().fillna(0)

    weighted_averages = (weighted_sums / total_weights).fillna(0)  # Handle NaN

    return {candidate: weighted_averages.get(candidate, 0) for candidate in candidate_names}

def combine_analysis(polling_metrics: Dict[str, float], favorability_differential: Dict[str, float], favorability_weight: float) -> Dict[str, float]:
    """
    Combine polling metrics and favorability differentials into a unified analysis.
    Handles cases where favorability_differential might be empty.
    """
    combined_metrics = {}
    for candidate in polling_metrics.keys():
        fav_diff = favorability_differential.get(candidate, 0)  # Get favorability, default to 0 if not available
        combined_metrics[candidate] = (
            polling_metrics[candidate][0] * (1 - favorability_weight) + fav_diff * favorability_weight,
            polling_metrics[candidate][1]
        )
    return combined_metrics

def print_with_color(text: str, color_code: int):
    """
    Print text with the specified color code using ANSI escape sequences.
    """
    print(f"\033[38;5;{color_code}m{text}\033[0m")

def output_results(combined_results: Dict[str, float], color_index: int, period_value: int, period_type: str, oob_variance: float):
    """
    Corrected output formatting to display percentages properly and include OOB variance.
    """
    harris_score, harris_margin = combined_results['Kamala Harris']
    trump_score, trump_margin = combined_results['Donald Trump']
    differential = trump_score - harris_score
    favored_candidate = "Harris" if differential < 0 else "Trump"
    color_code = start_color + (color_index * skip_color)
    print(f"\033[38;5;{color_code}m{period_value:2d}{period_type[0]:<4} H∙{harris_score:5.2f}%±{harris_margin:.2f} T∙{trump_score:5.2f}%±{trump_margin:.2f} {abs(differential):+5.2f} {favored_candidate} 𝛂{oob_variance:5.1f}\033[0m")

def _get_unsampled_indices(tree, n_samples):
    """Retrieves indices of out-of-bag samples for a given tree."""
    unsampled_mask = np.ones(n_samples, dtype=bool)
    unsampled_mask[tree.tree_.feature[tree.tree_.feature >= 0]] = False
    return np.arange(n_samples)[unsampled_mask]

def impute_data(X):
    """Imputes data for each column separately, only if the column has non-missing values."""
    imputer = SimpleImputer(strategy='median')
    for col in range(X.shape[1]):
        if np.any(~np.isnan(X[:, col])):
            X[:, col] = imputer.fit_transform(X[:, col].reshape(-1, 1)).ravel()
    return X

def main():
    polling_df = download_csv_data(polling_url)
    favorability_df = download_csv_data(favorability_url)

    polling_df = preprocess_data(polling_df)
    favorability_df = preprocess_data(favorability_df)

    polling_df = apply_time_decay_weight(polling_df, decay_rate, half_life_days)
    favorability_df = apply_time_decay_weight(favorability_df, decay_rate, half_life_days)

    min_samples_required = 5
    n_trees = 1000

    color_index = 0
    for period in [(12, 'months'), (6, 'months'), (3, 'months'), (1, 'months'), (21, 'days'), (14, 'days'), (7, 'days'), (3, 'days'), (1, 'days')]:
        period_value, period_type = period
        if period_type == 'months':
            start_period = pd.Timestamp.now() - pd.DateOffset(months=period_value)
        elif period_type == 'days':
            start_period = pd.Timestamp.now() - pd.Timedelta(days=period_value)

        filtered_polling_df = preprocess_data(polling_df[(polling_df['created_at'] >= start_period) &
                                                          (polling_df['candidate_name'].isin(candidate_names))].copy(), start_period)
        filtered_favorability_df = preprocess_data(favorability_df[(favorability_df['created_at'] >= start_period) &
                                                                    (favorability_df['politician'].isin(candidate_names))].copy(), start_period)

        # Check for sufficient polling data
        if filtered_polling_df.shape[0] >= min_samples_required: 
            polling_metrics = calculate_polling_metrics(filtered_polling_df, candidate_names)
            
            # Check for sufficient favorability data
            if filtered_favorability_df.shape[0] >= min_samples_required:
                favorability_differential = calculate_favorability_differential(filtered_favorability_df, candidate_names)

                combined_results = combine_analysis(polling_metrics, favorability_differential, favorability_weight)

                features_columns = ['normalized_numeric_grade', 'normalized_pollscore', 'normalized_transparency_score', 'sample_size_weight', 'state_rank', 'population_weight']

                X = filtered_favorability_df[features_columns].values
                y = filtered_favorability_df['favorable'].values
                
                # Create the pipeline with imputation
                pipeline = Pipeline(steps=[
                    ('imputer', FunctionTransformer(impute_data)), 
                    ('model', RandomForestRegressor(n_estimators=n_trees, oob_score=True, random_state=5000, bootstrap=True))
                ])

                pipeline.fit(X, y)

                oob_predictions = np.zeros(y.shape)
                oob_sample_counts = np.zeros(X.shape[0], dtype=int) # Initialize counts here

                for tree in pipeline.named_steps['model'].estimators_:
                    unsampled_indices = _get_unsampled_indices(tree, X.shape[0])
                    if len(unsampled_indices) > 0:
                        oob_predictions[unsampled_indices] += tree.predict(impute_data(X[unsampled_indices])) # Using function name directly
                        oob_sample_counts[unsampled_indices] += 1 # Update counts in the same loop 

                epsilon = np.finfo(float).eps  
                oob_predictions /= (oob_sample_counts + epsilon)

                oob_variance = np.var(y - oob_predictions)

                output_results(combined_results, color_index, period_value, period_type, oob_variance)

            else: # Not enough favorability data, use only polling
                print_with_color(f"Using only polling data for {period_value} {period_type} period.", color_index)
                combined_results = combine_analysis(polling_metrics, {}, 0.0)  # Pass 0.0 as favorability_weight
                output_results(combined_results, color_index, period_value, period_type, oob_variance=0)  # Set OOB variance to 0 
        else:
            print_with_color(f"Not enough data for prediction in {period_value} {period_type} period. Data count: {filtered_polling_df.shape[0]}", color_index)

        color_index += 1

if __name__ == "__main__":
    main()