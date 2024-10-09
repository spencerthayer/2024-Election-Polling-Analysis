# analysis.py

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from io import StringIO
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
import logging
import json
import re
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import norm

# Constants (You can adjust these as needed)
CANDIDATE_NAMES = ['Kamala Harris', 'Donald Trump']
FAVORABILITY_WEIGHT = 0.25
HEAVY_WEIGHT = True
DECAY_RATE = 2
HALF_LIFE_DAYS = 14
MIN_SAMPLES_REQUIRED = 5
N_TREES = 1000
RANDOM_STATE = 5000
POLLING_URL = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
FAVORABILITY_URL = "https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv"
PARTISAN_WEIGHT = {True: 0.01, False: 1}
POPULATION_WEIGHTS = {
    'lv': 1.0,  # Likely voters
    'rv': 0.6666666666666666,  # Registered voters
    'v': 0.5,  # Voters
    'a': 0.3333333333333333,  # Adults
    'all': 0.3333333333333333
}
START_COLOR = 164
SKIP_COLOR = 3

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_state_data() -> Dict[str, float]:
    """
    Retrieves state data from 270toWin.com and calculates a rank for each state
    based on electoral votes and political leanings.
    """
    mapurl = 'https://www.270towin.com/'
    electoral_total = 538

    pro_values = {
        'T': 0.8,  # Swing State
        'D1': 0.6,  # Tilts Democrat
        'D2': 0.4,  # Leans Democrat
        'D3': 0.2,  # Likely Democrat
        'D4': 0.1,  # Safe Democrat
        'R1': 0.6,  # Tilts Republican
        'R2': 0.4,  # Leans Republican
        'R3': 0.2,  # Likely Republican
        'R4': 0.1   # Safe Republican
    }

    try:
        response = requests.get(mapurl)
        response.raise_for_status()
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

                processed_data = {}
                for state_fips, seats in seats_data.items():
                    for seat in seats:
                        state_name = seat['state_name']
                        e_votes = seat['e_votes']
                        pro_status_code = seat['pro_status']
                        pro_status_value = pro_values.get(pro_status_code, 1)
                        normalized_e_votes = e_votes / electoral_total
                        state_rank = pro_status_value + normalized_e_votes
                        processed_data[state_name] = state_rank

                return processed_data
            else:
                logging.error("Failed to find the required script in the page")
                return {}
        else:
            logging.error("Failed to find the required script in the page")
            return {}
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while making the request: {str(e)}")
        return {}

def download_csv_data(url: str) -> pd.DataFrame:
    """
    Download CSV data from the specified URL.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = StringIO(response.content.decode('utf-8'))
        return pd.read_csv(csv_data)
    except requests.RequestException as e:
        logging.error(f"Network error while downloading data from {url}: {e}")
        return pd.DataFrame()
    except pd.errors.ParserError as e:
        logging.error(f"Parsing error while reading CSV data from {url}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Unexpected error while downloading data from {url}: {e}")
        return pd.DataFrame()

def preprocess_data(df: pd.DataFrame, start_period: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """
    Preprocess the data by converting date columns, handling missing values,
    and calculating necessary weights.
    """
    df = df.copy()
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at'])
    if start_period is not None:
        df = df[df['created_at'] >= start_period]

    # Handle sample_size_weight
    min_sample_size = df['sample_size'].min()
    max_sample_size = df['sample_size'].max()
    if max_sample_size - min_sample_size > 0:
        df['sample_size_weight'] = (df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)
    else:
        df['sample_size_weight'] = 1

    # Handle population_weight
    if 'population' in df.columns:
        df['population'] = df['population'].str.lower()
        df['population_weight'] = df['population'].map(lambda x: POPULATION_WEIGHTS.get(x, 1))
    else:
        logging.warning("'population' column is missing. Setting 'population_weight' to 1 for all rows.")
        df['population_weight'] = 1

    # Handle is_partisan and partisan_weight
    df['is_partisan'] = df['partisan'].notna() & df['partisan'].ne('')
    df['partisan_weight'] = df['is_partisan'].map(PARTISAN_WEIGHT)

    # Calculate state_rank using get_state_data()
    state_data = get_state_data()
    df['state_rank'] = df['state'].apply(lambda x: state_data.get(x, 1))

    return df

def apply_time_decay_weight(df: pd.DataFrame, decay_rate: float, half_life_days: int) -> pd.DataFrame:
    """
    Apply time decay weighting to the data based on the specified decay rate and half-life.
    """
    try:
        reference_date = pd.Timestamp.now()
        days_old = (reference_date - df['created_at']).dt.days
        df['time_decay_weight'] = np.exp(-np.log(decay_rate) * days_old / half_life_days)
        return df
    except Exception as e:
        logging.error(f"Error applying time decay: {e}")
        df['time_decay_weight'] = 1
        return df

def margin_of_error(n: int, p: float = 0.5, confidence_level: float = 0.95) -> float:
    """
    Calculate the margin of error for a proportion at a given confidence level.
    """
    z = norm.ppf((1 + confidence_level) / 2)
    moe = z * np.sqrt((p * (1 - p)) / n)
    return moe * 100  # Convert to percentage

def calculate_timeframe_specific_moe(df: pd.DataFrame, candidate_names: List[str]) -> float:
    """
    Calculate the average margin of error for the given candidates within the DataFrame.
    """
    moes = []
    for candidate in candidate_names:
        candidate_df = df[df['candidate_name'] == candidate]
        if candidate_df.empty:
            continue
        for _, poll in candidate_df.iterrows():
            if poll['sample_size'] > 0 and 0 <= poll['pct'] <= 100:
                moe = margin_of_error(n=poll['sample_size'], p=poll['pct'] / 100)
                moes.append(moe)
    return np.mean(moes) if moes else np.nan

def calculate_polling_metrics(df: pd.DataFrame, candidate_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Calculate polling metrics for the specified candidate names.
    """
    df = df.copy()
    df['pct'] = df['pct'].apply(lambda x: x if x > 1 else x * 100)

    # Combined weight
    list_weights = np.array([
        df['time_decay_weight'],
        df['sample_size_weight'],
        df['population_weight'],
        df['partisan_weight'],
        df['state_rank'],
    ])
    if HEAVY_WEIGHT:
        df['combined_weight'] = np.prod(list_weights, axis=0)
    else:
        df['combined_weight'] = np.mean(list_weights, axis=0)

    weighted_sums = df.groupby('candidate_name')['combined_weight'].apply(
        lambda x: (x * df.loc[x.index, 'pct']).sum()).fillna(0)
    total_weights = df.groupby('candidate_name')['combined_weight'].sum().fillna(0)

    weighted_averages = (weighted_sums / total_weights).fillna(0)

    weighted_margins = {candidate: calculate_timeframe_specific_moe(df, [candidate]) for candidate in candidate_names}

    return {candidate: (weighted_averages.get(candidate, 0), weighted_margins.get(candidate, 0)) for candidate in candidate_names}

def calculate_favorability_differential(df: pd.DataFrame, candidate_names: List[str]) -> Dict[str, float]:
    """
    Calculate favorability differentials for the specified candidate names.
    """
    df = df.copy()
    df['favorable'] = df['favorable'].apply(lambda x: x if x > 1 else x * 100)

    # Combined weight (since we don't have grades, we'll use sample_size_weight and time_decay_weight)
    list_weights = np.array([
        df['sample_size_weight'],
        df['time_decay_weight']
    ])
    df['combined_weight'] = np.prod(list_weights, axis=0)

    weighted_sums = df.groupby('politician')['combined_weight'].apply(
        lambda x: (x * df.loc[x.index, 'favorable']).sum()).fillna(0)
    total_weights = df.groupby('politician')['combined_weight'].sum().fillna(0)

    weighted_averages = (weighted_sums / total_weights).fillna(0)

    return {candidate: weighted_averages.get(candidate, 0) for candidate in candidate_names}

def combine_analysis(
    polling_metrics: Dict[str, Tuple[float, float]],
    favorability_differential: Dict[str, float],
    favorability_weight: float
) -> Dict[str, Tuple[float, float]]:
    """
    Combine polling metrics and favorability differentials into a unified analysis.
    """
    combined_metrics = {}
    for candidate in polling_metrics.keys():
        fav_diff = favorability_differential.get(candidate, polling_metrics[candidate][0])
        polling_score, margin = polling_metrics[candidate]
        combined_score = polling_score * (1 - favorability_weight) + fav_diff * favorability_weight
        combined_metrics[candidate] = (combined_score, margin)
    return combined_metrics

def calculate_oob_variance(df: pd.DataFrame) -> float:
    """
    Calculate the out-of-bag variance using Random Forest regression.
    """
    features_columns = [
        'sample_size_weight',
        'population_weight',
        'partisan_weight',
        'state_rank',
        'time_decay_weight'
    ]
    X = df[features_columns].values
    y = df['favorable'].values

    pipeline = Pipeline(steps=[
        ('imputer', FunctionTransformer(impute_data)),
        ('model', RandomForestRegressor(
            n_estimators=N_TREES,
            oob_score=True,
            random_state=RANDOM_STATE,
            bootstrap=True))
    ])

    pipeline.fit(X, y)

    oob_predictions = pipeline.named_steps['model'].oob_prediction_
    oob_variance = np.var(y - oob_predictions)

    return oob_variance

def impute_data(X: np.ndarray) -> np.ndarray:
    """
    Impute missing data for each column separately, only if the column has non-missing values.
    """
    imputer = SimpleImputer(strategy='median')
    for col in range(X.shape[1]):
        if np.any(~np.isnan(X[:, col])):
            X[:, col] = imputer.fit_transform(X[:, col].reshape(-1, 1)).ravel()
    return X

def get_analysis_results() -> pd.DataFrame:
    """
    Performs the full analysis and returns the results as a DataFrame.
    """
    polling_df, favorability_df = load_and_preprocess_data()
    results = calculate_results_for_all_periods(polling_df, favorability_df)
    return pd.DataFrame(results)

def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses polling and favorability data.
    """
    polling_df = download_csv_data(POLLING_URL)
    favorability_df = download_csv_data(FAVORABILITY_URL)

    polling_df = preprocess_data(polling_df)
    favorability_df = preprocess_data(favorability_df)

    polling_df = apply_time_decay_weight(polling_df, DECAY_RATE, HALF_LIFE_DAYS)
    favorability_df = apply_time_decay_weight(favorability_df, DECAY_RATE, HALF_LIFE_DAYS)

    return polling_df, favorability_df

def calculate_results_for_all_periods(
    polling_df: pd.DataFrame,
    favorability_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Calculates results for all predefined periods.
    """
    results = []
    periods = [
        (12, 'months'), (6, 'months'), (3, 'months'), (1, 'months'),
        (21, 'days'), (14, 'days'), (7, 'days'), (3, 'days'), (1, 'days')
    ]

    for period_value, period_type in periods:
        period_result = calculate_results_for_period(
            polling_df, favorability_df, period_value, period_type
        )
        results.append(period_result)

    return results

def calculate_results_for_period(
    polling_df: pd.DataFrame,
    favorability_df: pd.DataFrame,
    period_value: int,
    period_type: str
) -> Dict[str, Any]:
    """
    Calculate metrics and OOB variance for a single period.
    """
    period_map: Dict[str, Callable[[int], Union[pd.DateOffset, pd.Timedelta]]] = {
        'months': lambda x: pd.DateOffset(months=x),
        'days': lambda x: pd.Timedelta(days=x)
    }
    start_period = pd.Timestamp.now() - period_map[period_type](period_value)

    filtered_polling_df = polling_df[
        (polling_df['created_at'] >= start_period) &
        (polling_df['candidate_name'].isin(CANDIDATE_NAMES))
    ].copy()

    filtered_favorability_df = favorability_df[
        (favorability_df['created_at'] >= start_period) &
        (favorability_df['politician'].isin(CANDIDATE_NAMES))
    ].copy()

    if filtered_polling_df.shape[0] < MIN_SAMPLES_REQUIRED:
        return {
            'period': f"{period_value} {period_type}",
            'harris': None,
            'trump': None,
            'harris_fav': None,
            'trump_fav': None,
            'harris_moe': None,
            'trump_moe': None,
            'oob_variance': None,
            'message': "Not enough polling data"
        }

    polling_metrics = calculate_polling_metrics(filtered_polling_df, CANDIDATE_NAMES)

    if filtered_favorability_df.shape[0] >= MIN_SAMPLES_REQUIRED:
        favorability_differential = calculate_favorability_differential(
            filtered_favorability_df, CANDIDATE_NAMES
        )
        combined_results = combine_analysis(
            polling_metrics, favorability_differential, FAVORABILITY_WEIGHT
        )
        oob_variance = calculate_oob_variance(filtered_favorability_df)
        harris_fav = favorability_differential.get('Kamala Harris', None)
        trump_fav = favorability_differential.get('Donald Trump', None)
    else:
        combined_results = combine_analysis(
            polling_metrics, {}, 0.0  # Set favorability_weight to 0.0
        )
        oob_variance = None
        harris_fav = None
        trump_fav = None

    return {
        'period': f"{period_value} {period_type}",
        'harris': combined_results['Kamala Harris'][0],
        'trump': combined_results['Donald Trump'][0],
        'harris_fav': harris_fav,
        'trump_fav': trump_fav,
        'harris_moe': combined_results['Kamala Harris'][1],
        'trump_moe': combined_results['Donald Trump'][1],
        'oob_variance': oob_variance,
        'message': None
    }

def output_results(row: Dict[str, Any]):
    """
    Outputs the results for a period to the console.
    """
    period = row['period']
    harris_score = row['harris']
    trump_score = row['trump']
    harris_margin = row['harris_moe']
    trump_margin = row['trump_moe']
    oob_variance = row['oob_variance']
    message = row.get('message')

    if message:
        logging.warning(f"{period:<4} {message}")
        return

    differential = trump_score - harris_score
    favored_candidate = "Harris" if differential < 0 else "Trump"
    color_code = START_COLOR  # Adjust as needed
    print(f"\033[38;5;{color_code}m{period:>4} H∙{harris_score:5.2f}%±{harris_margin:.2f} "
          f"T∙{trump_score:5.2f}%±{trump_margin:.2f} {abs(differential):+5.2f} "
          f"{favored_candidate} 𝛂{oob_variance:5.1f}\033[0m")

def main():
    """
    Main function to perform analysis and output results.
    """
    results_df = get_analysis_results()
    for _, row in results_df.iterrows():
        output_results(row)

if __name__ == "__main__":
    main()
