# analysis.py

import pandas as pd
import numpy as np
import requests
import config
import logging
from states import get_state_data
from config import *
from io import StringIO
from typing import Dict, List, Tuple, Any, Optional, Callable, Union
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Configure logging
logging.basicConfig(level=config.LOGGING_LEVEL, format=config.LOGGING_FORMAT)

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
    Preprocess the data by converting date columns, handling missing values, and calculating necessary weights.
    """
    df = df.copy()

    # Filter out polls without numeric_grade or pollscore
    original_count = len(df)

    df['numeric_grade'] = df['numeric_grade'].fillna(config.ZERO_CORRECTION)
    df['pollscore'] = df['pollscore'].fillna(config.ZERO_CORRECTION)
    df = df[(df['numeric_grade'] != config.ZERO_CORRECTION) | (df['pollscore'] != config.ZERO_CORRECTION)]

    filtered_count = len(df)
    logging.info(f"Filtered out {original_count - filtered_count} polls without numeric_grade or pollscore")

    # Specify the date format based on your data
    date_format = '%m/%d/%y %H:%M'
    df['created_at'] = pd.to_datetime(df['created_at'], format=date_format, errors='coerce', utc=True)
    df = df.dropna(subset=['created_at'])
    if start_period is not None:
        df = df[df['created_at'] >= start_period]

    # Standardize candidate names
    if 'candidate_name' in df.columns:
        df['candidate_name'] = df['candidate_name'].str.strip()
    if 'politician' in df.columns:
        df['politician'] = df['politician'].str.strip()

    # Normalize 'numeric_grade'
    df['numeric_grade'] = pd.to_numeric(df['numeric_grade'], errors='coerce').fillna(0)
    max_numeric_grade = df['numeric_grade'].max()
    if max_numeric_grade != 0:
        df['normalized_numeric_grade'] = df['numeric_grade'] / max_numeric_grade
    else:
        df['normalized_numeric_grade'] = config.ZERO_CORRECTION
    df['normalized_numeric_grade'] = df['normalized_numeric_grade'].clip(0, 1)

    # Handle pollscore (lower is better, negative is best)
    df['pollscore'] = pd.to_numeric(df['pollscore'], errors='coerce').fillna(0)
    max_pollscore = df['pollscore'].max()
    min_pollscore = df['pollscore'].min()
    
    if max_pollscore != min_pollscore:
        # Invert and shift the scale so that the best (most negative) score gets the highest weight
        df['normalized_pollscore'] = (max_pollscore - df['pollscore']) / (max_pollscore - min_pollscore)
    else:
        df['normalized_pollscore'] = 1  # If all scores are the same, give them full weight

    # Clip to ensure all values are between 0 and 1
    df['normalized_pollscore'] = df['normalized_pollscore'].clip(0, 1)

    logging.info(f"Normalized pollscores range from {df['normalized_pollscore'].min():.4f} to {df['normalized_pollscore'].max():.4f}")
    logging.info(f"Best raw pollscore: {min_pollscore:.4f}, Worst raw pollscore: {max_pollscore:.4f}")

    # Normalize 'transparency_score'
    df['transparency_score'] = pd.to_numeric(df['transparency_score'], errors='coerce').fillna(0)
    max_transparency_score = df['transparency_score'].max()
    if max_transparency_score != 0:
        df['normalized_transparency_score'] = df['transparency_score'] / max_transparency_score
    else:
        df['normalized_transparency_score'] = config.ZERO_CORRECTION
    df['normalized_transparency_score'] = df['normalized_transparency_score'].clip(0, 1)

    # Handle sample_size_weight
    min_sample_size = df['sample_size'].min()
    max_sample_size = df['sample_size'].max()
    if max_sample_size - min_sample_size > 0:
        df['sample_size_weight'] = (df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)
    else:
        df['sample_size_weight'] = config.ZERO_CORRECTION

    # Handle population_weight
    if 'population' in df.columns:
        df['population'] = df['population'].str.lower()
        df['population_weight'] = df['population'].map(lambda x: config.POPULATION_WEIGHTS.get(x, 1.0))
    else:
        logging.warning("'population' column is missing. Setting 'population_weight' to 1 for all rows.")
        df['population_weight'] = config.ZERO_CORRECTION

    # Handle is_partisan flag
    df['partisan'] = df['partisan'].fillna('').astype(str).str.strip()
    df['is_partisan'] = df['partisan'] != ''

    # Apply partisan weight mapping here
    df['partisan_weight'] = df['is_partisan'].map({
        True: config.PARTISAN_WEIGHT[True],
        False: config.PARTISAN_WEIGHT[False]
    })

    # Calculate state_rank using get_state_data()
    state_data = get_state_data()
    df['state_rank'] = df['state'].apply(lambda x: state_data.get(x, 1.0))

    # Apply time decay weight
    df = apply_time_decay_weight(df, config.DECAY_RATE, config.HALF_LIFE_DAYS)

    # Apply multipliers to the weights
    df['time_decay_weight'] *= config.TIME_DECAY_WEIGHT_MULTIPLIER
    df['sample_size_weight'] *= config.SAMPLE_SIZE_WEIGHT_MULTIPLIER
    df['normalized_numeric_grade'] *= config.NORMALIZED_NUMERIC_GRADE_MULTIPLIER
    df['normalized_pollscore'] *= config.NORMALIZED_POLLSCORE_MULTIPLIER
    df['normalized_transparency_score'] *= config.NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER
    df['population_weight'] *= config.POPULATION_WEIGHT_MULTIPLIER
    df['partisan_weight'] *= config.PARTISAN_WEIGHT_MULTIPLIER
    df['state_rank'] *= config.STATE_RANK_MULTIPLIER

    return df

def apply_time_decay_weight(df: pd.DataFrame, decay_rate: float, half_life_days: int) -> pd.DataFrame:
    """
    Apply time decay weighting to the data based on the specified decay rate and half-life using a logarithmic scale.
    """
    try:
        reference_date = pd.Timestamp.now(tz='UTC')
        days_old = (reference_date - df['created_at']).dt.total_seconds() / (24 * 3600)
        
        # Calculate decay constant
        lambda_decay = np.log(decay_rate) / half_life_days
        
        # Apply exponential decay
        df['time_decay_weight'] = np.exp(-lambda_decay * days_old)
        
        # Apply logarithmic transformation
        df['time_decay_weight'] = np.log1p(df['time_decay_weight'])
        
        # Normalize weights to [0, 1] range
        max_weight = df['time_decay_weight'].max()
        min_weight = df['time_decay_weight'].min()
        if max_weight != min_weight:
            df['time_decay_weight'] = (df['time_decay_weight'] - min_weight) / (max_weight - min_weight)
        else:
            df['time_decay_weight'] = 1.0  # If all weights are the same, set to 1
        
        # Apply multiplier from config
        df['time_decay_weight'] *= config.TIME_DECAY_WEIGHT_MULTIPLIER
        
        # Clip weights to ensure they're within [0, 1]
        df['time_decay_weight'] = df['time_decay_weight'].clip(0, 1)
        
        return df
    except Exception as e:
        logging.error(f"Error applying time decay: {e}")
        df['time_decay_weight'] = 1.0
        return df

def margin_of_error(n: int, p: float = 0.5, confidence_level: float = 0.95) -> float:
    """
    Calculate the margin of error for a proportion at a given confidence level.
    """
    if n == 0:
        return 0.0
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

def calculate_polling(df: pd.DataFrame, candidate_names: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Calculate polling metrics for the specified candidate names.
    """
    df = df.copy()
    # Ensure 'pct' is correctly interpreted as a percentage
    df['pct'] = df['pct'].apply(lambda x: x if x > 1 else x * 100)

    # Apply partisan weight mapping here
    df['partisan_weight'] = df['is_partisan'].map({
        True: config.PARTISAN_WEIGHT[True],
        False: config.PARTISAN_WEIGHT[False]
    })

    # Prepare the weights with multipliers
    weight_components = {
        'time_decay': df['time_decay_weight'] * config.TIME_DECAY_WEIGHT_MULTIPLIER,
        'sample_size': df['sample_size_weight'] * config.SAMPLE_SIZE_WEIGHT_MULTIPLIER,
        'numeric_grade': df['normalized_numeric_grade'] * config.NORMALIZED_NUMERIC_GRADE_MULTIPLIER,
        'pollscore': df['normalized_pollscore'] * config.NORMALIZED_POLLSCORE_MULTIPLIER,
        'transparency': df['normalized_transparency_score'] * config.NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER,
        'population': df['population_weight'] * config.POPULATION_WEIGHT_MULTIPLIER,
        'partisan': df['partisan_weight'] * config.PARTISAN_WEIGHT_MULTIPLIER,
        'state_rank': df['state_rank'] * config.STATE_RANK_MULTIPLIER,
    }

    # Normalize weights after applying multipliers
    for component in weight_components:
        max_val = weight_components[component].max()
        if max_val != 0:
            weight_components[component] = weight_components[component] / max_val
        else:
            weight_components[component] = config.ZERO_CORRECTION

    if config.HEAVY_WEIGHT:
        df['combined_weight'] = np.prod([weight_components[comp] for comp in weight_components], axis=0)
    else:
        df['combined_weight'] = np.mean([weight_components[comp] for comp in weight_components], axis=0)

    # Handle national polls
    df['is_national'] = df['state'].isnull() | (df['state'] == '')
    df.loc[df['is_national'], 'combined_weight'] *= config.NATIONAL_POLL_WEIGHT

    # Normalize combined_weight
    if df['combined_weight'].max() != 0:
        df['combined_weight'] = df['combined_weight'] / df['combined_weight'].max()
    else:
        df['combined_weight'] = 0

    results = {}
    for candidate in candidate_names:
        candidate_df = df[df['candidate_name'] == candidate]
        
        weighted_sum = (candidate_df['combined_weight'] * candidate_df['pct']).sum()
        total_weight = candidate_df['combined_weight'].sum()
        
        weighted_average = weighted_sum / total_weight if total_weight > 0 else 0
        moe = calculate_timeframe_specific_moe(candidate_df, [candidate])
        
        results[candidate] = (weighted_average, moe)
        
        print(f"\nDetailed calculations for {candidate}:")
        print(f"  Total polls: {len(candidate_df)}")
        print("  Weight components (mean values):")
        for component, values in weight_components.items():
            mean_val = values[candidate_df.index].mean() if not candidate_df.empty else 0
            print(f"    {component}: {mean_val:.4f}")
        print(f"  Combined weight (sum): {total_weight:.4f}")
        print(f"  Weighted sum: {weighted_sum:.4f}")
        print(f"  Weighted average: {weighted_average:.2f}%")
        print(f"  Margin of Error: ±{moe:.2f}%")
        print(f"  National polls: {candidate_df['is_national'].sum()}")

    return results

def calculate_favorability(df: pd.DataFrame, candidate_names: List[str]) -> Dict[str, float]:
    """
    Calculate favorability differentials for the specified candidate names,
    aligning more closely with polling metrics calculation.
    """
    df = df.copy()
    # Ensure 'favorable' and 'unfavorable' are correctly interpreted as percentages
    for col in ['favorable', 'unfavorable']:
        df[col] = df[col].apply(lambda x: x if x > 1 else x * 100)

    # Apply partisan weight mapping
    df['partisan_weight'] = df['is_partisan'].map({
        True: config.PARTISAN_WEIGHT[True],
        False: config.PARTISAN_WEIGHT[False]
    })

    # Prepare weights with multipliers
    weight_components = {
        'time_decay': df.get('time_decay_weight', 1) * config.TIME_DECAY_WEIGHT_MULTIPLIER,
        'sample_size': df.get('sample_size_weight', 1) * config.SAMPLE_SIZE_WEIGHT_MULTIPLIER,
        'numeric_grade': df['normalized_numeric_grade'] * config.NORMALIZED_NUMERIC_GRADE_MULTIPLIER,
        'pollscore': df['normalized_pollscore'] * config.NORMALIZED_POLLSCORE_MULTIPLIER,
        'transparency': df['normalized_transparency_score'] * config.NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER,
        'population': df.get('population_weight', 1) * config.POPULATION_WEIGHT_MULTIPLIER,
        'partisan': df['partisan_weight'] * config.PARTISAN_WEIGHT_MULTIPLIER,
        'state_rank': df['state_rank'] * config.STATE_RANK_MULTIPLIER,
    }

    # Normalize weights after applying multipliers
    for component in weight_components:
        max_val = weight_components[component].max()
        if max_val != 0:
            weight_components[component] = weight_components[component] / max_val
        else:
            weight_components[component] = config.ZERO_CORRECTION

    if config.HEAVY_WEIGHT:
        df['combined_weight'] = np.prod([weight_components[comp] for comp in weight_components], axis=0)
    else:
        df['combined_weight'] = np.mean([weight_components[comp] for comp in weight_components], axis=0)

    # Handle national polls
    df['is_national'] = df['state'].isnull() | (df['state'] == '')
    df.loc[df['is_national'], 'combined_weight'] *= config.NATIONAL_POLL_WEIGHT

    # Normalize combined_weight
    if df['combined_weight'].max() != 0:
        df['combined_weight'] = df['combined_weight'] / df['combined_weight'].max()
    else:
        df['combined_weight'] = 0

    results = {}
    for candidate in candidate_names:
        candidate_df = df[df['politician'] == candidate]
        
        weighted_favorable = (candidate_df['combined_weight'] * candidate_df['favorable']).sum()
        weighted_unfavorable = (candidate_df['combined_weight'] * candidate_df['unfavorable']).sum()
        total_weight = candidate_df['combined_weight'].sum()
        
        if total_weight > 0:
            favorability = (weighted_favorable / total_weight)
        else:
            favorability = 0
        
        results[candidate] = favorability
        
        print(f"\nDetailed favorability calculations for {candidate}:")
        print(f"  Total polls: {len(candidate_df)}")
        print("  Weight components (mean values):")
        for component, values in weight_components.items():
            mean_val = values[candidate_df.index].mean() if not candidate_df.empty else 0
            print(f"    {component}: {mean_val:.4f}")
        print(f"  National poll weight: {config.NATIONAL_POLL_WEIGHT:.4f}")
        print(f"  Combined weight (sum): {total_weight:.4f}")
        print(f"  Weighted favorable sum: {weighted_favorable:.4f}")
        print(f"  Weighted unfavorable sum: {weighted_unfavorable:.4f}")
        print(f"  Favorability: {favorability:.2f}%")
        print(f"  National polls: {candidate_df['is_national'].sum()}")

    return results

def combine_analysis(
    polling_metrics: Dict[str, Tuple[float, float]],
    favorability_scores: Dict[str, float],
    favorability_weight: float
) -> Dict[str, Tuple[float, float]]:
    """
    Combine polling metrics and favorability scores into a unified analysis.
    """
    combined_metrics = {}
    for candidate in polling_metrics.keys():
        polling_score, margin = polling_metrics[candidate]
        favorability = favorability_scores.get(candidate, polling_score)
        
        # Combine polling score and favorability
        combined_score = polling_score * (1 - favorability_weight) + favorability * favorability_weight
        
        combined_metrics[candidate] = (combined_score, margin)
    return combined_metrics

def calculate_oob_variance(polling_df: pd.DataFrame, favorability_df: pd.DataFrame) -> float:
    """
    Calculate the out-of-bag variance using Random Forest regression on combined polling and favorability data.
    """
    if polling_df.empty and favorability_df.empty:
        return 0.0  # Return 0 variance if both DataFrames are empty

    # Combine polling and favorability data
    combined_df = pd.concat([polling_df, favorability_df], axis=0, sort=False)

    # Define all possible feature columns
    all_features = [
        'normalized_numeric_grade',
        'normalized_pollscore',
        'normalized_transparency_score',
        'sample_size_weight',
        'population_weight',
        'partisan_weight',
        'state_rank',
        'time_decay_weight'
    ]

    # Filter to only use columns that exist in the combined DataFrame
    features_columns = [col for col in all_features if col in combined_df.columns]

    if not features_columns:
        logging.warning("No valid feature columns found for OOB variance calculation.")
        return 0.0

    X = combined_df[features_columns].values
    
    # Use 'pct' from polling data and 'favorable' from favorability data as the target
    y = combined_df['pct'].fillna(combined_df['favorable']).values

    pipeline = Pipeline(steps=[
        ('imputer', FunctionTransformer(impute_data)),
        ('model', RandomForestRegressor(
            n_estimators=config.N_TREES,
            oob_score=True,
            random_state=config.RANDOM_STATE,
            bootstrap=True,
            n_jobs=-1  # Use all available cores
        ))
    ])

    try:
        pipeline.fit(X, y)
        oob_predictions = pipeline.named_steps['model'].oob_prediction_
        oob_variance = np.var(y - oob_predictions)
        return oob_variance
    except Exception as e:
        logging.error(f"Error in OOB variance calculation: {e}")
        return 0.0

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
    results_df = pd.DataFrame(results)

    # Ensure 'period' is a categorical variable with the specified order
    results_df['period'] = pd.Categorical(results_df['period'], categories=config.PERIOD_ORDER, ordered=True)
    results_df = results_df.sort_values('period')

    return results_df

def load_and_preprocess_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads and preprocesses polling and favorability data.
    """
    polling_df = download_csv_data(config.POLLING_URL)
    favorability_df = download_csv_data(config.FAVORABILITY_URL)

    logging.info(f"Polling data loaded with {polling_df.shape[0]} rows.")
    logging.info(f"Favorability data loaded with {favorability_df.shape[0]} rows.")

    polling_df = preprocess_data(polling_df)
    favorability_df = preprocess_data(favorability_df)

    return polling_df, favorability_df

def calculate_results_for_all_periods(
    polling_df: pd.DataFrame,
    favorability_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Calculates results for all predefined periods.
    """
    results = []
    periods = [(int(period.split()[0]), period.split()[1]) for period in config.PERIOD_ORDER]

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
    start_period = pd.Timestamp.now(tz='UTC') - period_map[period_type](period_value)

    filtered_polling_df = polling_df[
        (polling_df['created_at'] >= start_period) &
        (polling_df['candidate_name'].isin(config.CANDIDATE_NAMES))
    ].copy()

    filtered_favorability_df = favorability_df[
        (favorability_df['created_at'] >= start_period) &
        (favorability_df['politician'].isin(config.CANDIDATE_NAMES))
    ].copy()

    print(f"\n--- Period: {period_value} {period_type} ---")
    print(f"Filtered polling data size: {filtered_polling_df.shape}")
    print(f"Filtered favorability data size: {filtered_favorability_df.shape}")

    if filtered_polling_df.shape[0] < config.MIN_SAMPLES_REQUIRED:
        print("Not enough polling data for this period.")
        return {
            'period': f"{period_value} {period_type}",
            'harris_polling': None,
            'trump_polling': None,
            'harris_fav': None,
            'trump_fav': None,
            'harris_combined': None,
            'trump_combined': None,
            'harris_moe': None,
            'trump_moe': None,
            'oob_variance': None,
            'message': "Not enough polling data"
        }

    # Calculate polling metrics
    print("\nCalculating Polling Metrics:")
    polling_metrics = calculate_polling(filtered_polling_df, config.CANDIDATE_NAMES)

    # Initialize variables
    harris_fav = None
    trump_fav = None
    oob_variance = None
    favorability_differential = {}

    # Check if we have enough data
    if filtered_polling_df.shape[0] >= config.MIN_SAMPLES_REQUIRED or filtered_favorability_df.shape[0] >= config.MIN_SAMPLES_REQUIRED:
        print("\nCalculating Favorability Differential:")
        favorability_differential = calculate_favorability(
            filtered_favorability_df, config.CANDIDATE_NAMES
        )
        harris_fav = favorability_differential.get('Kamala Harris', None)
        trump_fav = favorability_differential.get('Donald Trump', None)
        
        # Calculate OOB variance using both polling and favorability data
        oob_variance = calculate_oob_variance(filtered_polling_df, filtered_favorability_df)
        
        print(f"\nOOB Variance (combined data): {oob_variance:.2f}")
    else:
        print("\nNot enough data for this period.")

    # Combine polling metrics and favorability
    combined_results = combine_analysis(
        polling_metrics, favorability_differential, config.FAVORABILITY_WEIGHT
    )
    
    print("\nCombined Results Details:")
    for candidate, (combined, moe) in combined_results.items():
        print(f"{candidate}:")
        print(f"  Polling: {polling_metrics[candidate][0]:.2f}%")
        print(f"  Favorability: {favorability_differential.get(candidate, 'N/A')}")
        print(f"  Combined: {combined:.2f}% ± {moe:.2f}%")
        print(f"  Calculation: {polling_metrics[candidate][0]:.2f} * (1 - {config.FAVORABILITY_WEIGHT}) + "
              f"{favorability_differential.get(candidate, 0):.2f} * {config.FAVORABILITY_WEIGHT}")

    harris_combined = combined_results['Kamala Harris'][0]
    trump_combined = combined_results['Donald Trump'][0]
    differential = harris_combined - trump_combined
    favored_candidate = "Harris" if differential > 0 else "Trump"

    print(f"\nDifferential: {differential:.2f}% favoring {favored_candidate}")

    return {
        'period': f"{period_value} {period_type}",
        'harris_polling': polling_metrics['Kamala Harris'][0],
        'trump_polling': polling_metrics['Donald Trump'][0],
        'harris_fav': harris_fav,
        'trump_fav': trump_fav,
        'harris_combined': harris_combined,
        'trump_combined': trump_combined,
        'harris_moe': polling_metrics['Kamala Harris'][1],
        'trump_moe': polling_metrics['Donald Trump'][1],
        'oob_variance': oob_variance,
        'message': None
    }

def output_results(row: Dict[str, Any]):
    """
    Outputs the results for a period to the console.
    """
    period = row['period']
    harris_score = row['harris_combined']
    trump_score = row['trump_combined']
    harris_margin = row['harris_moe']
    trump_margin = row['trump_moe']
    oob_variance = row['oob_variance']
    message = row.get('message')

    if message:
        logging.warning(f"{period:<4} {message}")
        return

    differential = harris_score - trump_score
    favored_candidate = "Harris" if differential > 0 else "Trump"
    color_code = config.START_COLOR  # Adjust as needed
    print(f"\033[38;5;{color_code}m{period:>4} H∙{harris_score:5.2f}%±{harris_margin:.2f} "
          f"T∙{trump_score:5.2f}%±{trump_margin:.2f} {differential:+5.2f} "
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
