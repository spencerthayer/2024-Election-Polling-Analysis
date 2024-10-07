import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from io import StringIO
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Constants
POLLING_URL = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
FAVORABILITY_URL = "https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv"
CANDIDATE_NAMES = ['Kamala Harris', 'Donald Trump']
FAVORABILITY_WEIGHT = 0.1
DECAY_RATE = 2
HALF_LIFE_DAYS = 28

# Define a custom order for the periods
period_order = [
    '1 days', '3 days', '7 days', '14 days', '21 days',
    '1 months', '3 months', '6 months', '12 months'
]

# Utility functions
@st.cache_data
def download_csv_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        csv_data = StringIO(response.content.decode('utf-8'))
        return pd.read_csv(csv_data)
    except (requests.RequestException, pd.errors.EmptyDataError, ValueError) as e:
        st.error(f"Error downloading data from {url}: {e}")
        return pd.DataFrame()

def preprocess_data(df, start_period=None):
    df['created_at'] = pd.to_datetime(df['created_at'], format='%m/%d/%y %H:%M', errors='coerce')
    df = df.dropna(subset=['created_at'])
    if start_period:
        df = df[df['created_at'] >= start_period]
    
    grade_map = {
        'A+': 1.0, 'A': 0.9, 'A-': 0.8, 'A/B': 0.75, 'B+': 0.7, 'B': 0.6, 'B-': 0.5,
        'B/C': 0.45, 'C+': 0.4, 'C': 0.3, 'C-': 0.2, 'C/D': 0.15, 'D+': 0.1, 'D': 0.05, 'D-': 0.025
    }

    # Check if 'fte_grade' column exists, if not, create a default grade
    if 'fte_grade' not in df.columns:
        print("Warning: 'fte_grade' column not found. Using default grade.")
        df['fte_grade'] = 'C'  # Or any other default grade you prefer

    df['numeric_grade'] = df['fte_grade'].map(grade_map).fillna(0.3)

    # Handle other potential missing columns
    for col in ['transparency_score', 'sample_size', 'population', 'partisan']:
        if col not in df.columns:
            print(f"Warning: '{col}' column not found. Using default values.")
            df[col] = None  # or any other appropriate default value

    df['transparency_score'] = pd.to_numeric(df['transparency_score'], errors='coerce').fillna(0)
    max_transparency = df['transparency_score'].max()
    df['normalized_transparency_score'] = df['transparency_score'] / max_transparency if max_transparency > 0 else 0

    df['sample_size'] = pd.to_numeric(df['sample_size'], errors='coerce').fillna(0)
    min_sample, max_sample = df['sample_size'].min(), df['sample_size'].max()
    df['sample_size_weight'] = (df['sample_size'] - min_sample) / (max_sample - min_sample) if max_sample > min_sample else 0

    population_weights = {'lv': 1.0, 'rv': 0.67, 'v': 0.5, 'a': 0.33, 'all': 0.33}
    df['population_weight'] = df['population'].str.lower().map(population_weights).fillna(1)

    df['partisan_weight'] = (~df['partisan'].isna() & (df['partisan'] != '')).map({True: 0.1, False: 1})

    df['time_decay_weight'] = np.exp(-np.log(DECAY_RATE) * (datetime.now() - df['created_at']).dt.days / HALF_LIFE_DAYS)

    return df
    
def calculate_polling_metrics(df, candidate_names):
    metrics = {}
    for candidate in candidate_names:
        candidate_df = df[df['candidate_name'] == candidate]
        if not candidate_df.empty:
            weights = candidate_df['numeric_grade'] * candidate_df['normalized_transparency_score'] * \
                      candidate_df['sample_size_weight'] * candidate_df['population_weight'] * \
                      candidate_df['partisan_weight'] * candidate_df['time_decay_weight']
            weighted_sum = (weights * candidate_df['pct']).sum()
            total_weight = weights.sum()
            weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
            moe = calculate_moe(candidate_df)
            metrics[candidate] = (weighted_avg, moe)
    return metrics

def calculate_favorability_differential(df, candidate_names):
    differentials = {}
    for candidate in candidate_names:
        candidate_df = df[df['politician'] == candidate]
        if not candidate_df.empty:
            weights = candidate_df['numeric_grade'] * candidate_df['normalized_transparency_score'] * \
                      candidate_df['sample_size_weight'] * candidate_df['population_weight'] * \
                      candidate_df['time_decay_weight']
            weighted_sum = (weights * candidate_df['favorable']).sum()
            total_weight = weights.sum()
            weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
            differentials[candidate] = weighted_avg
    return differentials

def calculate_moe(df, confidence_level=0.95):
    n = df['sample_size'].mean()
    p = df['pct'].mean() / 100
    z = norm.ppf((1 + confidence_level) / 2)
    moe = z * np.sqrt((p * (1 - p)) / n)
    return moe * 100  # Convert to percentage

def combine_analysis(polling_metrics, favorability_differential, favorability_weight):
    combined_metrics = {}
    for candidate in polling_metrics.keys():
        poll_result, moe = polling_metrics[candidate]
        fav_diff = favorability_differential.get(candidate, 0)
        combined_result = poll_result * (1 - favorability_weight) + fav_diff * favorability_weight
        combined_metrics[candidate] = (combined_result, moe)
    return combined_metrics

def create_chart(df, y_columns, title):
    # Ensure the period column is properly formatted
    df['period'] = pd.Categorical(df['period'], categories=period_order, ordered=True)
    df = df.sort_values('period')

    base = alt.Chart(df).encode(
        x=alt.X('period:O', sort=period_order),
        y=alt.Y('value:Q', scale=alt.Scale(domain=[30, 70]))
    )
    
    lines = base.mark_line().encode(
        color=alt.Color('candidate:N', scale=alt.Scale(domain=CANDIDATE_NAMES, range=['blue', 'red']))
    ).transform_fold(y_columns, as_=['candidate', 'value'])
    
    points = base.mark_point().encode(
        color=alt.Color('candidate:N', scale=alt.Scale(domain=CANDIDATE_NAMES, range=['blue', 'red']))
    ).transform_fold(y_columns, as_=['candidate', 'value'])
    
    if 'moe' in df.columns:
        error_bars = base.mark_errorbar(extent='ci').encode(
            y='low:Q',
            y2='high:Q',
            color=alt.Color('candidate:N', scale=alt.Scale(domain=CANDIDATE_NAMES, range=['blue', 'red']))
        ).transform_fold(
            y_columns,
            as_=['candidate', 'value']
        ).transform_calculate(
            low='datum.value - datum.moe',
            high='datum.value + datum.moe'
        )
        chart = (lines + points + error_bars)
    else:
        chart = (lines + points)
    
    return chart.properties(width=600, height=400, title=title).interactive()

def impute_data(X):
    imputer = SimpleImputer(strategy='median')
    for col in range(X.shape[1]):
        if np.any(~np.isnan(X[:, col])):
            X[:, col] = imputer.fit_transform(X[:, col].reshape(-1, 1)).ravel()
    return X

def _get_unsampled_indices(tree, n_samples):
    unsampled_mask = np.ones(n_samples, dtype=bool)
    unsampled_mask[tree.tree_.feature[tree.tree_.feature >= 0]] = False
    return np.arange(n_samples)[unsampled_mask]

# Main Streamlit app
st.set_page_config(page_title="Election Polling Analysis", layout="wide")
st.title("Election Polling Analysis")

# Download and preprocess data
polling_df = download_csv_data(POLLING_URL)
favorability_df = download_csv_data(FAVORABILITY_URL)

polling_df = preprocess_data(polling_df)
favorability_df = preprocess_data(favorability_df)

# Analyze data for different time periods
periods = [
    (1, 'days'), (3, 'days'), (7, 'days'), (14, 'days'), (21, 'days'),
    (1, 'months'), (3, 'months'), (6, 'months'), (12, 'months')
]

results = []
for period_value, period_type in periods:
    if period_type == 'months':
        start_period = datetime.now() - relativedelta(months=period_value)
    else:  # 'days'
        start_period = datetime.now() - timedelta(days=period_value)
    
    period_polling_df = polling_df[polling_df['created_at'] >= start_period]
    period_favorability_df = favorability_df[favorability_df['created_at'] >= start_period]
    
    polling_metrics = calculate_polling_metrics(period_polling_df, CANDIDATE_NAMES)
    favorability_differential = calculate_favorability_differential(period_favorability_df, CANDIDATE_NAMES)
    combined_results = combine_analysis(polling_metrics, favorability_differential, FAVORABILITY_WEIGHT)
    
    results.append({
        'period': f"{period_value} {period_type}",
        'harris_poll': combined_results['Kamala Harris'][0] if 'Kamala Harris' in combined_results else 0,
        'harris_moe': combined_results['Kamala Harris'][1] if 'Kamala Harris' in combined_results else 0,
        'trump_poll': combined_results['Donald Trump'][0] if 'Donald Trump' in combined_results else 0,
        'trump_moe': combined_results['Donald Trump'][1] if 'Donald Trump' in combined_results else 0,
        'harris_fav': favorability_differential.get('Kamala Harris', 0),
        'trump_fav': favorability_differential.get('Donald Trump', 0),
    })

results_df = pd.DataFrame(results)

# Display charts
st.header("Polling Results Over Time")
polling_chart = create_chart(results_df, ['harris_poll', 'trump_poll'], 'Polling Results Over Time')
st.altair_chart(polling_chart, use_container_width=True)

st.header("Favorability Over Time")
favorability_chart = create_chart(results_df, ['harris_fav', 'trump_fav'], 'Favorability Over Time')
st.altair_chart(favorability_chart, use_container_width=True)

st.header("Combined Analysis Over Time")
combined_chart = create_chart(results_df, ['harris_poll', 'trump_poll'], 'Combined Analysis Over Time')
st.altair_chart(combined_chart, use_container_width=True)

# Display raw data
st.header("Raw Data")
st.dataframe(results_df)

# Add OOB Random Forest analysis
st.header("Out-of-Bag (OOB) Random Forest Analysis")

features_columns = ['numeric_grade', 'normalized_transparency_score', 'sample_size_weight', 'population_weight', 'partisan_weight', 'time_decay_weight']
X = polling_df[features_columns].values
y = polling_df['pct'].values

pipeline = Pipeline(steps=[
    ('imputer', FunctionTransformer(impute_data)), 
    ('model', RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42, bootstrap=True))
])

pipeline.fit(X, y)

oob_predictions = np.zeros(y.shape)
oob_sample_counts = np.zeros(X.shape[0], dtype=int)

for tree in pipeline.named_steps['model'].estimators_:
    unsampled_indices = _get_unsampled_indices(tree, X.shape[0])
    if len(unsampled_indices) > 0:
        oob_predictions[unsampled_indices] += tree.predict(impute_data(X[unsampled_indices]))
        oob_sample_counts[unsampled_indices] += 1

epsilon = np.finfo(float).eps
oob_predictions /= (oob_sample_counts + epsilon)

oob_variance = np.var(y - oob_predictions)

st.write(f"OOB Variance: {oob_variance:.4f}")

# Feature importance
feature_importance = pipeline.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({'feature': features_columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

st.subheader("Feature Importance")
feature_chart = alt.Chart(feature_importance_df).mark_bar().encode(
    x='importance:Q',
    y=alt.Y('feature:N', sort='-x'),
    tooltip=['feature', 'importance']
).properties(width=600, height=300, title="Feature Importance in Random Forest Model")

st.altair_chart(feature_chart, use_container_width=True)