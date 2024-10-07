import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from io import StringIO
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from states import get_state_data

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

# Constants for the weighting calculations
partisan_weight = {True: 0.1, False: 1}
population_weights = {
    'lv': 1.0, 'rv': 0.6666666666666666, 'v': 0.5,
    'a': 0.3333333333333333, 'all': 0.3333333333333333
}

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
    st.write("Preprocessing data...")
    st.write("Columns in the DataFrame:", df.columns)
    
    df['created_at'] = pd.to_datetime(df['created_at'], format='%m/%d/%y %H:%M', errors='coerce')
    df = df.dropna(subset=['created_at'])
    if start_period:
        df = df[df['created_at'] >= start_period]
    
    # Normalizing numeric_grade
    df['numeric_grade'] = pd.to_numeric(df['numeric_grade'], errors='coerce').fillna(0)
    max_numeric_grade = df['numeric_grade'].max()
    df['normalized_numeric_grade'] = df['numeric_grade'] / max_numeric_grade

    # Inverting and normalizing pollscore
    df['pollscore'] = pd.to_numeric(df['pollscore'], errors='coerce')
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

    df['sample_size'] = pd.to_numeric(df['sample_size'], errors='coerce').fillna(0)
    min_sample_size, max_sample_size = df['sample_size'].min(), df['sample_size'].max()
    df['sample_size_weight'] = (df['sample_size'] - min_sample_size) / (max_sample_size - min_sample_size)

    state_data = get_state_data()
    df['state_rank'] = df['state'].apply(lambda x: state_data.get(x, 1))

    if 'population' in df.columns:
        df['population'] = df['population'].str.lower()
        df['population_weight'] = df['population'].map(lambda x: population_weights.get(x, 1))
    else:
        st.warning("'population' column not found. Using default population weight.")
        df['population_weight'] = 1

    # Apply time decay weight
    reference_date = pd.Timestamp.now()
    days_old = (reference_date - df['created_at']).dt.days
    df['time_decay_weight'] = np.exp(-np.log(DECAY_RATE) * days_old / HALF_LIFE_DAYS)

    st.write("Preprocessed data shape:", df.shape)
    return df

def calculate_polling_metrics(df, candidate_names):
    df = df.copy()
    df['pct'] = df['pct'].apply(lambda x: x if x > 1 else x * 100)

    df['is_partisan'] = df['partisan'].notna() & df['partisan'].ne('')
    df['partisan_weight'] = df['is_partisan'].map(partisan_weight)

    list_weights = np.array([
        df['time_decay_weight'],
        df['sample_size_weight'],
        df['normalized_numeric_grade'],
        df['normalized_transparency_score'],
        df['population_weight'],
        df['partisan_weight'],
        df['state_rank'],
    ])
    df['combined_weight'] = np.prod(list_weights, axis=0)

    # Reset index before grouping
    df = df.reset_index(drop=True)

    weighted_sums = df.groupby('candidate_name').apply(lambda x: (x['combined_weight'] * x['pct']).sum()).fillna(0)
    total_weights = df.groupby('candidate_name')['combined_weight'].sum().fillna(0)
    
    # Ensure we're working with Series, not DataFrames
    weighted_sums = pd.Series(weighted_sums)
    total_weights = pd.Series(total_weights)
    
    weighted_averages = (weighted_sums / total_weights).fillna(0)

    moes = {}
    for candidate in candidate_names:
        candidate_df = df[df['candidate_name'] == candidate]
        if not candidate_df.empty:
            moe = calculate_moe(candidate_df)
            moes[candidate] = moe

    return {candidate: (weighted_averages.get(candidate, 0), moes.get(candidate, 0)) for candidate in candidate_names}

    df = df.copy()
    df['pct'] = df['pct'].apply(lambda x: x if x > 1 else x * 100)

    df['is_partisan'] = df['partisan'].notna() & df['partisan'].ne('')
    df['partisan_weight'] = df['is_partisan'].map(partisan_weight)

    list_weights = np.array([
        df['time_decay_weight'],
        df['sample_size_weight'],
        df['normalized_numeric_grade'],
        df['normalized_transparency_score'],
        df['population_weight'],
        df['partisan_weight'],
        df['state_rank'],
    ])
    df['combined_weight'] = np.prod(list_weights, axis=0)

    weighted_sums = df.groupby('candidate_name').apply(lambda x: (x['combined_weight'] * x['pct']).sum()).fillna(0)
    total_weights = df.groupby('candidate_name')['combined_weight'].sum().fillna(0)
    weighted_averages = (weighted_sums / total_weights).fillna(0)

    moes = {}
    for candidate in candidate_names:
        candidate_df = df[df['candidate_name'] == candidate]
        if not candidate_df.empty:
            moe = calculate_moe(candidate_df)
            moes[candidate] = moe

    return {candidate: (weighted_averages.get(candidate, 0), moes.get(candidate, 0)) for candidate in candidate_names}

def calculate_favorability_differential(df, candidate_names):
    df = df.copy()
    df['favorable'] = df['favorable'].apply(lambda x: x if x > 1 else x * 100)

    list_weights = np.array([
        df['normalized_numeric_grade'],
        df['normalized_pollscore'],
        df['normalized_transparency_score']
    ])
    df['combined_weight'] = np.prod(list_weights, axis=0)

    # Reset index before grouping
    df = df.reset_index(drop=True)

    weighted_sums = df.groupby('politician').apply(lambda x: (x['combined_weight'] * x['favorable']).sum()).fillna(0)
    total_weights = df.groupby('politician')['combined_weight'].sum().fillna(0)
    
    # Ensure we're working with Series, not DataFrames
    weighted_sums = pd.Series(weighted_sums)
    total_weights = pd.Series(total_weights)
    
    weighted_averages = (weighted_sums / total_weights).fillna(0)

    return {candidate: weighted_averages.get(candidate, 0) for candidate in candidate_names}

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

# Function to create a line chart
def create_line_chart(df, y_columns, title):
    st.line_chart(df.set_index('period')[y_columns], use_container_width=True)
    st.write(title)

# Main Streamlit app
st.set_page_config(page_title="Election Polling Analysis", layout="wide")
st.title("Election Polling Analysis")

# Download and preprocess data
st.write("Downloading polling data...")
polling_df = download_csv_data(POLLING_URL)
st.write("Polling data shape:", polling_df.shape)

st.write("Downloading favorability data...")
favorability_df = download_csv_data(FAVORABILITY_URL)
st.write("Favorability data shape:", favorability_df.shape)

polling_df = preprocess_data(polling_df)
favorability_df = preprocess_data(favorability_df)

# Analyze data for different time periods
periods = [
    (1, 'days'), (3, 'days'), (7, 'days'), (14, 'days'), (21, 'days'),
    (1, 'months'), (3, 'months'), (6, 'months'), (12, 'months')
]

results = []
# In the main part of your script where you process different time periods

for period_value, period_type in periods:
    st.write(f"Processing period: {period_value} {period_type}")
    if period_type == 'months':
        start_period = datetime.now() - relativedelta(months=period_value)
    else:  # 'days'
        start_period = datetime.now() - timedelta(days=period_value)
    
    period_polling_df = polling_df[polling_df['created_at'] >= start_period]
    period_favorability_df = favorability_df[favorability_df['created_at'] >= start_period]
    
    st.write(f"Polling data for period: {period_polling_df.shape}")
    st.write(f"Favorability data for period: {period_favorability_df.shape}")
    
    try:
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
    except Exception as e:
        st.error(f"Error processing data for period {period_value} {period_type}: {str(e)}")
        st.write("Skipping this period due to error.")
        continue
#####
    st.write(f"Processing period: {period_value} {period_type}")
    if period_type == 'months':
        start_period = datetime.now() - relativedelta(months=period_value)
    else:  # 'days'
        start_period = datetime.now() - timedelta(days=period_value)
    
    period_polling_df = polling_df[polling_df['created_at'] >= start_period]
    period_favorability_df = favorability_df[favorability_df['created_at'] >= start_period]
    
    st.write(f"Polling data for period: {period_polling_df.shape}")
    st.write(f"Favorability data for period: {period_favorability_df.shape}")
    
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
# 
results_df = pd.DataFrame(results)
st.write("Results DataFrame:")
st.write(results_df)
st.write(results_df.dtypes)

# Display charts
st.header("Polling Results Over Time")
create_line_chart(results_df, ['harris_poll', 'trump_poll'], "Polling Results Over Time")

st.header("Favorability Over Time")
create_line_chart(results_df, ['harris_fav', 'trump_fav'], "Favorability Over Time")

st.header("Combined Analysis Over Time")
create_line_chart(results_df, ['harris_poll', 'trump_poll'], "Combined Analysis Over Time")

# Display polling results with error bars
st.header("Polling Results with Error Bars")
st.error_bar_chart(
    results_df.set_index('period')[['harris_poll', 'trump_poll']],
    error_data=results_df.set_index('period')[['harris_moe', 'trump_moe']],
    use_container_width=True
)
st.write("Polling Results with Error Bars")

# Display raw data
st.header("Raw Data")
st.dataframe(results_df)

# Add OOB Random Forest analysis
st.header("Out-of-Bag (OOB) Random Forest Analysis")

features_columns = ['normalized_numeric_grade', 'normalized_pollscore', 'normalized_transparency_score', 'sample_size_weight', 'state_rank', 'population_weight', 'time_decay_weight']
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

# Streamlit native bar chart
st.bar_chart(feature_importance_df.set_index('feature')['importance'])

# Altair chart for more detailed visualization
feature_chart = alt.Chart(feature_importance_df).mark_bar().encode(
    x='importance:Q',
    y=alt.Y('feature:N', sort='-x'),
    tooltip=['feature', 'importance']
).properties(width=600, height=300, title="Feature Importance in Random Forest Model")

st.altair_chart(feature_chart, use_container_width=True)

# Add explanations for feature importance
st.write("Feature Importance Explanation:")
st.write("- The chart above shows the relative importance of each feature in the Random Forest model.")
st.write("- Features with higher importance have a greater impact on the model's predictions.")
st.write("- This can help identify which factors are most influential in the polling and favorability analysis.")

# You can add more detailed explanations for each feature if desired
for feature, importance in feature_importance_df.itertuples(index=False):
    st.write(f"- {feature}: {importance:.4f}")

# Additional context and interpretation
st.subheader("Interpretation of Results")
st.write("""
The analysis presented above combines polling data, favorability ratings, and various weighting factors to provide 
a comprehensive view of the current state of the presidential race between Kamala Harris and Donald Trump.

Key points to consider:
1. The polling results show the estimated support for each candidate over different time periods.
2. The favorability ratings indicate the public's general sentiment towards each candidate.
3. The combined analysis integrates both polling and favorability data for a more nuanced perspective.
4. The feature importance chart highlights which factors have the most significant impact on the model's predictions.

It's important to note that this analysis is based on historical data and current trends. Political landscapes can 
change rapidly, and unforeseen events can significantly impact public opinion. Always consider this analysis as 
one of many tools for understanding the political climate, rather than a definitive prediction of election outcomes.
""")

# Disclaimer
st.sidebar.header("Disclaimer")
st.sidebar.write("""
This analysis is for educational and informational purposes only. It does not constitute an official election forecast 
or prediction. The results presented here are based on publicly available data and should be interpreted with caution.
Always refer to official sources and professional pollsters for the most accurate and up-to-date information on 
election-related matters.
""")

# Data sources and methodology
st.sidebar.header("Data Sources and Methodology")
st.sidebar.write("""
- Polling data: FiveThirtyEight (https://projects.fivethirtyeight.com/)
- Methodology: Weighted average of polls, adjusted for poll quality, recency, sample size, and other factors.
- Random Forest model used for feature importance analysis.
- Time decay factor applied to give more weight to recent polls.
""")

# Allow users to download the results
st.download_button(
    label="Download Results as CSV",
    data=results_df.to_csv(index=False).encode('utf-8'),
    file_name="election_polling_analysis_results.csv",
    mime="text/csv",
)

# Footer
st.markdown("---")
st.write("Developed by [Your Name/Organization]. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))