# app.py

import os
import json
import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from analysis import get_analysis_results
from analysis import load_invalid_pollsters
import config
from config import *

# Constants imported from config.py
TRUMP_COLOR_DARK = config.TRUMP_COLOR_DARK
TRUMP_COLOR = config.TRUMP_COLOR
TRUMP_COLOR_LIGHT = config.TRUMP_COLOR_LIGHT
HARRIS_COLOR_DARK = config.HARRIS_COLOR_DARK
HARRIS_COLOR = config.HARRIS_COLOR
HARRIS_COLOR_LIGHT = config.HARRIS_COLOR_LIGHT
PERIOD_ORDER = config.PERIOD_ORDER
CANDIDATE_NAMES = config.CANDIDATE_NAMES
POLLING_URL = config.POLLING_URL
FAVORABILITY_URL = config.FAVORABILITY_URL

# Constants for caching
DATA_DIR = "data"
CACHED_DATA_FILE = os.path.join(DATA_DIR, "sufficient_data.csv")
CACHED_CONFIG_FILE = os.path.join(DATA_DIR, "config.json")
CACHED_RESULTS_FILE = os.path.join(DATA_DIR, "results_df.csv")

def ensure_data_dir():
    """Ensure the data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_cached_data():
    """Load cached sufficient_data_df if it exists."""
    if os.path.exists(CACHED_DATA_FILE):
        try:
            df = pd.read_csv(CACHED_DATA_FILE)
            return df
        except Exception as e:
            st.error(f"Error loading cached data: {e}")
    return None

def save_cached_data(df):
    """Save sufficient_data_df to cache."""
    ensure_data_dir()
    try:
        df.to_csv(CACHED_DATA_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving cached data: {e}")

def load_cached_results_df():
    """Load cached results_df if it exists."""
    if os.path.exists(CACHED_RESULTS_FILE):
        try:
            df = pd.read_csv(CACHED_RESULTS_FILE)
            return df
        except Exception as e:
            st.error(f"Error loading cached results: {e}")
    return None

def save_cached_results_df(df):
    """Save results_df to cache."""
    ensure_data_dir()
    try:
        df.to_csv(CACHED_RESULTS_FILE, index=False)
    except Exception as e:
        st.error(f"Error saving cached results: {e}")

def load_cached_config():
    """Load cached configuration if it exists."""
    if os.path.exists(CACHED_CONFIG_FILE):
        try:
            with open(CACHED_CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading cached configuration: {e}")
    return None

def save_cached_config(config_dict):
    """Save configuration to cache."""
    ensure_data_dir()
    try:
        with open(CACHED_CONFIG_FILE, 'w') as f:
            json.dump(config_dict, f)
    except Exception as e:
        st.error(f"Error saving cached configuration: {e}")

def clear_config_cache():
    """Clear the cached configuration file."""
    if os.path.exists(CACHED_CONFIG_FILE):
        os.remove(CACHED_CONFIG_FILE)
        st.success("Configuration cache cleared. Default values will be used.")
    else:
        st.info("No cached configuration found.")

@st.cache_data
def preprocess_data(results_df):
    """Preprocess the data."""
    try:
        # Process the results_df to return sufficient_data_df
        sufficient_data_df = results_df[results_df['message'].isnull()]
        return sufficient_data_df
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

@st.cache_data
def load_and_process_data(config_vars, force_refresh=False):
    """
    Loads and processes data using the analysis module with user-defined configuration.
    Utilizes caching to improve performance on subsequent runs.

    Args:
        config_vars (dict): Dictionary containing user-defined configuration variables.
        force_refresh (bool): If True, forces data to be reprocessed even if cached data exists.

    Returns:
        tuple: (sufficient_data_df, results_df)
    """
    cached_data = load_cached_data()
    cached_results = load_cached_results_df()
    cached_config = load_cached_config()

    if not force_refresh and cached_data is not None and cached_config == config_vars:
        st.info("Using cached data.")
        sufficient_data_df = cached_data
        results_df = cached_results
        return sufficient_data_df, results_df

    try:
        # Update config with user-defined values
        for key, value in config_vars.items():
            setattr(config, key, value)
        
        invalid_pollsters = load_invalid_pollsters() if config.PURGE_POLLS else set()
        results_df = get_analysis_results(invalid_pollsters)
        sufficient_data_df = preprocess_data(results_df)
        
        save_cached_data(sufficient_data_df)
        save_cached_results_df(results_df)
        save_cached_config(config_vars)
        
        return sufficient_data_df, results_df
    except Exception as e:
        st.error(f"An error occurred while processing data: {e}")
        st.stop()

def create_line_chart(df: pd.DataFrame, y_columns: list, title: str):
    """
    Creates a line chart using Altair with the formatting from version 1.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        y_columns (list): List of column names to plot on the y-axis.
        title (str): Title of the chart.
    """
    df = df.dropna(subset=y_columns, how='all')
    df_melted = df.melt(
        id_vars=['period'],
        value_vars=y_columns,
        var_name='candidate',
        value_name='value'
    )

    if df_melted.empty:
        st.warning(f"No data available for {title}.")
        return

    y_min = df_melted['value'].min() - 0.5
    y_max = df_melted['value'].max() + 0.5

    # Create a color mapping dictionary
    color_mapping = {
        'harris_polling': HARRIS_COLOR,
        'harris_fav': HARRIS_COLOR_LIGHT,
        'harris_combined': HARRIS_COLOR_DARK,
        'trump_polling': TRUMP_COLOR,
        'trump_fav': TRUMP_COLOR_LIGHT,
        'trump_combined': TRUMP_COLOR_DARK
    }

    # Create the color scale using the mapping
    color_scale = alt.Scale(
        domain=y_columns,
        range=[color_mapping[col] for col in y_columns]
    )

    chart = alt.Chart(df_melted).mark_line(point=True).encode(
        x=alt.X('period:N', sort=PERIOD_ORDER, title='Period'),
        y=alt.Y('value:Q', scale=alt.Scale(domain=[y_min, y_max]), title='Percentage'),
        color=alt.Color('candidate:N', scale=color_scale)
    ).properties(
        width=800,
        height=400,
        title=title
    )

    st.altair_chart(chart, use_container_width=True)

def create_differential_bar_chart(df: pd.DataFrame):
    """
    Creates a differential bar chart using Altair with the formatting from version 1.
    Includes OOB variance in the tooltip and adds a simple key at the bottom.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
    """
    df = df.dropna(subset=['harris_combined', 'trump_combined'])
    df['differential'] = df['harris_combined'] - df['trump_combined']

    if df.empty:
        st.warning("No data available for Differential Analysis.")
        return

    max_abs_diff = max(abs(df['differential'].min()), abs(df['differential'].max()))
    max_moe = max(df['harris_moe'].max(), df['trump_moe'].max())
    y_range = max(max_abs_diff, max_moe) + 0.1
    y_min, y_max = -y_range, y_range

    base = alt.Chart(df).encode(
        x=alt.X('period:N', sort=PERIOD_ORDER, title='Period')
    )

    bars = base.mark_bar(size=4).encode(
        y=alt.Y(
            'differential:Q', 
            title='Trump            Harris', 
            scale=alt.Scale(domain=[y_min, y_max])
        ),
        color=alt.condition(
            alt.datum.differential > 0,
            alt.value(HARRIS_COLOR),
            alt.value(TRUMP_COLOR)
        ),
        tooltip=[
            alt.Tooltip('differential:Q', format='+.2f', title='Differential'),
            alt.Tooltip('harris_combined:Q', format='.2f', title='Harris'),
            alt.Tooltip('trump_combined:Q', format='.2f', title='Trump'),
            alt.Tooltip('oob_variance:Q', format='.2f', title='OOB Variance')
        ]
    )

    trump_moe_area = base.mark_area(
        opacity=0.25,
        color=TRUMP_COLOR_LIGHT
    ).encode(
        y=alt.Y('zero:Q'),
        y2=alt.Y2('low:Q')
    ).transform_calculate(
        oob = '(datum.differential * (datum.oob_variance/100)) /datum.trump_moe',
        dif = '(datum.differential / datum.trump_moe)'
    ).transform_calculate(
        zero='(datum.trump_moe)',
        low='(datum.trump_moe + datum.oob)*-1'
    )

    harris_moe_area = base.mark_area(
        opacity=0.25,
        color=HARRIS_COLOR_LIGHT
    ).encode(
        y=alt.Y('zero:Q'),
        y2=alt.Y2('high:Q')
    ).transform_calculate(
        oob = '(datum.differential * (datum.oob_variance/100)) /datum.harris_moe',
        dif = '(datum.differential / datum.harris_moe)'
    ).transform_calculate(
        zero='(datum.harris_moe)*-1',
        high='(datum.harris_moe + datum.oob)'
    )

    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='#666', 
        strokeWidth=1,
        strokeDash=[10, 5]
    ).encode(y='y')

    text_labels = base.mark_text(
        align='center',
        baseline='middle',
        dy=alt.expr('datum.differential > 0 ? -15 : 15'),
        fontSize=20
    ).encode(
        y=alt.Y('differential:Q'),
        text=alt.Text('differential:Q', format='+.2f'),
        color=alt.condition(
            alt.datum.differential > 0,
            alt.value(HARRIS_COLOR),
            alt.value(TRUMP_COLOR)
        )
    )

    # Create a DataFrame for the key
    key_data = pd.DataFrame({
        'label': [
            'Harris Lead', 'Harris Uncertainty',
            'Trump Lead', 'Trump Uncertainty',
            'Margin of Error'
            ],
        'color': [
            HARRIS_COLOR, HARRIS_COLOR_LIGHT,
            TRUMP_COLOR, TRUMP_COLOR_LIGHT,
            '#7D7A82'
            ],
        'x': [0, 0, 1, 1, 2],
        'y': [0, 1, 0, 1, 0]
    })

    # Create the key chart
    key_chart = alt.Chart(key_data).mark_circle(size=60).encode(
        x=alt.X('x:O', axis=None, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('y:O', axis=None, scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('color:N', scale=None)
    ).properties(width=800, height=30)

    # Create labels for the key
    key_labels = alt.Chart(key_data).mark_text(
        align='left',
        baseline='middle',
        dx=15,
        fontSize=12
    ).encode(
        x=alt.X('x:O', axis=None, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('y:O', axis=None, scale=alt.Scale(domain=[0, 1])),
        text='label'
    ).properties(width=800, height=30)

    # Combine the main chart with the key
    final_chart = alt.vconcat(
        alt.layer(
            trump_moe_area,
            harris_moe_area,
            bars,
            zero_line,
            text_labels
        ).properties(
            title="Differential Between Harris and Trump Over Time",
            width=800,
            height=400
        ),
        (key_chart + key_labels).properties()
    )

    st.altair_chart(final_chart, use_container_width=True)

def configuration_form():
    with st.sidebar:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.header("Polling Configuration")
        with col2:
            if st.button("⟲", help="Reset to default configuration"):
                clear_config_cache()
                st.rerun()
        
        st.markdown("<sup>Adjust the configuration weights for the polling analysis.</sup>", unsafe_allow_html=True)
        with st.form("config_form"):
            favorability_weight = st.slider("Favorability Weight", 0.01, 1.0, float(config.FAVORABILITY_WEIGHT), 0.01)
            heavy_weight = st.checkbox("Heavy Weight", config.HEAVY_WEIGHT)
            st.markdown("<sup>Check for multiplicative, uncheck for additive.</sup>", unsafe_allow_html=True)
            
            purge_polls = st.checkbox("Purge Polls", config.PURGE_POLLS)
            st.markdown("<sup>Check to remove pollsters who are trying to game the system.</sup>", unsafe_allow_html=True)
            
            st.subheader("Time Weight")
            half_life_days = st.number_input("Half Life in Days", 1, 365, int(config.HALF_LIFE_DAYS), 1)
            st.markdown("<sup>Time decay parameter that controls the influence of older polls.</sup>", unsafe_allow_html=True)
            decay_rate = st.slider("Decay Rate", 0.01, 2.000, float(config.DECAY_RATE), 0.1)
            st.markdown("<sup>The rate at which older polls lose influence.</sup>", unsafe_allow_html=True)
            min_samples_required = st.slider("Minimum Samples Required", 2, 12, int(config.MIN_SAMPLES_REQUIRED), 2)
            st.markdown("<sup>The minimum number of samples required to perform analysis for a period.</sup>", unsafe_allow_html=True)
            
            st.subheader("Partisan Polling Weight")
            partisan_weight_true = st.slider("Partisan Polls Weight", 0.01, 1.0, float(config.PARTISAN_WEIGHT[True]), 0.01)
            partisan_weight_false = st.slider("Non-Partisan Polls Weight", 0.01, 1.0, float(config.PARTISAN_WEIGHT[False]), 0.01)

            st.subheader("Voter Weights")
            lv_weight = st.slider("Likely Voters", 0.01, 1.0, float(config.POPULATION_WEIGHTS['lv']), 0.01)
            rv_weight = st.slider("Registered Voters", 0.01, 1.0, float(config.POPULATION_WEIGHTS['rv']), 0.01)
            v_weight = st.slider("Past Voters", 0.01, 1.0, float(config.POPULATION_WEIGHTS['v']), 0.01)
            a_weight = st.slider("Eligible Voters", 0.01, 1.0, float(config.POPULATION_WEIGHTS['a']), 0.01)
            all_weight = st.slider("All Respondents", 0.01, 1.0, float(config.POPULATION_WEIGHTS['all']), 0.01)
            
            st.subheader("Weight Multipliers")
            time_decay_weight_multiplier = st.slider("Time Decay Weight Multiplier", 0.01, 2.0, float(config.TIME_DECAY_WEIGHT_MULTIPLIER), 0.1)
            sample_size_weight_multiplier = st.slider("Sample Size Weight Multiplier", 0.01, 2.0, float(config.SAMPLE_SIZE_WEIGHT_MULTIPLIER), 0.1)
            normalized_numeric_grade_multiplier = st.slider("Numeric Grade Multiplier", 0.01, 2.0, float(config.NORMALIZED_NUMERIC_GRADE_MULTIPLIER), 0.1)
            normalized_pollscore_multiplier = st.slider("Poll Score Multiplier", 0.01, 2.0, float(config.NORMALIZED_POLLSCORE_MULTIPLIER), 0.1)
            normalized_transparency_score_multiplier = st.slider("Transparency Score Multiplier", 0.01, 2.0, float(config.NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER), 0.1)
            population_weight_multiplier = st.slider("Population Weight Multiplier", 0.01, 2.0, float(config.POPULATION_WEIGHT_MULTIPLIER), 0.1)
            partisan_weight_multiplier = st.slider("Partisan Weight Multiplier", 0.01, 2.0, float(config.PARTISAN_WEIGHT_MULTIPLIER), 0.1)
            state_rank_multiplier = st.slider("State Rank Multiplier", 0.01, 2.0, float(config.STATE_RANK_MULTIPLIER), 0.1)
            national_poll_weight = st.slider("National Poll Weight", 0.01, 2.0, float(config.NATIONAL_POLL_WEIGHT), 0.1)
            
            force_refresh = st.checkbox("Force Refresh Data", False)
            
            submitted = st.form_submit_button("Apply Changes and Run Analysis")
    
    if submitted:
        return {
            "FAVORABILITY_WEIGHT": favorability_weight,
            "HEAVY_WEIGHT": heavy_weight,
            "PURGE_POLLS": purge_polls,
            "DECAY_RATE": decay_rate,
            "HALF_LIFE_DAYS": half_life_days,
            "MIN_SAMPLES_REQUIRED": min_samples_required,
            "PARTISAN_WEIGHT": {True: partisan_weight_true, False: partisan_weight_false},
            "POPULATION_WEIGHTS": {
                'lv': lv_weight,
                'rv': rv_weight,
                'v': v_weight,
                'a': a_weight,
                'all': all_weight
            },
            "TIME_DECAY_WEIGHT_MULTIPLIER": time_decay_weight_multiplier,
            "SAMPLE_SIZE_WEIGHT_MULTIPLIER": sample_size_weight_multiplier,
            "NORMALIZED_NUMERIC_GRADE_MULTIPLIER": normalized_numeric_grade_multiplier,
            "NORMALIZED_POLLSCORE_MULTIPLIER": normalized_pollscore_multiplier,
            "NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER": normalized_transparency_score_multiplier,
            "POPULATION_WEIGHT_MULTIPLIER": population_weight_multiplier,
            "PARTISAN_WEIGHT_MULTIPLIER": partisan_weight_multiplier,
            "STATE_RANK_MULTIPLIER": state_rank_multiplier,
            "NATIONAL_POLL_WEIGHT": national_poll_weight,
            "FORCE_REFRESH": force_refresh
        }
    return None

def main():
    """
    Main function to run the Streamlit app.
    """
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Election Polling Analysis", layout="wide")
    
    # Custom CSS to style the button (place this after set_page_config)
    st.markdown("""
    <style>
        button.ef3psqc16,
        button.ef3psqc16
        div.e1nzilvr5
        p {
            font-size: 32px !important;
            line-height: 0px !important;
            border: none !important;
            background: none !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Election Polling Analysis")
    
    # Configuration form
    new_config = configuration_form()
    force_refresh = False  # default

    if new_config:
        # Apply new configuration
        for key, value in new_config.items():
            if key != "FORCE_REFRESH":
                setattr(config, key, value)
        st.success("Configuration updated. Reprocessing data, this might take a while...")
        force_refresh = new_config.get("FORCE_REFRESH", False)
        if force_refresh:
            st.info("Forcing data refresh...")
            # Clear the cache to force a refresh of the data
            st.cache_data.clear()

    # Load and process data
    config_vars = {
        key: getattr(config, key)
        for key in dir(config)
        if key.isupper() and not key.startswith("__")
    }

    sufficient_data_df, results_df = load_and_process_data(config_vars, force_refresh)

    if not sufficient_data_df.empty:
        st.header("Differential Analysis")
        create_differential_bar_chart(sufficient_data_df)

        st.header("Combined Analysis Over Time")
        create_line_chart(sufficient_data_df, [
            'harris_polling',
            'harris_fav',
            'harris_combined',
            'trump_polling',
            'trump_fav',
            'trump_combined'
        ], "Combined Analysis Over Time")

        st.header("Polling Results Over Time")
        create_line_chart(sufficient_data_df, ['harris_polling', 'trump_polling'], "Polling Results Over Time")

        st.header("Favorability Over Time")
        if (
            'harris_fav' in sufficient_data_df.columns and
            'trump_fav' in sufficient_data_df.columns and
            (sufficient_data_df['harris_fav'].notnull().any() or sufficient_data_df['trump_fav'].notnull().any())
        ):
            create_line_chart(
                sufficient_data_df,
                ['harris_fav', 'trump_fav'],
                "Favorability Over Time"
            )
        else:
            st.warning("No favorability data available.")

        st.header("Analysis Results (Table)")
        st.write(sufficient_data_df)

        st.header("Analysis Results (JSON)")
        st.json(sufficient_data_df.to_json(orient='records'))
        
    else:
        st.error("No data available.")

    # Display messages for periods with insufficient data
    if results_df is not None:
        insufficient_data_periods = results_df[results_df['message'].notnull()]
        if not insufficient_data_periods.empty:
            st.write("The following periods have insufficient data:")
            for _, row in insufficient_data_periods.iterrows():
                st.write(f"- {row['period']}: {row['message']}")
    else:
        st.warning("Unable to retrieve results for periods with insufficient data.")


    # Add logging to verify configuration
    st.write("Current Configuration:")
    st.json(config_vars)

    # Add download links for CSV files
    st.header("Download Raw Data")
    st.markdown(f"[Download Polling Data CSV]({POLLING_URL})")
    st.markdown(f"[Download Favorability Data CSV]({FAVORABILITY_URL})")

    # Embed and render readme.md
    st.header("Project Documentation")
    readme_path = os.path.join(os.path.dirname(__file__), 'readme.md')
    with open(readme_path, 'r') as readme_file:
        readme_content = readme_file.read()
        st.markdown(readme_content, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.write(f"Developed by Spencer Thayer. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
