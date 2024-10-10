# app.py

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from analysis import get_analysis_results
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

# Update the load_and_process_data function to use st.cache_data and accept config_vars
@st.cache_data
def load_and_process_data(config_vars) -> pd.DataFrame:
    """
    Loads and processes data using the analysis module with user-defined configuration.

    Args:
        config_vars (dict): Dictionary containing user-defined configuration variables.

    Returns:
        pd.DataFrame: DataFrame containing analysis results.
    """
    try:
        # Update config with user-defined values
        for key, value in config_vars.items():
            setattr(config, key, value)
        
        results_df = get_analysis_results()
        return results_df
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
    y_range = max(max_abs_diff, max_moe) + 0
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
            alt.Tooltip('trump_combined:Q', format='.2f', title='Trump')
        ]
    )

    trump_moe_area = base.mark_area(
        opacity=0.25,
        color=TRUMP_COLOR_LIGHT
    ).encode(
        y=alt.Y('zero:Q'),
        y2=alt.Y2('low:Q')
    ).transform_calculate(
        zero='datum.trump_moe*((100-(datum.differential*10))*.01)*1',
        low='datum.trump_moe*-1'
    )

    harris_moe_area = base.mark_area(
        opacity=0.25,
        color=HARRIS_COLOR_LIGHT
    ).encode(
        y=alt.Y('zero:Q'),
        y2=alt.Y2('high:Q')
    ).transform_calculate(
        zero='datum.harris_moe*((100-(datum.differential*10))*.01)*-1',
        high='datum.harris_moe'
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

    final_chart = alt.layer(
        trump_moe_area,
        harris_moe_area,
        bars,
        zero_line,
        text_labels
    ).properties(
        title="Differential Between Harris and Trump Over Time",
        width=800,
        height=400
    )

    st.altair_chart(final_chart, use_container_width=True)

def configuration_form():
    with st.sidebar:
        st.header("Polling Configuration")
        st.html("<sup>Adjust the configuration weights for the polling analysis.</sup>")
        with st.form("config_form"):
            favorability_weight = st.slider("Favorability Weight", 0.0, 1.0, float(FAVORABILITY_WEIGHT), 0.01)
            # favorability_weight = st.number_input(
            #     "Favorability Weight", 
            #     min_value=0.000, 
            #     max_value=1.000, 
            #     value=float(FAVORABILITY_WEIGHT), 
            #     step=0.001,
            #     format="%.3f"
            # )
            heavy_weight = st.checkbox("Heavy Weight", HEAVY_WEIGHT)
            st.html("<sup>Check for multiplicative, uncheck for additive.</sup>")
            
            st.subheader("Time Weight")
            half_life_days = st.number_input("Half Life in Days", 1, 365, int(HALF_LIFE_DAYS), 1)
            st.html("<sup>Time decay parameter that controls the inlfuence of older polls.</sup>")
            decay_rate = st.number_input(
                "Decay Rate", 
                min_value=0.001, 
                max_value=10.000, 
                value=float(DECAY_RATE), 
                step=0.001,
                format="%.3f"
            )
            st.html("<sup>The rate at which older polls lose influence.</sup>")
            min_samples_required = st.number_input("Minimum Samples Required", 1, 100, int(MIN_SAMPLES_REQUIRED), 1)
            st.html("<sup>The minimum number of samples required to perform analysis for a period.</sup>")
            
            st.subheader("Partisan Polling Weight")
            partisan_weight_true = st.number_input(
                "Partisan Polls Weight", 
                min_value=0.000, 
                max_value=1.000, 
                value=float(PARTISAN_WEIGHT[True]), 
                step=0.001,
                format="%.3f"
            )
            partisan_weight_false = st.number_input(
                "Non-Partisan Polls Weight", 
                min_value=0.000, 
                max_value=1.000, 
                value=float(PARTISAN_WEIGHT[False]), 
                step=0.001,
                format="%.3f"
            )
            
            st.subheader("Voter Weights")
            lv_weight = st.number_input(
                "Likely Voters", 
                min_value=0.000, 
                max_value=1.000, 
                value=float(POPULATION_WEIGHTS['lv']), 
                step=0.001,
                format="%.3f"
            )
            rv_weight = st.number_input(
                "Registered Voters", 
                min_value=0.000, 
                max_value=1.000, 
                value=float(POPULATION_WEIGHTS['rv']), 
                step=0.001,
                format="%.3f"
            )
            v_weight = st.number_input(
                "Past Vosters", 
                min_value=0.000, 
                max_value=1.000, 
                value=float(POPULATION_WEIGHTS['v']), 
                step=0.001,
                format="%.3f"
            )
            a_weight = st.number_input(
                "Eligible Voters", 
                min_value=0.000, 
                max_value=1.000, 
                value=float(POPULATION_WEIGHTS['a']), 
                step=0.001,
                format="%.3f"
            )
            all_weight = st.number_input(
                "Unlikely Voters",
                min_value=0.000, 
                max_value=1.000, 
                value=float(POPULATION_WEIGHTS['all']), 
                step=0.001,
                format="%.3f"
            )
            
            refresh_data = st.checkbox("Refresh Data", True)
            
            submitted = st.form_submit_button("Apply Changes and Run Analysis")
    
    if submitted:
        return {
            "FAVORABILITY_WEIGHT": favorability_weight,
            "HEAVY_WEIGHT": heavy_weight,
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
            "REFRESH_DATA": refresh_data
        }
    return None

def main():
    """
    Main function to run the Streamlit app.
    """
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Election Polling Analysis", layout="wide")
    st.title("Election Polling Analysis")
    
    # Configuration form
    config_vars = configuration_form()
    
    if config_vars:
        st.success("Configuration updated. Reprocessing data...")
        if config_vars.pop("REFRESH_DATA", False):
            st.info("Refreshing data from CSV URLs...")
            # Clear the cache to force a refresh of the data
            st.cache_data.clear()
    else:
        config_vars = {
            "FAVORABILITY_WEIGHT": FAVORABILITY_WEIGHT,
            "HEAVY_WEIGHT": HEAVY_WEIGHT,
            "DECAY_RATE": DECAY_RATE,
            "HALF_LIFE_DAYS": HALF_LIFE_DAYS,
            "MIN_SAMPLES_REQUIRED": MIN_SAMPLES_REQUIRED,
            "PARTISAN_WEIGHT": PARTISAN_WEIGHT,
            "POPULATION_WEIGHTS": POPULATION_WEIGHTS
        }

    # Load and process data
    results_df = load_and_process_data(config_vars)

    if not results_df.empty:
        sufficient_data_df = results_df[results_df['message'].isnull()]

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

            st.header("Analysis Results")
            st.write(sufficient_data_df)
        else:
            st.warning("No sufficient data available for the selected periods.")

        # Display messages for periods with insufficient data
        insufficient_data_periods = results_df[results_df['message'].notnull()]
        if not insufficient_data_periods.empty:
            st.write("The following periods have insufficient data:")
            for _, row in insufficient_data_periods.iterrows():
                st.write(f"- {row['period']}: {row['message']}")
    else:
        st.error("No data available.")

    # Add download links for CSV files
    st.header("Download Raw Data")
    st.markdown(f"[Download Polling Data CSV]({POLLING_URL})")
    st.markdown(f"[Download Favorability Data CSV]({FAVORABILITY_URL})")

    # Footer
    st.markdown("---")
    st.write(f"Developed by Spencer Thayer. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
if __name__ == "__main__":
    main()
