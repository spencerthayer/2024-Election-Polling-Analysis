# app.py

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
from analysis import get_analysis_results
import config

# Constants imported from config.py
TRUMP_COLOR_DARK = config.TRUMP_COLOR_DARK
TRUMP_COLOR = config.TRUMP_COLOR
TRUMP_COLOR_LIGHT = config.TRUMP_COLOR_LIGHT
HARRIS_COLOR_DARK = config.HARRIS_COLOR_DARK
HARRIS_COLOR = config.HARRIS_COLOR
HARRIS_COLOR_LIGHT = config.HARRIS_COLOR_LIGHT
PERIOD_ORDER = config.PERIOD_ORDER
CANDIDATE_NAMES = config.CANDIDATE_NAMES

@st.cache_data
def load_and_process_data() -> pd.DataFrame:
    """
    Loads and processes data using the analysis module.

    Returns:
        pd.DataFrame: DataFrame containing analysis results.
    """
    try:
        results_df = get_analysis_results()
        return results_df
    except Exception as e:
        st.error(f"An error occurred while processing data: {e}")
        st.stop()

def create_line_chart(df: pd.DataFrame, y_columns: list, title: str):
    """
    Creates a line chart using Altair.

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

    # Adjust color_scale based on the number of y_columns
    color_range = []
    for col in y_columns:
        if 'harris' in col.lower():
            if 'fav' in col.lower():
                color_range.append(HARRIS_COLOR_LIGHT)
            else:
                color_range.append(HARRIS_COLOR)
        elif 'trump' in col.lower():
            if 'fav' in col.lower():
                color_range.append(TRUMP_COLOR_LIGHT)
            else:
                color_range.append(TRUMP_COLOR)
        else:
            color_range.append('gray')  # Default color

    color_scale = alt.Scale(
        domain=y_columns,
        range=color_range
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
    Creates a differential bar chart using Altair.

    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
    """
    df = df.dropna(subset=['harris', 'trump'])
    df['differential'] = df['trump'] - df['harris']

    if df.empty:
        st.warning("No data available for Differential Analysis.")
        return

    max_abs_diff = max(abs(df['differential'].min()), abs(df['differential'].max()))
    y_range = max_abs_diff + 1
    y_min, y_max = -y_range, y_range

    base = alt.Chart(df).encode(
        x=alt.X('period:N', sort=PERIOD_ORDER, title='Period')
    )

    bars = base.mark_bar(size=30).encode(
        y=alt.Y('differential:Q',
                title='Trump Advantage (+) vs Harris Advantage (-)',
                scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.condition(
            alt.datum.differential > 0,
            alt.value(TRUMP_COLOR),
            alt.value(HARRIS_COLOR)
        ),
        tooltip=[
            alt.Tooltip('period:N', title='Period'),
            alt.Tooltip('differential:Q', title='Differential', format='+.2f')
        ]
    )

    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='black',
        strokeWidth=2
    ).encode(y='y')

    final_chart = (bars + zero_line).properties(
        title="Differential Between Trump and Harris Over Time",
        width=800,
        height=400
    )

    st.altair_chart(final_chart, use_container_width=True)

def main():
    """
    Main function to run the Streamlit app.
    """
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Election Polling Analysis", layout="wide")
    st.title("Election Polling Analysis")

    # Load and process data
    results_df = load_and_process_data()

    if not results_df.empty:
        # Filter out periods with insufficient data
        sufficient_data_df = results_df[results_df['message'].isnull()]

        if not sufficient_data_df.empty:
            st.header("Differential Analysis")
            create_differential_bar_chart(sufficient_data_df)

            st.header("Combined Analysis Over Time")
            create_line_chart(sufficient_data_df, [
                'harris',
                'trump',
            ], "Combined Polling and Favorability Results Over Time")

            st.header("Polling Results Over Time")
            create_line_chart(sufficient_data_df, ['harris', 'trump'], "Polling Results Over Time")

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

            st.header("Grouped Analysis")
            y_columns = ['harris', 'trump']
            if 'harris_fav' in sufficient_data_df.columns and 'trump_fav' in sufficient_data_df.columns:
                y_columns.extend(['harris_fav', 'trump_fav'])
            create_line_chart(sufficient_data_df, y_columns, "Grouped Results Over Time")

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

    # Footer
    st.markdown("---")
    st.write(f"Developed by Spencer Thayer. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
