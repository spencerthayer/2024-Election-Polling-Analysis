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

def main():
    """
    Main function to run the Streamlit app.
    """
    # Set up the Streamlit page configuration
    st.set_page_config(page_title="Election Polling Analysis", layout="wide")
    st.title("Election Polling Analysis")
    
    

    # Load and process data
    results_df = load_and_process_data()

    # Display the available columns for debugging
    # st.write("Available columns in results_df:", results_df.columns.tolist())

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

            # st.header("Grouped Analysis")
            # y_columns = ['harris_polling', 'trump_polling']
            # if 'harris_fav' in sufficient_data_df.columns and 'trump_fav' in sufficient_data_df.columns:
            #     y_columns.extend(['harris_fav', 'trump_fav'])
            # y_columns.extend(['harris_combined', 'trump_combined'])
            # create_line_chart(sufficient_data_df, y_columns, "Grouped Results Over Time")

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
    
    # Add this code at the bottom of your app.py file, just before the if __name__ == "__main__": line

if __name__ == "__main__":
    main()
