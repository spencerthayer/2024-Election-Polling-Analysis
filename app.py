import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from analysis import (
    download_csv_data,
    preprocess_data,
    apply_time_decay_weight,
    calculate_polling_metrics,
    calculate_favorability_differential,
    combine_analysis,
    candidate_names,
    favorability_weight,
    decay_rate,
    half_life_days
)

# Constants
POLLING_URL = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
FAVORABILITY_URL = "https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv"
TRUMP_COLOR = '#D13838'
HARRIS_COLOR = '#3838D1'

# Define a custom order for the periods
period_order = [
    '1 days', '3 days', '7 days', '14 days', '21 days',
    '1 months', '3 months', '6 months', '12 months'
]

def create_line_chart(df, y_columns, title):
    y_min = min(df[y_columns].min().min() - 0.5, 42)  # Ensure lower bound is at least 42
    y_max = max(df[y_columns].max().max() + 0.5, 49)  # Ensure upper bound is at least 49

    color_scale = alt.Scale(
        domain=['harris', 'trump'],
        range=[HARRIS_COLOR, TRUMP_COLOR]
    )

    chart = alt.Chart(df).transform_fold(
        y_columns,
        as_=['candidate', 'value']
    ).mark_line().encode(
        x=alt.X('period:N', sort=period_order, title='Period'),
        y=alt.Y('value:Q', scale=alt.Scale(domain=[y_min, y_max]), title='Polling Percentage'),
        color=alt.Color('candidate:N', scale=color_scale)
    ).properties(
        width=600,
        height=400,
        title=title
    )
    
    st.altair_chart(chart, use_container_width=True)

def create_differential_bar_chart(df):
    df['differential'] = df['harris'] - df['trump']
    df['differential_label'] = df.apply(
        lambda row: f"{row['harris']:.2f}%±{row['harris_moe']:.2f} vs {row['trump']:.2f}%±{row['trump_moe']:.2f} ({'Harris' if row['differential'] > 0 else 'Trump'})",
        axis=1
    )

    bar_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('differential:Q', title='Differential (Harris - Trump)', scale=alt.Scale(domain=[-10, 10])),
        y=alt.Y('period:N', sort=period_order, title='Period'),
        color=alt.condition(
            alt.datum.differential > 0,
            alt.value(HARRIS_COLOR),
            alt.value(TRUMP_COLOR)
        ),
        tooltip=[
            alt.Tooltip('differential_label:N', title='Differential'),
            alt.Tooltip('harris:Q', format='.2f', title='Harris (%)'),
            alt.Tooltip('trump:Q', format='.2f', title='Trump (%)'),
            alt.Tooltip('harris_moe:Q', format='.2f', title='Harris MOE'),
            alt.Tooltip('trump_moe:Q', format='.2f', title='Trump MOE')
        ]
    ).properties(
        title="Differential Between Harris and Trump Over Time",
        width=600,
        height=400
    )

    st.altair_chart(bar_chart, use_container_width=True)

def run_analysis():
    try:
        polling_df = download_csv_data(POLLING_URL)
        favorability_df = download_csv_data(FAVORABILITY_URL)

        polling_df = preprocess_data(polling_df)
        favorability_df = preprocess_data(favorability_df)

        polling_df = apply_time_decay_weight(polling_df, decay_rate, half_life_days)
        favorability_df = apply_time_decay_weight(favorability_df, decay_rate, half_life_days)

        results = []
        periods = [
            (1, 'days'), (3, 'days'), (7, 'days'), (14, 'days'), (21, 'days'),
            (1, 'months'), (3, 'months'), (6, 'months'), (12, 'months')
        ]

        for period_value, period_type in periods:
            if period_type == 'months':
                start_period = datetime.now() - relativedelta(months=period_value)
            else:  # 'days'
                start_period = datetime.now() - timedelta(days=period_value)
            
            period_polling_df = polling_df[polling_df['created_at'] >= start_period]
            period_favorability_df = favorability_df[favorability_df['created_at'] >= start_period]
            
            polling_metrics = calculate_polling_metrics(period_polling_df, candidate_names)
            favorability_differential = calculate_favorability_differential(period_favorability_df, candidate_names)
            combined_results = combine_analysis(polling_metrics, favorability_differential, favorability_weight)
            
            st.write(f"Data for {period_value} {period_type}:")
            st.write("Polling metrics:", polling_metrics)
            st.write("Favorability differential:", favorability_differential)
            st.write("Combined results:", combined_results)
            
            results.append({
                'period': f"{period_value} {period_type}",
                'harris': combined_results['Kamala Harris'][0],
                'harris_moe': combined_results['Kamala Harris'][1],
                'trump': combined_results['Donald Trump'][0],
                'trump_moe': combined_results['Donald Trump'][1],
                'harris_fav': favorability_differential.get('Kamala Harris', 0),
                'trump_fav': favorability_differential.get('Donald Trump', 0),
            })

        return pd.DataFrame(results)

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.write("Please check the analysis.py script for potential issues.")
        return pd.DataFrame()

# Main Streamlit app
st.set_page_config(page_title="Election Polling Analysis", layout="wide")
st.title("Election Polling Analysis")

results_df = run_analysis()

if not results_df.empty:
    results_df = results_df.sort_values('period')
    results_df['period'] = pd.Categorical(results_df['period'], categories=period_order, ordered=True)

    st.write("Results DataFrame:")
    st.write(results_df)
    st.write("Data types of results:", results_df.dtypes)

    st.header("Polling Results Over Time")
    create_line_chart(results_df, ['harris', 'trump'], "Polling Results Over Time")

    st.header("Favorability Over Time")
    st.write("Favorability data:", results_df[['period', 'harris_fav', 'trump_fav']])
    if results_df['harris_fav'].sum() == 0 and results_df['trump_fav'].sum() == 0:
        st.warning("No favorability data available. Check the calculation in analysis.py")
    else:
        create_line_chart(results_df, ['harris_fav', 'trump_fav'], "Favorability Over Time")

    st.header("Combined Analysis Over Time")
    create_line_chart(results_df, ['harris', 'trump'], "Combined Analysis Over Time")

    st.header("Differential Analysis")
    create_differential_bar_chart(results_df)

    st.header("Comparison with Terminal Output")
    for _, row in results_df.iterrows():
        st.write(f"{row['period']:<4} H∙{row['harris']:5.2f}%±{row['harris_moe']:.2f} "
                 f"T∙{row['trump']:5.2f}%±{row['trump_moe']:.2f} "
                 f"{abs(row['harris'] - row['trump']):+5.2f} "
                 f"{'Harris' if row['harris'] > row['trump'] else 'Trump'}")

else:
    st.write("No data available for the selected periods.")

# Display raw data
st.header("Raw Data")
st.dataframe(results_df)

# Footer
st.markdown("---")
st.write("Developed by Spencer Thayer. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
