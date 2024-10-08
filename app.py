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
TRUMP_COLOR = "#D13838"
TRUMP_COLOR_LIGHT = "#FFA07A"
HARRIS_COLOR = "#3838D1"
HARRIS_COLOR_LIGHT = "#6495ED"

period_order = [
    '12 months',
    '6 months',
    '3 months',
    '1 months',
    '21 days',
    '14 days',
    '7 days',
    '3 days',
    '1 days'
]
# Caching the data download to avoid re-fetching on each rerun
@st.cache_data
def load_data():
    polling_df = download_csv_data(POLLING_URL)
    favorability_df = download_csv_data(FAVORABILITY_URL)
    return polling_df, favorability_df

def create_line_chart(df, y_columns, title):
    df_melted = df.melt(
        id_vars=['period'],
        value_vars=y_columns,
        var_name='candidate',
        value_name='value'
    )

    y_min = df_melted['value'].min() - 0.5
    y_max = df_melted['value'].max() + 0.5

    color_scale = alt.Scale(
        domain=y_columns,
        range=[HARRIS_COLOR, TRUMP_COLOR, HARRIS_COLOR_LIGHT, TRUMP_COLOR_LIGHT]
    )

    chart = alt.Chart(df_melted).mark_line().encode(
        x=alt.X('period:N', sort=period_order, title='Period'),
        y=alt.Y('value:Q', scale=alt.Scale(domain=[y_min, y_max]), title='Percentage'),
        color=alt.Color('candidate:N', scale=color_scale)
    ).properties(
        width=800,
        height=400,
        title=title
    )
    
    st.altair_chart(chart, use_container_width=True)
# START create_grouped_bar_chart
def create_grouped_bar_chart(df):
    # Melt the dataframe to create a long format suitable for grouping
    df_melted = df.melt(
        id_vars=['period'],
        value_vars=['harris', 'trump', 'harris_fav', 'trump_fav'],
        var_name='metric',
        value_name='value'
    )

    # Create the grouped bar chart
    chart = alt.Chart(df_melted).mark_bar().encode(
        x=alt.X('period:N', sort=period_order, title='Period'),
        xOffset='metric:N',  # Offset bars within each period
        y=alt.Y('value:Q', title='Percentage'),
        color=alt.Color('metric:N', scale=alt.Scale(
            domain=['harris', 'trump', 'harris_fav', 'trump_fav'],
            range=[HARRIS_COLOR, TRUMP_COLOR, HARRIS_COLOR_LIGHT, TRUMP_COLOR_LIGHT]
        )),
        tooltip=['period', 'metric', 'value']
    ).properties(
        width=800,
        height=400,
        title="Grouped Analysis: Polling and Favorability"
    )

    st.altair_chart(chart, use_container_width=True)
# START create_differential_bar_chart
def create_differential_bar_chart(df):
    df['differential'] = df['harris'] - df['trump']
    df['differential_label'] = df.apply(
        lambda row: f"H: {row['harris']:.2f}%±{row['harris_moe']:.2f}\nT: {row['trump']:.2f}%±{row['trump_moe']:.2f}",
        axis=1
    )

    # Calculate the symmetric range around zero
    max_abs_diff = max(abs(df['differential'].min()), abs(df['differential'].max()))
    max_moe = max(df['harris_moe'].max(), df['trump_moe'].max())
    y_range = max(max_abs_diff, max_moe) + 1  # Add 1 for some padding
    y_min, y_max = -y_range, y_range

    # Base chart
    base = alt.Chart(df).encode(
        x=alt.X('period:N', sort=period_order, title='Period')
    )

    # Differential bars
    bars = base.mark_bar(size=30).encode(
        y=alt.Y('differential:Q', 
                title='Trump            Harris', 
                scale=alt.Scale(domain=[y_min, y_max])),
        color=alt.condition(
            alt.datum.differential > 0,
            alt.value(HARRIS_COLOR),
            alt.value(TRUMP_COLOR)
        ),
        tooltip=[alt.Tooltip('differential_label:N', title='Results')]
    )

    # Trump MOE area (negative side)
    trump_moe_area = base.mark_area(
        opacity=0.25,
        color=TRUMP_COLOR_LIGHT
    ).encode(
        y=alt.Y('zero:Q'),
        y2=alt.Y2('low:Q')
    ).transform_calculate(
        zero='datum.trump_moe/2',
        low='datum.trump_moe*-1'
        # low='datum.differential - datum.harris_moe - datum.trump_moe',
    )

    # Harris MOE area (positive side)
    harris_moe_area = base.mark_area(
        opacity=0.25,
        color=HARRIS_COLOR_LIGHT
    ).encode(
        y=alt.Y('zero:Q'),
        y2=alt.Y2('high:Q')
    ).transform_calculate(
        zero='datum.harris_moe/2*-1',
        high='datum.harris_moe'
        # high='datum.differential + datum.harris_moe + datum.trump_moe'
    )

    # Zero line
    zero_line = alt.Chart(pd.DataFrame({'y': [0]})).mark_rule(
        color='black', 
        strokeWidth=1,
        strokeDash=[5, 5]
    ).encode(y='y')

    # Text labels
    text_labels = base.mark_text(
        align='center',
        baseline='middle',
        dy=alt.expr('datum.differential > 0 ? -10 : 10'),
        fontSize=12,
        fontWeight='bold'
    ).encode(
        y=alt.Y('differential:Q'),
        text=alt.Text('differential:Q', format='+.2f'),
        color=alt.condition(
            alt.datum.differential > 0,
            alt.value(HARRIS_COLOR),
            alt.value(TRUMP_COLOR)
        )
    )

    # Combine all chart elements
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
# END create_differential_bar_chart
@st.cache_data
def run_analysis():
    try:
        polling_df, favorability_df = load_data()

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
            
            # st.write(f"Data for {period_value} {period_type}:")
            # st.write("Polling metrics:", polling_metrics)
            # st.write("Favorability differential:", favorability_differential)
            # st.write("Combined results:", combined_results)
            
            results.append({
                'period': f"{period_value} {period_type}",
                'harris': combined_results['Kamala Harris'][0],
                'trump': combined_results['Donald Trump'][0],
                'harris_fav': favorability_differential.get('Kamala Harris', 0),
                'trump_fav': favorability_differential.get('Donald Trump', 0),
                'harris_moe': combined_results['Kamala Harris'][1],
                'trump_moe': combined_results['Donald Trump'][1]
            })

        results_df = pd.DataFrame(results)
        results_df['period'] = pd.Categorical(results_df['period'], categories=period_order, ordered=True)
        results_df = results_df.sort_values('period')

        return results_df

    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.write("Please check the analysis.py script for potential issues.")
        return pd.DataFrame()

# Main Streamlit app
st.set_page_config(page_title="Election Polling Analysis", layout="wide")
st.title("Election Polling Analysis")

results_df = run_analysis()

if not results_df.empty:

    st.header("Differential Analysis")
    create_differential_bar_chart(results_df)

    st.header("Polling Results Over Time")
    create_line_chart(results_df, ['harris', 'trump'], "Polling Results Over Time")

    st.header("Favorability Over Time")
    # st.write("Favorability data:", results_df[['period', 'harris_fav', 'trump_fav']])
    if results_df['harris_fav'].sum() == 0 and results_df['trump_fav'].sum() == 0:
        st.warning("No favorability data available. Check the calculation in analysis.py")
    else:
        create_line_chart(results_df, ['harris_fav', 'trump_fav'], "Favorability Over Time")

    st.header("Combined Analysis Over Time")
    create_line_chart(results_df, ['harris', 'trump'], "Combined Analysis Over Time")

    st.header("Grouped Analysis")
    create_grouped_bar_chart(results_df)
    
    st.write("Results DataFrame:")
    st.write(results_df)
    # st.write("Data types of results:", results_df.dtypes)

    # st.header("Comparison with Terminal Output")
    # for _, row in results_df.iterrows():
    #     st.write(f"{row['period']:<4} H∙{row['harris']:5.2f}%±{row['harris_moe']:.2f} "
    #              f"T∙{row['trump']:5.2f}%±{row['trump_moe']:.2f} "
    #              f"{abs(row['harris'] - row['trump']):+5.2f} "
    #              f"{'Harris' if row['harris'] > row['trump'] else 'Trump'}")

else:
    st.write("No data available for the selected periods.")

# Footer
st.markdown("---")
st.write("Developed by Spencer Thayer. Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
