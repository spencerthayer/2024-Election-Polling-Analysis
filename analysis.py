import pandas as pd
from polls import download_csv_data as download_polls_data
from liked import download_csv_data as download_favorability_data

def combine_data(polls_df, favorability_df, period_value, period_type='months'):
    # Convert 'created_at' column to datetime format
    polls_df['created_at'] = pd.to_datetime(polls_df['created_at'], format='%m/%d/%y %H:%M')
    favorability_df['created_at'] = pd.to_datetime(favorability_df['created_at'], format='%m/%d/%y %H:%M')

    # Filter the data for the specified period
    if period_type == 'months':
        polls_filtered_df = polls_df[(polls_df['created_at'] > (pd.Timestamp.now() - pd.DateOffset(months=period_value))) &
                                     (polls_df['candidate_name'].isin(['Joe Biden', 'Donald Trump']))]
        favorability_filtered_df = favorability_df[(favorability_df['created_at'] > (pd.Timestamp.now() - pd.DateOffset(months=period_value))) &
                                                    (favorability_df['politician'].isin(['Joe Biden', 'Donald Trump']))]
    elif period_type == 'days':
        polls_filtered_df = polls_df[(polls_df['created_at'] > (pd.Timestamp.now() - pd.Timedelta(days=period_value))) &
                                     (polls_df['candidate_name'].isin(['Joe Biden', 'Donald Trump']))]
        favorability_filtered_df = favorability_df[(favorability_df['created_at'] > (pd.Timestamp.now() - pd.Timedelta(days=period_value))) &
                                                    (favorability_df['politician'].isin(['Joe Biden', 'Donald Trump']))]

    # Assign weights to each dataset
    polls_filtered_df = polls_filtered_df.copy()
    favorability_filtered_df = favorability_filtered_df.copy()
    polls_filtered_df['weight'] = 0.6
    favorability_filtered_df['weight'] = 0.4

    # Concatenate the polling and favorability data
    combined_df = pd.concat([polls_filtered_df.rename(columns={'candidate_name': 'politician', 'pct': 'value'}),
                             favorability_filtered_df.rename(columns={'favorable': 'value'})], ignore_index=True)

    # Convert 'value' to decimal between 0 and 1
    combined_df['value'] = combined_df['value'] / 100

    # Calculate the combined weighted average
    combined_weighted_avg = combined_df.groupby('politician').apply(lambda x: (x['value'] * x['weight']).sum() / x['weight'].sum())

    biden_average = combined_weighted_avg.get('Joe Biden', 0)
    trump_average = combined_weighted_avg.get('Donald Trump', 0)
    differential = biden_average - trump_average

    favored_candidate = "Biden" if differential > 0 else "Trump"

    print(f"\nCombined Analysis for {period_value} {period_type}:")
    print(f"Biden: {biden_average:.2%}")
    print(f"Trump: {trump_average:.2%}")
    print(f"Differential: {differential:.2%} in favor of {favored_candidate}")

if __name__ == "__main__":
    polls_csv_url = 'https://projects.fivethirtyeight.com/polls/data/president_polls.csv'
    favorability_csv_url = 'https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv'

    polls_df = download_polls_data(polls_csv_url)
    favorability_df = download_favorability_data(favorability_csv_url)

    periods = [
        (12, 'months'),
        (6, 'months'),
        (3, 'months'),
        (1, 'months'),
        (21, 'days'),
        (14, 'days'),
        (7, 'days')
    ]

    for period_value, period_type in periods:
        combine_data(polls_df, favorability_df, period_value, period_type)