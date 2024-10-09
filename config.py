# config.py

import logging

# Candidate names to be analyzed
CANDIDATE_NAMES = ['Kamala Harris', 'Donald Trump']

# Weight given to favorability in the combined analysis (between 0 and 1)
FAVORABILITY_WEIGHT = 0.25

# Flag to control weighting strategy (True for multiplicative, False for additive)
HEAVY_WEIGHT = True

# Time decay parameters
DECAY_RATE = 2.0          # Decay rate for time decay weighting
HALF_LIFE_DAYS = 14       # Half-life in days for time decay weighting

# Minimum number of samples required for analysis
MIN_SAMPLES_REQUIRED = 5

# Random Forest parameters
N_TREES = 1000            # Number of trees in the Random Forest
RANDOM_STATE = 42         # Random state for reproducibility

# Logging configuration
LOGGING_LEVEL = logging.INFO  # Set logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)

# URL constants for polling and favorability data
POLLING_URL = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv"
FAVORABILITY_URL = "https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv"

# Partisan weight mapping
PARTISAN_WEIGHT = {
    True: 0.01,  # Apply a low weight to partisan polls
    False: 1     # Apply normal weight to non-partisan polls
}

# Population weights mapping
POPULATION_WEIGHTS = {
    'lv': 1.0,                 # Likely voters
    'rv': 0.6666666666666666,  # Registered voters
    'v': 0.5,                  # Voters
    'a': 0.3333333333333333,   # Adults
    'all': 0.3333333333333333  # All respondents
}

# Coloring constants for terminal output
START_COLOR = 164  # Starting color code for ANSI escape sequences
SKIP_COLOR = 3     # Color code increment for each line of output

# Period order for analysis (used in visualizations and sorting)
PERIOD_ORDER = [
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

# Color definitions for visualizations
TRUMP_COLOR_DARK = "#8B0000"
TRUMP_COLOR = "#D13838"
TRUMP_COLOR_LIGHT = "#FFA07A"
HARRIS_COLOR_DARK = "#00008B"
HARRIS_COLOR = "#3838D1"
HARRIS_COLOR_LIGHT = "#6495ED"

# Additional configuration variables (if any)
# For example, thresholds, scaling factors, or other constants used in calculations
# These can be added here as needed to ensure consistency across the application

# Example of an additional constant (uncomment and modify as needed)
# SAMPLE_SIZE_THRESHOLD = 1000  # Threshold for sample size weighting

# Logging format (optional)
LOGGING_FORMAT = '%(levelname)s: %(message)s'

# Configure logging with the specified level and format
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
