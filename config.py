# config.py

import logging

# Candidate names to be analyzed
CANDIDATE_NAMES = ['Kamala Harris', 'Donald Trump']

# Configurable weight multipliers
TIME_DECAY_WEIGHT_MULTIPLIER = 1.0
SAMPLE_SIZE_WEIGHT_MULTIPLIER = 1.0
NORMALIZED_NUMERIC_GRADE_MULTIPLIER = 1.0
NORMALIZED_POLLSCORE_MULTIPLIER = 1.0
NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER = 1.0
POPULATION_WEIGHT_MULTIPLIER = 1.0
PARTISAN_WEIGHT_MULTIPLIER = 1.0
STATE_RANK_MULTIPLIER = 1.0

# Weight given to favorability in the combined analysis (between 0 and 1)
FAVORABILITY_WEIGHT = 0.15  # Adjust this value between 0 (no favorability influence) and 1 (only favorability)

# Flag to control weighting strategy (True for multiplicative, False for additive)
HEAVY_WEIGHT = True  # Set to True to use multiplicative weighting, False for additive weighting

# Time decay parameters
DECAY_RATE = 1.0     # Decay rate for time decay weighting (e.g., 0.5 for a slow 50% decay use 2 for a faster 200% decay)
HALF_LIFE_DAYS = 14  # Half-life in days for time decay weighting

# Minimum number of samples required for analysis
MIN_SAMPLES_REQUIRED = 4 # Minimum number of data points required to perform analysis for a period

ZERO_CORRECTION = 0.0001  # Small value to prevent division by zero in calculations

# Random Forest parameters
N_TREES = 1000       # Number of trees in the Random Forest
RANDOM_STATE = 42    # Random state for reproducibility

# Logging configuration
LOGGING_LEVEL = logging.INFO  # Set logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'  # Format for logging messages

# URL constants for polling and favorability data
POLLING_URL = "https://projects.fivethirtyeight.com/polls-page/data/president_polls.csv"
FAVORABILITY_URL = "https://projects.fivethirtyeight.com/polls-page/data/favorability_polls.csv"

# Partisan weight mapping
PARTISAN_WEIGHT = {
    True:  0.01, # Apply a reduced weight to partisan polls
    False: 1.0   # Apply full weight to non-partisan polls
}

# Population weights mapping
POPULATION_WEIGHTS = {
    'lv':  1.0,  # Likely voters
    'rv':  0.75, # Registered voters
    'v':   0.5,  # Voters (general)
    'a':   0.25, # Adults
    'all': 0.01  # All respondents
}

# Coloring constants for visualizations
TRUMP_COLOR_DARK = "#8B0000"
TRUMP_COLOR = "#D13838"
TRUMP_COLOR_LIGHT = "#FFA07A"
HARRIS_COLOR_DARK = "#00008B"
HARRIS_COLOR = "#3838D1"
HARRIS_COLOR_LIGHT = "#6495ED"

# Period order for analysis (used in visualizations and sorting)
PERIOD_ORDER = [
    '12 months',
    '6 months',
    '5 months',
    '4 months',
    '3 months',
    '2 months',
    '1 months',
    '21 days',
    '14 days',
    '7 days',
    '5 days',
    '3 days',
    '1 days'
]

# Starting color code for terminal output (adjust as needed)
START_COLOR = 164  # Starting color code for ANSI escape sequences
SKIP_COLOR = 3     # Color code increment for each line of output

# Additional configuration variables (if any)
# Example: Thresholds, scaling factors, or other constants used in calculations
# These can be added here as needed to ensure consistency across the application

# Logging format (optional)
LOGGING_FORMAT = '%(levelname)s: %(message)s'

# Configure logging with the specified level and format
logging.basicConfig(level=LOGGING_LEVEL, format=LOGGING_FORMAT)
