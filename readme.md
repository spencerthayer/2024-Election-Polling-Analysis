# Election Polling Analysis

This Python project is designed to fetch, process, and analyze presidential polling data. It consists of several main scripts: `analysis.py`, `app.py`, `states.py`, and `config.py`. The `analysis.py` script fetches data from [FiveThirtyEight](https://projects.fivethirtyeight.com/)'s publicly available CSV files for [presidential polls](https://projects.fivethirtyeight.com/polls/data/president_polls.csv) and [favorability polls](https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv), and applies a series of weightings to adjust for various factors such as poll quality, partisanship, and sample population type. The `states.py` script scrapes data from the [270 To Win](https://www.270towin.com/) website to obtain information about the electoral votes and political leanings of each state to calculate the state-specific electoral significance.

The project includes a Streamlit app (`app.py`) that provides an interactive interface for visualizing the analysis results and allows users to adjust configuration parameters dynamically.

## Table of Contents

- [File Structure](#file-structure)
- [Data Acquisition](#data-acquisition)
- [Weighting Calculations](#weighting-calculations)
  - [1. Time Decay Weight](#1-time-decay-weight)
  - [2. Grade Weight](#2-grade-weight)
  - [3. Transparency Weight](#3-transparency-weight)
  - [4. Sample Size Weight](#4-sample-size-weight)
  - [5. Partisan Weight](#5-partisan-weight)
  - [6. Population Weight](#6-population-weight)
  - [7. State Rank Weight](#7-state-rank-weight)
  - [8. Combining Weights](#8-combining-weights)
  - [9. Calculating Polling Metrics](#9-calculating-polling-metrics)
  - [10. Calculating Favorability Differential](#10-calculating-favorability-differential)
  - [11. Combining Polling Metrics and Favorability Differential](#11-combining-polling-metrics-and-favorability-differential)
  - [12. Out-of-Bag (OOB) Variance Calculation](#12-out-of-bag-oob-variance-calculation)
- [Error Handling and Normalization](#error-handling-and-normalization)
- [Data Caching Mechanisms](#data-caching-mechanisms)
- [Configuration Options](#configuration-options)
- [Conclusion](#conclusion)
- [Possible Next Steps](#possible-next-steps)

## File Structure

```
.
├── analysis.py
├── app.py
├── config.py
├── readme.md
├── requirements.txt
├── states.py
└── streamlit
```

## Data Acquisition

- **Presidential Polling Data**: Sourced from [FiveThirtyEight](https://projects.fivethirtyeight.com/), this dataset is accessed via the Python `requests` library, ensuring real-time relevance by incorporating the latest available data into a `pandas` DataFrame for subsequent analysis.

- **Favorability Polling Data**: Complementing presidential polling, FiveThirtyEight's favorability data offers insight into public sentiment regarding candidates, fetched similarly and integrated into the analysis to enhance depth.

- **State Data**: The `states.py` script enriches the analysis by integrating state-specific electoral information, offering a more granular view of the electoral landscape.

## Weighting Calculations

The `analysis.py` script employs several mathematical principles to calculate the final weight of each poll. Below we will explore how these weights are derived and applied to the polling data to calculate the adjusted polls, resulting in a nuanced analysis of polling data.

### 1. Time Decay Weight

The weight of a poll decreases exponentially based on how long ago it was conducted, ensuring that more recent polls have greater influence. The calculation uses fractional days to increase precision:

$$
W_{\text{time decay}} = e^{-\lambda \cdot t}
$$

Where:

- \( t \) is the age of the poll in fractional days, calculated as:

  $$
  t = \frac{\text{Current Timestamp} - \text{Poll Timestamp}}{86400 \text{ seconds/day}}
  $$

- \( \lambda \) is the decay constant:

  $$
  \lambda = \frac{\ln(\text{DECAY\_RATE})}{\text{HALF\_LIFE\_DAYS}}
  $$

**Default Values in `config.py`:**

- `DECAY_RATE = 1.0`
- `HALF_LIFE_DAYS = 14`

These values are configurable in the Streamlit app or by editing the `config.py` file.

### 2. Grade Weight

Polls are weighted based on the numeric grade assigned to the polling organization, reflecting their historical accuracy and methodological quality. The normalized numeric grade is calculated as:

$$
W_{\text{grade}} = \frac{\text{Numeric Grade}}{\text{Max Numeric Grade}}
$$

**Error Handling:**

- If the maximum numeric grade is zero, a small value (`ZERO_CORRECTION = 0.0001`) is used to prevent division by zero.

### 3. Transparency Weight

The transparency weight is calculated based on the transparency score provided in the polling data, normalized as:

$$
W_{\text{transparency}} = \frac{\text{Transparency Score}}{\text{Max Transparency Score}}
$$

**Error Handling:**

- If the maximum transparency score is zero, `ZERO_CORRECTION` is used.

### 4. Sample Size Weight

Sample size weight accounts for the reliability of a poll based on its sample size:

$$
W_{\text{sample size}} = \frac{\text{Sample Size} - \text{Min Sample Size}}{\text{Max Sample Size} - \text{Min Sample Size}}
$$

**Error Handling:**

- If `Max Sample Size - Min Sample Size` is zero, `ZERO_CORRECTION` is used.

### 5. Partisan Weight

Adjusts the weight based on whether a poll is partisan:

$$
W_{\text{partisan}} =
\begin{cases}
\text{PARTISAN\_WEIGHT[True]}, & \text{if partisan} \\
\text{PARTISAN\_WEIGHT[False]}, & \text{if non-partisan}
\end{cases}
$$

**Default Values in `config.py`:**

- `PARTISAN_WEIGHT = {True: 0.01, False: 1.0}`

### 6. Population Weight

Weights polls based on the population type:

| Population Type     | Weight                               |
|---------------------|--------------------------------------|
| Likely Voters (`lv`)    | `POPULATION_WEIGHTS['lv'] = 1.0`      |
| Registered Voters (`rv`)| `POPULATION_WEIGHTS['rv'] = 0.75`     |
| Voters (`v`)            | `POPULATION_WEIGHTS['v'] = 0.5`       |
| Adults (`a`)            | `POPULATION_WEIGHTS['a'] = 0.25`      |
| All Respondents (`all`)| `POPULATION_WEIGHTS['all'] = 0.01`    |

These values are configurable.

### 7. State Rank Weight

The state rank is calculated using data from `states.py`, considering both the electoral votes and the partisan lean of each state:

$$
W_{\text{state}} = \text{Pro Status Value} + \left( \frac{\text{Electoral Votes}}{\text{Total Electoral Votes}} \right)
$$

- **Pro Status Values:**

  | Code | Pro Status Value |
  |------|------------------|
  | `T`  | 0.8              |
  | `D1` | 0.6              |
  | `D2` | 0.4              |
  | `D3` | 0.2              |
  | `D4` | 0.1              |
  | `R1` | 0.6              |
  | `R2` | 0.4              |
  | `R3` | 0.2              |
  | `R4` | 0.1              |

- **Total Electoral Votes:** 538

### 8. Combining Weights

The script uses either multiplicative or additive methods based on the `HEAVY_WEIGHT` parameter to combine individual weights.

**Formula:**

- **Multiplicative (when `HEAVY_WEIGHT = True`):**

  $$
  W_{\text{combined}} = \prod_{i} \left( W_i \times \text{Multiplier}_i \right)
  $$

- **Additive (when `HEAVY_WEIGHT = False`):**

  $$
  W_{\text{combined}} = \frac{\sum_{i} \left( W_i \times \text{Multiplier}_i \right)}{n}
  $$

**Default Values for Multipliers in `config.py`:**

- `TIME_DECAY_WEIGHT_MULTIPLIER = 1.0`
- `SAMPLE_SIZE_WEIGHT_MULTIPLIER = 1.0`
- `NORMALIZED_NUMERIC_GRADE_MULTIPLIER = 1.0`
- `NORMALIZED_POLLSCORE_MULTIPLIER = 1.0`
- `NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER = 1.0`
- `POPULATION_WEIGHT_MULTIPLIER = 1.0`
- `PARTISAN_WEIGHT_MULTIPLIER = 1.0`
- `STATE_RANK_MULTIPLIER = 1.0`

These multipliers are configurable in the Streamlit app or by editing `config.py`.

### 9. Calculating Polling Metrics

For each candidate:

1. **Data Filtering:** Polling data is filtered based on the time period and candidate names.

2. **Percentage Handling:** Ensures that percentages are correctly interpreted, adjusting if necessary.

3. **Combined Weight Calculation:** Using the method described in [Combining Weights](#8-combining-weights).

4. **Weighted Sum and Total Weights:**

   $$
   \text{Weighted Sum}_c = \sum_{i \in c} W_{\text{combined}, i} \times \text{pct}_i
   $$

   $$
   \text{Total Weight}_c = \sum_{i \in c} W_{\text{combined}, i}
   $$

5. **Weighted Average:**

   $$
   \text{Weighted Average}_c = \frac{\text{Weighted Sum}_c}{\text{Total Weight}_c}
   $$

6. **Margin of Error Calculation:**

   $$
   \text{Margin of Error}_c = z \times \sqrt{\frac{p(1 - p)}{n}} \times 100\%
   $$

   Where:

   - \( z \) is the z-score (default 1.96 for 95% confidence).
   - \( p \) is the proportion (in decimal form).
   - \( n \) is the sample size.

### 10. Calculating Favorability Differential

Similar steps are followed for favorability data, with normalization and error handling:

- **Normalization:** Ensures 'favorable' percentages are correctly interpreted.

- **Combined Weight Calculation:** Uses only relevant weights (e.g., `normalized_numeric_grade`, `normalized_pollscore`, `normalized_transparency_score`).

- **Weighted Average Favorability:**

  $$
  \text{Weighted Favorability}_c = \frac{\sum_{i \in c} W_{\text{combined}, i} \times \text{Favorable}_i}{\sum_{i \in c} W_{\text{combined}, i}}
  $$

### 11. Combining Polling Metrics and Favorability Differential

The polling metrics and favorability differentials are combined using a weighted average:

$$
\text{Combined Result}_c = (1 - \alpha) \times \text{Polling Metric}_c + \alpha \times \text{Favorability Differential}_c
$$

Where:

- \( \alpha \) is the `FAVORABILITY_WEIGHT`.

**Default Value in `config.py`:**

- `FAVORABILITY_WEIGHT = 0.15`

### 12. Out-of-Bag (OOB) Variance Calculation

The OOB variance is calculated using the built-in `oob_prediction_` attribute of `RandomForestRegressor`:

$$
\sigma_{\text{OOB}}^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i^{\text{OOB}})^2
$$

**Benefits:**

- Simplifies the calculation.
- Reduces potential implementation errors.
- Leverages optimized internal methods for accuracy.

## Error Handling and Normalization

The updated script includes robust error handling to ensure mathematical integrity:

- **Division by Zero Prevention:** Checks are added before divisions to prevent division by zero, setting default values (`ZERO_CORRECTION = 0.0001`) when necessary.

- **Missing Data Handling:** Missing or `NaN` values are handled gracefully, with default assignments to ensure calculations proceed correctly.

- **Percentage Interpretation:** Improved handling of percentages less than or equal to 1 to avoid misclassification.

- **Time Calculations:** Uses fractional days and ensures timestamps are timezone-aware to prevent errors in time decay calculations.

## Data Caching Mechanisms

The `app.py` script includes data caching to improve performance:

- **Caching Data Files:** Processed data is saved to cache files to avoid reprocessing on subsequent runs.

- **Configuration Cache:** User-defined configuration settings are cached to maintain consistency between sessions.

- **Force Refresh Option:** Users can force a data refresh, clearing the cache and reprocessing the data with new configurations.

## Configuration Options

All configuration options are centralized in the `config.py` file and can be adjusted through the Streamlit app or by editing the file directly.

**Key Configurable Parameters:**

- **Candidates to Analyze:**

  ```python
  CANDIDATE_NAMES = ['Kamala Harris', 'Donald Trump']
  ```

- **Weight Multipliers:**

  ```python
  TIME_DECAY_WEIGHT_MULTIPLIER = 1.0
  SAMPLE_SIZE_WEIGHT_MULTIPLIER = 1.0
  NORMALIZED_NUMERIC_GRADE_MULTIPLIER = 1.0
  NORMALIZED_POLLSCORE_MULTIPLIER = 1.0
  NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER = 1.0
  POPULATION_WEIGHT_MULTIPLIER = 1.0
  PARTISAN_WEIGHT_MULTIPLIER = 1.0
  STATE_RANK_MULTIPLIER = 1.0
  ```

- **Favorability Weight:**

  ```python
  FAVORABILITY_WEIGHT = 0.15
  ```

- **Weighting Strategy:**

  ```python
  HEAVY_WEIGHT = True  # True for multiplicative, False for additive
  ```

- **Time Decay Parameters:**

  ```python
  DECAY_RATE = 1.0
  HALF_LIFE_DAYS = 14
  ```

- **Minimum Samples Required:**

  ```python
  MIN_SAMPLES_REQUIRED = 4
  ```

- **Partisan Weights:**

  ```python
  PARTISAN_WEIGHT = {
      True: 0.01,
      False: 1.0
  }
  ```

- **Population Weights:**

  ```python
  POPULATION_WEIGHTS = {
      'lv': 1.0,
      'rv': 0.75,
      'v': 0.5,
      'a': 0.25,
      'all': 0.01
  }
  ```

- **Random Forest Parameters:**

  ```python
  N_TREES = 1000
  RANDOM_STATE = 42
  ```

These parameters can be adjusted to fine-tune the analysis and are accessible via the Streamlit app's configuration sidebar.

## Conclusion

By incorporating favorability data, state-specific weights, and various other factors into the analysis, this project provides a nuanced and comprehensive assessment of presidential polling data. The integration of data from `states.py` allows for the consideration of each state's unique electoral dynamics, ensuring that the adjusted poll results reflect the significance and political leanings of individual states.

This approach aims to strike a balance between the broad insights provided by national polls, the detailed, state-specific information captured by local polls, and the additional context provided by favorability ratings. By carefully normalizing and combining these various weights, the scripts produce adjusted results that offer a more accurate and representative picture of the current state of the presidential race.

As with any polling analysis, there is always room for further refinement and improvement. The modular design of the scripts allows for the incorporation of additional factors and adjustments as needed. Collaboration and feedback from the community are welcome to enhance the methodology and ensure the most accurate and meaningful analysis possible.

## Possible Next Steps

To further enhance the project, several next steps can be considered:

1. **Sensitivity Analysis:** Assess the impact of different weight assignments and parameter values on the final results.

2. **Incorporation of Additional Data Sources:** Integrate polling data from other reputable sources to enhance robustness.

3. **Advanced Modeling Techniques:** Explore advanced modeling techniques to capture more complex patterns.

4. **Uncertainty Quantification:** Implement techniques to quantify the uncertainty associated with the adjusted poll results.

5. **User Interface and Visualization:** Continue to develop the Streamlit app for a more user-friendly interface and improved data visualization components.

6. **Sophisticated Stratification Frame Construction:** Introduce data integration techniques to merge disparate data sources, enhancing the completeness of the stratification frame.

7. **Incorporation of Uncertainty Estimates:** Apply techniques to estimate uncertainty around predictions, offering a nuanced view of reliability.

8. **Integration with Multiple Forecasting Models:** Develop an ensemble method that averages forecasts from multiple sources.

9. **Benchmarking Turnout Modeling Strategies:** Explore and benchmark alternative turnout modeling strategies against the current approach.

**Practical Steps:**

- **Data Preparation:** Integrate multiple data sources, ensuring they are preprocessed and harmonized.

- **Model Development:** Adapt the script to incorporate advanced algorithms and uncertainty estimation methods.

- **Evaluation Framework:** Establish metrics and validation procedures to assess performance.

- **Iterative Testing:** Systematically test and refine the forecasting model.

- **Documentation and Reporting:** Update documentation to reflect new methodologies and findings.