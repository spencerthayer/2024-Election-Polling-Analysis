# Election Polling Analysis

This Python project is designed to fetch, process, and analyze presidential polling data, providing a comprehensive and nuanced assessment of the current electoral landscape. It consists of several main scripts:

- **`analysis.py`**: The core script responsible for data fetching, processing, and applying various weighting mechanisms to adjust poll results.
- **`states.py`**: A script that scrapes state-specific electoral data from both the [270 To Win](https://www.270towin.com/) website and [FiveThirtyEight](https://projects.fivethirtyeight.com/), enhancing the analysis with state-level insights.
- **`app.py`**: A Streamlit application that provides an interactive user interface for visualizing results and adjusting configuration parameters dynamically.
- **`config.py`**: A configuration file containing adjustable parameters that control the behavior of the analysis.

The project leverages data from [FiveThirtyEight](https://projects.fivethirtyeight.com/)'s publicly available CSV files for both [presidential polls](https://projects.fivethirtyeight.com/polls/data/president_polls.csv) and [favorability polls](https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv). By applying a series of weightings to adjust for various factors—such as poll quality, partisanship, sample population type, and state significance—the analysis aims to produce an adjusted polling metric that more accurately reflects the true state of the presidential race.

---

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

---

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

- **`analysis.py`**: Core script for fetching and analyzing polling data.
- **`app.py`**: Streamlit application providing an interactive user interface.
- **`config.py`**: Configuration file containing adjustable parameters.
- **`states.py`**: Script for scraping state-specific electoral data.
- **`readme.md`**: Comprehensive project documentation.
- **`requirements.txt`**: List of Python dependencies.
- **`streamlit`**: Directory containing additional resources for the Streamlit app.

---

## Data Acquisition

The project relies on three primary data sources to ensure a robust and comprehensive analysis:

1. **Presidential Polling Data**:
   - **Source**: [FiveThirtyEight](https://projects.fivethirtyeight.com/polls/data/president_polls.csv)
   - **Description**: Provides detailed polling information for presidential candidates, including pollster ratings, sample sizes, and polling dates.
   - **Method of Acquisition**: Data is fetched using the Python `requests` library and loaded into a `pandas` DataFrame for analysis.

2. **Favorability Polling Data**:
   - **Source**: [FiveThirtyEight](https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv)
   - **Description**: Offers insights into public sentiment regarding candidates, capturing favorability ratings which are crucial for understanding broader public perceptions.
   - **Method of Acquisition**: Similar to the presidential polling data, it is fetched and processed into a `pandas` DataFrame.

3. **State Data**:
   - **Sources**:
     - [270 To Win](https://www.270towin.com/)
     - [FiveThirtyEight Forecast Data](https://projects.fivethirtyeight.com/2024-election-forecast/priors.json)
   - **Description**: Contains information about each state's electoral votes, political leanings, and forecasted election outcomes, essential for calculating state-specific weights.
   - **Method of Acquisition**: The `states.py` script scrapes and processes data from both websites to obtain up-to-date state rankings and forecasts.

**Justification for Data Sources**:

- **FiveThirtyEight**: Renowned for its rigorous methodology and comprehensive data, making it a reliable source for polling and forecast information.
- **270 To Win**: Provides up-to-date and detailed electoral data, essential for state-level analysis.

---

## Weighting Calculations

To adjust raw polling data and produce a more accurate reflection of the electoral landscape, the project applies several weighting mechanisms. Each weight addresses a specific factor that can influence the reliability or relevance of a poll.

### 1. Time Decay Weight

**Objective**: To prioritize recent polls over older ones, acknowledging that public opinion can change rapidly.

**Mathematical Formulation**:

The weight decreases exponentially with the age of the poll using NumPy's exponential function:

$$
W_{\text{time decay}} = \exp\left(-\lambda t\right)
$$

- **\( t \)**: The age of the poll in fractional days.

  $$
  t = \frac{\text{Current Timestamp} - \text{Poll Timestamp}}{86400}
  $$

  - *Justification*: Using fractional days increases precision, especially for recent polls where hours can make a difference.

- **\( \lambda \)**: The decay constant, representing how quickly the weight decreases over time.

  $$
  \lambda = \frac{\ln(\text{DECAY\_RATE})}{\text{HALF\_LIFE\_DAYS}}
  $$

  - **Parameters**:
    - `DECAY_RATE`: The rate at which the weight decays over the half-life period (default is `1.0`, meaning no decay).
    - `HALF_LIFE_DAYS`: The half-life period in days (default is `14` days).

**Justification for Exponential Decay**:

- Exponential decay reflects the idea that the influence of a poll diminishes over time.
- The half-life parameter allows for control over how quickly this influence wanes.
- Exponential functions are continuous and smooth, providing a realistic decay model.

**Implementation**:

- Adjust the `DECAY_RATE` and `HALF_LIFE_DAYS` in `config.py` or via the Streamlit app to reflect the desired decay behavior.

### 2. Grade Weight

**Objective**: To adjust poll weights based on the historical accuracy and methodological quality of the polling organizations.

**Mathematical Formulation**:

The normalized grade weight is calculated as:

$$
W_{\text{grade}} = \frac{\text{Numeric Grade}}{\text{Max Numeric Grade}} \\
W_{\text{grade}} = \text{clip}(W_{\text{grade}}, 0, 1)
$$

- **Numeric Grade**: A numerical representation of the pollster's grade assigned by FiveThirtyEight.
- **Max Numeric Grade**: The highest numeric grade among all pollsters in the dataset.

**Justification**:

- Pollsters with higher grades have historically produced more accurate polls.
- Normalization ensures that grades are scaled between 0 and 1, allowing for consistent weighting across different datasets.

**Error Handling**:

- If `Max Numeric Grade` is zero (which could happen if grades are missing), a small value (`ZERO_CORRECTION = 0.0001`) is used to prevent division by zero.

**Implementation**:

- Ensure that the grades are properly converted to numeric values, handling any non-standard grades or missing values.
- The normalized grade is clipped to ensure it remains within the [0, 1] range using Pandas' `clip()` function.

### 3. Transparency Weight

**Objective**: To reward polls that are transparent about their methodologies, which can be an indicator of reliability.

**Mathematical Formulation**:

$$
W_{\text{transparency}} = \frac{\text{Transparency Score}}{\text{Max Transparency Score}} \\
W_{\text{transparency}} = \text{clip}(W_{\text{transparency}}, 0, 1)
$$

- **Transparency Score**: A score provided by FiveThirtyEight that reflects the level of methodological disclosure by the pollster.
- **Max Transparency Score**: The highest transparency score among all polls in the dataset.

**Justification**:

- Transparency allows for better assessment of a poll's quality.
- Polls that disclose their methods fully are more trustworthy.

**Error Handling**:

- If `Max Transparency Score` is zero, `ZERO_CORRECTION` is used to prevent division by zero.

**Implementation**:

- Convert transparency scores to numeric values and handle any non-standard or missing values.
- Normalize and clip the transparency scores to maintain consistency.

### 4. Sample Size Weight

**Objective**: To account for the reliability of polls based on the number of respondents.

**Mathematical Formulation**:

$$
W_{\text{sample size}} = \frac{\text{Sample Size} - \text{Min Sample Size}}{\text{Max Sample Size} - \text{Min Sample Size}}
$$

- **Sample Size**: The number of respondents in the poll.
- **Min Sample Size** and **Max Sample Size**: The minimum and maximum sample sizes across all polls, determined using Pandas' `min()` and `max()` functions.

**Justification**:

- Larger sample sizes generally lead to more accurate and reliable results due to reduced sampling error.
- Normalizing the sample size ensures that weights are proportionate across the range of sample sizes.

**Error Handling**:

- If `Max Sample Size - Min Sample Size` is zero, `ZERO_CORRECTION` is used.

**Implementation**:

- Calculate the minimum and maximum sample sizes using Pandas' `min()` and `max()` functions.
- Normalize the sample sizes and handle cases where all sample sizes are identical.

### 5. Partisan Weight

**Objective**: To mitigate potential biases introduced by polls sponsored by partisan organizations.

**Mathematical Formulation**:

$$
W_{\text{partisan}} =
\begin{cases}
\text{PARTISAN\_WEIGHT}[True], & \text{if the poll is partisan} \\
\text{PARTISAN\_WEIGHT}[False], & \text{if the poll is non-partisan}
\end{cases}
$$

**Default Values in `config.py`**:

```python
PARTISAN_WEIGHT = {True: 0.01, False: 1.0}
```

**Justification**:

- Partisan polls may exhibit bias toward a particular candidate or party.
- Assigning a significantly lower weight to partisan polls reduces their impact on the overall analysis.

**Implementation**:

- Weights are assigned using a dictionary lookup for efficiency.
- The weight values can be adjusted in `config.py` or via the Streamlit app to reflect the desired level of influence from partisan polls.

### 6. Population Weight

**Objective**: To adjust poll weights based on the population type surveyed, reflecting the likelihood that respondents will vote.

**Population Types and Weights**:

| Population Type          | Weight                                |
|--------------------------|---------------------------------------|
| Likely Voters (`lv`)     | `POPULATION_WEIGHTS['lv'] = 1.0`      |
| Registered Voters (`rv`) | `POPULATION_WEIGHTS['rv'] = 0.75`     |
| Voters (`v`)             | `POPULATION_WEIGHTS['v'] = 0.5`       |
| Adults (`a`)             | `POPULATION_WEIGHTS['a'] = 0.25`      |
| All Respondents (`all`)  | `POPULATION_WEIGHTS['all'] = 0.01`    |

**Justification**:

- **Likely Voters** are most representative of the actual electorate, so they receive the highest weight.
- **Registered Voters** are somewhat less predictive, as not all registered voters turn out.
- **Voters** and **Adults** include individuals who may not be eligible or likely to vote, so they receive lower weights.
- **All Respondents** include the broadest population, many of whom may not be eligible or likely to vote, thus receiving the lowest weight.

**Implementation**:

- These weights are assigned using a dictionary and can be adjusted in `config.py` or via the Streamlit app to reflect changes in voter behavior or to conduct sensitivity analyses.

### 7. State Rank Weight

The **State Rank Weight** integrates the electoral significance, political leaning, and current forecasts of each state into the overall weighting of polls. This weight ensures that polls from states that are more influential in the electoral college, have competitive political landscapes, and are projected to have close races are given greater consideration in the analysis.

**Objective**: To calculate a weight for each poll based on the state's electoral importance, partisan classification, and current election forecasts, thereby prioritizing polls from significant and competitive states.

**Mathematical Formulation**:

The state rank for each state is calculated as a weighted sum of three components:

$$
\text{State Rank} = (\text{Pro Status Value} \times 0.4) + (\text{Normalized Electoral Votes} \times 0.3) + (\text{Forecast Weight} \times 0.3)
$$

Where:

- **Pro Status Value**: A numerical representation of the state's partisan lean, derived from the `pro_status` codes provided by 270 To Win.
- **Normalized Electoral Votes**: The state's electoral votes divided by the total electoral votes (538), representing the state's relative electoral significance.
- **Forecast Weight**: Based on FiveThirtyEight's forecast data, representing the closeness of the race in each state.

**Components Explanation**:

1. **Pro Status Value (40% of State Rank)**:
   - Derived from the state's political classification:
     - `T`: Toss-up state (0.8)
     - `D1`, `R1`: Tilt Democrat/Republican (0.6)
     - `D2`, `R2`: Lean Democrat/Republican (0.4)
     - `D3`, `R3`: Likely Democrat/Republican (0.2)
     - `D4`, `R4`: Safe Democrat/Republican (0.1)
   - **Justification**: Reflects the competitiveness of the state based on historical and current political leanings, with higher values for more competitive states.

2. **Normalized Electoral Votes (30% of State Rank)**:
   - Calculated as:
     $$
     \text{Normalized Electoral Votes} = \frac{\text{State's Electoral Votes}}{538}
     $$
   - **Justification**: Gives more weight to states with more electoral votes, reflecting their greater potential impact on the election outcome.

3. **Forecast Weight (30% of State Rank)**:
   - Calculated as:
     $$
     \text{Forecast Weight} = 1 - \left( \frac{|\text{Forecast Median}|}{100} \right)
     $$
     - **Forecast Median**: The median forecasted margin between the candidates from FiveThirtyEight's data.
   - **Justification**: Prioritizes states with closer races, as they are more likely to influence the election outcome.

**Implementation Details**:

- **Data Retrieval**:
  - The `states.py` script fetches:
    - `pro_status` codes and electoral votes from 270 To Win.
    - Forecast medians from FiveThirtyEight's forecast JSON data.
- **State Rank Calculation**:
  - Each state's rank is calculated using the weighted sum formula above.
  - The ranks are normalized and used as weights in the polling analysis.
- **Incorporation into Combined Weight**:
  - The State Rank Weight is included as one of the factors in the combined weight calculation (see [Combining Weights](#8-combining-weights)).
  - Its influence can be adjusted using the `STATE_RANK_MULTIPLIER` in `config.py`:

    ```python
    STATE_RANK_MULTIPLIER = 1.0  # Adjust to increase or decrease influence
    ```

**Handling Missing Forecast Data**:

- If forecast data for a state is missing, a default forecast weight is applied to prevent computational errors.

**Considerations**:

- **Dynamic Political Landscape**:
  - The state's `pro_status` and forecast data are regularly updated to reflect the most current information.
- **Data Handling**:
  - The `states.py` script ensures that if forecast data is missing or incomplete, default values are used to maintain computational integrity.
- **Weight Sensitivity**:
  - The weighting percentages (40%, 30%, 30%) can be adjusted to emphasize different components based on analytical needs.

**Example Calculation**:

Suppose we have a state with the following characteristics:

- **Pro Status**: `T` (Toss-up state), so `Pro Status Value = 0.8`
- **Electoral Votes**: 20, so `Normalized Electoral Votes = 20 / 538 ≈ 0.0372`
- **Forecast Median**: 2.5 (indicating a close race), so:
  $$
  \text{Forecast Weight} = 1 - \left( \frac{2.5}{100} \right) = 0.975
  $$

The State Rank would be:

$$
\text{State Rank} = (0.8 \times 0.4) + (0.0372 \times 0.3) + (0.975 \times 0.3) \\
= 0.32 + 0.01116 + 0.2925 \\
\approx 0.62366
$$

This high rank indicates that the state is both competitive and significant in terms of electoral votes and current forecasts.

### 8. Combining Weights

An essential step in the analysis is to aggregate the individual weights calculated from various factors into a single **Combined Weight** for each poll. This combined weight determines the overall influence each poll will have on the final polling metrics.

**Objective**: To combine individual weights into a single weight that reflects all factors influencing poll reliability and relevance.

**Methods of Combining Weights**:

1. **Multiplicative Combination** (when `HEAVY_WEIGHT = True`):

   $$
   W_{\text{combined}} = \prod_{k} \left( W_k \times \text{Multiplier}_k \right)
   $$

   - **Pros**:
     - Strongly penalizes polls weak in any single criterion.
     - Emphasizes high-quality polls.
   - **Cons**:
     - Can overly penalize polls with minor weaknesses.

   *Implementation Note*: Utilizes NumPy's `prod()` function for efficient computation.

2. **Additive Combination** (when `HEAVY_WEIGHT = False`):

   $$
   W_{\text{combined}} = \frac{\sum_{k} \left( W_k \times \text{Multiplier}_k \right)}{n}
   $$

   - **Pros**:
     - Balances the influence of each weight.
     - More forgiving of polls with mixed strengths and weaknesses.
   - **Cons**:
     - May allow lower-quality polls to have more influence than desired.

   *Implementation Note*: Utilizes NumPy's `mean()` function to calculate the average of weighted components.

**Multipliers**:

Multipliers adjust the influence of each individual weight:

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

**Implementation Steps**:

1. **Calculate Individual Weights**: Compute each weight as described in the previous sections.
2. **Apply Multipliers**: Multiply each weight by its corresponding multiplier.
3. **Combine Weights**: Use the chosen method (multiplicative or additive) to compute the combined weight.
4. **Normalization** (Optional): Ensure combined weights are on a consistent scale.

**Example**:

If `HEAVY_WEIGHT` is set to `True`, the combined weight for a poll would be the product of all individual weights multiplied by their respective multipliers. If set to `False`, it would be the average of these weighted components.

### 9. Calculating Polling Metrics

**Objective**: To compute an adjusted polling metric for each candidate by combining poll results with their respective combined weights.

**Methodology**:

1. **Data Filtering**: Select relevant polls for the candidates within the specified time frame.
2. **Percentage Handling**: Standardize percentage values to ensure consistency.
   - If a percentage value (`pct`) is less than or equal to 1, it's assumed to be a proportion and multiplied by 100.
3. **Combined Weight Calculation**: Calculate the combined weight for each poll using the methods described in [Combining Weights](#8-combining-weights).
4. **Weighted Sum and Total Weights**:
   - **Weighted Sum**:
     $$
     \text{Weighted Sum}_c = \sum_{i \in c} W_{\text{combined}, i} \times \text{pct}_i
     $$
   - **Total Weight**:
     $$
     \text{Total Weight}_c = \sum_{i \in c} W_{\text{combined}, i}
     $$
5. **Weighted Average**:
   $$
   \text{Weighted Average}_c = \frac{\text{Weighted Sum}_c}{\text{Total Weight}_c}
   $$

**Margin of Error Calculation**:

- **Effective Sample Size**:
  $$
  n_{\text{effective}} = \sum_{i} W_{\text{combined}, i} \times n_i
  $$
- **Margin of Error**:
  $$
  \text{Margin of Error}_c = z \times \sqrt{\frac{p(1 - p)}{n_{\text{effective}}}} \times 100\%
  $$
  
  - **\( p \)**: Proportion (Weighted Average divided by 100).
  - **\( z \)**: Z-score (default is 1.96 for 95% confidence).

**Implementation**:

- Utilize Pandas for efficient data filtering, aggregation, and computation.
- Handle edge cases where `Total Weight` is zero to prevent division by zero errors.

**Example Calculation**:

For a candidate with the following polls:

| Poll | Combined Weight (\( W_{\text{combined}} \)) | Percentage (\( \text{pct} \)) |
|------|---------------------------------------------|--------------------------------|
| 1    | 0.8                                         | 50                             |
| 2    | 1.0                                         | 55                             |
| 3    | 0.6                                         | 45                             |

- **Weighted Sum**:
  $$
  \text{Weighted Sum} = (0.8 \times 50) + (1.0 \times 55) + (0.6 \times 45) = 40 + 55 + 27 = 122
  $$
  
- **Total Weight**:
  $$
  \text{Total Weight} = 0.8 + 1.0 + 0.6 = 2.4
  $$
  
- **Weighted Average**:
  $$
  \text{Weighted Average} = \frac{122}{2.4} \approx 50.83\%
  $$

- **Margin of Error**:
  Assuming \( n_{\text{effective}} = 0.8 \times 1000 + 1.0 \times 800 + 0.6 \times 1200 = 800 + 800 + 720 = 2320 \),
  $$
  \text{Margin of Error} = 1.96 \times \sqrt{\frac{0.5083 \times (1 - 0.5083)}{2320}} \times 100\% \approx 1.96 \times 0.0103 \times 100\% \approx 2.02\%
  $$

**Output**:

- **Weighted Average**: 50.83%
- **Margin of Error**: ±2.02%

### 10. Calculating Favorability Differential

**Objective**: To calculate a weighted favorability differential for each candidate, reflecting net public sentiment.

**Methodology**:

1. **Data Filtering**: Extract favorability polls relevant to the candidates.
2. **Normalization**: Standardize 'favorable' and 'unfavorable' percentages.
   - If a favorability percentage is less than or equal to 1, it's assumed to be a proportion and multiplied by 100.
3. **Combined Weight Calculation**: Calculate weights relevant to favorability data using similar methods as polling metrics.
4. **Weighted Favorability Differential**:
   $$
   \text{Favorability Differential}_c = \frac{\sum_{i \in c} W_{\text{combined}, i} \times \text{favorable}_i - \sum_{i \in c} W_{\text{combined}, i} \times \text{unfavorable}_i}{\sum_{i \in c} W_{\text{combined}, i}}
   $$

**Implementation**:

- Utilize Pandas for efficient data manipulation and aggregation.
- Handle cases where `Total Weight` is zero to prevent division by zero errors.

**Example Calculation**:

For a candidate with the following favorability polls:

| Poll | Combined Weight (\( W_{\text{combined}} \)) | Favorable (\( \text{favorable} \)) | Unfavorable (\( \text{unfavorable} \)) |
|------|---------------------------------------------|--------------------------------------|----------------------------------------|
| 1    | 0.9                                         | 60                                   | 30                                     |
| 2    | 1.1                                         | 55                                   | 35                                     |
| 3    | 0.7                                         | 50                                   | 40                                     |

- **Weighted Favorable Sum**:
  $$
  (0.9 \times 60) + (1.1 \times 55) + (0.7 \times 50) = 54 + 60.5 + 35 = 149.5
  $$
  
- **Weighted Unfavorable Sum**:
  $$
  (0.9 \times 30) + (1.1 \times 35) + (0.7 \times 40) = 27 + 38.5 + 28 = 93.5
  $$
  
- **Total Weight**:
  $$
  \text{Total Weight} = 0.9 + 1.1 + 0.7 = 2.7
  $$
  
- **Favorability Differential**:
  $$
  \text{Favorability Differential} = \frac{149.5 - 93.5}{2.7} \approx \frac{56}{2.7} \approx 20.74\%
  $$

**Output**:

- **Favorability Differential**: +20.74%

### 11. Combining Polling Metrics and Favorability Differential

**Objective**: To produce a final adjusted result by blending the weighted polling metrics with the favorability differential.

**Mathematical Formulation**:

$$
\text{Combined Result}_c = (1 - \alpha) \times \text{Polling Metric}_c + \alpha \times \text{Favorability Differential}_c
$$

- **\( \alpha \)**: Favorability Weight (default is `0.15`).

**Implementation**:

- Adjust the `FAVORABILITY_WEIGHT` in `config.py` or via the Streamlit app.
- Compute the final result for each candidate using the formula above.
- Utilize Pandas operations to merge and compute these metrics across the dataset efficiently.

**Example Calculation**:

Given:

- **Polling Metric**: 50.83%
- **Favorability Differential**: +20.74%
- **Favorability Weight (\( \alpha \))**: 0.15

The Combined Result would be:

$$
\text{Combined Result} = (1 - 0.15) \times 50.83 + 0.15 \times 20.74 \\
= 0.85 \times 50.83 + 0.15 \times 20.74 \\
= 43.188 + 3.111 \\
= 46.299\%
$$

**Output**:

- **Combined Result**: 46.30%

### 12. Out-of-Bag (OOB) Variance Calculation

**Objective**: To estimate the variance associated with the polling metrics using a Random Forest model.

**Methodology**:

- **Random Forest Model**: Utilize the `RandomForestRegressor` with `oob_score=True`.
- **Pipeline Components**:
  1. **Imputation**:
     - **Strategy**: `SimpleImputer` with a median strategy to handle missing data.
     - **Function**: Ensures that all feature columns are free of missing values before model training.
  2. **Model Training**:
     - **Model**: `RandomForestRegressor` with specified parameters from `config.py`.
     - **Parameters**:
       ```python
       N_TREES = 1000
       RANDOM_STATE = 42
       ```
  
- **OOB Variance**:
  $$
  \sigma_{\text{OOB}}^2 = \frac{1}{N} \sum_{i=1}^{N} \left( y_i - \hat{y}_i^{\text{OOB}} \right)^2
  $$
  - **\( y_i \)**: Actual value.
  - **\( \hat{y}_i^{\text{OOB}} \)**: OOB prediction.

**Justification**:

- The OOB error provides an unbiased estimate of the model's prediction error.
- Enhances the reliability of the analysis by quantifying uncertainty.

**Implementation**:

- **Data Preparation**:
  - Combine polling and favorability data into a single DataFrame.
  - Select relevant feature columns based on availability.
  
- **Pipeline Execution**:
  - Implement a `Pipeline` that first imputes missing data and then fits the Random Forest model.
  
- **Variance Calculation**:
  - After fitting, extract the OOB predictions and calculate the variance between actual values and OOB predictions.
  
- **Error Handling**:
  - If feature columns are missing or data is insufficient, log appropriate warnings and return a default variance value.

**Example Calculation**:

Suppose after model training:

- **Actual Values (\( y \))**: [52, 48, 50, 51, 49]
- **OOB Predictions (\( \hat{y}^{\text{OOB}} \))**: [51, 49, 50, 50, 50]

Then:

$$
\sigma_{\text{OOB}}^2 = \frac{(52-51)^2 + (48-49)^2 + (50-50)^2 + (51-50)^2 + (49-50)^2}{5} \\
= \frac{1 + 1 + 0 + 1 + 1}{5} \\
= \frac{4}{5} \\
= 0.8
$$

**Output**:

- **OOB Variance**: 0.8

**Implementation Note**:

- The variance is reported as a single numerical value representing the average squared difference between actual and predicted values, providing insight into the model's prediction accuracy.

---

## Error Handling and Normalization

To ensure mathematical integrity and robustness, the project includes comprehensive error handling and data normalization procedures.

**Key Strategies**:

- **Division by Zero Prevention**:
  - **Method**: Use of a small constant (`ZERO_CORRECTION = 0.0001`) to prevent division by zero in weight calculations.
  - **Implementation**: Applied in scenarios where the denominator could potentially be zero, such as when normalizing weights.

- **Missing Data Handling**:
  - **Method**: Assign default values or exclude data points with missing critical information.
  - **Implementation**: Utilizes Pandas functions like `dropna()` and `fillna()` to manage missing data effectively.

- **Percentage Interpretation**:
  - **Method**: Adjust percentages that are likely misformatted (e.g., values less than or equal to 1) by multiplying them by 100 to convert proportions to percentages.
  - **Implementation**: Applies lambda functions within Pandas `apply()` methods to standardize percentage values.

- **Time Calculations**:
  - **Method**: Utilize timezone-aware timestamps and fractional days to accurately compute time-related weights.
  - **Implementation**: Employs Pandas' `to_datetime()` with UTC time zones and calculates time differences in fractional days for precision.

**Justification**:

- These measures prevent computational errors and ensure that the analysis remains accurate and reliable.
- Proper handling of data anomalies enhances the robustness of the results.

**Implementation Example**:

```python
# Prevent division by zero by using ZERO_CORRECTION
if max_numeric_grade != 0:
    df['normalized_numeric_grade'] = df['numeric_grade'] / max_numeric_grade
else:
    df['normalized_numeric_grade'] = config.ZERO_CORRECTION

# Normalize and clip to [0, 1]
df['normalized_numeric_grade'] = df['normalized_numeric_grade'].clip(0, 1)
```

---

## Data Caching Mechanisms

To improve performance and user experience, the project implements data caching.

**Features**:

- **Caching Data Files**:
  - **Purpose**: Processed data is saved locally, reducing the need to re-fetch and re-process data on each run.
  - **Implementation**: Utilizes local CSV files to store processed polling and favorability data.

- **Configuration Cache**:
  - **Purpose**: User settings are cached to maintain consistency across sessions.
  - **Implementation**: Stores configuration parameters in a JSON file to persist user-defined settings.

- **Force Refresh Option**:
  - **Purpose**: Users can clear caches and refresh data to incorporate the latest information or configuration changes.
  - **Implementation**: Provides a checkbox in the Streamlit app to force data refresh, bypassing cached data.

**Justification**:

- Enhances performance, especially when dealing with large datasets or complex computations.
- Provides flexibility for users to control when data and settings are refreshed.

**Implementation Details**:

- **Cache Files**:
  - `sufficient_data.csv`: Stores the processed polling data that meets the minimum sample requirements.
  - `config.json`: Stores the current configuration parameters.
  - `results_df.csv`: Stores the results of the analysis across different periods.

- **Caching Strategy**:
  - On initial run, data is fetched, processed, and cached.
  - On subsequent runs, cached data is loaded if available and configuration parameters haven't changed.
  - If `force_refresh` is enabled, caches are cleared, and data is re-fetched and processed.

**Implementation Example**:

```python
def load_and_process_data(config_vars, force_refresh=False):
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
        
        results_df = get_analysis_results()
        sufficient_data_df = preprocess_data(results_df)
        
        save_cached_data(sufficient_data_df)
        save_cached_results_df(results_df)
        save_cached_config(config_vars)
        
        return sufficient_data_df, results_df
    except Exception as e:
        st.error(f"An error occurred while processing data: {e}")
        st.stop()
```

---

## Configuration Options

All configuration parameters are centralized in `config.py` and can be adjusted via the Streamlit app or directly in the file.

**Key Parameters**:

- **Candidates to Analyze**:

  ```python
  CANDIDATE_NAMES = ['Kamala Harris', 'Donald Trump']
  ```

- **Weight Multipliers**:

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

- **Favorability Weight**:

  ```python
  FAVORABILITY_WEIGHT = 0.15
  ```

- **Weighting Strategy**:

  ```python
  HEAVY_WEIGHT = True  # True for multiplicative, False for additive
  ```

- **Time Decay Parameters**:

  ```python
  DECAY_RATE = 1.0
  HALF_LIFE_DAYS = 14
  ```

- **Minimum Samples Required**:

  ```python
  MIN_SAMPLES_REQUIRED = 4
  ```

- **Partisan and Population Weights**:

  ```python
  PARTISAN_WEIGHT = {True: 0.01, False: 1.0}
  POPULATION_WEIGHTS = {
      'lv': 1.0,
      'rv': 0.75,
      'v': 0.5,
      'a': 0.25,
      'all': 0.01
  }
  ```

- **Random Forest Parameters**:

  ```python
  N_TREES = 1000
  RANDOM_STATE = 42
  ```

**Adjusting Configuration**:

- **Via `config.py`**:
  - Directly edit the `config.py` file to set desired values.
  
- **Via Streamlit App**:
  - The Streamlit interface provides sliders and input fields to adjust configuration parameters dynamically.
  - Changes made through the app are saved and cached, ensuring consistency across sessions.

**Implementation Example**:

```python
def configuration_form():
    with st.sidebar:
        # ... (UI components)
        with st.form("config_form"):
            favorability_weight = st.slider("Favorability Weight", 0.01, 1.0, float(config.FAVORABILITY_WEIGHT), 0.01)
            heavy_weight = st.checkbox("Heavy Weight", config.HEAVY_WEIGHT)
            # ... (additional configuration inputs)
            submitted = st.form_submit_button("Apply Changes and Run Analysis")

    if submitted:
        return {
            "FAVORABILITY_WEIGHT": favorability_weight,
            "HEAVY_WEIGHT": heavy_weight,
            # ... (additional configuration parameters)
        }
    return None
```

**Justification**:

- Centralizing configuration parameters allows for easy adjustments and experimentation.
- Providing both file-based and UI-based configuration options caters to different user preferences and workflows.

---

## Conclusion

By meticulously integrating multiple data sources and applying a comprehensive set of weighting factors—including the enhanced State Rank Weight that incorporates current forecasts—this project offers a detailed and accurate analysis of presidential polling data. The consideration of factors such as pollster quality, sample size, partisanship, population type, and state significance ensures that the adjusted poll results provide a realistic reflection of the electoral landscape.

**Key Strengths**:

- **Robust Methodology**: The use of mathematical models and justifiable weighting mechanisms enhances the credibility of the analysis.
- **Incorporation of Current Forecasts**: By integrating FiveThirtyEight's forecast data into the State Rank Weight, the model stays updated with the latest electoral dynamics.
- **Customizability**: Users can adjust parameters to explore different analytical perspectives or to align with specific research questions.
- **Interactivity**: The Streamlit app provides a user-friendly interface, making the analysis accessible to a broader audience.

**Impact of the Weighting Choices**:

- Each weighting factor addresses a specific aspect that can influence poll accuracy or relevance.
- The mathematical formulations are designed to be fair and justifiable, based on statistical principles and practical considerations.
- By providing transparency in the weighting mechanisms, users can understand and trust the adjustments made to the raw polling data.

---

## Possible Next Steps

To further enhance the project, several avenues can be explored:

1. **Sensitivity Analysis**:

   - **Objective**: Assess how changes in weight assignments and parameter values affect the final results.
   - **Method**: Systematically vary one parameter at a time while keeping others constant.
   - **Justification**: Helps identify which factors have the most significant impact and ensures robustness.

2. **Incorporation of Additional Data Sources**:

   - **Objective**: Enhance the comprehensiveness and robustness of the analysis by integrating more polling data.
   - **Method**: Fetch and process data from other reputable sources, ensuring proper alignment with existing datasets.
   - **Justification**: Diversifies the data pool and reduces potential biases from a single source.

3. **Advanced Modeling Techniques**:

   - **Objective**: Capture more complex patterns and relationships in the data.
   - **Method**: Implement machine learning models such as Gradient Boosting Machines, Neural Networks, or Bayesian models.
   - **Justification**: May improve predictive accuracy and provide deeper insights.

4. **Uncertainty Quantification**:

   - **Objective**: Provide more nuanced estimates of the uncertainty associated with predictions.
   - **Method**: Use techniques like bootstrap resampling, Bayesian credible intervals, or probabilistic models.
   - **Justification**: Enhances the interpretation of results, especially for decision-making purposes.

5. **User Interface and Visualization Enhancements**:

   - **Objective**: Improve the accessibility and interpretability of the analysis.
   - **Method**: Add interactive charts, maps, and explanatory texts to the Streamlit app.
   - **Justification**: Makes the analysis more engaging and easier to understand for non-technical users.

6. **Sophisticated Stratification Frame Construction**:

   - **Objective**: Enhance the representativeness of the sample by merging disparate data sources.
   - **Method**: Integrate demographic and socioeconomic data to create a more complete stratification frame.
   - **Justification**: Improves the accuracy of weight adjustments and the generalizability of results.

7. **Integration with Multiple Forecasting Models**:

   - **Objective**: Improve predictive performance by combining forecasts.
   - **Method**: Develop an ensemble method that averages or weights forecasts from multiple models.
   - **Justification**: Leverages the strengths of different models and mitigates individual weaknesses.

8. **Benchmarking Turnout Modeling Strategies**:

   - **Objective**: Evaluate and compare different approaches to modeling voter turnout.
   - **Method**: Implement alternative turnout models and assess their impact on results.
   - **Justification**: Ensures that the chosen approach is the most appropriate for the data and context.

9. **Documentation and Reporting**:

   - **Objective**: Maintain clear and up-to-date documentation to facilitate collaboration and transparency.
   - **Method**: Regularly update the readme and other documentation files to reflect new methodologies and findings.
   - **Justification**: Enhances reproducibility and fosters community engagement.

**Practical Steps**:

- **Data Preparation**: Acquire and preprocess new data sources, ensuring compatibility with existing structures.
- **Model Development**: Experiment with and implement advanced algorithms, testing their performance.
- **Evaluation Framework**: Establish clear metrics and validation procedures to assess improvements.
- **Iterative Testing**: Use cross-validation and other techniques to refine models and prevent overfitting.
- **Community Engagement**: Encourage feedback and contributions from other analysts and stakeholders.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- `pip` package manager

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/election-polling-analysis.git
   cd election-polling-analysis
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Analysis

1. **Execute the Core Analysis**:

   ```bash
   python analysis.py
   ```

   - This will fetch the latest polling and state data, perform the analysis, and output results to the console.

2. **Launch the Streamlit App**:

   ```bash
   streamlit run app.py
   ```

   - This will open the interactive user interface in your default web browser.

### Configuration

- **Adjusting Parameters**:
  - Modify `config.py` to change default parameters.
  - Alternatively, use the Streamlit app to dynamically adjust settings via the sidebar.

### Caching

- **Cached Data Files**:
  - Located in the `data` directory.
  - Includes `sufficient_data.csv`, `config.json`, and `results_df.csv`.

- **Clearing Cache**:
  - Use the "Force Refresh Data" option in the Streamlit app or manually delete cache files from the `data` directory.

### Documentation and Support

- **Project Documentation**:
  - Comprehensive details available in this `readme.md` file.
  
- **Support**:
  - For issues or feature requests, please open an issue on the [GitHub repository](https://github.com/yourusername/election-polling-analysis/issues).

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- **FiveThirtyEight**: For providing reliable and comprehensive polling and favorability data.
- **270 To Win**: For offering detailed state-specific electoral data.

---

## Contact

For any inquiries or feedback, please contact [polling@spencerthayer.com](mailto:polling@spencerthayer.com).