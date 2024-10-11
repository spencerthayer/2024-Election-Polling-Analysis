# Election Polling Analysis

This Python project is designed to fetch, process, and analyze presidential polling data. It consists of two main scripts: `analysis.py` and `states.py`. The `analysis.py` script fetches data from [FiveThirtyEight](https://projects.fivethirtyeight.com/)'s publicly available CSV files for [presidential polls](https://projects.fivethirtyeight.com/polls/data/president_polls.csv) and [favorability polls](https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv), and applies a series of weightings to adjust for various factors such as poll quality, partisanship, and sample population type. The `states.py` script scrapes data from the [270 To Win](https://www.270towin.com/) website to obtain information about the electoral votes and political leaning of each state to calculate the state-specific electoral significance.

## File Structure

```
[drwxr-xr-x  74k]  .
[-rw-r--r--  147]  ├── ./Election-Polling.code-workspace
[-rw-r--r--  18k]  ├── ./analysis.py
[-rw-r--r-- 6.9k]  ├── ./app.py
[-rw-r--r-- 2.7k]  ├── ./config.py
[-rw-r--r-- 2.4k]  ├── ./install
[-rw-r--r--  26k]  ├── ./readme.md
[-rw-r--r--   88]  ├── ./requirements.txt
[-rwxr-xr-x  642]  ├── ./start
[-rw-r--r-- 3.6k]  ├── ./states.py
[-rwxr-xr-x 1.9k]  └── ./streamlit
```

## Data Acquisition

- **Presidential Polling Data**: Sourced from [FiveThirtyEight](https://projects.fivethirtyeight.com/), this dataset is accessed via the Python `requests` library, ensuring real-time relevance by incorporating the latest available data into a `pandas` DataFrame for subsequent analysis.

- **Favorability Polling Data**: Complementing presidential polling, [FiveThirtyEight](https://projects.fivethirtyeight.com/)'s favorability data offers insight into public sentiment regarding candidates, fetched similarly and integrated into the analysis to enhance depth.

- **State Data**: The `states.py` script enriches the analysis by integrating state-specific electoral information, offering a more granular view of the electoral landscape.

## Weighting Calculations

The `analysis.py` script employs several mathematical principles to calculate the final weight of each poll. Below we will explore how these weights are derived and applied to the polling data to calculate the adjusted polls, resulting in a nuanced analysis of polling data.

### 1. Time Decay Weight

The weight of a poll decreases exponentially based on how long ago it was conducted, ensuring that more recent polls have greater influence.

In the updated version, the calculation of the time decay weight uses fractional days to increase precision:

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
  \lambda = \frac{\ln(r)}{h}
  $$

  - \( r \) is the decay rate.
  - \( h \) is the half-life in days.

This change from integer days to fractional days ensures a more precise weighting, especially for recent polls.

### 2. Grade Weight

Polls are weighted based on the grade assigned to the polling organization, reflecting their historical accuracy and methodological quality.

**Normalization with Error Handling:**

In the updated version, the normalization of the numeric grades includes error handling to prevent division by zero:

- If the maximum numeric grade is zero, the normalized numeric grade is set to 1.0 to avoid division by zero.

The normalized numeric grade is calculated as:

$$
W_{\text{grade}} = \frac{\text{Numeric Grade}}{\text{Max Numeric Grade}}
$$

### 3. Transparency Weight

The transparency weight is calculated based on the transparency score provided in the polling data. The transparency score indicates the level of disclosure and methodological transparency of the polling organization. The transparency weight is computed by normalizing the transparency score of each poll with respect to the maximum transparency score among all polls.

$$
W_{\text{transparency}} = \frac{\text{Transparency Score}}{\text{Max Transparency Score}}
$$

This normalization ensures that the transparency weight falls within the range [0, 1], with higher transparency scores resulting in higher weights.

- If the maximum transparency score is zero, the transparency weight is set to 1.0 to prevent division by zero.

### 4. Sample Size Weight

The sample size weight is calculated based on the sample size of each poll. Polls with larger sample sizes are generally considered more reliable and representative of the population. The sample size weight is computed by normalizing the sample size of each poll with respect to the minimum and maximum sample sizes among all polls.

$$
W_{\text{sample size}} = \frac{\text{Sample Size} - \text{Min Sample Size}}{\text{Max Sample Size} - \text{Min Sample Size}}
$$

This normalization ensures that the sample size weight falls within the range [0, 1], with larger sample sizes resulting in higher weights.

- If \(\text{Max Sample Size} - \text{Min Sample Size} = 0\), the sample size weight is set to 1.0 to prevent division by zero.

### 5. Partisan Weight

Partisan-sponsored polls are adjusted using a correction factor:

- Partisan polls: weight of 0.01.
- Non-partisan polls: weight of 1.0.

This is formalized as:

$$
W_{\text{partisan}} =
\begin{cases}
0.01, & \text{if the poll is partisan} \\
1.0, & \text{if the poll is non-partisan}
\end{cases}
$$

### 6. Population Weight

Polls targeting different population types are weighted accordingly:

- Likely voters (lv): 1.0
- Registered voters (rv): \( \frac{2}{3} \) or approximately 0.6667
- Voters (v): 0.5
- Adults (a): \( \frac{1}{3} \) or approximately 0.3333
- All: \( \frac{1}{3} \) or approximately 0.3333

This can be expressed as:

$$
W_{\text{population}}(P) =
\begin{cases}
1.0, & \text{if } P = \text{lv} \\
0.6667, & \text{if } P = \text{rv} \\
0.5, & \text{if } P = \text{v} \\
0.3333, & \text{if } P = \text{a or all}
\end{cases}
$$

Where \( P \) represents the population type of the poll.

### 7. State Rank Weight

The state rank is calculated using the `states.py` script and considers both the electoral votes and the partisan lean of each state.

The state rank is calculated as:

$$
W_{\text{state}} = R_s = \frac{E_s}{E_{\text{total}}} + P_s
$$

Where:

- \( R_s \): The rank of state \( s \).
- \( E_s \): The number of electoral votes of state \( s \).
- \( E_{\text{total}} \): The total number of electoral votes across all states (538).
- \( P_s \): The projected partisan lean value of state \( s \), based on the `pro_values` dictionary.

**Note on Error Handling:**

- If a state is not found in the `state_data`, a default rank of 1.0 is assigned to prevent errors.

### 8. **Combining Weights**

The script uses multiplicative or additive combinatorial methods based on the `HEAVY_WEIGHT` parameter, combining individual weights such as time decay, pollster grade, transparency, sample size, population, partisan status, and state rank. In the `calculate_polling_metrics` function of the `analysis.py` script, the combined weight is calculated as follows:

```python
# Prepare the weights with multipliers
list_weights = np.array([
    df['time_decay_weight'] * config.TIME_DECAY_WEIGHT_MULTIPLIER,
    df['sample_size_weight'] * config.SAMPLE_SIZE_WEIGHT_MULTIPLIER,
    df['normalized_numeric_grade'] * config.NORMALIZED_NUMERIC_GRADE_MULTIPLIER,
    df['normalized_pollscore'] * config.NORMALIZED_POLLSCORE_MULTIPLIER,
    df['normalized_transparency_score'] * config.NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER,
    df['population_weight'] * config.POPULATION_WEIGHT_MULTIPLIER,
    df['partisan_weight'] * config.PARTISAN_WEIGHT_MULTIPLIER,
    df['state_rank'] * config.STATE_RANK_MULTIPLIER,
])

# Combine weights according to the HEAVY_WEIGHT flag
if config.HEAVY_WEIGHT:
    df['combined_weight'] = np.prod(list_weights, axis=0)
else:
    df['combined_weight'] = np.mean(list_weights, axis=0)
```

### **Explanation of the Weight Combination**

- **When `HEAVY_WEIGHT` is `True`:**

  The combined weight is calculated by **multiplying** all individual weights:

  $$ W_{\text{combined}} = \prod_{i} W_i $$

  This multiplicative approach emphasizes the impact of each weight. If any individual weight is small (e.g., close to zero), it will significantly reduce the combined weight.

- **When `HEAVY_WEIGHT` is `False`:**

  The combined weight is calculated by **averaging** all individual weights:

  $$ W_{\text{combined}} = \frac{\sum_{i} W_i}{n} $$

  where \( n \) is the number of individual weights.

  This additive approach balances the weights, ensuring that no single weight disproportionately influences the combined weight.

### **Implications of the Weighting Methods**

- **Multiplicative Weighting (`HEAVY_WEIGHT = True`):**

  - **Pros:**
    - Ensures that all criteria must be met to achieve a high combined weight.
    - Any low individual weight will reduce the overall weight, highlighting polls that excel across all metrics.

  - **Cons:**
    - Can overly penalize polls due to a single low weight.
    - May result in very small combined weights, making it harder for differences to emerge.

- **Additive Weighting (`HEAVY_WEIGHT = False`):**

  - **Pros:**
    - Allows for compensations between weights; a low score in one area can be offset by higher scores in others.
    - Provides a more balanced assessment when individual weights vary.

  - **Cons:**
    - May not penalize polls sufficiently for poor performance in critical areas.
    - Can dilute the impact of particularly important weights.

### **Adjusting the Weighting Strategy**

You can control how the weights are combined by setting the `HEAVY_WEIGHT` parameter in your `config.py` file:

```python
# config.py

HEAVY_WEIGHT = True  # Use multiplicative weighting
# or
HEAVY_WEIGHT = False  # Use additive weighting
```

### **Customizing Individual Weights**

Additionally, you can adjust the multipliers for each individual weight to fine-tune their impact:

```python
# config.py

TIME_DECAY_WEIGHT_MULTIPLIER = 1.0
SAMPLE_SIZE_WEIGHT_MULTIPLIER = 1.0
NORMALIZED_NUMERIC_GRADE_MULTIPLIER = 1.0
NORMALIZED_POLLSCORE_MULTIPLIER = 1.0
NORMALIZED_TRANSPARENCY_SCORE_MULTIPLIER = 1.0
POPULATION_WEIGHT_MULTIPLIER = 1.0
PARTISAN_WEIGHT_MULTIPLIER = 1.0
STATE_RANK_MULTIPLIER = 1.0
```

By increasing or decreasing these multipliers, you can amplify or diminish the influence of specific weights in the combined calculation.

### 9. Calculating Polling Metrics

To calculate the adjusted poll results for each candidate:

1. **Data Filtering**: Polling data is filtered based on the time period and candidates.

2. **Percentage Handling**: Ensures that percentages are correctly interpreted. If the 'pct' value is less than or equal to 1, it is assumed to be in decimal form and multiplied by 100.

3. **Weight Calculations**: All individual weights are calculated with robust error handling to prevent division by zero.

4. **Weighted Averages**: The weighted sum of poll results is divided by the total combined weights, with careful handling of potential `NaN` values.

   The weighted average for each candidate is calculated as:

   $$
   \text{Weighted Average}_c = \frac{\sum_{i \in c} W_{\text{combined}, i} \cdot \text{pct}_i}{\sum_{i \in c} W_{\text{combined}, i}}
   $$

   Where:

   - \( c \) represents the candidate.
   - \( W_{\text{combined}, i} \) is the combined weight for poll \( i \).
   - \( \text{pct}_i \) is the poll percentage for poll \( i \).

5. **Margin of Error Calculation**: The margin of error is calculated using the standard formula, considering the sample size and proportion.

   $$
   \text{Margin of Error} = z \cdot \sqrt{\frac{p(1 - p)}{n}} \times 100\%
   $$

   Where:

   - \( z \) is the z-score corresponding to the desired confidence level (e.g., 1.96 for 95% confidence).
   - \( p \) is the estimated proportion (in decimal form).
   - \( n \) is the sample size.

### 10. Calculating Favorability Differential

Favorability data is incorporated similarly, with updated normalization and error handling:

- **Normalization**: All scores are normalized with checks to prevent division by zero.

- **Combined Weight**: Calculated using the product of the normalized weights.

- **Weighted Averages**: The weighted favorability percentages are calculated with proper handling of missing data.

The weighted average favorability for each candidate is calculated as:

$$
\text{Weighted Favorability}_c = \frac{\sum_{i \in c} W_{\text{combined}, i} \cdot \text{Favorable}_i}{\sum_{i \in c} W_{\text{combined}, i}}
$$

### 11. Combining Polling Metrics and Favorability Differential

The polling metrics and favorability differentials are combined using a weighted average:

$$
\text{Combined Result}_c = (1 - \alpha) \cdot \text{Polling Metric}_c + \alpha \cdot \text{Favorability Differential}_c
$$

Where:

- \( \alpha \) is the `FAVORABILITY_WEIGHT`.
- \( c \) represents the candidate.

The margin of error for the combined result is directly obtained from the polling metrics.

### 12. Out-of-Bag (OOB) Variance Calculation

The OOB variance is calculated using the built-in `oob_prediction_` attribute of `RandomForestRegressor`, simplifying the process and reducing potential errors.

- **Methodology**:

  - The Random Forest model is trained with `oob_score=True`.
  - The OOB predictions are obtained directly from `model.oob_prediction_`.
  - The OOB variance is calculated as the variance of the residuals between actual values and OOB predictions.

  $$
  \sigma_{\text{OOB}}^2 = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i^{\text{OOB}})^2
  $$

  Where:

  - \( N \) is the number of samples.
  - \( y_i \) is the actual value for sample \( i \).
  - \( \hat{y}_i^{\text{OOB}} \) is the OOB prediction for sample \( i \).

- **Benefits**:

  - Simplifies the calculation.
  - Reduces potential implementation errors.
  - Leverages optimized internal methods for accuracy.

## Error Handling

The updated script includes robust error handling to ensure mathematical integrity:

- **Division by Zero Prevention**: Checks are added before divisions to prevent division by zero, setting default values when necessary.

- **Missing Data Handling**: Missing or `NaN` values are handled gracefully, with default assignments to ensure calculations proceed correctly.

- **Percentage Interpretation**: Improved handling of percentages less than or equal to 1 to avoid misclassification.

- **Time Calculations**: Uses fractional days and ensures timestamps are timezone-aware to prevent errors in time decay calculations.

## Conclusion

By incorporating favorability data, state-specific weights, and various other factors into the analysis, this project provides a nuanced and comprehensive assessment of presidential polling data. The integration of data from `states.py` allows for the consideration of each state's unique electoral dynamics, ensuring that the adjusted poll results reflect the significance and political leanings of individual states.

This approach aims to strike a balance between the broad insights provided by national polls, the detailed, state-specific information captured by local polls, and the additional context provided by favorability ratings. By carefully normalizing and combining these various weights, the scripts produce adjusted results that offer a more accurate and representative picture of the current state of the presidential race.

As with any polling analysis, there is always room for further refinement and improvement. The modular design of the scripts allows for the incorporation of additional factors and adjustments as needed. Collaboration and feedback from the community are welcome to enhance the methodology and ensure the most accurate and meaningful analysis possible.

---

## Possible Next Steps

To further enhance the project, several next steps can be considered:

1. **Sensitivity Analysis**: Assess the impact of different weight assignments and parameter values on the final results.

2. **Incorporation of Additional Data Sources**: Integrate polling data from other reputable sources to enhance robustness.

3. **Advanced Modeling Techniques**: Explore advanced modeling techniques to capture more complex patterns.

4. **Uncertainty Quantification**: Implement techniques to quantify the uncertainty associated with the adjusted poll results.

5. **User Interface and Visualization**: Develop a user-friendly interface and data visualization components.

6. **Sophisticated Stratification Frame Construction**: Introduce data integration techniques to merge disparate data sources, enhancing the completeness of the stratification frame.

7. **Incorporation of Uncertainty Estimates**: Apply techniques to estimate uncertainty around predictions, offering a nuanced view of reliability.

8. **Integration with Multiple Forecasting Models**: Develop an ensemble method that averages forecasts from multiple sources.

9. **Benchmarking Turnout Modeling Strategies**: Explore and benchmark alternative turnout modeling strategies against the current approach.

**Practical Steps**:

- **Data Preparation**: Integrate multiple data sources, ensuring they are preprocessed and harmonized.

- **Model Development**: Adapt the script to incorporate advanced algorithms and uncertainty estimation methods.

- **Evaluation Framework**: Establish metrics and validation procedures to assess performance.

- **Iterative Testing**: Systematically test and refine the forecasting model.

- **Documentation and Reporting**: Update documentation to reflect new methodologies and findings.