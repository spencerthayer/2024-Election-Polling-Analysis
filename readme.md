# Election Polling Analysis

This Python project is designed to fetch, process, and analyze presidential polling data. It consists of two main scripts: `analysis.py` and `states.py`. The `analysis.py` script fetches data from [FiveThirtyEight](https://projects.fivethirtyeight.com/)'s publicly available CSV files for [presidential polls](https://projects.fivethirtyeight.com/polls/data/president_polls.csv) and [favorability polls](https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv), and applies a series of weightings to adjust for various factors such as poll quality, partisanship, and sample population type. The `states.py` script scrapes data from the [270 To Win](https://www.270towin.com/) website to obtain information about the electoral votes and political leaning of each state to calculate the state-specific electoral significance.

## Data Acquisition

Presidential Polling Data: Sourced from [FiveThirtyEight](https://projects.fivethirtyeight.com/), this dataset is accessed via the Python `Requests` library, ensuring real-time relevance by incorporating the latest available data into a `Pandas` DataFrame for subsequent analysis.

Favorability Polling Data: Complementing presidential polling, [FiveThirtyEight](https://projects.fivethirtyeight.com/)'s favorability data offers insight into public sentiment regarding candidates, fetched similarly and integrated into the analysis to enhance depth.

State Data: The states data enriches the analysis by integrating state-specific electoral information, offering a more granular view of the electoral landscape.

## Weighting Calculations

The `analysis.py` script employs several mathematical principles to calculate the final weight of each poll. Below we will explore how these weights are derived and applied to the polling data to calculate the adjusted polls, which should result in a nuanced analysis of polling data.

### 1. Time Decay Weight

The weight of a poll decreases exponentially based on how long ago it was conducted, underpinning the principle that more recent polls are more reflective of the current state of public opinion.

$$ W_{time decay} = e^{-\lambda \cdot t} $$

- $W_{time decay}$: Weight of the poll at time $t$ days from the present.
- $\lambda$: Decay constant, calculated as $\lambda = \ln(\text{decay rate}) / \text{half life}$.
- $t$: Time in days between the poll's conduct date and the present day.

The half-life, $h$, represents the period over which the poll's weight is reduced by the decay rate, $r$, reflecting the decreasing influence of older polls over time. Thus, $\lambda$ is given by:

$$ \lambda = \frac{\ln(r)}{h} $$

To determine $t$ for each poll, the difference in days between the poll's conduct date and the current date is calculated. This adjustment ensures that the poll's weight is inversely proportional to its age, with more recent polls being given greater emphasis in the analysis.

$$ t = \text{Current Date} - \text{Poll Conduct Date} $$

### 2. Grade Weight

Polls are weighted based on the grade assigned to the polling organization, which reflects their historical accuracy and methodology quality. [FiveThirtyEight](https://projects.fivethirtyeight.com/) categorizes these grades, and in the script, each grade is associated with a specific numerical weight. This numerical weight translates the qualitative assessment of a poll's reliability into a quantitative factor that can be used in further calculations. The mapping from grades to numerical weights is as follows:

```
A+: 1.0, A: 0.9, A-: 0.8, A/B: 0.75, B+: 0.7, B: 0.6, B-: 0.5, B/C: 0.45, C+: 0.4, C: 0.3, C-: 0.2, C/D: 0.15, D+: 0.1, D: 0.05, D-: 0.025
```

Each grade is assigned a weight that diminishes as the grade decreases, with 'A+' polls being considered the most reliable (and thus given a weight of 1.0) and 'D-' polls being considered the least reliable (with a weight of 0.025). This numerical representation of grades allows for a standardized and objective approach to adjust the impact of each poll based on the credibility and track record of the polling organization.

For a poll with a specific grade, its grade weight $W_{grade}$ is directly fetched from this predefined mapping.

### 3. Transparency Weight

The transparency weight is calculated based on the transparency score provided in the polling data. The transparency score indicates the level of disclosure and methodological transparency of the polling organization. The transparency weight is computed by normalizing the transparency score of each poll with respect to the maximum transparency score among all polls.

$$ W_{transparency} = \frac{\text{Transparency Score}}{\text{Max Transparency Score}} $$

This normalization ensures that the transparency weight falls within the range [0, 1], with higher transparency scores resulting in higher weights.

### 4. Sample Size Weight

The sample size weight is calculated based on the sample size of each poll. Polls with larger sample sizes are generally considered more reliable and representative of the population. The sample size weight is computed by normalizing the sample size of each poll with respect to the minimum and maximum sample sizes among all polls.

$$ W_{sample size} = \frac{\text{Sample Size} - \text{Min Sample Size}}{\text{Max Sample Size} - \text{Min Sample Size}} $$

This normalization ensures that the sample size weight falls within the range [0, 1], with larger sample sizes resulting in higher weights.

### 5. Partisan Weight

Partisan-sponsored polls may have a bias toward their sponsor. The script applies a correction factor to account for this bias:

- If a poll is partisan (true), a weight of $0.1$ is applied.
- If a poll is non-partisan (false), a weight of $1$ is applied.

This adjustment, $W_{partisan}$, is applied directly based on the poll's partisanship status.

### 6. Population Weight

Different polls target different segments of the population (e.g., likely voters, registered voters). The reliability of these polls varies with the population segment, so weights are applied accordingly:

- Likely voters (lv): $1.0$
- Registered voters (rv): $\frac{2}{3}$
- Voters (v): $\frac{1}{2}$
- Adults (a): $\frac{1}{3}$
- All: $\frac{1}{3}$

This is formalized as $W_{population}(P)$ where $P$ stands for the population type of the poll.

### 7. State Rank Weight

The `states.py` script calculates a `state_rank` for each state based on its electoral significance and political leaning. The `state_rank` is a weighted sum of two components: the normalized electoral vote count and the partisan lean of the state.

The script first retrieves data from the [270 To Win](https://www.270towin.com/) website using web scraping techniques. It extracts information about each state's electoral votes and projected partisan lean (pro_status_code).

The partisan lean is mapped to a numerical value using the `pro_values` dictionary:

```python
    pro_values = {
        'T' : 0.8,  # Swing State
        'D1': 0.6,  # Tilts Democrat
        'D2': 0.4,  # Leans Democrat
        'D3': 0.2,  # Likely Democrat
        'D4': 0.1,  # Safe Democrat
        'R1': 0.6,  # Tilts Republican
        'R2': 0.4,  # Leans Republican
        'R3': 0.2,  # Likely Republican
        'R4': 0.1   # Safe Republican
    }
```

The electoral votes of each state are normalized by dividing them by $538$, the total number of electoral votes:

$$ \text{Normalized Electoral Votes} = \frac{\text{State Electoral Votes}}{\text{Total Electoral Votes}} $$

The state rank is then calculated as the sum of the normalized electoral votes and the partisan lean value:

$$ \text{State Rank} = \text{Normalized Electoral Votes} + \text{Partisan Lean Value} $$

Mathematically, this can be expressed as:

$$ R_s = \frac{E_s}{E_{total}} + P_s $$

Where:
- $R_s$: The rank of state $s$
- $E_s$: The number of electoral votes of state $s$
- $E_{total}$: The total number of electoral votes across all states (538)
- $P_s$: The projected partisan lean value of state $s$, based on the `pro_values` dictionary

The state rank is stored in the `state_data` dictionary, with the state name as the key and the rank as the value.

In the `analysis.py` script, the state rank is retrieved from the `state_data` dictionary based on the state name of each poll. The state rank is then used as a weight in the overall poll weighting calculation.

$$ W_{state} = R_s $$

Where $W_{state}$ is the state rank weight used in the `analysis.py` script.

By incorporating the state rank, which considers both the electoral significance and partisan lean of each state, the analysis takes into account the unique electoral dynamics of individual states. This allows for a more nuanced and comprehensive assessment of the polling data.

### 8. Combining Weights

After calculating individual weights, the combined weight of a poll is given by:

$$ W_{combined} = \prod_{i} W_i $$

Where $W_i$ represents each individual weight (time decay, grade, transparency, sample size, population, partisan, state rank).

Alternatively, if the `heavy_weight` parameter is set to `False`, the combined weight is calculated as the average of the individual weights:

$$ W_{combined} = \frac{\sum_{i} W_i}{n} $$

Where $n$ is the number of individual weights.

### 9. Calculating Polling Metrics

To calculate the adjusted poll results for each candidate, the script follows these steps:

1. Filter the polling data for the desired time period (e.g., last 12 months, last 6 months, etc.) and candidates (Kamala Harris and Donald Trump).
2. Calculate the individual weights for each poll based on the factors mentioned above.
3. Compute the combined weight for each poll using the selected method (`heavy_weight` or average).
4. Calculate the weighted sum of poll results for each candidate by multiplying the poll result percentage by the combined weight.
5. Sum the weighted poll results for each candidate.
6. Divide the weighted sum by the total combined weights for each candidate to obtain the weighted average poll result. This step includes careful handling of potential `NaN` (Not a Number) values that might arise due to missing data. `NaN` values are replaced with 0 to prevent them from propagating through calculations and causing errors.
7. Calculate the margin of error for each candidate's poll results using the `margin_of_error` function, which takes into account the sample size and poll result percentage.
8. Return a dictionary containing the weighted average poll result and margin of error for each candidate.

The `margin_of_error` function calculates the margin of error for a given sample size $n$ and population proportion estimate $p$ at a specified confidence level (default $95\%$):

$$ \text{MoE} = z \sqrt{\frac{p(1-p)}{n}} $$

Where $z$ is the z-score corresponding to the desired confidence level (e.g., 1.96 for 95% confidence).

### 10. Calculating Favorability Differential

To incorporate favorability polling data into the analysis, the script calculates the favorability differential for each candidate using the following steps:

1. Filter the favorability polling data for the desired time period and candidates.
2. Calculate the individual weights for each poll based on the grade, population, and time decay factors.
3. Compute the combined weight for each poll by multiplying the individual weights.
4. Calculate the weighted sum of favorability percentages for each candidate by multiplying the favorability percentage by the combined weight.
5. Sum the weighted favorability percentages for each candidate.
6. Divide the weighted sum by the total combined weights for each candidate to obtain the weighted average favorability percentage. This step, like in the polling metrics calculation, includes robust handling of potential `NaN` values.
7. Return a dictionary containing the weighted average favorability percentage for each candidate.

### 11. Combining Polling Metrics and Favorability Differential

The script combines the polling metrics and favorability differential using a weighted average approach:

$$ \text{Combined Result} = (1 - \alpha) \cdot \text{Polling Metric} + \alpha \cdot \text{Favorability Differential} $$

Where $\alpha$ is the `favorability_weight` parameter that determines the relative importance of the favorability differential in the combined result.

The margin of error for the combined result is directly obtained from the polling metrics.

## Out-of-Bag (OOB) Random Forest Implementation (Beta Test)

The `analysis.py` script incorporates an Out-of-Bag (OOB) Random Forest implementation to estimate the variance of the model's predictions. This approach leverages the inherent properties of Random Forests, where each tree is trained on a different bootstrap sample of the data, and the samples not included in the bootstrap sample (i.e., left out of the bag) are used to calculate the OOB error or variance.

#### Mathematical Formulation

Let's denote the training dataset as:

$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$$

Where $\mathbf{x}_i \in \mathbb{R}^d$ represents the input features and $y_i \in \mathbb{R}$ represents the target variable for the $i$-th sample. The Random Forest model consists of $M$ decision trees, where each tree $T_j$ is trained on a bootstrap sample $\mathcal{D}_j$ drawn with replacement from the original dataset $\mathcal{D}$.

For each tree $T_j$, the OOB sample $\mathcal{D}_j^{OOB}$ is defined as the set of samples from $\mathcal{D}$ that were not included in the bootstrap sample $\mathcal{D}_j$:

$$\mathcal{D}_j^{OOB} = \mathcal{D} \setminus \mathcal{D}_j$$

The OOB prediction for the $i$-th sample, denoted as $\hat{y}_i^{OOB}$, is obtained by averaging the predictions of the trees for which the $i$-th sample was in the OOB set:

$$\hat{y}_i^{OOB} = \frac{1}{|\{j: i \in \mathcal{D}_j^{OOB}\}|} \sum_{j: i \in \mathcal{D}_j^{OOB}} T_j(\mathbf{x}_i)$$

Where $T_j(\mathbf{x}_i)$ represents the prediction of the $j$-th tree for the input features $\mathbf{x}_i$.

The OOB variance, denoted as $\sigma_{OOB}^2$, is then calculated as the variance of the differences between the true target values and the OOB predictions:

$$ \sigma_{OOB}^2 = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i^{OOB})^2 $$

#### Implementation Details

In the `analysis.py` script, the OOB variance estimation is implemented as follows:

1. The Random Forest model is initialized with `n_estimators=100` (number of trees), `oob_score=True` (to enable OOB scoring), `random_state=42` (for reproducibility), and `bootstrap=True` (to perform bootstrap sampling).
2. The `impute_data(X)` function handles potential missing data during the OOB estimation by imputing missing values with the median of the corresponding feature.
3. The model is trained on the input features `X` and target variable `y` using the `fit()` method within a `Pipeline`, which includes the `impute_data` function to handle potential missing values.
4. The `_get_unsampled_indices()` function is defined to retrieve the indices of the OOB samples for each tree in the forest.
5. The OOB predictions are calculated by iterating over each tree in the forest, obtaining the unsampled indices using `_get_unsampled_indices()`, and accumulating the predictions for the OOB samples.
6. The OOB sample counts are computed using `np.bincount()` to determine the number of times each sample was included in the OOB sets.
7. The OOB predictions are divided by the OOB sample counts (with a small epsilon value added to avoid division by zero) to obtain the final OOB predictions.
8. The OOB variance is calculated using the formula: $\sigma_{OOB}^2 = \text{Var}(y - \hat{y}^{OOB})$, where $y$ represents the true target values and $\hat{y}^{OOB}$ represents the OOB predictions.

#### Interpretation and Implications

The OOB variance provides insights into the variability and reliability of the Random Forest model's predictions. Here are some key points to consider:

1. **Consistency Across Time Frames**: The OOB variance indicates the stability or variability of the model's predictions over different time periods leading up to an event (e.g., an election). A lower OOB variance suggests more consistent predictions across different bootstrap samples, while a higher OOB variance indicates greater variability.

2. **Model Reliability**: Generally, a lower OOB variance is desirable as it suggests that the model's predictions are relatively stable and not overly dependent on the specific subset of data used for training. However, some variability is expected, especially in complex datasets with many variables.

3. **Overfitting Detection**: Significantly low OOB variance might indicate that the model is overfitting the data, particularly if the predictive performance on external validation sets is poor. This would imply that the model is too closely tailored to the training data, including its noise, rather than capturing the underlying patterns applicable to unseen data.

4. **Predictive Power and Uncertainty**: The OOB results, when combined with the predicted percentages (e.g., H:41.55% vs. T:44.15%), provide a more comprehensive understanding of the model's predictive power and the uncertainty associated with those predictions. The variance measures how much the model's predictions for the outcome (favorability or voting percentages) might vary, adding a layer of understanding to the confidence in the model's output.

The OOB variance should be used as a tool for model evaluation alongside other metrics, guiding model improvement efforts, informing decision-making, and monitoring temporal dynamics. It offers valuable insights into the reliability and stability of the Random Forest model's predictions, aiding in the interpretation and utilization of the model's outputs.

## Error Handling

The script incorporates robust error handling to address various potential issues, particularly related to missing or incomplete data. Here's a summary of the error handling strategies implemented:

- **Data Download Errors:** The `download_csv_data` function uses a `try-except` block to catch potential errors during data download, such as network issues or invalid URLs. If an error occurs, the script prints an error message and returns an empty DataFrame to prevent the script from crashing.
- **Missing 'Population' Column:** The `preprocess_data` function includes a warning message if the 'population' column is missing from the DataFrame. In this case, the 'population_weight' is set to 1 for all rows.
- **Missing Values in Calculations:**  The `calculate_polling_metrics` and `calculate_favorability_differential` functions use `.fillna(0)` to replace `NaN` values that might arise during intermediate calculations (e.g., in `weighted_sums` or `total_weights`). This prevents `NaN` values from propagating through the calculations and ensures that the script produces numerical outputs even if there are missing values.
- **Insufficient Data for OOB Estimation:** The script checks for sufficient data both before and after the OOB variance calculation. If there are not enough samples for reliable estimation (`X.shape[0] < min_samples_required`), the script prints an informative message indicating that there is insufficient data for the specific time period. This is done to handle cases where the `NaN` values might originate from the OOB estimation process itself, even if there was enough initial data.

## Output

The `analysis.py` script processes the polling and favorability data for different time periods (e.g., 12 months, 6 months, 3 months, 21 days, 14 days, 7 days, 3 days, and 1 day) and prints the analyzed results for each period. The output includes the weighted averages and margins of error for each candidate, `Harris` and `Trump`, the differential between them, and the favored candidate based on the differential. The output is color-coded based on the time period to provide a visual representation of the trends.

## Conclusion

By incorporating favorability data, state-specific weights, and various other factors into the analysis, this project provides a nuanced and comprehensive assessment of presidential polling data. The integration of data from `states.py` allows for the consideration of each state's unique electoral dynamics, ensuring that the adjusted poll results reflect the significance and political leanings of individual states.

This approach aims to strike a balance between the broad insights provided by national polls, the detailed, state-specific information captured by local polls, and the additional context provided by favorability ratings. By carefully normalizing and combining these various weights, the scripts produce adjusted results that offer a more accurate and representative picture of the current state of the presidential race.

As with any polling analysis, there is always room for further refinement and improvement. The modular design of the scripts allows for the incorporation of additional factors and adjustments as needed. Collaboration and feedback from the community are welcome to enhance the methodology and ensure the most accurate and meaningful analysis possible.

---

## Possible Next Steps

To further enhance the project, several next steps can be considered:

1. **Sensitivity analysis**: Conduct sensitivity analyses to assess the impact of different weight assignments and parameter values on the final results. This will help identify the most influential factors and guide the refinement of the weighting scheme.

2. **Incorporation of additional data sources**: Explore and integrate polling data from other reputable sources to enhance the robustness and reliability of the analysis. This may involve adapting the data processing pipeline to handle different data formats and structures.

3. **Exploration of advanced modeling techniques**: Investigate and experiment with advanced modeling techniques, such as time series analysis or machine learning algorithms, to capture more complex patterns and relationships in the polling data.

4. **Uncertainty quantification**: Implement techniques like bootstrapping or Bayesian inference to quantify the uncertainty associated with the adjusted poll results. This will provide a more comprehensive understanding of the range of possible outcomes and help convey the level of confidence in the estimates.

5. **User interface and visualization**: Develop a user-friendly interface and data visualization components to make the project more accessible and informative to a wider audience. This could include interactive dashboards, maps, and charts that allow users to explore the polling data and adjusted results in a more intuitive manner.

6. **Sophisticated Stratification Frame Construction**: Introduce data integration techniques such as multiple imputation to merge disparate data sources, enhancing the completeness and representativeness of the stratification frame. This involves identifying common variables across datasets, using them to predict missing values, and creating a more comprehensive dataset that improves upon relying solely on census tables.

7. **Advanced Weighting with Machine Learning**: ~~Replace traditional raking methods with machine learning algorithms such as random forests for weighting. This approach would involve using random forests to identify and automatically adjust for interactions between variables at both individual and aggregate levels, which could enhance the model's accuracy and reduce specification errors.~~

8. **Incorporation of Uncertainty Estimates**: ~~Apply techniques like the infinitesimal jackknife and out-of-bag neighbors within a random forest framework to estimate uncertainty around cell-level predictions. This addition would require modifications to the forecasting model to include these techniques and present uncertainty estimates alongside the forecasts, offering a more nuanced view of the predictions' reliability.~~

9. **Integration with Election Forecasting Models from Multiple Sources**: Develop an ensemble method that averages forecasts from multiple online sample sources. This strategy would involve aggregating predictions from these varied sources and calculating a weighted average, aiming to cancel out differential biases and improve overall forecast accuracy.

10. **Benchmarking Turnout Modeling Strategies**: Explore and benchmark alternative turnout modeling strategies, including ecological regression on historical data, against the script’s current post-stratification approach. This would involve setting up a comparative analysis framework to evaluate the performance of different turnout modeling techniques on historical election data to identify the most effective method.

**Practical Steps**:
- **Data Preparation**: Integrate multiple data sources, including online non-probability samples, ensuring they are preprocessed and harmonized for analysis.
- **Model Development**: Adapt the script to incorporate random forest algorithms for weighting, along with implementing methods for uncertainty estimation.
- **Evaluation Framework**: Establish metrics and validation procedures to assess the performance of new strategies against existing methods, particularly focusing on forecast accuracy and uncertainty quantification.
- **Iterative Testing**: Systematically test and refine the enhanced forecasting model, using historical data and real-time predictions to validate improvements.
- **Documentation and Reporting**: Update the script’s documentation to reflect new methodologies and findings, including detailed explanations of the techniques used and the rationale behind their selection.