# Election Polling Analysis

This Python project is designed to fetch, process, and analyze presidential polling data. It consists of two main scripts: `polls.py` and `states.py`. The `polls.py` script fetches data from FiveThirtyEight's publicly available [CSV file](https://projects.fivethirtyeight.com/polls/data/president_polls.csv) and applies a series of weightings to adjust for various factors such as poll quality, partisanship, sample population type, and state-specific electoral significance. The `states.py` script scrapes data from the "270towin.com" website to obtain information about the electoral votes and political leaning of each state.

## Data Acquisition

The `polls.py` script fetches polling data using the Python `Requests` library. The data is then loaded into a `Pandas` DataFrame for analysis. This approach ensures that the analysis is always up-to-date with the latest available data.

The `states.py` script uses the `BeautifulSoup` library to scrape data from the "270towin.com" website. It parses the scraped data to extract the relevant information about each state's electoral votes and political leaning.

## Weighting Calculations

The `polls.py` script employs several mathematical principles to calculate the final weight of each poll. Below we will explore how these weights are derived and applied to the polling data to calculate the adjusted polls, which should result in a nuanced analysis of polling data.

### 1. Time Weights

The weight of a poll decreases exponentially based on how long ago it was conducted, underpinning the principle that more recent polls are more reflective of the current state of public opinion.

$$ w(t) = e^{-\lambda \cdot t} $$

- $w(t)$: Weight of the poll at time $t$ days from the present.
- $\lambda$: Decay constant, calculated as $\lambda = \ln(\text{rate-of-decay}) / \text{half-life}$.
- $t$: Time in days between the poll's conduct date and the present day.

The half-life, $h$, represents the period over which the poll's weight is reduced by the rate-of-decay, $r$, reflecting the decreasing influence of older polls over time. Thus, $\lambda$ is given by:

$$ \lambda = \frac{\ln(\text{r})}{h} $$

To determine $t$ for each poll, the difference in days between the poll's conduct date and the current date is calculated. This adjustment ensures that the poll's weight is inversely proportional to its age, with more recent polls being given greater emphasis in the analysis.

$$ t = \text{Current Date} - \text{Poll Conduct Date} $$

### 2. Grade Weights Calculation

Polls are weighted based on the grade assigned to the polling organization, which reflects their historical accuracy and methodology quality. FiveThirtyEight categorizes these grades, and in the script, each grade is associated with a specific numerical weight. This numerical weight translates the qualitative assessment of a poll's reliability into a quantitative factor that can be used in further calculations. The mapping from grades to numerical weights is as follows:

```
A+: 1.0, A: 0.9, A-: 0.8, A/B: 0.75, B+: 0.7, B: 0.6, B-: 0.5, B/C: 0.45, C+: 0.4, C: 0.3, C-: 0.2, C/D: 0.15, D+: 0.1, D: 0.05, D-: 0.025
```

Each grade is assigned a weight that diminishes as the grade decreases, with 'A+' polls being considered the most reliable (and thus given a weight of 1.0) and 'D-' polls being considered the least reliable (with a weight of 0.025). This numerical representation of grades allows for a standardized and objective approach to adjust the impact of each poll based on the credibility and track record of the polling organization.

For a poll with a specific grade, its grade weight is directly fetched from this predefined mapping. This explicit numerical representation ensures that the weight calculation process is transparent and consistent across all polls analyzed.

### 3. Transparency Weight

The transparency weight is calculated based on the transparency score provided in the polling data. The transparency score indicates the level of disclosure and methodological transparency of the polling organization. The transparency weight is computed by normalizing the transparency score of each poll with respect to the maximum transparency score among all polls.

$$ W_{transparency} = \frac{\text{Transparency Score}}{\text{Max Transparency Score}} $$

This normalization ensures that the transparency weight falls within the range [0, 1], with higher transparency scores resulting in higher weights.

### 4. Sample Size Weight

The sample size weight is calculated based on the sample size of each poll. Polls with larger sample sizes are generally considered more reliable and representative of the population. The sample size weight is computed by normalizing the sample size of each poll with respect to the minimum and maximum sample sizes among all polls.

$$ W_{sample size} = \frac{\text{Sample Size} - \text{Min Sample Size}}{\text{Max Sample Size} - \text{Min Sample Size}} $$

This normalization ensures that the sample size weight falls within the range [0, 1], with larger sample sizes resulting in higher weights.

### 5. Partisan Weight Adjustment

Partisan-sponsored polls may have a bias toward their sponsor. The script applies a correction factor to account for this bias:

- If a poll is partisan (true), a weight of $0.1$ is applied.
- If a poll is non-partisan (false), a weight of $1$ is applied.

This adjustment, $W_{partisan}$, is applied directly based on the poll's partisanship status.

### 6. Population Sample Weights

Different polls target different segments of the population (e.g., likely voters, registered voters). The reliability of these polls varies with the population segment, so weights are applied accordingly:

- Likely voters (lv): $1.0$
- Registered voters (rv): $\frac{2}{3}$
- Voters (v): $\frac{1}{2}$
- Adults (a): $\frac{1}{3}$
- All: $\frac{1}{3}$

This is formalized as $W_{population}(P)$ where $P$ stands for the population type of the poll.

### 7. State-Specific Weights

To incorporate state-specific polling data into the broader analysis, the script integrates data from `states.py`, which calculates a `state_rank` based on political projections and the proportion of electoral votes for each state. This rank represents the state's electoral significance and political leaning.

The state rank $R_s$ is calculated as a weighted sum of two components: the normalized electoral vote count and the partisan lean of the state.

$$ R_s = \alpha \cdot \frac{E_s}{E_{total}} + \beta \cdot P_s $$

Where:
- $R_s$: The rank of state $s$
- $E_s$: The number of electoral votes of state $s$
- $E_{total}$: The total number of electoral votes across all states
- $P_s$: The projected partisan lean of state $s$, where $P_s \in [0, 1]$. A value of 0 indicates a strong Democratic lean, while a value of 1 indicates a strong Republican lean.
- $\alpha$: A weighting factor for the electoral vote component
- $\beta$: A weighting factor for the partisan lean component, and $\alpha + \beta = 1$

The weighting factors $\alpha$ and $\beta$ determine the relative importance of the electoral vote count and the partisan lean in the overall state ranking. These factors can be adjusted based on the desired emphasis on each component.

### 8. Combining Weights

After calculating individual weights, the combined weight of a poll is given by:

$$ W_{combined} = W_{time decay} \times W_{grade} \times W_{transparency} \times W_{sample size} \times W_{population} \times W_{partisan} \times W_{state} $$

This formula incorporates the time decay weight, grade weight, transparency weight, sample size weight, population weight, partisan weight, and state-specific weight to provide a comprehensive assessment of each poll's significance.

### 9. Calculating Adjusted Poll Results

To calculate the adjusted poll results for each candidate, the script follows these steps:

1. Filter the polling data for the desired time period (e.g., last 12 months, last 6 months, etc.) and candidates (Joe Biden and Donald Trump).
2. Calculate the individual weights for each poll based on the factors mentioned above.
3. Compute the combined weight for each poll by multiplying the individual weights.
4. Calculate the weighted sum of poll results for each candidate by multiplying the poll result percentage by the combined weight.
5. Sum the weighted poll results for each candidate.
6. Divide the weighted sum by the total combined weights for each candidate to obtain the weighted average poll result.
7. Calculate the differential between the weighted average poll results of the two candidates.
8. Determine the favored candidate based on the differential.
9. Print the results, including the weighted averages for each candidate, the differential, and the favored candidate, using colored output.

## Output

The `polls.py` script processes the polling data for different time periods (e.g., 12 months, 6 months, 3 months, 21 days, 14 days, etc.) and prints the analyzed results for each period. The output includes the weighted averages for each candidate (Biden and Trump), the differential between them, and the favored candidate based on the differential. The output is color-coded based on the time period to provide a visual representation of the trends.

## Conclusion

By incorporating state-specific weights and various other factors into the analysis, this project provides a nuanced and comprehensive assessment of presidential polling data. The integration of data from `states.py` allows for the consideration of each state's unique electoral dynamics, ensuring that the adjusted poll results reflect the significance and political leanings of individual states.

This approach aims to strike a balance between the broad insights provided by national polls and the detailed, state-specific information captured by local polls. By carefully normalizing and combining these various weights, the scripts produce adjusted results that offer a more accurate and representative picture of the current state of the presidential race.

As with any polling analysis, there is always room for further refinement and improvement. The modular design of the scripts allows for the incorporation of additional factors and adjustments as needed. Collaboration and feedback from the community are welcome to enhance the methodology and ensure the most accurate and meaningful analysis possible.

## Criticisms and Possible Next Steps

While this project provides a comprehensive approach to analyzing presidential polling data, there are some potential criticisms and areas for improvement:

1. **Subjectivity in weight assignments**: The assignment of weights to various factors, such as poll grades and population types, involves a degree of subjectivity. Different analysts might assign different weights based on their own judgment and experience. To mitigate this, it would be beneficial to conduct sensitivity analyses to assess how changes in weight assignments affect the final results.

2. **Limited data sources**: Currently, the project relies on polling data from a single source (FiveThirtyEight) and state-specific data from one website (270towin.com). Incorporating data from additional reputable sources could enhance the robustness and reliability of the analysis. This would help to reduce potential biases and provide a more comprehensive view of the polling landscape.

3. **Assumption of linear relationships**: The weighting calculations assume linear relationships between various factors and poll reliability. However, some relationships might be non-linear in nature. Exploring and incorporating non-linear weighting schemes could potentially improve the accuracy of the adjusted poll results.

4. **Lack of uncertainty quantification**: The current analysis does not provide a measure of uncertainty or confidence intervals for the adjusted poll results. Incorporating techniques such as bootstrapping or Bayesian inference could help quantify the uncertainty associated with the estimates and provide a more complete picture of the range of possible outcomes.

To address these criticisms and further enhance the project, several next steps can be considered:

1. **Sensitivity analysis**: Conduct sensitivity analyses to assess the impact of different weight assignments on the final results. This will help identify the most influential factors and guide the refinement of the weighting scheme.

2. **Incorporation of additional data sources**: Explore and integrate polling data from other reputable sources to enhance the robustness and reliability of the analysis. This may involve adapting the data processing pipeline to handle different data formats and structures.

3. **Exploration of non-linear weighting schemes**: Investigate and experiment with non-linear weighting schemes to capture potential non-linear relationships between various factors and poll reliability. This could involve techniques such as polynomial regression or machine learning algorithms.

4. **Uncertainty quantification**: Implement techniques like bootstrapping or Bayesian inference to quantify the uncertainty associated with the adjusted poll results. This will provide a more comprehensive understanding of the range of possible outcomes and help convey the level of confidence in the estimates.

5. **User interface and visualization**: Develop a user-friendly interface and data visualization components to make the project more accessible and informative to a wider audience. This could include interactive dashboards, maps, and charts that allow users to explore the polling data and adjusted results in a more intuitive manner.