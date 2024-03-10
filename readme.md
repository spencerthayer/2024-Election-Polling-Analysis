# Election Polling Analysis

This Python script is designed to fetch, process, and analyze presidential polling data. Using data from FiveThirtyEight's publicly available [CSV file](https://projects.fivethirtyeight.com/polls/data/president_polls.csv), it applies a series of weightings to adjust for various factors such as poll quality, partisanship, sample population type, and state-specific electoral significance.

## Data Acquisition

The script fetches polling data using the Python `Requests` library. The data is then loaded into a `Pandas` DataFrame for analysis. This approach ensures that the analysis is always up-to-date with the latest available data.

## Weighting Calculations

The script employs several mathematical principles to calculate the final weight of each poll. Below we will explore how these weights are derived and applied to the polling data to calculate the adjusted polls which should result in a nuanced analysis of polling data.

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

Polls are weighted based on the grade assigned to the polling organization, which reflects their historical accuracy and methodology quality. FiveThirtyEight categorizes these grades, and in our script, each grade is associated with a specific numerical weight. This numerical weight translates the qualitative assessment of a poll's reliability into a quantitative factor that can be used in further calculations. The mapping from grades to numerical weights is as follows:

```
A+: 1.0, A: 0.9, A-: 0.8, A/B: 0.75, B+: 0.7, B: 0.6, B-: 0.5, B/C: 0.45, C+: 0.4, C: 0.3, C-: 0.2, C/D: 0.15, D+: 0.1, D: 0.05, D-: 0.025, `null`: 0.0125
```

Each grade is assigned a weight that diminishes as the grade decreases, with 'A+' polls being considered the most reliable (and thus given a weight of 1.0) and 'D-' polls being considered the least reliable (with a weight of 0.025). This numerical representation of grades allows for a standardized and objective approach to adjust the impact of each poll based on the credibility and track record of the polling organization.

The logic behind this grading system is straightforward: higher grades correspond to higher numerical weights, indicating a greater degree of reliability and historical accuracy. Each grade step down represents a decrease in reliability, and this is reflected in the proportional decrease in the numerical weight. The specific weights were chosen to reflect a balance between acknowledging the quality of higher-graded polls and still allowing lower-graded polls to contribute to the overall analysis, albeit to a much lesser extent.

For a poll with a specific grade, its grade weight is directly fetched from this predefined mapping. This explicit numerical representation ensures that the weight calculation process is transparent and consistent across all polls analyzed.

### 3. Partisan Weight Adjustment

Partisan-sponsored polls may have a bias toward their sponsor. The script applies a correction factor to account for this bias:

- If a poll is partisan (true), a weight of $0.25$ is applied.
- If a poll is non-partisan (false), a weight of $1$ is applied.

This adjustment, $W_{partisan}$, is applied directly based on the poll's partisanship status.

### 4. Population Sample Weights

Different polls target different segments of the population (e.g., likely voters, registered voters). The reliability of these polls varies with the population segment, so weights are applied accordingly:

- Likely voters (lv): $1.0$
- Registered voters (rv): $\frac{2}{3}$
- Voters (v): $\frac{1}{2}$
- Adults (a): $\frac{1}{3}$
- All: $\frac{1}{3}$

This is formalized as $W_{population}(P)$ where $P$ stands for the population type of the poll.

### 5. State-Specific Weights

To incorporate state-specific polling data into the broader analysis, the script integrates data from `states.py`, which calculates a `state_rank` based on political projections and the proportion of electoral votes for each state. This rank represents the state's electoral significance and political leaning.

To derive a mathematical formula for state ranking, I am considering various factors that contribute to a state's electoral significance and political leaning.

- $R_s$: The rank of state $s$
- $E_s$: The number of electoral votes of state $s$
- $E_{total}$: The total number of electoral votes across all states
- $P_s$: The projected partisan lean of state $s$, where $P_s \in [-1, 1]$. A value of -1 indicates a strong Democratic lean, while a value of 1 indicates a strong Republican lean.
- $\alpha$: A weighting factor for the electoral vote component, where $\alpha \in [0, 1]$
- $\beta$: A weighting factor for the partisan lean component, where $\beta \in [0, 1]$, and $\alpha + \beta = 1$

The state rank $R_s$ can be calculated as a weighted sum of two components: the normalized electoral vote count and the partisan lean of the state.

1. Electoral Vote Component:
   
   The electoral vote component is calculated by normalizing the number of electoral votes of state $s$ with respect to the total number of electoral votes across all states.
   
   $$ \text{Electoral Vote Component} = \frac{E_s}{E_{total}} $$

2. Partisan Lean Component:
   
   The partisan lean component is represented by $P_s$, which is a value between -1 and 1, indicating the projected partisan lean of the state.

3. State Rank Formula:
   
   The state rank $R_s$ is calculated by combining the electoral vote component and the partisan lean component using their respective weighting factors:
   
   $$ R_s = \alpha \cdot \frac{E_s}{E_{total}} + \beta \cdot P_s $$

   Where:
   - $\alpha$: The weighting factor for the electoral vote component
   - $\beta$: The weighting factor for the partisan lean component

   The weighting factors $\alpha$ and $\beta$ determine the relative importance of the electoral vote count and the partisan lean in the overall state ranking. These factors can be adjusted based on the desired emphasis on each component.

4. Normalization (Optional):
   
   To ensure that the state ranks are within a specific range (e.g., [0, 1]), we can apply a normalization step. One common approach is min-max normalization:
   
   $$ \text{Normalized } R_s = \frac{R_s - \min(R)}{\max(R) - \min(R)} $$

   Where:
   - $\min(R)$: The minimum state rank across all states
   - $\max(R)$: The maximum state rank across all states

   After normalization, the state ranks will be scaled to the range [0, 1], with 0 representing the state with the lowest rank and 1 representing the state with the highest rank.

By applying this formula, we can calculate the state rank $R_s$ for each state, considering both its electoral vote count and its projected partisan lean. The resulting state ranks can be used to weight the significance of state-specific polls in the overall polling analysis.

### 6. Combining Weights

After calculating individual weights, the combined weight of a poll is given by:

$$ W_{combined} = W_{grade} \times W_{partisan} \times W_{population} \times w(t) \times W_{state} $$

This formula incorporates the time decay weight, grade weight, partisan weight, population weight, and state-specific weight to provide a comprehensive assessment of each poll's significance.

### 7. Normalization

The normalization process ensures that the sum of weights across all polls equals 1, allowing for a fair comparison and aggregation:

$$ W_{normalized,i} = \frac{W_{combined,i}}{\sum_{j=1}^{N} W_{combined,j}} $$

- $W_{normalized,i}$: Normalized weight of the $i$th poll.
- $W_{combined,i}$: Combined weight of the $i$th poll before normalization.
- $N$: Total number of polls considered.
- $j$: Is the index of each poll in the series of polls from 1 to $N$ where $N$ is the total number of polls.

### 8. Adjusted Poll Results Calculation

Finally, to calculate adjusted poll results, each poll's result is multiplied by its normalized weight. This yields a weighted average that accounts for the reliability and relevance of each poll.

$$ \text{Adjusted Result} = \sum_{i=1}^{N} (W_{normalized,i} \times \text{Poll Result}_i) $$

This formula ensures that polls that are more recent, from higher-grade organizations, non-partisan, targeting more reliable population samples, and from states with greater electoral significance have a greater influence on the adjusted result.

## Conclusion

By incorporating state-specific weights into the analysis, this script provides a more nuanced and comprehensive assessment of presidential polling data. The integration of data from `states.py` allows for the consideration of each state's unique electoral dynamics, ensuring that the adjusted poll results reflect the significance and political leanings of individual states.

This approach aims to strike a balance between the broad insights provided by national polls and the detailed, state-specific information captured by local polls. By carefully normalizing and combining these various weights, the script produces an adjusted result that offers a more accurate and representative picture of the current state of the presidential race.

As with any polling analysis, there is always room for further refinement and improvement. The script's modular design allows for the incorporation of additional factors and adjustments as needed. Collaboration and feedback from the community are welcome to enhance the methodology and ensure the most accurate and meaningful analysis possible.