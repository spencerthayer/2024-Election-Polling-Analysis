<?php

// Download the Data
$polling_url = "https://projects.fivethirtyeight.com/polls/data/president_polls.csv";
$favorability_url = "https://projects.fivethirtyeight.com/polls/data/favorability_polls.csv";

// Data Parsing
$candidate_names = ['Joe Biden', 'Donald Trump'];
$favorability_weight = 0.2;
$heavy_weight = true;

// Define the time decay weighting
$decay_rate = 2;
$half_life_days = 28;

// Constants for the weighting calculations
$grade_weights = [
    'A+' => 1.0, 'A' => 0.9, 'A-' => 0.8, 'A/B' => 0.75, 'B+' => 0.7,
    'B' => 0.6, 'B-' => 0.5, 'B/C' => 0.45, 'C+' => 0.4, 'C' => 0.3,
    'C-' => 0.2, 'C/D' => 0.15, 'D+' => 0.1, 'D' => 0.05, 'D-' => 0.025
];
$partisan_weight = [true => 0.1, false => 1];
$population_weights = [
    'lv' => 1.0, 'rv' => 0.6666666666666666, 'v' => 0.5,
    'a' => 0.3333333333333333, 'all' => 0.3333333333333333
];

function margin_of_error($n, $p = 0.5, $confidence_level = 0.95) {
    $z = 1.96; // Assuming a 95% confidence level
    $moe = $z * sqrt(($p * (1 - $p)) / $n);
    return $moe * 100; // Convert to percentage
}

function download_csv_data($url) {
    try {
        $csv_data = file_get_contents($url);
        return array_map('str_getcsv', explode("\n", $csv_data));
    } catch (Exception $e) {
        echo "Error downloading data from {$url}: {$e->getMessage()}\n";
        return [];
    }
}

function preprocess_data($data, $candidate_names) {
    $processed_data = [];
    foreach ($data as $row) {
        if (count($row) < 47) {
            continue;
        }
        if (!in_array($row[45], $candidate_names)) {
            continue;
        }
        $created_at = DateTime::createFromFormat('m/d/y H:i', $row[26]);
        if ($created_at === false) {
            continue;
        }
        $processed_row = [
            'created_at' => $created_at->format('Y-m-d H:i:s'),
            'candidate_name' => $row[45],
            'pct' => floatval($row[46]),
            'sample_size' => intval($row[21]),
            'population' => $row[22],
            'fte_grade' => $row[8],
            'partisan' => !empty($row[31]),
            'transparency_score' => floatval($row[10]),
        ];
        $processed_data[] = $processed_row;
    }

    if (empty($processed_data)) {
        echo "No data to process after filtering.\n";
        return $processed_data;
    }

    $transparency_scores = array_column($processed_data, 'transparency_score');
    $max_transparency_score = !empty($transparency_scores) ? max($transparency_scores) : 0;

    $sample_sizes = array_column($processed_data, 'sample_size');
    $min_sample_size = !empty($sample_sizes) ? min($sample_sizes) : 0;
    $max_sample_size = !empty($sample_sizes) ? max($sample_sizes) : 0;

    foreach ($processed_data as &$row) {
        $row['transparency_weight'] = $max_transparency_score > 0 ? $row['transparency_score'] / $max_transparency_score : 0;
        $row['sample_size_weight'] = $max_sample_size > $min_sample_size ? ($row['sample_size'] - $min_sample_size) / ($max_sample_size - $min_sample_size) : 0;
        $row['grade_weight'] = $grade_weights[$row['fte_grade']] ?? 0.0125;
    }

    return $processed_data;
}

function apply_time_decay_weight($data, $decay_rate, $half_life_days) {
    $reference_date = new DateTime();
    foreach ($data as &$row) {
        $days_old = $reference_date->diff($row['created_at'])->days;
        $row['time_decay_weight'] = exp(-log($decay_rate) * $days_old / $half_life_days);
    }
    return $data;
}

function calculate_timeframe_specific_moe($data, $candidate_names) {
    $moes = [];
    foreach ($candidate_names as $candidate) {
        $candidate_data = array_filter($data, function ($row) use ($candidate) {
            return $row['candidate_name'] === $candidate;
        });
        if (empty($candidate_data)) {
            continue;
        }
        foreach ($candidate_data as $poll) {
            if ($poll['sample_size'] > 0 && $poll['pct'] >= 0 && $poll['pct'] <= 100) {
                $moe = margin_of_error($poll['sample_size'], $poll['pct'] / 100);
                $moes[] = $moe;
            }
        }
    }
    return !empty($moes) ? array_sum($moes) / count($moes) : NAN;
}

function calculate_polling_metrics($data, $candidate_names) {
    global $grade_weights, $partisan_weight, $population_weights, $heavy_weight;

    $transparency_scores = array_column($data, 'transparency_score');
    $max_transparency_score = !empty($transparency_scores) ? max($transparency_scores) : 0;

    $sample_sizes = array_column($data, 'sample_size');
    $min_sample_size = !empty($sample_sizes) ? min($sample_sizes) : 0;
    $max_sample_size = !empty($sample_sizes) ? max($sample_sizes) : 0;

    $data = array_map(function ($row) use ($grade_weights, $max_transparency_score, $min_sample_size, $max_sample_size, $partisan_weight, $population_weights) {
        $row['pct'] = $row['pct'] > 1 ? $row['pct'] : $row['pct'] * 100;
        $row['grade_weight'] = $grade_weights[$row['fte_grade']] ?? 0.0125;
        $row['transparency_weight'] = $max_transparency_score > 0 ? $row['transparency_score'] / $max_transparency_score : 0;
        $row['sample_size_weight'] = $max_sample_size > $min_sample_size ? ($row['sample_size'] - $min_sample_size) / ($max_sample_size - $min_sample_size) : 0;
        $row['is_partisan'] = $row['partisan'];
        $row['partisan_weight'] = $partisan_weight[$row['is_partisan']];
        $row['population_weight'] = $population_weights[$row['population']] ?? 1;
        return $row;
    }, $data);

    $list_weights = [];
    foreach (['time_decay_weight', 'sample_size_weight', 'grade_weight', 'transparency_weight', 'population_weight', 'partisan_weight'] as $weight) {
        $list_weights[] = array_column($data, $weight);
    }
    $combined_weights = $heavy_weight ? array_product($list_weights) : array_sum($list_weights) / count($list_weights);

    $weighted_sums = [];
    $total_weights = [];
    foreach ($data as $row) {
        $candidate = $row['candidate_name'];
        $weighted_sums[$candidate] = ($weighted_sums[$candidate] ?? 0) + $combined_weights[$candidate] * $row['pct'];
        $total_weights[$candidate] = ($total_weights[$candidate] ?? 0) + $combined_weights[$candidate];
    }
    $weighted_averages = array_map(function ($sum, $weight) {
        return $weight > 0 ? $sum / $weight : 0;
    }, $weighted_sums, $total_weights);

    $weighted_margins = array_map(function ($candidate) use ($data) {
        return calculate_timeframe_specific_moe(array_filter($data, function ($row) use ($candidate) {
            return $row['candidate_name'] === $candidate;
        }), [$candidate]);
    }, $candidate_names);

    return array_combine($candidate_names, array_map(null, $weighted_averages, $weighted_margins));
}

function calculate_favorability_differential($data, $candidate_names) {
    global $grade_weights, $population_weights;

    $data = array_map(function ($row) use ($grade_weights, $population_weights) {
        $row['favorable'] = $row['favorable'] > 1 ? $row['favorable'] : $row['favorable'] * 100;
        $row['grade_weight'] = $grade_weights[$row['fte_grade']] ?? 0.0125;
        $row['population_weight'] = $population_weights[strtolower($row['population'])] ?? 1;
        return $row;
    }, $data);

    $list_weights = [];
    foreach (['grade_weight', 'population_weight', 'time_decay_weight'] as $weight) {
        $list_weights[] = array_column($data, $weight);
    }
    $combined_weights = array_product($list_weights);

    $weighted_sums = [];
    $total_weights = [];
    foreach ($data as $row) {
        $candidate = $row['politician'];
        $weighted_sums[$candidate] = ($weighted_sums[$candidate] ?? 0) + $combined_weights[$candidate] * $row['favorable'];
        $total_weights[$candidate] = ($total_weights[$candidate] ?? 0) + $combined_weights[$candidate];
    }
    $weighted_averages = array_map(function ($sum, $weight) {
        return $weight > 0 ? $sum / $weight : 0;
    }, $weighted_sums, $total_weights);

    return array_fill_keys($candidate_names, 0) + $weighted_averages;
}

function combine_analysis($polling_metrics, $favorability_differential, $favorability_weight) {
    $combined_metrics = [];
    foreach ($polling_metrics as $candidate => $polling_data) {
        $combined_metrics[$candidate] = [
            $polling_data[0] * (1 - $favorability_weight) + $favorability_differential[$candidate] * $favorability_weight,
            $polling_data[1]
        ];
    }
    return $combined_metrics;
}

function output_results($combined_results, $period_value, $period_type) {
    if (isset($combined_results['Joe Biden']) && isset($combined_results['Donald Trump'])) {
        [$biden_score, $biden_margin] = $combined_results['Joe Biden'];
        [$trump_score, $trump_margin] = $combined_results['Donald Trump'];
        $differential = $trump_score - $biden_score;
        $favored_candidate = $differential < 0 ? "Biden" : "Trump";
        $output = sprintf(
            "%2d%-4s B %5.2f%% ±%.2f | T %5.2f%% ±%.2f | %+5.2f %s\n",
            $period_value, substr($period_type, 0, 1),
            $biden_score, $biden_margin, $trump_score, $trump_margin,
            abs($differential), $favored_candidate
        );
        echo $output;
    } else {
        echo "Insufficient data to generate results for the {$period_value} {$period_type} period.\n";
    }
}

function main() {
    global $polling_url, $favorability_url, $candidate_names, $favorability_weight, $decay_rate, $half_life_days;

    $polling_data = download_csv_data($polling_url);
    $favorability_data = download_csv_data($favorability_url);

    $polling_data = preprocess_data($polling_data, $candidate_names);
    $favorability_data = preprocess_data($favorability_data, $candidate_names);

    $polling_data = apply_time_decay_weight($polling_data, $decay_rate, $half_life_days);
    $favorability_data = apply_time_decay_weight($favorability_data, $decay_rate, $half_life_days);

    foreach ([
        [12, 'months'], [6, 'months'], [3, 'months'], [1, 'months'],
        [21, 'days'], [14, 'days'], [7, 'days'], [3, 'days'], [1, 'days']
    ] as [$period_value, $period_type]) {
        $start_period = $period_type === 'months'
            ? (new DateTime())->sub(new DateInterval("P{$period_value}M"))
            : (new DateTime())->sub(new DateInterval("P{$period_value}D"));

        $filtered_polling_data = preprocess_data(
            array_filter($polling_data, function ($row) use ($start_period, $candidate_names) {
                return $row['created_at'] >= $start_period && in_array($row['candidate_name'], $candidate_names);
            }),
            $candidate_names
        );
        $filtered_favorability_data = preprocess_data(
            array_filter($favorability_data, function ($row) use ($start_period, $candidate_names) {
                return $row['created_at'] >= $start_period && in_array($row['politician'], $candidate_names);
            }),
            $candidate_names
        );

        $polling_metrics = calculate_polling_metrics($filtered_polling_data, $candidate_names);
        $favorability_differential = calculate_favorability_differential($filtered_favorability_data, $candidate_names);

        $combined_results = combine_analysis($polling_metrics, $favorability_differential, $favorability_weight);

        output_results($combined_results, $period_value, $period_type);
    }
}

main();