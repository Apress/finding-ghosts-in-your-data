# Finding Ghosts in Your Data
# Multiple time series anomaly detection
# For more information on this, review chapters 16-17

import pandas as pd
import numpy as np
from pandas.core import base
from tslearn.piecewise import SymbolicAggregateApproximation

def detect_multi_timeseries(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
    weights = { "DIFFSTD": 1.0, "SAX": 1.0 }

    # Ensure that everything is sorted by dt and series key
    df = df.sort_values(["dt", "series_key"], axis=0, ascending=True)
    
    num_series = len(df["series_key"].unique())
    num_data_points = df['value'].count()
    if (num_data_points / num_series < 15):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, f"Must have a minimum of at least fifteen data points per time series for anomaly detection.  You sent {num_data_points} per series.")
    elif (num_series < 2):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, f"Must have a minimum of at least two time series for anomaly detection.  You sent {num_series} series.")
    elif (max_fraction_anomalies <= 0.0 or max_fraction_anomalies > 1.0):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid max fraction of anomalies, 0 < x <= 1.0.")
    elif (sensitivity_score <= 0 or sensitivity_score > 100 ):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid sensitivity score, 0 < x <= 100.")
    else:
        (df_tested, tests_run, diagnostics) = run_tests(df)
        (df_scored, diag_scored) = score_results(df_tested, tests_run, sensitivity_score)
        (df_out, diag_outliers) = determine_outliers(df_scored, max_fraction_anomalies)
        return (df_out, weights, { "message": "Result of single time series statistical tests.", "Tests run": tests_run, "Test diagnostics": diagnostics, "Outlier scoring": diag_scored, "Outlier determination": diag_outliers})

def run_tests(df):
    tests_run = {
        "DIFFSTD": 1,
        "SAX": 1
    }

    # Break out each series to operate on individually.
    series = [y for x, y in df.groupby("series_key", as_index=False)]
    # Grab basic information:  number of series, length of series.
    num_series = len(series)
    l = len(series[0])

    diagnostics = {
        "Number of time series": num_series,
        "Time series length": l
    }

    # Perform SAX first while everything is still in individual series.
    # This will add sax_distance to each point in each series.
    (series, diag_sax) = check_sax(series, num_series, l)
    diagnostics["SAX"] = diag_sax    

    # Break out the series into segments of approximately 7 data points.
    # 7 data points allows us to have at least 2 segments given our 15-point minimum.
    # We use integer math here to ensure no segment has just 1-2 records and no segment
    # is wildly unbalanced in size compared to the others.  At a minimum,
    # we should have 6 data points per segment.  At a maximum, we can end up with 10.
    series_segments = [np.array_split(series[x], (l // 7)) for x in range(num_series)]
    num_segments = len(series_segments[0])
    num_records = df['key'].shape[0]

    diagnostics["Number of records"] = num_records
    diagnostics["Number of segments per time series"] = num_segments

    segment_means = generate_segment_means(series_segments, num_series, num_segments)
    diagnostics["Segment means"] = segment_means
    segments_diffstd = check_diffstd(series_segments, segment_means, num_series, num_segments)
    # Merge segments together as df.  Segments comes in as a list of lists, each of which contains a DataFrame.
    # First, flatten out the list of lists, giving us a list of DataFrames.
    flattened = [item for sublist in segments_diffstd for item in sublist]
    # Next, concatenate the DataFrames together.
    df = pd.concat(flattened)

    return (df, tests_run, diagnostics)

def generate_segment_means(series_segments, num_series, num_segments):
    means = []
    for j in range(num_segments):
        C = [series_segments[i][j]['value'] for i in range(num_series)]
        means.append([sum(x)/num_series for x in zip(*C)])
    return means

def diffstd(s1v, s2v):
    # Find the differences between the two input segments.
    dt = [x1 - x2 for (x1, x2) in zip(s1v, s2v)]
    n = len(s1v)
    mu = np.mean(dt)
    # For each difference, square its distance from the mean.  This guarantees all numbers are positive.
    diff2 = [(d-mu)**2 for d in dt]
    # Sum the squared differences, divide by the number of data points (to get an average),
    # and take the square root of the result.  This returns a single number, the DIFFSTD comparing
    # these two segments.
    return (np.sum(diff2)/n)**0.5

def check_diffstd(series_segments, segment_means, num_series, num_segments):
    # For each series, make a pairwise comparison against the average.
    for i in range(num_series):
        for j in range(num_segments):
            series_segments[i][j]['segment_number'] = j
            series_segments[i][j]['diffstd_distance'] = diffstd(series_segments[i][j]['value'], segment_means[j])
    return series_segments

def check_sax(series, num_series, l):
    if (l < 100):
        segment_split = 2
    elif (l < 1000):
        segment_split = 3
    else:
        segment_split = 5

    # The current recommendation for SAX is that you limit the alphabet size to 3-5, with 4 being
    # the typical sweet spot.  We also want to normalize our input data, so scale = True.
    # We determine each alphabet character based on 2-5 data points (depending on total data length)
    sax = SymbolicAggregateApproximation(n_segments= l//segment_split, alphabet_size_avg=4, scale=True)

    # slist is a list of lists:  one list per series, NOT per segment!
    slist = [series[i]['value'].tolist() for i in range(num_series)]

    # sax_data is an array containing a list (one per series) of lists (containing the "letter")
    #    eg:  array([ [[1],[1],[1]], [[1],[0],[2]], [[2],[2],[1]] ])
    # Note that tslearn doesn't use a letter-based alphabet but instead a numeric one:  0, 1, 2, 3.
    sax_data = sax.fit_transform(slist)

    # tslearn gives us the ability to perform pairwise comparisons of SAX results using a distance measure.
    # We will break things into fixed-size chunks of 4 letters, e.g. 1103 | 3111 | 2203
    # Then, we can perform 1-versus-all comparisons of each word versus the other words in the same position.
    word_size = 4
    num_words = len(sax_data[0])//word_size

    # Create a matrix which will hold the mean score of these pairwise comparisons.
    m = np.empty((num_series, num_words))
    for i in range(num_series):
        for j in range(num_words):
            # Calculate pairwise distances for each word of SAX results
            # For example, given three series:
            #  1103 | 1111 | 2203
            #  1100 | 2111 | 2202
            #  1211 | 3111 | 2201
            # We would find the distance between 1103 and each of 1100 and 1211 and average it out.
            # That result would go into m[0][0].
            # m[0][1] would be the average distance between 1111 and 2111 / 3111, etc.
            # The calculation here technically also includes the distance between 1103 and 1103, which is always 0.
            # Therefore, we subtract 1 from num_series and we still get a good average.
            m[i][j] = sum(sax.distance_sax(sax_data[i][j*word_size:(1+j)*word_size],sax_data[k][j*word_size:(1+j)*word_size])
                for k in range(num_series))/(num_series-1)

    diagnostics = {
        "Segment size per letter": segment_split,
        "Number of segments":  l//segment_split,
        "Word size": word_size,
        "Number of words": num_words,
        "SAX matrix": m.tolist()
    }

    # Set the SAX distance for each section of each series.
    for i in range(num_series):
        # If we have "overflow" (e.g., 19 data points and segment_split=2, use the final word)
        series[i]['sax_distance'] = [m[i][min(j//(word_size*segment_split), num_words-1)] for j,val in enumerate(series[0].index)]

    return (series, diagnostics)

def score_results(df, tests_run, sensitivity_score):
    # Calculate anomaly score for each series independently.
    # This is because DIFFSTD distances are not normalized across series.
    series = [y for x, y in df.groupby("series_key", as_index=False)]
    num_series = len(series)
    diagnostics = { }

    for i in range(num_series):
        # DIFFSTD doesn't have a hard cutoff point describing when something is (or is not) an outlier.
        # Therefore, to reduce the number of results, we'll start with 1.5 * mean of diffstd distances as a max distance score.
        diffstd_mean = series[i]['diffstd_distance'].mean()

        # Subtract from 1.5 the sensitivity_score/100.0, so at 100 sensitivity, we use 0.5 * mean as a max distance from the mean.
        # Ex:  if the mean is 10 and sensitivity_score is 0, we'll look for segments with DIFFSTD above (10 + 1.5*10) = 25
        # With sensitivity_score 100, the cutoff score will be 15.
        diffstd_sensitivity_threshold = diffstd_mean + ((1.5 - (sensitivity_score / 100.0)) * diffstd_mean)

        # The diffstd_score is the percentage difference between the distance and the sensitivity threshold.
        series[i]['diffstd_score'] = (series[i]['diffstd_distance'] - diffstd_sensitivity_threshold) / diffstd_sensitivity_threshold

        # SAX also doesn't have a hard cutoff point so we will use a rule of thumb here as well.
        # Some divergence is noticeable at approximately 2.5 and major divergence is notable at about 3-4.
        # If we multiply by 15, we can calculate the percentage of this score versus (100 - sensitivity_score).
        # This will not necessarily put us on the same scale as DIFFSTD but will ensure that for higher sensitivity
        # scores, 1.5 will trigger with a SAX score > 0, indicating at least a small outlier.
        # Also, cap the threshold at a floor value of 25.0 to prevent absurd results.
        sax_sensitivity_threshold = max(100.0 - sensitivity_score, 25.0)
        series[i]['sax_score'] = ((series[i]['sax_distance'] * 15.0) - sax_sensitivity_threshold) / sax_sensitivity_threshold

        # Our anomaly score is the sum of diffstd_score and sax_score.  Because DIFFSTD and SAX
        # split data different ways, this helps us at the margin with determining *which* data points in the series
        # are the biggest outliers, as the intersection of high SAX + high DIFFSTD will be the most likely culprits.
        series[i]['anomaly_score'] = series[i]['sax_score'] + series[i]['diffstd_score']

        diagnostics["Series " + str(i)] = {
            "Mean DIFFSTD distance": diffstd_mean,
            "DIFFSTD sensitivity threshold": diffstd_sensitivity_threshold,
            "SAX sensitivity threshold": sax_sensitivity_threshold
        }

    return (pd.concat(series), diagnostics)

def determine_outliers(
    df,
    max_fraction_anomalies
):
    series = [y for x, y in df.groupby("series_key", as_index=False)]

    # Get the 100-Nth percentile of anomaly score.
    # Ex:  if max_fraction_anomalies = 0.1, get the
    # 90th percentile anomaly score.
    max_fraction_anomaly_scores = [np.quantile(s['anomaly_score'], 1.0 - max_fraction_anomalies) for s in series]
    diagnostics = {"Max fraction anomaly scores":  max_fraction_anomaly_scores }

    # When scoring outliers, we made 0.01 the sensitivity threshold, as 0 means no differences.
    # If the max fraction anomaly score is greater than 0, it means that we have MORE outliers
    # than our max_fraction_anomalies supports, and therefore we
    # need to cut it off before we get down to our sensitivity score.
    # Otherwise, sensitivity score stays the same and we operate as normal.
    sensitivity_thresholds = [max(0.01, mfa) for mfa in max_fraction_anomaly_scores]
    diagnostics["Sensitivity scores"] = sensitivity_thresholds

    # We treat segments as outliers, not individual data points.  Mark each segment with a sufficiently large
    # anomaly score as an outlier for subsequent review.
    for i in range(len(series)):
        series[i]['is_anomaly'] = [score >= sensitivity_thresholds[i] for score in series[i]['anomaly_score']]

    return (pd.concat(series), diagnostics)
