# Finding Ghosts in Your Data
# Time series anomaly detection
# For more information on this, review chapters 13-14

import pandas as pd
import numpy as np
from pandas.core import base
import ruptures as rpt

def detect_single_timeseries(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
    # Weights is here as a future-proofing measure.
    weights = { "time_series": 1.0 }

    # Ensure that everything is sorted by dt
    df = df.sort_values("dt", axis=0, ascending=True)
    
    num_data_points = df['value'].count()
    if (num_data_points < 15):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, f"Must have a minimum of at least fifteen data points for anomaly detection.  You sent {num_data_points}.")
    elif (max_fraction_anomalies <= 0.0 or max_fraction_anomalies > 1.0):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid max fraction of anomalies, 0 < x <= 1.0.")
    elif (sensitivity_score <= 0 or sensitivity_score > 100 ):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid sensitivity score, 0 < x <= 100.")
    else:
        (df_tested, tests_run, diagnostics) = run_tests(df)
        (df_out, diag_outliers) = determine_outliers(df_tested, tests_run, diagnostics["num_iterations"], sensitivity_score, max_fraction_anomalies)
        return (df_out, weights, { "message": "Result of single time series statistical tests.", "Tests run": tests_run, "Test diagnostics": diagnostics, "Outlier determination": diag_outliers})

def run_tests(df):
    tests_run = {
        "changepoint": 1
    }

    num_records = df['key'].shape[0]
    diagnostics = {
        "Number of records": num_records
    }
    signal = df['value'].to_numpy()

    # Not knowing the shape of the data, we will try each of the three kernels.
    # We will also try a variety of penalty values across 7 orders of magnitude.
    # The combination of results will allow us to develop a sensitivity score.
    kernels = { "linear", "rbf", "cosine" }
    penalties = { 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50, 80, 100, 200, 500, 800, 1000 }
    diagnostics["kernels"] = kernels
    diagnostics["penalties"] = penalties
    diagnostics["num_iterations"] = len(kernels) * len(penalties)

    scores = np.zeros([num_records])
    for idx,k in enumerate(kernels):
        algo = rpt.KernelCPD(kernel=k).fit(signal)
        for idxp,p in enumerate(penalties):
            # Get the set of results and add them to the scores array
            result = algo.predict(pen=p)
            for ix,r in enumerate(result[:-1]):
                scores[r] += 1

    df["anomaly_score"] = scores
    return (df, tests_run, diagnostics)

def determine_outliers(
    df,
    tests_run,
    num_iterations,
    sensitivity_score,
    max_fraction_anomalies
):
    # To deal with lower-sensitivity iterations not always picking up valid changepoints, divide iterations by 1.5.
    # Then multiply by the inverse of sensitivity score to get our cutoff.
    sensitivity_threshold = (num_iterations / 1.5) * ((100.0 - sensitivity_score) / 100.0)
    diagnostics = { "Sensitivity threshold": sensitivity_threshold }

    # Get the 100-Nth percentile of anomaly score.
    # Ex:  if max_fraction_anomalies = 0.1, get the
    # 90th percentile anomaly score.
    max_fraction_anomaly_score = np.quantile(df['anomaly_score'], 1.0 - max_fraction_anomalies)
    diagnostics["Max fraction anomaly score"] = max_fraction_anomaly_score
    # If the max fraction anomaly score is greater than
    # the sensitivity score, it means that we have MORE outliers
    # than our max_fraction_anomalies supports, and therefore we
    # need to cut it off before we get down to our sensitivity score.
    # Otherwise, sensitivity score stays the same and we operate as normal.
    if max_fraction_anomaly_score > sensitivity_threshold and max_fraction_anomalies < 1.0:
        sensitivity_threshold = max_fraction_anomaly_score
    diagnostics["Sensitivity score"] = sensitivity_threshold
    return (df.assign(is_anomaly=df['anomaly_score'] > sensitivity_threshold), diagnostics)
