# Finding Ghosts in Your Data
# Multivariate anomaly detection
# For more information on this, review chapters 10-12

import pandas as pd
import numpy as np
from pandas.core import base
from pyod.models.cof import COF
from pyod.models.loci import LOCI
from pyod.models.copod import COPOD
from pyod.models.combination import aom, moa, average, median, maximization, majority_vote
from pyod.utils.data import evaluate_print
from sklearn.preprocessing import OrdinalEncoder

def detect_multivariate_statistical(
    df,
    sensitivity_score,
    max_fraction_anomalies,
    n_neighbors
):
    # Unlike univariate ensembling, we don't weight any of
    # our multivariate ensemble specially.  We do need a
    # sensitivity factor because they will be on different scales.
    # COF has a minimum threshold of 1.35 (estimated by us).
    # LOCI has a threshold of 3.0 (estimated by paper authors).
    weights = { "cof": 1.0, "loci": 1.0, "copod": 1.0 }
    # For COPOD, we get 2.3 from -ln(0.10).  This is a little low but because
    # we're adding the median COPOD value in the calculation, this puts us
    # well above the expected median.
    sensitivity_factors = { "cof": 1.35, "loci": 3.0, "copod":2.3 }

    num_data_points = df['vals'].count()
    if (num_data_points < 15):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, f"Must have a minimum of at least fifteen data points for anomaly detection.  You sent {num_data_points}.")
    elif (max_fraction_anomalies <= 0.0 or max_fraction_anomalies > 1.0):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid max fraction of anomalies, 0 < x <= 1.0.")
    elif (sensitivity_score <= 0 or sensitivity_score > 100 ):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid sensitivity score, 0 < x <= 100.")
    elif (df['vals'].count() < (n_neighbors - 5)):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, f"You sent in {num_data_points} data points, so n_neighbors should be no more than {num_data_points - 5}--that is, n_neighbors should be at least 5 less than the number of observations.")
    else:
        # Max fraction of anomalies must be no more than 0.5 for COF.
        if max_fraction_anomalies > 0.5:
            max_fraction_anomalies = 0.5
        # Number of neighbors should be no more than 5 if we have fewer than 16 data points.
        # Once we have more data points, we can switch to larger counts.  This prevents an issue with 15 data points
        # where we look at an incomplete range.
        if num_data_points < 16:
            n_neighbors = min(n_neighbors, 5)
        (df_encoded, diagnostics) = encode_string_data(df)
        (df_tested, tests_run, diagnostics) = run_tests(df_encoded, max_fraction_anomalies, n_neighbors)
        (df_out, diag_outliers) = determine_outliers(df_tested, tests_run, sensitivity_factors, sensitivity_score, max_fraction_anomalies)
        return (df_out, weights, { "message": "Result of multivariate statistical tests.", "Tests run": tests_run, "Test diagnostics": diagnostics, "Outlier determination": diag_outliers})

def encode_string_data(df):
    # df comes in with two columns:  key and vals.
    # We want to break out the list in vals and turn it into a set of columns.
    # Column names don't matter here.
    df2 = pd.DataFrame([pd.Series(x) for x in df.vals])
    string_cols = df2.select_dtypes(include=[object]).columns.values
    diagnostics = { "Number of string columns in input": len(string_cols) }
    if (len(string_cols) > 0):
        diagnostics["Encoding Operation"] = "Encoding performed on string columns."
        # If there are any string columns in our list, convert them to ordinals.
        # CRITICAL NOTE:  this is not a great practice!  We don't have a mechanism (here)
        # to determine string nearness, so "cat" might get a value of 1.0 and "cats" may be 900.0.
        # Our outlier detection engine really depends on numeric inputs, though, so the options
        # are to avoid encoding altogether and simply fail on string inputs or perform the
        # encoding and potentially lose information if the strings are not truly ordinal.
        enc = OrdinalEncoder()
        # Look for any inputs of type object; numeric values will come in as float64 or int64.
        # Generate a float value for each unique string in the input dataset.
        enc.fit(df2[string_cols])
        # Transform any string columns into their encoded values.
        df2[string_cols] = enc.transform(df2[string_cols])
    else:
        diagnostics["Encoding Operation"] = "No encoding necessary because all columns are numeric."
    # Merge together the two DataFrames.  They will have the same number of rows and will
    # remain in the same order.

    return (pd.concat([df, df2], axis=1), diagnostics)

def run_tests(df, max_fraction_anomalies, n_neighbors):
    num_records = df['key'].shape[0]
    if (num_records > 1000):
        run_loci = 0
    else:
        run_loci = 1

    tests_run = {
        "cof": 1,
        "loci": run_loci,
        "copod": 1
    }
    diagnostics = {
        "Number of records": num_records
    }
    # Remove key and vals, leaving the split-out and encoded versions of values.
    # Bring them back in as an array, as that's what our tests will require.
    col_array = df.drop(["key", "vals"], axis=1).to_numpy()

    # Determine numbers of neighbors
    # Ensure we have n_neighbors at least 5 below the number of records.
    # Ensure we have a boundary on number of tests.  100 above n_neighbors is a bit arbitrary
    # if we have extremely large datasets but should be fine for 1k-10k.
    n_neighbor_range = range(n_neighbors, min(num_records - 5, n_neighbors + 100), 5)
    n_neighbor_range_len = len(n_neighbor_range)

    # COF
    labels_cof = np.zeros([num_records, n_neighbor_range_len])
    scores_cof = np.zeros([num_records, n_neighbor_range_len])
    for idx,n in enumerate(n_neighbor_range):
        (labels_cof[:, idx], scores_cof[:, idx], diag_idx) = check_cof(col_array, max_fraction_anomalies=max_fraction_anomalies, n_neighbors=n)
        k = "Neighbors_" + str(n)
        diagnostics[k] = diag_idx


    df["is_raw_anomaly_cof"] = majority_vote(labels_cof)
    anomaly_score = median(scores_cof)
    df["anomaly_score_cof"] = anomaly_score

    # LOCI
    if (run_loci == 1):
        (labels_loci, scores_loci, diag_loci) = check_loci(col_array)
        df["is_raw_anomaly_loci"] = labels_loci
        anomaly_score = anomaly_score + scores_loci
        diagnostics["LOCI"] = diag_loci
        df["anomaly_score_loci"] = scores_loci

    # COPOD
    (labels_copod, scores_copod, diag_copod) = check_copod(col_array)
    df["is_raw_anomaly_copod"] = labels_copod
    diagnostics["COPOD"] = diag_copod
    df["anomaly_score_copod"] = scores_copod
    anomaly_score = anomaly_score + scores_copod

    df["anomaly_score"] = anomaly_score
    return (df, tests_run, diagnostics)


def check_cof(col_array, max_fraction_anomalies, n_neighbors):
    clf = COF(n_neighbors=n_neighbors, contamination=max_fraction_anomalies)
    clf.fit(col_array)
    diagnostics = {
        "COF Contamination": clf.contamination,
        "COF Threshold": clf.threshold_
    }
    return (clf.labels_, clf.decision_scores_, diagnostics)

# LOCI doesn't use contamination and has good defaults of k=3 and alpha=0.5.
def check_loci(col_array):
    clf = LOCI()
    clf.fit(col_array)
    diagnostics = {
        "LOCI Threshold": clf.threshold_
    }
    return (clf.labels_, clf.decision_scores_, diagnostics)

def check_copod(col_array):
    clf = COPOD()
    clf.fit(col_array)
    diagnostics = {
        "COPOD Threshold": clf.threshold_
    }
    return (clf.labels_, clf.decision_scores_, diagnostics)

def determine_outliers(
    df,
    tests_run,
    sensitivity_factors,
    sensitivity_score,
    max_fraction_anomalies
):
    # Need to multiply this because we don't know up-front if we ran, e.g., LOCI.
    tested_sensitivity_factors = {sf: sensitivity_factors.get(sf, 0) * tests_run.get(sf, 0) for sf in set(sensitivity_factors).union(tests_run)}
    # COPOD typically has a fairly consistent spread but the median point may be quite different,
    # so we will start from the median and add our sensitivity factor to it.
    median_copod = df["anomaly_score_copod"].median()
    sensitivity_threshold = sum([tested_sensitivity_factors[w] for w in tested_sensitivity_factors]) + median_copod
    diagnostics = { "Sensitivity threshold": sensitivity_threshold, "COPOD Median": median_copod }
    # Convert sensitivity score to be approximately the same
    # scale as anomaly score.  Note that sensitivity score is "reversed",
    # such that 100 is the *most* sensitive.
    # Multiply this by the second-largest anomaly score to scale appropriately.
    second_largest = df['anomaly_score'].nlargest(2).iloc[1]
    sensitivity_score = (100 - sensitivity_score) * second_largest / 100.0
    diagnostics["Raw sensitivity score"] = sensitivity_score
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
    if max_fraction_anomaly_score > sensitivity_score and max_fraction_anomalies < 1.0:
        sensitivity_score = max_fraction_anomaly_score
    diagnostics["Sensitivity score"] = sensitivity_score
    return (df.assign(is_anomaly=df['anomaly_score'] > np.max([sensitivity_score, sensitivity_threshold])), diagnostics)
