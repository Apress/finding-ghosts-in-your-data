# Finding Ghosts in Your Data
# Univariate statistical anomaly detection
# For more information on this, review chapters 6-9

import pandas as pd
import numpy as np
from pandas.core import base
from statsmodels import robust
# Chapter 7
from scipy.stats import shapiro, normaltest, anderson, boxcox
import scikit_posthocs as ph
import math
# Chapter 9
from sklearn.mixture import GaussianMixture

def detect_univariate_statistical(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
    # Standard deviation is not a very robust measure, so we weigh this lowest.
    # IQR is a reasonably good measure, so we give it the second-highest weight.
    # MAD is a robust measure for deviation, so we give it the highest weight.
    # The normal distribution tests are generally pretty good if we have the right
    # shape of the data and the correct number of observations.
    # The reason Grubbs' and Dixon's tests are so low is that they capture at most
    # 1 (Grubbs) or 2 (Dixon) outliers.
    weights = {"sds": 0.25, "iqrs": 0.35, "mads": 0.45,
               "grubbs": 0.05, "dixon": 0.15, "gesd": 0.3,
               "gaussian_mixture": 1.5}

    if (df['value'].count() < 3):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a minimum of at least three data points for anomaly detection.")
    elif (max_fraction_anomalies <= 0.0 or max_fraction_anomalies > 1.0):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid max fraction of anomalies, 0 < x <= 1.0.")
    elif (sensitivity_score <= 0 or sensitivity_score > 100 ):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid sensitivity score, 0 < x <= 100.")
    else:
        (df_tested, tests_run, diagnostics) = run_tests(df)
        df_scored = score_results(df_tested, tests_run, weights)
        df_out = determine_outliers(df_scored, sensitivity_score, max_fraction_anomalies)
        return (df_out, weights, { "message": "Ensemble of univariate statistical tests.", "Test diagnostics": diagnostics})

def run_tests(df):
    # Get our baseline calculations, prior to any data transformations.
    base_calculations = perform_statistical_calculations(df['value'])

    diagnostics = { "Base calculations": base_calculations }

    (use_fitted_results, fitted_data, normalization_diagnostics) = perform_normalization(base_calculations, df)
    diagnostics.update(normalization_diagnostics)

    # for each test, execute and add a new score
    # Initial tests should NOT use the fitted calculations.
    b = base_calculations
    df['sds'] = [check_sd(val, b["mean"], b["sd"], 3.0) for val in df['value']]
    df['mads'] = [check_mad(val, b["median"], b["mad"], 3.0) for val in df['value']]
    df['iqrs'] = [check_iqr(val, b["median"], b["p25"], b["p75"], b["iqr"], 1.5) for val in df['value']]
    tests_run = {
        "sds": 1,
        "mads": 1,
        "iqrs": 1,
        # Mark these as 0s to start and set them on if we run them.
        "grubbs": 0,
        "gesd": 0,
        "dixon": 0,
        "gaussian_mixture": 0
    }
    # Start off with values of -1.  If we run a test, we'll populate it with a valid value.
    df['grubbs'] = -1
    df['gesd'] = -1
    df['dixon'] = -1
    df['gaussian_mixture'] = -1

    # Grubbs, GESD, and Dixon's Q tests all require that the input data be normally distributed.
    # Further, Dixon requires no more than 25 observations.
    # Grubbs requires at least 7 observations.
    # GESD requires at least 15 observations.
    if (use_fitted_results):
        df['fitted_value'] = fitted_data
        col = df['fitted_value']
        c = perform_statistical_calculations(col)
        diagnostics["Fitted calculations"] = c

        if (b['len'] >= 7):
            df['grubbs'] = check_grubbs(col)
            tests_run['grubbs'] = 1
        else:
            diagnostics["Grubbs' Test"] = f"Did not run Grubbs' test because we need at least 7 observations but only had {b['len']}."

        if (b['len'] >= 3 and b['len'] <= 25):
            df['dixon'] = check_dixon(col)
            tests_run['dixon'] = 1
        else:
            diagnostics["Dixon's Q Test"] = f"Did not run Dixon's Q test because we need between 3 and 25 observations but had {b['len']}."

        if (b['len'] >= 15):
            # Ensure we have at least 1 outlier allowed and there are still enough
            # degrees of freedom to analyze the data.
            max_num_outliers = math.floor(b['len'] / 3)
            df['gesd'] = check_gesd(col, max_num_outliers)
            tests_run['gesd'] = 1
    else:
        diagnostics["Extended tests"] = "Did not run extended tests because the dataset was not normal and could not be normalized."

    if b['len'] >= 15:
        num_clusters = get_number_of_gaussian_mixture_clusters(df['value'])
        if (num_clusters > 1):
            df['gaussian_mixture'] = check_gaussian_mixture(df['value'], num_clusters)
            diagnostics["Gaussian mixture test"] = f"Ran Gaussian mixture test with {num_clusters} clusters."
            tests_run['gaussian_mixture'] = 1
        else:
            diagnostics["Gaussian mixture test"] = "Did not run Gaussian mixture test because the dataset appears to contain one cluster."
    else:
        diagnostics["Gaussian mixture test"] = "Did not run Gaussian mixture test because we need at least 15 data points to run this test."
    
    diagnostics["Tests Run"] = tests_run

    return (df, tests_run, diagnostics)

def perform_normalization(base_calculations, df):
    use_fitted_results = False
    fitted_data = None

    (is_naturally_normal, natural_normality_checks) = is_normally_distributed(df['value'])
    diagnostics = {"Initial normality checks": natural_normality_checks}
    # If we already have normal-looking data, just use it without reshaping.
    if is_naturally_normal:
        fitted_data = df['value']
        use_fitted_results = True

    # Perform a Box-Cox normalization test if we meet all of the criteria:
    # 1. The data is not already normally distributed
    # 2. The data is not all the same value
    # 3. All values are greater than zero
    # 4. We have at least eight observations
    if ((not is_naturally_normal)
        and base_calculations["min"] < base_calculations["max"]
        and base_calculations["min"] > 0
        and df['value'].shape[0] >= 8):

        (fitted_data, fitted_lambda) = normalize(df['value'])
        (is_fitted_normal, fitted_normality_checks) = is_normally_distributed(fitted_data)
        # The output dataset might not be totally normal, but it should be a lot closer.
        use_fitted_results = True
        diagnostics["Fitted Lambda"] = fitted_lambda
        diagnostics["Fitted normality checks"] = fitted_normality_checks
    else:
        has_variance = base_calculations["min"] < base_calculations["max"]
        all_gt_zero = base_calculations["min"] > 0
        enough_observations = df['value'].shape[0] >= 8
        diagnostics["Fitting Status"] = f"Did not attempt to normalize the data.  Is naturally normal?  {is_naturally_normal}.  Has variance?  {has_variance}.  All values above 0?  {all_gt_zero}.  Has at least 8 observations?  {enough_observations}"

    return (use_fitted_results, fitted_data, diagnostics)

def perform_statistical_calculations(col):
    mean = col.mean()
    sd = col.std()
    # Inter-Quartile Range (IQR) = 75th percentile - 25th percentile
    p25 = np.quantile(col, 0.25)
    p75 = np.quantile(col, 0.75)
    iqr = p75 - p25
    median = col.median()
    # Median Absolute Deviation (MAD)
    mad = robust.mad(col)
    min = col.min()
    max = col.max()
    len = col.shape[0]

    return { "mean": mean, "sd": sd, "min": min, "max": max,
        "p25": p25, "median": median, "p75": p75, "iqr": iqr, "mad": mad, "len": len }

def check_sd(val, mean, sd, min_num_sd):
    return check_stat(val, mean, sd, min_num_sd)

def check_mad(val, median, mad, min_num_mad):
    return check_stat(val, median, mad, min_num_mad)

def check_stat(val, midpoint, distance, n):
    # In the event that we are less than n times the distance
    # beyond the midpoint (mean or median) return the 
    # percent of the way we are to the extremity measure
    # and let the weighting process
    # figure out what to make of it.
    # If distance is 0, then distance-based calculations aren't meaningful--there is no spread.
    if (abs(val - midpoint) < (n * distance)):
        return abs(val - midpoint)/(n * distance)
    else:
        return 1.0

def check_iqr(val, median, p25, p75, iqr, min_iqr_diff):
    # We only want to check one direction, based on whether
    # the value is below the median or at/above.
    if (val < median):
        # If the value is in the p25-median range, it's
        # definitely not an outlier.  Return a value of 0.
        if (val > p25):
            return 0.0
        # If the value is between p25 and the outlier break point,
        # return a fractional score representing how distant it is.
        elif (p25 - val) < (min_iqr_diff * iqr):
            return abs(p25 - val)/(min_iqr_diff * iqr)
        # If the value is far enough away that it's definitely
        # an outlier, return 1.0
        else:
            return 1.0
    else:
        # If the value is in the median-p75 range, it's
        # definitely not an outlier.  Return a value of 0.
        if (val < p75):
            return 0.0
        # If the value is between p75 and the outlier break point,
        # return a fractional score representing how distant it is.
        elif (val - p75) < (min_iqr_diff * iqr):
            return abs(val - p75)/(min_iqr_diff * iqr)
        # If the value is far enough away that it's definitely
        # an outlier, return 1.0
        else:
            return 1.0

def is_normally_distributed(col):
    alpha = 0.05

    # The Shapiro-Wilk test works best for datasets with fewer than
    # 1000 or so observations, though it can work up to ~5K.
    if col.shape[0] < 5000:
        (shapiro_normal, shapiro_exp) = check_shapiro(col, alpha)
    else:
        # Start with the assumption that data is (close enough to)
        # normally distributed.
        shapiro_normal = True
        shapiro_exp = f"Shapiro-Wilk test did not run because n >= 5k.  n = {col.shape[0]}"


    # D'Agostino's K^2 Test can handle larger datasets
    # It requires at least 8 observations, though.
    if col.shape[0] >= 8:
        (dagostino_normal, dagostino_exp) = check_dagostino(col, alpha)
    else:
        dagostino_normal = True
        dagostino_exp = f"D'Agostino's test did not run because n < 8. n = {col.shape[0]}"

    # Anderson-Darling can also handle larger datasets.
    (anderson_normal, anderson_exp) = check_anderson(col)

    diagnostics = {"Shapiro-Wilk": shapiro_exp, "D'Agostino": dagostino_exp, "Anderson-Darling": anderson_exp}
    # Only consider the distribution normal if all three tests believe it.
    # Otherwise, we don't want to run tests which assume normality!
    return (shapiro_normal and dagostino_normal and anderson_normal, diagnostics)

def check_shapiro(col, alpha=0.05):
    return check_basic_normal_test(col, alpha, "Shaprio-Wilk test", shapiro)

def check_dagostino(col, alpha=0.05):
    return check_basic_normal_test(col, alpha, "D'Agostino's K^2 test", normaltest)

def check_basic_normal_test(col, alpha, name, f):
    stat, p = f(col)
    return ( (p > alpha), (f"{name} test, W = {stat}, p = {p}, alpha = {alpha}.") )

def check_anderson(col):
    # Start by assuming normality.
    anderson_normal = True
    return_str = "Anderson-Darling test.  "

    result = anderson(col)
    return_str = return_str + f"Result statistic:  {result.statistic}.  "
    for i in range(len(result.critical_values)):
        sl, cv = result.significance_level[i], result.critical_values[i]
        if result.statistic < cv:
            return_str = return_str + f"Significance Level {sl}: Critical Value = {cv}, looks normally distributed.  "
        else:
            anderson_normal = False
            return_str = return_str + f"Significance Level {sl}: Critical Value = {cv}, does NOT look normally distributed!  "

    return ( anderson_normal, return_str )

def normalize(col):
    # Perform Box-Cox transformation.  We don't know the right lambda
    # to choose, so let the algorithm figure this out.
    # Take the middle 90% of data sorted as the basis for calculating lambda.
    # This way, if there are outliers at the edge, they'll not affect the
    # translation as much.
    l = col.shape[0]
    col80 = col[ math.floor(.1 * l) + 1 : math.floor(.9 * l) ]
    temp_data, fitted_lambda = boxcox(col80)
    # Now use the fitted lambda on the entire dataset.
    fitted_data = boxcox(col, fitted_lambda)
    return (fitted_data, fitted_lambda)

def check_grubbs(col):
    out = ph.outliers_grubbs(col)
    return find_differences(col, out)

def check_gesd(col, max_num_outliers):
    out = ph.outliers_gesd(col, max_num_outliers)
    return find_differences(col, out)

def find_differences(col, out):
    # Convert column and output to sets to see what's missing.
    # Those are the outliers that we need to report back.
    scol = set(col)
    sout = set(out)
    sdiff = scol - sout

    res = [0.0 for val in col]
    # Find the positions of missing inputs and mark them
    # as outliers.
    for val in sdiff:
        indexes = col[col == val].index
        for i in indexes: res[i] = 1.0

    return res

def check_dixon(col):
    q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
        0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
        0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
        0.308, 0.305, 0.301, 0.29]
    Q95 = {n:q for n, q in zip(range(3, len(q95) + 1), q95)}

    Q_mindiff, Q_maxdiff = (0,0), (0,0)
    sorted_data = sorted(col)

    # Check the left-hand side to see if there are any min outliers
    Q_min = (sorted_data[1] - sorted_data[0])
    try:
        Q_min = Q_min / (sorted_data[-1] - sorted_data[0])
    except ZeroDivisionError:
        pass

    Q_mindiff = (Q_min - Q95[len(col)], sorted_data[0])

    Q_max = abs(sorted_data[-2] - sorted_data[-1])
    try:
        Q_max = Q_max / abs(sorted_data[0] - sorted_data[-1])
    except ZeroDivisionError:
        pass

    Q_maxdiff = (Q_max - Q95[len(col)], sorted_data[-1])

    # If the resulting calculation is greater than 0, we have an outlier.
    res = [0.0 for val in col]
    # Dixon's Q test only lets us test the edges, so if there are multiple
    # outliers on a side, we only get to see one.
    if Q_maxdiff[0] >= 0:
        indexes = col[col == Q_maxdiff[1]].index
        for i in indexes: res[i] = 1.0

    if Q_mindiff[0] >= 0:
        indexes = col[col == Q_mindiff[1]].index
        for i in indexes: res[i] = 1.0

    return res

def get_number_of_gaussian_mixture_clusters(col):
    X = np.array(col).reshape(-1,1)
    bic_vals = []
    # Have a minimum of 2 clusters (if 10 rows come in)
    # and a maximum of 9 clusters.
    max_clusters = math.floor(min(col.shape[0]/5.0, 9))
    for c in range(1, max_clusters, 1):
        gm = GaussianMixture(n_components = c, random_state = 0, max_iter = 250, covariance_type='full').fit(X)
        bic_vals.append(gm.bic(X))
    return np.argmin(bic_vals) + 1

def check_gaussian_mixture(col, best_fit_cluster_count):
    # Because this is univariate, we need to reshape the array using -1,1 as our parameters.
    # That will create a list per data point.
    X = np.array(col).reshape(-1,1)
    gm_model = GaussianMixture(n_components = best_fit_cluster_count, random_state = 0, max_iter = 250, covariance_type='full').fit(X)
    xdf = pd.DataFrame(X, columns={"value"})
    xdf["grp"] = list(gm_model.predict(X))
    xdf["score"] = list(gm_model.score_samples(X))
    # Clusters containing less than 5% of data will be marked as outliers.
    min_num_items = math.ceil(xdf.shape[0] * .05)
    small_groups = xdf.groupby('grp').count().reset_index().query('value <= @min_num_items')
    small_groups["small_cluster"] = 1.0
    xdf = xdf.merge(small_groups[['grp', 'small_cluster']], on='grp', how='left').fillna(0)
    # Run MAD check per cluster to see if scores are more than 3 MAD from the median.
    # If so, mark them as outliers.
    for g in xdf["grp"].unique():
        xdf_g = xdf[xdf["grp"] == g]
        calc = perform_statistical_calculations(xdf_g["value"])
        # In case there are duplicate values, we can drop them.
        # The MAD score will be the same for each duplicated item.
        xdf_g = xdf_g.drop_duplicates()
        # If there is no spread within a cluster, we can't calculate MAD.
        if calc["mad"] > 0.0:
            xdf_g['far_off'] = [check_mad(val, calc["median"], calc["mad"], 3.0) for val in xdf_g['value']]
        else:
            xdf_g['far_off'] = 0.0
        for r in range(len(xdf_g)):
            xdf.loc[xdf['value']==xdf_g.iloc[r,0], "far_off"]=xdf_g.iloc[r,4]
    return [max(sc, fo) for (sc, fo) in zip(xdf["small_cluster"], xdf["far_off"])]

def score_results(df, tests_run, weights):
    # Chapter 7:  add in normal distribution checks
    # Add in observation length tests (n <= 25 for Dixon, n >= 7 for Grubbs, n >= 15 for GESD)
    # Determine maximum weight, sum of weights of included tests, and proportionally allocate tests based on available weight
    # Ex:  max weight = 1.05.  Sum of tests = 0.8.  1.05 / 0.8 = 1.3125 * weights of individual tests will sum to 1.05.
    # Perform this calculation as a separate function

    # Because some tests do not run, we want to factor that into our anomaly score.
    tested_weights = {w: weights.get(w, 0) * tests_run.get(w, 0) for w in set(weights).union(tests_run)}
    max_weight = sum([tested_weights[w] for w in tested_weights])

    # If a test was not run, its tests_run[] result will be 0, so we won't include it in our score.
    # If we divide by max weight, we end up with possible values of [0-1].
    # Multiplying by 0.95 makes it a little more likely that we mark an item as an outlier.
    return df.assign(anomaly_score=(
       df['sds'] * tested_weights['sds'] +
       df['iqrs'] * tested_weights['iqrs'] +
       df['mads'] * tested_weights['mads'] +
       df['grubbs'] * tested_weights['grubbs'] +
       df['gesd'] * tested_weights['gesd'] +
       df['dixon'] * tested_weights['dixon'] +
       df['gaussian_mixture'] * tested_weights['gaussian_mixture']
    ) / (max_weight * 0.95))

def determine_outliers(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
    # Convert sensitivity score to be approximately the same
    # scale as anomaly score.  Note that sensitivity score is "reversed",
    # such that 100 is the *most* sensitive.
    sensitivity_score = (100 - sensitivity_score) / 100.0
    # Get the 100-Nth percentile of anomaly score.
    # Ex:  if max_fraction_anomalies = 0.1, get the
    # 90th percentile anomaly score.
    max_fraction_anomaly_score = np.quantile(df['anomaly_score'], 1.0 - max_fraction_anomalies)
    # If the max fraction anomaly score is greater than
    # the sensitivity score, it means that we have MORE outliers
    # than our max_fraction_anomalies supports, and therefore we
    # need to cut it off before we get down to our sensitivity score.
    # Otherwise, sensitivity score stays the same and we operate as normal.
    if max_fraction_anomaly_score > sensitivity_score and max_fraction_anomalies < 1.0:
        sensitivity_score = max_fraction_anomaly_score
    return df.assign(is_anomaly=(df['anomaly_score'] >= sensitivity_score))
