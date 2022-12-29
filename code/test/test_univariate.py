from src.app.models.univariate import *
import pandas as pd
import pytest

@pytest.mark.parametrize("df_input", [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1],
    [1, 2, 3, 4.5, 6.78, 9.10],
    [],
    [1000, 1500, 2230, 13, 1780, 1629, 2202, 2025]
])
def test_detect_univariate_statistical_returns_correct_number_of_rows(df_input):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    sensitivity_score = 50
    max_fraction_anomalies = 0.20
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    # Assert:  the DataFrame is the same length
    assert(df_out.shape[0] == df.shape[0])

@pytest.mark.parametrize("df_input", [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 90],
    [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 5, -13],
    [0.01, 0.03, 0.05, 0.02, 0.01, 0.03, 0.40],
    [1000, 1500, 1230, 13, 1780, 1629, 1450, 1106],
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19.4]
])
def test_detect_univariate_statistical_returns_single_anomaly(df_input):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    sensitivity_score = 50
    max_fraction_anomalies = 0.50
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have exactly one anomaly
    assert(num_anomalies == 1)

@pytest.mark.parametrize("df_input", [
    [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 2550, 9000],
    [1, 1, 1, 2, 3, 3, 5, -13],
    [1, 1, 1, 2, 3, 5, 5, 5, -13, 18],
    [0.01, 0.03, 2, 0.02, 0.01, 0.03, -2.8],
    [1000, 1250, 1173, 13, 1306, 1222, 1064, 1071, 6]
])
def test_detect_univariate_statistical_returns_two_anomalies(df_input):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    sensitivity_score = 50
    max_fraction_anomalies = 0.50
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have exactly two anomalies
    assert(num_anomalies == 2)

@pytest.mark.parametrize("df_input", [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    [1000, 1500, 2230, 13, 1780, 1629, 3202, 3025, 6]
])
def test_detect_univariate_statistical_returns_zero_anomalies(df_input):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    sensitivity_score = 30
    max_fraction_anomalies = 0.50
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have exactly zero anomalies
    assert(num_anomalies == 0)

anomalous_sample = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 2550, 9000]
@pytest.mark.parametrize("df_input, sensitivity_score, number_of_anomalies", [
    (anomalous_sample, 100, 17),
    (anomalous_sample, 95, 13),# In chapter 6, this is 15; in chapter 7, it decreases to 12; in chapter 9, it increases again to 13.
    (anomalous_sample, 85, 9), # In chapter 6, this is 8; in chapter 7, it decreases to 5; in chapter 9, it increases again to 9.
    (anomalous_sample, 75, 6), # In chapter 6, this is 5; in chapter 7, it decreases to 2; in chapter 9, it increases again to 6.
    (anomalous_sample, 50, 2),
    (anomalous_sample, 25, 2),
    (anomalous_sample, 1, 1)   # In chapter 6, this is 1; in chapter 7, it decreases to 0; in chapter 9, it increases back to 1.
])
def test_detect_univariate_statistical_sensitivity_affects_anomaly_count(df_input, sensitivity_score, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    max_fraction_anomalies = 1.0
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have the correct number of anomalies
    assert(num_anomalies == number_of_anomalies)

@pytest.mark.parametrize("df_input, max_fraction_anomalies, number_of_anomalies", [
    (anomalous_sample, 0.0, 0),
    (anomalous_sample, 0.01, 1),
    (anomalous_sample, 0.1, 2),
    (anomalous_sample, 0.2, 4), # In chapter 7, this is 3; in chapter 9, it increases to 4.
    (anomalous_sample, 0.3, 6), # In chapter 7, this is 5; in chapter 9, it decreases to 4 without the equivalency change and increases to 6 with it.
    (anomalous_sample, 0.4, 7), # In chapter 7, this is 6; in chapter 9, it increases to 7.
    (anomalous_sample, 0.5, 9), # In chapter 7, this is 8; in chapter 9, it decreases to 7 without the equivalency change and increases to 9 with it.
    (anomalous_sample, 0.6, 10), # In chapter 7, this is 9; in chapter 9, it increases to 10.
    (anomalous_sample, 0.7, 12),
    (anomalous_sample, 0.8, 13), # In chapter 7, this is 12; in chapter 9, it increases to 13.
    (anomalous_sample, 0.9, 15),
    (anomalous_sample, 1.0, 17)
])
def test_detect_univariate_statistical_max_fraction_anomalies_affects_anomaly_count(df_input, max_fraction_anomalies, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    sensitivity_score = 100.0
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have the correct number of anomalies
    assert(num_anomalies == number_of_anomalies)

uniform_data = [*range(1, 11, 1)]
# calculated from 5 * numpy.random.randn(100) + 50
normal_data = [50.889644  , 43.76473582, 56.5839747 , 51.72014519, 53.14307162,
       43.07173121, 49.28591931, 50.54496406, 53.61226964, 51.1278367 ,
       62.21414378, 46.93479261, 41.84814308, 50.43334892, 57.83838476,
       53.93178873, 52.37157216, 55.70397115, 51.14399302, 53.48140015,
       47.69145951, 51.794747  , 58.22842126, 43.46932648, 54.87004177,
       46.13997236, 53.28952737, 52.23761332, 56.72513471, 44.25527539,
       48.13523092, 55.41576003, 47.03209083, 50.64850049, 59.08908616,
       47.65692877, 46.22832137, 47.09839784, 51.95538775, 41.29499205,
       37.84900822, 51.74245934, 43.71558111, 44.23337738, 53.97194902,
       42.80893996, 54.90576302, 48.96645042, 50.52938633, 52.15746222,
       50.93766447, 43.57955326, 40.52347981, 48.96973179, 58.04508942,
       52.63913521, 58.24929646, 47.38716853, 45.48571601, 52.98105463,
       54.25642131, 49.8524963 , 51.46675059, 43.85321409, 58.64683516,
       51.65353608, 42.96118045, 44.83723836, 49.32198714, 43.40417526,
       56.58462218, 46.98532745, 49.59893321, 46.70444558, 50.63678109,
       43.04565447, 46.5164825 , 51.53128191, 41.93250858, 44.36218222,
       45.06016335, 52.32830471, 45.20161102, 37.43406243, 48.99640973,
       54.7477201 , 50.85640222, 41.81992917, 56.42927902, 52.10116186,
       49.4996767 , 50.00334951, 49.86474579, 47.80312399, 43.41820043,
       46.79107853, 52.1616614 , 50.45914774, 41.6903255 , 48.78243381]
# Normal data, but four values had the decimal moved over three spots.
skewed_data = [49.889644  , 43.76473582, 56.5839747 , 51.72014519, 53.14307162,
       43.07173121, 49.28591931, 50.54496406, 53.61226964, 51.1278367 ,
       62.21414378, 46.93479261, 41.84814308, 50.43334892, 57.83838476,
       53.93178873, 52.37157216, 55.70397115, 51.14399302, 53.48140015,
       47.69145951, 51.794747  , 58.22842126, 43.46932648, 54.87004177,
       46.13997236, 53.28952737, 52.23761332, 56.72513471, 44.25527539,
       48.13523092, 55.41576003, 47.03209083, 50.64850049, 59.08908616,
       47.65692877, 46.22832137, 47.09839784, 51.95538775, 41.29499205,
       37.84900822, 51.74245934, 43.71558111, 44.23337738, 53.97194902,
       42.80893996, 54.90576302, 48.96645042, 50.52938633, 52.15746222,
       50.93766447, 43.57955326, 40.52347981, 48.96973179, 58.04508942,
       52.63913521, 58.24929646, 47.38716853, 45.48571601, 52.98105463,
       54.25642131, 49.8524963 , 51.46675059, 43.85321409, 58.64683516,
       51.65353608, 42.96118045, 44.83723836, 49.32198714, 43.40417526,
       56.58462218, 46.98532745, 49.59893321, 46.70444558, 50.63678109,
       43.04565447, 46.5164825 , 51.53128191, 41.93250858, 44.36218222,
       45.06016335, 52.32830471, 45.20161102, 37.43406243, 48996.40973,
       54.7477201 , 50.85640222, 41.81992917, 56.42927902, 52101.16186,
       49.4996767 , 50.00334951, 49.86474579, 47.80312399, 43418.20043,
       46.79107853, 52.1616614 , 50.45914774, 41.6903255 , 48782.43381]
# Normal data but one skewed value.
single_skewed_data = [48.889644  , 43.76473582, 56.5839747 , 51.72014519, 53.14307162,
       43.07173121, 49.28591931, 50.54496406, 53.61226964, 51.1278367 ,
       62.21414378, 46.93479261, 41.84814308, 50.43334892, 57.83838476,
       53.93178873, 52.37157216, 55.70397115, 51.14399302, 53.48140015,
       47.69145951, 51.794747  , 58.22842126, 43.46932648, 54.87004177,
       46.13997236, 53.28952737, 52.23761332, 56.72513471, 44.25527539,
       48.13523092, 55.41576003, 47.03209083, 50.64850049, 59.08908616,
       47.65692877, 46.22832137, 47.09839784, 51.95538775, 41.29499205,
       37.84900822, 51.74245934, 43.71558111, 44.23337738, 53.97194902,
       42.80893996, 54.90576302, 48.96645042, 50.52938633, 52.15746222,
       50.93766447, 43.57955326, 40.52347981, 48.96973179, 58.04508942,
       52.63913521, 58.24929646, 47.38716853, 45.48571601, 52.98105463,
       54.25642131, 49.8524963 , 51.46675059, 43.85321409, 58.64683516,
       51.65353608, 42.96118045, 44.83723836, 49.32198714, 43.40417526,
       56.58462218, 46.98532745, 49.59893321, 46.70444558, 50.63678109,
       43.04565447, 46.5164825 , 51.53128191, 41.93250858, 44.36218222,
       45.06016335, 52.32830471, 45.20161102, 37.43406243, 48.99640973,
       54.7477201 , 50.85640222, 41.81992917, 56.42927902, 52.10116186,
       49.4996767 , 50.00334951, 49.86474579, 47.80312399, 43.41820043,
       46.79107853, 52.1616614 , 50.45914774, 41.6903255 , 48782.43381]
# Larger uniform data
larger_uniform_data = [*range(50, 0, -1)]

@pytest.mark.parametrize("df_input, function, expected_normal", [
    (uniform_data, check_shapiro, True),
    (normal_data, check_shapiro, True),
    (skewed_data, check_shapiro, False),
    (single_skewed_data, check_shapiro, False),
    (larger_uniform_data, check_shapiro, True),

    (uniform_data, check_dagostino, True),
    (normal_data, check_dagostino, True),
    (skewed_data, check_dagostino, False),
    (single_skewed_data, check_dagostino, False),
    (larger_uniform_data, check_dagostino, False), # Note that this is different from the other tests!

    (uniform_data, check_anderson, True),
    (normal_data, check_anderson, True),
    (skewed_data, check_anderson, False),
    (single_skewed_data, check_anderson, False),
    (larger_uniform_data, check_anderson, True),

])
def test_normalization_call_returns_expected_results(df_input, function, expected_normal):
    # Arrange
    # Act
    (is_normal, result_str) = function(df_input)
    # Assert:  the distribution is/is not normal, based on our expectations.
    assert(expected_normal == is_normal)

@pytest.mark.parametrize("df_input, function, expected_normal", [
    (uniform_data, is_normally_distributed, True),
    (normal_data, is_normally_distributed, True),
    (skewed_data, is_normally_distributed, False),
    (single_skewed_data, is_normally_distributed, False),
    (larger_uniform_data, is_normally_distributed, False), # False because D'Agostino is false.
])
def test_normalization_call_returns_expected_results(df_input, function, expected_normal):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    col = df['value']
    # Act
    (is_normal, result_str) = function(col)
    # Assert:  the distribution is/is not normal, based on our expectations.
    assert(expected_normal == is_normal)

@pytest.mark.parametrize("df_input, should_normalize", [
    (uniform_data, True), # Is naturally normal (enough)
    (normal_data, True), # Is naturally normal
    (skewed_data, True),
    (single_skewed_data, True),
    (larger_uniform_data, True), # Should normalize because D'Agostino is false.
    ([1,2,3], False), # Not enough datapoints to normalize.
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], False), # No variance in the data.
    ([100,20,3,40,500,6000,70,800,9,10,11,12,13,-1], False), # Not naturally normal and has a negative value--all values must be > 0.
    ([100,20,3,40,500,6000,70,800,0,10,11,12,13,14], False), # Not naturally normal and has a zero value--all values must be > 0.
])
def test_perform_normalization_only_works_on_valid_datasets(df_input, should_normalize):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    base_calculations = perform_statistical_calculations(df['value'])
    # Act
    (did_normalize, fitted_data, normalization_diagnostics) = perform_normalization(base_calculations, df)
    # Assert:  the distribution is/is not normal, based on our expectations.
    assert(should_normalize == did_normalize)

normal_with_negatives = [3.86094301, -17.6370525, 4.67057423, -9.43551162, -12.83771971, -3.41792716, 0.52941589, -4.83401163, -5.03106433, -8.04018073]
@pytest.mark.parametrize("df_input, test_name, test_should_run", [
    (uniform_data, "grubbs", True), # At least 7 data points, normal-enough data.
    (uniform_data, "dixon", True), # Between 3 and 25 data points, normal-enough data.
    (uniform_data, "gesd", False), # Not enough data points.

    (normal_data, "grubbs", True), # At least 7 data points, normal-enough data.
    (normal_data, "dixon", False), # Not between 3 and 25 data points.
    (normal_data, "gesd", True), # At least 15 data points, normal-enough data.

    (skewed_data, "grubbs", True), # At least 7 data points, normal-enough data.
    (skewed_data, "dixon", False), # Not between 3 and 25 data points.
    (skewed_data, "gesd", True), # At least 15 data points, normal-enough data.

    (single_skewed_data, "grubbs", True), # At least 7 data points, normal-enough data.
    (single_skewed_data, "dixon", False), # Not between 3 and 25 data points.
    (single_skewed_data, "gesd", True), # At least 15 data points, normal-enough data.

    (larger_uniform_data, "grubbs", True), # At least 7 data points, normal-enough data.
    (larger_uniform_data, "dixon", False), # Not between 3 and 25 data points.
    (larger_uniform_data, "gesd", True), # At least 15 data points, normal-enough data.

    ([1,2,3], "grubbs", False), # Not enough data points to normalize.
    ([1,2,3], "dixon", False), # Normality checks insufficient to determine if we can run this test.
    ([1,2,3], "gesd", False), # Not enough data points to normalize.

    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "grubbs", False), # No variance in the data.
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "dixon", False), # No variance in the data.
    ([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], "gesd", False), # No variance in the data.

    ([100,20,3,40,500,6000,70,800,9,10,11,12,13,-1], "grubbs", False), # Has a negative value--all values must be > 0.
    ([100,20,3,40,500,6000,70,800,9,10,11,12,13,-1], "dixon", False), # Has a negative value--all values must be > 0.
    ([100,20,3,40,500,6000,70,800,9,10,11,12,13,-1], "gesd", False), # Has a negative value--all values must be > 0.

    ([220,20,3,40,500,6000,70,800,0,10,11,12,13,14], "grubbs", False), # Has a zero value--all values must be > 0.
    ([220,20,3,40,500,6000,70,800,0,10,11,12,13,14], "dixon", False), # Has a zero value--all values must be > 0.
    ([220,20,3,40,500,6000,70,800,0,10,11,12,13,14], "gesd", False), # Has a zero value--all values must be > 0.

    (normal_with_negatives, "grubbs", True), # Has negative values but is already normal-enough.
    (normal_with_negatives, "dixon", True), # Has negative values but is already normal-enough.
    (normal_with_negatives, "gesd", False), # Not enough datapoints to run the test.
])
def test_normalization_required_for_certain_tests(df_input, test_name, test_should_run):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    # Act
    (df_tested, tests_run, diagnostics) = run_tests(df)
    # Assert:  the distribution is/is not normal, based on our expectations.
    assert(test_should_run == diagnostics["Tests Run"][test_name])


@pytest.mark.parametrize("df_input, number_of_anomalies", [
    ([1, 1, 1, 2, 2, 2, 3, 3, 98, 98, 98, 99, 99, 99, 100, 100], 0), # Two clusters with no outliers
    ([1, 1, 1, 2, 2, 2, 3, 3, 50, 98, 98, 98, 99, 99, 99, 100, 100], 1), # Two clusters with one outlier
    ([1, 1, 1, 2, 2, 2, 3, 3, 50, -50, 98, 98, 98, 99, 99, 99, 100, 100], 2), # Two clusters with two outliers
    ([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 98, 98, 99, 99, 100], 0), # 4 clusters (due to whole numbers and cap), with no outliers
    ([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 50, 98, 98, 99, 99, 100], 1), # 5 clusters (due to whole numbers and cap) with 50 as outlier
    ([1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, -50, 50, 98, 98, 99, 99, 100], 5), # 5 clusters (due to whole numbers and cap) with -50, 50, 98, and 100 as outliers
    ([1.4, 1.2, 1.0, 1.8, 1.4, 1.3, 1.8, 2.0, 2.1, 2.3, 2.5, 2.3, 2.6, 2.8, 2.4, 2.3, 3.1, 3.9, 3.2, 3.7, 3.1, 3.0, 3.4, 3.3, 98.6, 98.3, 99.6, 99.9, 100.2], 2), # Two mismatched-sized clusters.  98.3 and 98.6 are outliers because of MAD and IQR.
    ([1.4, 1.2, 1.0, 1.8, 1.4, 1.3, 1.8, 2.0, 2.1, 2.3, 2.5, 2.3, 2.6, 2.8, 2.4, 2.3, 3.1, 3.9, 3.2, 3.7, 3.1, 3.0, 3.4, 3.3, 50.2, 98.6, 98.3, 99.6, 99.9, 100.2], 3), # Three clusters.  98.3 and 98.6 marked as outliers because of high MAD and IQR scores.
    ([1.4, 1.2, 1.0, 1.8, 1.4, 1.3, 1.8, 2.0, 2.1, 2.3, 2.5, 2.3, 2.6, 2.8, 2.4, 2.3, 3.1, 3.9, 3.2, 3.7, 3.1, 3.0, 3.4, 3.3, 50.2, -50.1, 98.6, 98.3, 99.6, 99.9, 100.2], 5), # Four clusters.  98.3, 98.6, and 100.2 marked as outliers because of high MAD and IQR scores.
])
def test_detect_univariate_statistical_multi_cluster(df_input, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns={"value"})
    sensitivity_score = 50.0
    max_fraction_anomalies = 1.0
    # Act
    (df_out, weights, details) = detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    num_anomalies = df_out[df_out['is_anomaly'] == True].shape[0]
    # Assert:  we have the correct number of anomalies
    assert(num_anomalies == number_of_anomalies)