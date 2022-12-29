from numpy import number
from app.models.single_timeseries import detect_single_timeseries
from src.app.models.single_timeseries import *
import pandas as pd
import pytest
import ruptures as rpt
from ruptures.metrics import precision_recall

# Perform test and get results.  Should be within 2% of the actual change for 1000 points.
def test_ruptures_basic_usage():
    # Arrange
    # Generate artificial test data with Gaussian noise and 4 breakpoints
    # https://centre-borelli.github.io/ruptures-docs/getting-started/basic-usage/
    n_samples, n_dims, sigma = 1000, 3, 2
    n_bkps = 4
    signal, bkps = rpt.pw_constant(n_samples, n_dims, n_bkps, noise_std=sigma)
    # Act
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = algo.predict(pen=10)
    # Assert
    # Test one:  no single element is more than 2% off
    max_difference = n_samples * 0.02
    differences = [abs(r-b) for (r,b) in zip(result, bkps)]
    num_differences = len([i for i in differences if i > max_difference])
    assert(num_differences == 0)
    # Test 2:  we correctly detect every change point (within 2%) and have zero spurious detections.
    p, r = precision_recall(result, bkps, margin=max_difference)
    assert(p == 1.0)
    assert(r == 1.0)

sample_input = [["k1", "2021-12-11T08:00:00Z", 14.3],
["k2", "2021-12-11T09:00:00Z", 15.3],
["k3", "2021-12-11T10:00:00Z", 15.8],
["k4", "2021-12-11T11:00:00Z", 16.2],
["k5", "2021-12-11T12:00:00Z", 16.4],
["k6", "2021-12-11T13:00:00Z", 16.5],
["k7", "2021-12-11T14:00:00Z", 16.3],
["k8", "2021-12-11T15:00:00Z", 16.0],
["k9", "2021-12-11T16:00:00Z", 15.5],
["k10", "2021-12-11T17:00:00Z", 15.1],
["k11", "2021-12-11T18:00:00Z", 14.6],
["k12", "2021-12-11T19:00:00Z", 14.4],
["k13", "2021-12-11T20:00:00Z", 14.1],
["k14", "2021-12-11T21:00:00Z", 13.9],
["k15", "2021-12-11T22:00:00Z", 13.7],
["k16", "2021-12-11T23:00:00Z", 190.8],
["k17", "2021-12-12T00:00:00Z", 193.7]]

@pytest.mark.parametrize("df_input, sensitivity_score, number_of_anomalies", [
    (sample_input, 100, 8),
    (sample_input, 90, 7),
    (sample_input, 80, 7),
    (sample_input, 70, 4),
    (sample_input, 60, 3),
    (sample_input, 50, 1),
    (sample_input, 40, 1),
    (sample_input, 25, 0),
    (sample_input, 5, 0),
    (sample_input, 1, 0),
    (sample_input, 0, 0),
])
def test_detect_single_timeseries_sample_sensitivity(df_input, sensitivity_score, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "dt", "value"])
    max_fraction_anomalies = 1.0
    # Act
    (df_out, weights, diagnostics) = detect_single_timeseries(df, sensitivity_score, max_fraction_anomalies)
    print(df_out.sort_values(by=['dt']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])

@pytest.mark.parametrize("df_input, max_fraction_anomalies, number_of_anomalies", [
    (sample_input, 1.0, 4),
    (sample_input, 0.9, 4),
    (sample_input, 0.8, 4),
    (sample_input, 0.7, 4),
    (sample_input, 0.6, 4),
    (sample_input, 0.5, 4),
    (sample_input, 0.4, 4),
    (sample_input, 0.3, 4),
    (sample_input, 0.2, 4),
    (sample_input, 0.1, 1),
    (sample_input, 0, 0),
])
def test_detect_single_timeseries_sample_fraction(df_input, max_fraction_anomalies, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "dt", "value"])
    sensitivity_score = 70
    # Act
    (df_out, weights, diagnostics) = detect_single_timeseries(df, sensitivity_score, max_fraction_anomalies)
    print(df_out.sort_values(by=['dt']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])
