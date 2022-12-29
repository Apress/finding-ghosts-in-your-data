from numpy import number
from app.models.multi_timeseries import detect_multi_timeseries
from src.app.models.multi_timeseries import *
import pandas as pd
import pytest

sample_input = [["k1",  "s1", "2021-12-11T08:00:00Z", 14.3],
["k2",  "s1", "2021-12-11T09:00:00Z", 15.3],
["k3",  "s1", "2021-12-11T10:00:00Z", 15.8],
["k4",  "s1", "2021-12-11T11:00:00Z", 16.2],
["k5",  "s1", "2021-12-11T12:00:00Z", 16.4],
["k6",  "s1", "2021-12-11T13:00:00Z", 16.5],
["k7",  "s1", "2021-12-11T14:00:00Z", 16.3],
["k8",  "s1", "2021-12-11T15:00:00Z", 16.0],
["k9",  "s1", "2021-12-11T16:00:00Z", 15.5],
["k10", "s1", "2021-12-11T17:00:00Z", 15.1],
["k11", "s1", "2021-12-11T18:00:00Z", 14.6],
["k12", "s1", "2021-12-11T19:00:00Z", 14.4],
["k13", "s1", "2021-12-11T20:00:00Z", 14.1],
["k14", "s1", "2021-12-11T21:00:00Z", 13.9],
["k15", "s1", "2021-12-11T22:00:00Z", 13.7],
["k16", "s1", "2021-12-11T23:00:00Z", 190.8],
["k17", "s1", "2021-12-12T00:00:00Z", 193.7],
["k1a",  "s2", "2021-12-11T08:00:00Z", 24.3],
["k2a",  "s2", "2021-12-11T09:00:00Z", 25.3],
["k3a",  "s2", "2021-12-11T10:00:00Z", 25.8],
["k4a",  "s2", "2021-12-11T11:00:00Z", 26.2],
["k5a",  "s2", "2021-12-11T12:00:00Z", 26.4],
["k6a",  "s2", "2021-12-11T13:00:00Z", 26.5],
["k7a",  "s2", "2021-12-11T14:00:00Z", 26.3],
["k8a",  "s2", "2021-12-11T15:00:00Z", 26.0],
["k9a",  "s2", "2021-12-11T16:00:00Z", 25.5],
["k10a", "s2", "2021-12-11T17:00:00Z", 25.1],
["k11a", "s2", "2021-12-11T18:00:00Z", 24.6],
["k12a", "s2", "2021-12-11T19:00:00Z", 24.4],
["k13a", "s2", "2021-12-11T20:00:00Z", 4.1],
["k14a", "s2", "2021-12-11T21:00:00Z", 213.9],
["k15a", "s2", "2021-12-11T22:00:00Z", 23.7],
["k16a", "s2", "2021-12-11T23:00:00Z", 17.8],
["k17a", "s2", "2021-12-12T00:00:00Z", 183.7]]

@pytest.mark.parametrize("df_input, sensitivity_score, number_of_anomalies", [
    (sample_input, 100, 16),
    (sample_input, 90, 16),
    (sample_input, 80, 16),
    (sample_input, 70, 16),
    (sample_input, 60, 0), # Was 16 in chapter 16
    (sample_input, 50, 0),
    (sample_input, 40, 0),
    (sample_input, 25, 0),
    (sample_input, 5, 0),
    (sample_input, 1, 0),
    (sample_input, 0, 0),
])
def test_detect_multi_timeseries_sample_sensitivity(df_input, sensitivity_score, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "series_key", "dt", "value"])
    max_fraction_anomalies = 1.0
    # Act
    (df_out, weights, diagnostics) = detect_multi_timeseries(df, sensitivity_score, max_fraction_anomalies)
    print(df_out.sort_values(by=['dt']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])

@pytest.mark.parametrize("df_input, max_fraction_anomalies, number_of_anomalies", [
    (sample_input, 1.0, 16),
    (sample_input, 0.9, 16),
    (sample_input, 0.8, 16),
    (sample_input, 0.7, 16),
    (sample_input, 0.6, 16),
    (sample_input, 0.5, 16),
    (sample_input, 0.4, 16),
    (sample_input, 0.3, 16),
    (sample_input, 0.2, 16),
    (sample_input, 0.1, 16),
    (sample_input, 0, 0),
])
def test_detect_single_timeseries_sample_fraction(df_input, max_fraction_anomalies, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "series_key", "dt", "value"])
    sensitivity_score = 70
    # Act
    (df_out, weights, diagnostics) = detect_multi_timeseries(df, sensitivity_score, max_fraction_anomalies)
    print(df_out.sort_values(by=['dt']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])
