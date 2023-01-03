from numpy import number
from src.app.models.multivariate import *
import pandas as pd
import pytest

# Test encoding
@pytest.mark.parametrize("df_input, requires_encoding, number_of_string_columns", [
    ([["s1", [1, 30.1, 2, -1, 3]], ["s2", [4, 19.6, 5, -2, 6]], ["s3", [7, 17.3, 8, -3, 9]]], False, 0),
    ([["s1a", [1, "Bob", 2, "Janice", 3]], ["s2", [4, "Jim", 5, "Alice", 6]], ["s3", [7, "Teddy", 8, "Mercedes", 9]]], True, 2),
    ([["s1b", [1, 30.1, 2, "Janice", 3]], ["s2", [4, 19.6, 5, "Alice", 6]], ["s3", [7, 17.3, 8, "Mercedes", 9]]], True, 1),
    ([["s1c", [1, 30.1, 2, -1, 3]], ["s2", [4, 19.6, 5, -2, 6]], ["s3", [7, 17.3, 8, -3, 9]]], False, 0),
    ([["s1d", [1, 30.1, 2, "-1", 3]], ["s2", [4, 19.6, 5, "-2", 6]], ["s3", [7, 17.3, 8, "Mercedes", 9]]], True, 1),
])
def test_detect_multivariate_encoding_string_columns(df_input, requires_encoding, number_of_string_columns):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "vals"])
    # Act
    (df_encoded, diagnostics) = encode_string_data(df)
    encoding_performed = (diagnostics["Encoding Operation"] == "Encoding performed on string columns.")
    num_string_columns = diagnostics["Number of string columns in input"]
    # Assert
    assert(requires_encoding == encoding_performed)
    assert(number_of_string_columns == num_string_columns)

sample_input = [["1604", [87,16,6184.90844,0.771,11.72]],
["91849", [7921,12,6337.69829,0.919,11.55]],
["55194", [4497,5,5639.15773,0.678,4.71]],
["63735", [5969,18,29296.31524,0.605,26.42]],
["66847", [6166,18,29250.95360,0.616,25.23]],
["40687", [2508,12,6002.59147,0.847,9.34]],
["68406", [6276,18,30314.86144,0.639,23.94]],
["58621", [5637,13,28236.24848,0.760,3.70]],
["26993", [1580,7,14753.01818,0.844,46.33]],
["72869", [6529,8,12119.75085,0.599,20.10]],
["34158", [2057,20,45522.06835,0.749,13.12]],
["82060", [7054,18,31684.05119,0.641,22.87]],
["6267", [350,8,12218.47201,0.597,21.06]],
["82746", [7096,18,30300.03729,0.615,20.71]],
["5690", [314,18,31649.29269,0.637,24.16]],
["12827", [732,16,6074.75177,0.758,5.61]],
["33968", [2044,12,5423.63057,0.758,11.68]],
["34886", [2110,20,47921.24632,0.812,17.00]],
["35040", [2120,12,5952.07984,0.838,5.73]],
["47898", [3300,4,5699.15996,0.273,37.04]],
["15481", [882,8,15107.77381,0.741,18.99]],
["83709", [7147,18,30420.61485,0.631,26.39]],
["13128", [754,4,16087.76200,0.783,33.88]],
["2754", [152,18,37380.01275,0.782,25.17]],
["29174", [1726,20,47167.66052,0.780,13.62]],
["66683", [6159,18,29469.04211,0.607,21.31]],
["32633", [1955,8,19139.68534,0.962,16.63]],
["93150", [8087,12,6073.21747,0.872,4.47]],
["14351", [823,18,35772.06830,0.728,17.34]],
["13319", [765,10,16204.94920,0.781,22.20]],
["27798", [1632,5,6149.61433,0.739,5.97]],
["6504", [364,8,15087.57297,0.734,19.90]],
["76904", [6761,4,12751.12407,0.607,29.56]],
["90966", [7867,7,15906.27698,0.914,41.44]],
["63117", [5935,16,5081.29066,0.635,8.63]],
["88182", [7566,7,15048.51140,0.894,42.44]],
["16379", [929,8,15935.15582,0.775,9.52]],
["80830", [6994,4,12070.31544,0.566,29.62]],
["13704", [790,5,6415.32667,0.796,9.15]],
["52555", [3968,12,5364.16998,0.772,4.28]],
["64886", [6061,5,4680.25499,0.552,3.32]],
["18676", [1054,8,21683.18660,1.052,16.97]],
["61652", [5858,4,12309.67692,0.580,31.31]],
["96201", [8454,12,5415.56050,0.796,3.84]],
["64417", [6008,8,10123.78001,0.491,20.05]],
["55924", [4711,4,12427.68119,0.584,28.23]],
["80712", [6989,18,28451.14311,0.572,17.41]],
["50423", [3621,12,5225.04197,0.746,5.89]],
["83832", [7155,4,10930.86162,0.520,33.42]],
["55333", [4534,4,10292.80772,0.510,33.41]],
["28299", [1664,14,6141.59160,0.775,6.61]],
["43797", [2731,4,9399.95612,0.459,32.55]],
["97702", [8658,7,17132.85035,0.990,51.12]],
["64752", [6054,18,27087.69870,0.554,21.76]],
["89281", [7689,8,14659.85335,0.717,17.63]],
["85605", [7248,16,4803.94926,0.608,4.86]],
["98327", [8745,4,12951.71462,0.629,32.55]],
["1676", [92,16,5413.43709,0.673,8.01]],
["12434", [711,16,6894.92535,0.864,7.77]],
["88699", [7638,7,14768.86617,0.851,47.91]],
["92540", [8010,7,12860.19754,0.723,45.23]],
["93748", [8168,7,15228.72446,0.858,50.28]],
["69400", [6331,18,28540.14690,0.576,28.48]],
["80550", [6980,18,28277.89489,0.587,21.27]],
["67481", [6206,18,31529.09017,0.644,22.85]],
["68156", [6262,18,33981.25580,0.695,29.16]],
["34374", [2073,12,5748.41137,0.851,6.64]],
["36682", [2230,12,6289.54273,0.887,5.33]],
["6593", [369,8,20171.80051,0.977,21.31]],
["56195", [4833,20,42826.20755,0.726,17.20]],
["66359", [6145,8,16752.59312,0.811,13.68]],
["74576", [6639,4,13488.83315,0.627,37.04]],
["92035", [7941,14,7111.64887,0.878,5.40]],
["86848", [7413,7,13515.33168,0.801,44.27]],
["41796", [2588,12,5869.83301,0.822,9.34]],
["50294", [3613,12,5780.46619,0.812,6.96]],
["69040", [6314,8,15286.35829,0.747,9.16]],
["42733", [2656,12,5878.78651,0.800,7.07]],
["72647", [6521,4,14758.57022,0.713,30.81]],
["42512", [2640,20,49648.53389,0.865,18.17]],
["47445", [3239,12,5869.98514,0.822,4.90]],
["52906", [4002,4,10514.68364,0.518,28.61]],
["26699", [1559,12,6195.28802,0.849,3.25]],
["7650", [440,16,6458.71299,0.803,6.16]],
["9062", [520,16,5592.13467,0.703,6.48]],
["83664", [7145,18,28942.30785,0.608,22.73]],
["27557", [1617,5,7544.19022,0.924,4.55]],
["43827", [2742,4,11095.23574,0.532,33.33]],
["87553", [7479,8,14166.60125,0.689,18.40]],
["40095", [2466,14,6407.21687,0.826,6.65]],
["25421", [1468,20,46906.47825,0.789,14.70]],
["85720", [7256,8,16811.41031,0.787,20.51]],
["22029", [1246,8,11724.22073,0.591,20.69]],
["72258", [6502,18,29708.30412,0.592,29.87]],
["59805", [5732,4,11377.78071,0.562,33.84]],
["34981", [2116,14,6746.98314,0.854,7.79]],
["70172", [6379,18,30335.06885,0.617,23.15]],
["44208", [2796,4,12504.24445,0.597,29.56]],
["88612", [7626,19,9082.90086,0.836,28.43]],
["74610", [6640,18,32264.85829,0.647,25.97]]]

@pytest.mark.parametrize("df_input, should_run_cof", [
    (sample_input, 1),
    ([["k1", [1,2,3]]], 0),
])
def test_detect_multivariate_runs_cof(df_input, should_run_cof):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "vals"])
    sensitivity_score = 50
    max_fraction_anomalies = 1.0
    n_neighbors = 10
    # Act
    (df_out, weights, diagnostics) = detect_multivariate_statistical(df, sensitivity_score, max_fraction_anomalies, n_neighbors)
    # Assert
    if "Tests run" in diagnostics:
        did_run_cof = diagnostics["Tests run"]["cof"]
    else:
        did_run_cof = 0
    assert(should_run_cof == did_run_cof)

@pytest.mark.parametrize("df_input, sensitivity_score, number_of_anomalies", [
    (sample_input, 100, 6),     # Was 11 in chapter 10, 8 in chapter 11
    (sample_input, 50, 6),      # Was 11 in chapter 10, 8 in chapter 11
    (sample_input, 40, 6),      # Was 11 in chapter 10, 8 in chapter 11
    (sample_input, 25, 6),      # Was 4 in chapter 10, 8 in chapter 11
    (sample_input, 5, 5),       # Was 2 in chapter 10
    (sample_input, 1, 3),       # Was 1 in chapter 10, 2 in chapter 11
    (sample_input, 0, 0),
])
def test_detect_multivariate_cof_sample_sensitivity(df_input, sensitivity_score, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "vals"])
    max_fraction_anomalies = 1.0
    n_neighbors = 10
    # Act
    (df_out, weights, diagnostics) = detect_multivariate_statistical(df, sensitivity_score, max_fraction_anomalies, n_neighbors)
    print(df_out.sort_values(by=['anomaly_score']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])

@pytest.mark.parametrize("df_input, max_fraction_anomalies, number_of_anomalies", [
    (sample_input, 1.0, 6),     # Was 8 in chapter 11
    (sample_input, 0.8, 6),     # Was 8 in chapter 11
    (sample_input, 0.6, 6),     # Was 8 in chapter 11
    (sample_input, 0.4, 6),     # Was 8 in chapter 11
    (sample_input, 0.2, 6),     # Was 8 in chapter 11
    (sample_input, 0.1, 6),     # Was 8 in chapter 11
    (sample_input, 0.01, 1),
])
def test_detect_multivariate_cof_sample_fraction(df_input, max_fraction_anomalies, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "vals"])
    sensitivity_score = 50
    n_neighbors = 10
    # Act
    (df_out, weights, diagnostics) = detect_multivariate_statistical(df, sensitivity_score, max_fraction_anomalies, n_neighbors)
    print(df_out.sort_values(by=['anomaly_score']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])

# Note that even when we don't see outliers, COF and LOCI may still catch minor differences.
sample_input_one_outlier = [["1604", [87,16,6184.90844,0.771,11.72]],
["1604", [87.0,16,6184.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.4,16,6186.90844,0.771,11.72]],
["1604", [87.5,16,6185.90844,0.771,11.72]],
["1604", [87.6,16,6184.90844,0.771,11.72]],
["1604", [87.7,16,6183.90844,0.771,11.72]],
["1604", [87.8,16,6182.90844,0.771,11.72]],
["1604", [87.9,16,6181.90844,0.771,11.72]],
["1604", [87.0,16,6180.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.4,16,6186.90844,0.771,11.72]],
["1604", [87.5,16,6185.90844,0.771,11.72]],
["1604", [87.6,16,6184.90844,0.771,11.72]],
["1604", [87.7,16,6183.90844,0.771,11.72]],
["1604", [87.8,16,6182.90844,0.771,11.72]],
["1604", [87.9,16,6181.90844,0.771,11.72]],
["1604", [87.0,16,6180.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.4,16,6186.90844,0.771,11.72]],
["1604", [87.5,16,6185.90844,0.771,11.72]],
["1604", [87.6,16,6184.90844,0.771,11.72]],
["1604", [87.7,16,6183.90844,0.771,11.72]],
["1604", [87.8,16,6182.90844,0.771,11.72]],
["1604", [87.9,16,6181.90844,0.771,11.72]],
["1604", [87.0,16,6180.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.4,16,6186.90844,0.771,11.72]],
["1604", [87.5,16,6185.90844,0.771,11.72]],
["1604", [87.6,16,6184.90844,0.771,11.72]],
["1604", [87.7,16,6183.90844,0.771,11.72]],
["1604", [87.8,16,6182.90844,0.771,11.72]],
["1604", [87.9,16,6181.90844,0.771,11.72]],
["1604", [87.0,16,6180.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.3,16,6186.90844,0.771,11.72]],
["1604", [87.4,16,6185.90844,0.771,11.72]],
["1604", [87.5,16,6184.90844,0.771,11.72]],
["1604", [87.6,16,6183.90844,0.771,11.72]],
["1604", [87.7,16,6182.90844,0.771,11.72]],
["1604", [87.8,16,6181.90844,0.771,11.72]],
["1604", [87.9,16,6185.90844,0.771,11.72]], # Changed because LOCI catches the "double-extreme" test case.
["1604", [87.0,16,6189.90844,0.771,11.72]],
["1604", [87.1,16,6188.90844,0.771,11.72]],
["1604", [87.2,16,6187.90844,0.771,11.72]],
["74610", [6640,18,32264.85829,0.647,25.97]]]

@pytest.mark.parametrize("df_input, sensitivity_score, number_of_anomalies", [
    (sample_input_one_outlier, 100, 1),
    (sample_input_one_outlier, 98, 1),
    (sample_input_one_outlier, 50, 1),
    (sample_input_one_outlier, 40, 1),
    (sample_input_one_outlier, 25, 1),
    (sample_input_one_outlier, 5, 1),
    (sample_input_one_outlier, 1, 1),
    (sample_input_one_outlier, 0, 0),
])
def test_detect_multivariate_cof_single_sensitivity(df_input, sensitivity_score, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "vals"])
    max_fraction_anomalies = 1.0
    n_neighbors = 10
    # Act
    (df_out, weights, diagnostics) = detect_multivariate_statistical(df, sensitivity_score, max_fraction_anomalies, n_neighbors)
    print(df_out.sort_values(by=['anomaly_score']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])

@pytest.mark.parametrize("df_input, max_fraction_anomalies, number_of_anomalies", [
    (sample_input_one_outlier, 1.0, 1),
    (sample_input_one_outlier, 0.4, 1),
    (sample_input_one_outlier, 0.3, 1),
    (sample_input_one_outlier, 0.2, 1),
    (sample_input_one_outlier, 0.1, 1),
    (sample_input_one_outlier, 0.05, 1),
    (sample_input_one_outlier, 0, 0),
])
def test_detect_multivariate_cof_single_fraction(df_input, max_fraction_anomalies, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "vals"])
    sensitivity_score = 75
    n_neighbors = 10
    # Act
    (df_out, weights, diagnostics) = detect_multivariate_statistical(df, sensitivity_score, max_fraction_anomalies, n_neighbors)
    print(df_out.sort_values(by=['anomaly_score']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])

sample_input_no_outliers = [["1604", [87,16,6184.90844,0.771,11.72]],
["1604", [87.0,16,6184.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.4,16,6186.90844,0.771,11.72]],
["1604", [87.5,16,6185.90844,0.771,11.72]],
["1604", [87.6,16,6184.90844,0.771,11.72]],
["1604", [87.7,16,6183.90844,0.771,11.72]],
["1604", [87.8,16,6182.90844,0.771,11.72]],
["1604", [87.9,16,6181.90844,0.771,11.72]],
["1604", [87.0,16,6180.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.4,16,6186.90844,0.771,11.72]],
["1604", [87.5,16,6185.90844,0.771,11.72]],
["1604", [87.6,16,6184.90844,0.771,11.72]],
["1604", [87.7,16,6183.90844,0.771,11.72]],
["1604", [87.8,16,6182.90844,0.771,11.72]],
["1604", [87.9,16,6181.90844,0.771,11.72]],
["1604", [87.0,16,6180.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.4,16,6186.90844,0.771,11.72]],
["1604", [87.5,16,6185.90844,0.771,11.72]],
["1604", [87.6,16,6184.90844,0.771,11.72]],
["1604", [87.7,16,6183.90844,0.771,11.72]],
["1604", [87.8,16,6182.90844,0.771,11.72]],
["1604", [87.9,16,6181.90844,0.771,11.72]],
["1604", [87.0,16,6180.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.4,16,6186.90844,0.771,11.72]],
["1604", [87.5,16,6185.90844,0.771,11.72]],
["1604", [87.6,16,6184.90844,0.771,11.72]],
["1604", [87.7,16,6183.90844,0.771,11.72]],
["1604", [87.8,16,6182.90844,0.771,11.72]],
["1604", [87.9,16,6181.90844,0.771,11.72]],
["1604", [87.0,16,6180.90844,0.771,11.72]],
["1604", [87.1,16,6189.90844,0.771,11.72]],
["1604", [87.2,16,6188.90844,0.771,11.72]],
["1604", [87.3,16,6187.90844,0.771,11.72]],
["1604", [87.3,16,6186.90844,0.771,11.72]],
["1604", [87.4,16,6185.90844,0.771,11.72]],
["1604", [87.5,16,6184.90844,0.771,11.72]],
["1604", [87.6,16,6183.90844,0.771,11.72]],
["1604", [87.7,16,6182.90844,0.771,11.72]],
["1604", [87.8,16,6181.90844,0.771,11.72]],
["1604", [87.9,16,6185.90844,0.771,11.72]], # Changed because LOCI catches the "double-extreme" test case.
["1604", [87.0,16,6189.90844,0.771,11.72]],
["1604", [87.1,16,6188.90844,0.771,11.72]],
["1604", [87.2,16,6187.90844,0.771,11.72]]]

@pytest.mark.parametrize("df_input, sensitivity_score, number_of_anomalies", [
    (sample_input_no_outliers, 100, 0),
    (sample_input_no_outliers, 50, 0),
    (sample_input_no_outliers, 40, 0),
    (sample_input_no_outliers, 25, 0),
    (sample_input_no_outliers, 9, 0),
    (sample_input_no_outliers, 8, 0),
    (sample_input_no_outliers, 5, 0),
    (sample_input_no_outliers, 1, 0),
    (sample_input_no_outliers, 0, 0),
])
def test_detect_multivariate_cof_none_sensitivity(df_input, sensitivity_score, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "vals"])
    max_fraction_anomalies = 1.0
    n_neighbors = 10
    # Act
    (df_out, weights, diagnostics) = detect_multivariate_statistical(df, sensitivity_score, max_fraction_anomalies, n_neighbors)
    print(df_out.sort_values(by=['anomaly_score']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])

@pytest.mark.parametrize("df_input, max_fraction_anomalies, number_of_anomalies", [
    (sample_input_no_outliers, 1.0, 0),
    (sample_input_no_outliers, 0.4, 0),
    (sample_input_no_outliers, 0.3, 0),
    (sample_input_no_outliers, 0.2, 0),
    (sample_input_no_outliers, 0.1, 0),
    (sample_input_no_outliers, 0.05, 0),
    (sample_input_no_outliers, 0, 0),
])
def test_detect_multivariate_cof_none_fraction(df_input, max_fraction_anomalies, number_of_anomalies):
    # Arrange
    df = pd.DataFrame(df_input, columns=["key", "vals"])
    sensitivity_score = 75
    n_neighbors = 10
    # Act
    (df_out, weights, diagnostics) = detect_multivariate_statistical(df, sensitivity_score, max_fraction_anomalies, n_neighbors)
    print(df_out.sort_values(by=['anomaly_score']))
    # Assert
    assert(number_of_anomalies == df_out[df_out['is_anomaly'] == True].shape[0])