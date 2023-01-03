# If you do not already have Azure Cognitive Services Anomaly Detector installed:
# pip install azure-ai-anomalydetector

from regex import F
import requests
import pandas as pd
import json
from pathlib import Path
import glob
from azure.ai.anomalydetector import AnomalyDetectorClient
from azure.ai.anomalydetector.models import DetectRequest, TimeSeriesPoint, TimeGranularity, AnomalyDetectorError
from azure.core.credentials import AzureKeyCredential
import os
import datetime as dt

# Helper functions for processing our anomaly detection engine.
def read_file_book(input_file):
    input_df = pd.read_csv(input_file)
    input_df['key'] = input_df.index
    input_df['dt'] = pd.to_datetime(input_df['timestamp'])
    return input_df


def detect_outliers_book(server_url, method, sensitivity_score, max_fraction_anomalies, debug, input_df):
    input_data_set = input_df[['key', 'dt', 'value']].to_json(orient='records', date_format='iso')
    full_server_url = f"{server_url}/{method}?sensitivity_score={sensitivity_score}&max_fraction_anomalies={max_fraction_anomalies}&debug={debug}"
    r = requests.post(
        full_server_url,
        data=input_data_set,
        headers={"Content-Type": "application/json"}
    )
    res = json.loads(r.content)
    cutoff = res['debug_details']['Outlier determination']['Sensitivity score']
    df = pd.DataFrame(res['anomalies'])
    df = df.drop('key', axis=1)
    df['anomaly_score'] = 0.5 * df['anomaly_score'] / cutoff
    # If anomaly score is greater than 1, set it to 1.0.
    df.loc[df['anomaly_score'] > 1.0, 'anomaly_score'] = 1.0
    df['label'] = 1 * df['is_anomaly']
    df = df.drop('dt', axis=1)
    return df


def write_file_book(df, output_file):
    output_filepath = Path(output_file)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def process_book(data_folder, results_folder):
    server_url = "http://localhost/detect"
    method = "timeseries/single"
    sensitivity_score = 55
    max_fraction_anomalies = 0.25
    debug = True
    # If you are using Linux or MacOS, change any \\ reference to a /.
    for input_file in glob.iglob(data_folder + '**\\*.csv', recursive=True):
        file_location = input_file.replace(data_folder, "")
        file_name = Path(input_file).name
        input_data = read_file_book(input_file)
        df = detect_outliers_book(server_url, method, sensitivity_score, max_fraction_anomalies, debug, input_data)
        output_file = results_folder + "book\\" + file_location.replace(file_name, "book_" + file_name)
        write_file_book(df, output_file)
        print('Completed file ' + file_name)

# Helper functions for processing Azure anomaly detection.
def read_file_azure(input_file):
    input_df = pd.read_csv(input_file)
    input_df['timestamp'] = pd.to_datetime(input_df['timestamp'])
    return input_df


def detect_outliers_azure(client, df):
    sensitivity = 55
    max_anomaly_ratio = 0.25
    series = []
    for index, row in df.iterrows():
        series.append(TimeSeriesPoint(timestamp = dt.datetime.strftime(row[0], '%Y-%m-%dT%H:%M:%SZ'), value = row[1]))
    request = DetectRequest(series=series, custom_interval=5, sensitivity=sensitivity, max_anomaly_ratio=max_anomaly_ratio)
    # Alternatively, we can run this against Azure's change point detection mechanism.
    # return client.detect_change_point(request)
    return client.detect_entire_series(request)


def write_file_azure(df, output_file):
    output_filepath = Path(output_file)
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)


def process_azure(data_folder, results_folder):
    # NOTE:  you must have these two environment variables set before running the code!
    ANOMALY_DETECTOR_KEY = os.environ["ANOMALY_DETECTOR_KEY"]
    ANOMALY_DETECTOR_ENDPOINT = os.environ["ANOMALY_DETECTOR_ENDPOINT"]
    client = AnomalyDetectorClient(AzureKeyCredential(ANOMALY_DETECTOR_KEY), ANOMALY_DETECTOR_ENDPOINT)
    # If you are using Linux or MacOS, change any \\ reference to a /.
    for input_file in glob.iglob(data_folder + '**\\*.csv', recursive=True):
        file_location = input_file.replace(data_folder, "")
        file_name = Path(input_file).name
        df = read_file_azure(input_file)
        try:
            response = detect_outliers_azure(client, df)
            df['is_anomaly'] = [v for i,v in enumerate(response.is_anomaly)]
            df['anomaly_score'] = 1.0 * df['is_anomaly']
            df['label'] = 1 * df['is_anomaly']
            df = df.drop('is_anomaly', axis=1)
            output_file = results_folder + "azure\\" + file_location.replace(file_name, "azure_" + file_name)
            write_file_azure(df, output_file)
            print('Completed file ' + file_name)
        except Exception as e:
            print('Skipping this file because Cognitive Services failed to return a result. {}'.format(file_location), 'Exception: {}'.format(e))
            


# The entry point to our code.
def main():
    # Change these to where you have cloned the NAB repo.
    # https://github.com/numenta/NAB/
    # If you are using Linux or MacOS, change any \\ reference to a /.
    data_folder = "D:\\SourceCode\\NAB\\data\\"
    results_folder = "D:\\SourceCode\\NAB\\results\\"
    # NOTE:  running this will probably require the S0 tier of Anomaly Detection.
    # As a result, it will likely cost you money.  You're welcome to run it if you'd
    # like but there enough data points to process across all of the files that 
    # you will probably run up a non-trivial bill.
    # Uncomment the following line if you want to run the Cognitive Services process.
    # process_azure(data_folder, results_folder)
    process_book(data_folder, results_folder)

main()
