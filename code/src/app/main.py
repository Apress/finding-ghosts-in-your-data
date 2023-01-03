# Finding Ghosts in Your Data
from typing import Optional, List
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import json
import datetime
from app.models import univariate, multivariate, single_timeseries, multi_timeseries

app = FastAPI()

@app.get("/")
def doc():
    return {
        "message": "Welcome to the anomaly detector service, based on the book Finding Ghosts in Your Data!",
        "documentation": "If you want to see the OpenAPI specification, navigate to the /redoc/ path on this server."
    }

# Univariate statistical anomaly detection
# For more information on this, review chapters 6-8
class Univariate_Statistical_Input(BaseModel):
    key: str
    value: float

    
@app.post("/detect/univariate")
def post_univariate(
    input_data: List[Univariate_Statistical_Input],
    sensitivity_score: float = 50,
    max_fraction_anomalies: float = 1.0,
    debug: bool = False
):
    df = pd.DataFrame(i.__dict__ for i in input_data)

    (df, weights, details) = univariate.detect_univariate_statistical(df, sensitivity_score, max_fraction_anomalies)
    
    # If debug = False, include only key, value, is_anomaly, and anomaly_score.  Remove other values
    results = { "anomalies": json.loads(df.to_json(orient='records')) }
    
    if (debug):
        results.update({ "debug_weights": weights })
        results.update({ "debug_details": details })
    return results
    
    
# Multivariate anomaly detection with clustering and COPOD
# For more information on this, review chapters 9-12
class Multivariate_Input(BaseModel):
    key: str
    vals: list = []
    
@app.post("/detect/multivariate")
def post_multivariate(
    input_data: List[Multivariate_Input],
    sensitivity_score: float = 50,
    max_fraction_anomalies: float = 1.0,
    n_neighbors: int = 10,
    debug: bool = False
):
    df = pd.DataFrame(i.__dict__ for i in input_data)
    
    (df, weights, details) = multivariate.detect_multivariate_statistical(df, sensitivity_score, max_fraction_anomalies, n_neighbors)
    
    results = { "anomalies": json.loads(df.to_json(orient='records')) }
    
    if (debug):
        results.update({ "debug_weights": weights })
        results.update({ "debug_details": details })
    return results
    

# Time series anomaly detection
# For more information on this, review chapters 13-14
class Single_TimeSeries_Input(BaseModel):
    key: str
    dt: datetime.datetime
    value: float
    
@app.post("/detect/timeseries/single")
def post_time_series_single(
    input_data: List[Single_TimeSeries_Input],
    sensitivity_score: float = 50,
    max_fraction_anomalies: float = 1.0,
    debug: bool = False
):
    df = pd.DataFrame(i.__dict__ for i in input_data)
    
    (df, weights, details) = single_timeseries.detect_single_timeseries(df, sensitivity_score, max_fraction_anomalies)
    
    results = { "anomalies": json.loads(df.to_json(orient='records', date_format='iso')) }
    
    if (debug):
        results.update({ "debug_weights": weights })
        results.update({ "debug_details": details })
    return results
    

# Multiple time series anomaly detection
# For more information on this, review chapters 15-17
class Multi_TimeSeries_Input(BaseModel):
    key: str
    series_key: str
    dt: datetime.datetime
    value: float
    
@app.post("/detect/timeseries/multiple")
def post_time_series_multiple(
    input_data: List[Multi_TimeSeries_Input],
    sensitivity_score: float = 50,
    max_fraction_anomalies: float = 1.0,
    debug: bool = False
):
    df = pd.DataFrame(i.__dict__ for i in input_data)

    (df, weights, details) = multi_timeseries.detect_multi_timeseries(df, sensitivity_score, max_fraction_anomalies)
    
    results = { "anomalies": json.loads(df.to_json(orient='records', date_format='iso')) }
    
    if (debug):
        results.update({ "debug_weights": weights })
        results.update({ "debug_details": details })
    return results
