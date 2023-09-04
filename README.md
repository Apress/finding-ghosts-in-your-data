# Finding Ghosts in Your Data:  The Code Base

This code base includes a **completed** form of everything created over the course of _Finding Ghosts in Your Data_.  If you wish to follow along with the book and create your own solution in line with mine, please read the next section.  If you prefer instead simply to review my code and run it directly, read the following section.

## Play Along at Home

If you wish to create your own solution, I recommend renaming the `app` folder in the `code\src` directory to something like `app_complete`.  That way, you still have a completed version of the application to refer back against.  Then, when following along with the book, you can create your own `app` folder.  Note that not all of the code will be in the book but it will be in the `app` folder here.

If you follow this practice and want to use a Docker-based solution, you should not need to modify the `Dockerfile` at all.  If you are running things locally, kick off the API process with the following command:

```python
uvicorn app.main:app --host 0.0.0.0 --port 80
```

And should you wish instead to see how the completed version of the app works, make a minor change to the call:

```python
uvicorn app_complete.main:app --host 0.0.0.0 --port 80
```

Should you wish to deploy the completed version as a Docker container, you can edit the `Dockerfile` to copy `./src/app_complete` instead of `./src/app`, though leave the destination directory alone.

If you are proficient with Docker, you may also wish to build one image based on the completed code and tag it separately:

```python
docker build -t anomalydetector_complete .
```

Then, when building your own version of the detector, tag it as mentioned in the book:

```python
docker build -t anomalydetector .
```

This will allow you to compare how your service behaves compared to the version in the book.

## Try the Completed Product

If you simply want to run the completed product, execute the following command inside the `code` directory:

```python
uvicorn app.main:app --host 0.0.0.0 --port 80
```

## Running Tests

All of the integration and unit tests we use in this book are included in the `\code\test` folder.

### Unit Tests

All of the unit tests for this book are located in `code\test\` and have names starting with `test_`.  If you wish to run the unit tests, ensure that you have the PyTest library installed--which you will if you installed all packages in `requirements.txt`--and run the following command:  `pytest test_{file to run}`.  For example:

```python
pytest test_univariate.py
```

Note that this does **not** require that you have the API running, as these are unit tests.  It does require you to be in the `\code\test` folder and that relevant code be in `\code\src\app\`.

### Postman Integration Tests

The [Postman](https://www.postman.com/) application will allow you to run all of the integration tests, which you can find in `code\test\postman_integration_tests\`.  Import each collection once you have completed the code for each relevant chapter, so for example, you should wait until chapter 7 to import `Finding Ghosts in Your Data - Chapter 07.json`.

In the Postman app, select the "Import" button and select the appropriate file to import all integration tests.  Note that there are no environment settings for these tests and they assume that you are running the API on port 80.

## Errata

This section is dedicated to any book errata.  This may include any errors at the time of publication or notes about code changes after release to deal with outdated packages or other issues.

If you have found any issues, please reach out to me at feasel@catallaxyservices.com.

### Listing 6-7

The print copy of Listing 6-7 has the `return` assignment indented in further than it should be.  This is the corrected listing.

```python
def determine_outliers(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
    sensitivity_score = (100 - sensitivity_score) / 100.0
    max_fraction_anomaly_score = np.quantile(df['anomaly_score'], 1.0 - max_fraction_anomalies)
    if max_fraction_anomaly_score > sensitivity_score and max_fraction_anomalies < 1.0:
        sensitivity_score = max_fraction_anomaly_score
    return df.assign(is_anomaly=(df['anomaly_score'] > sensitivity_score))
```

Thanks to:  Andy Huber.

### Listing 6-8

The print copy of Listing 6-8 has an incorrect output assignment for the `run_tests(df)` function.  At this point in the book, we get back a tuple of results, `df_tested` and `calculations`.

```python
def detect_univariate_statistical(
    df,
    sensitivity_score,
    max_fraction_anomalies
):
    weights = {"sds": 0.25, "iqrs": 0.35, "mads": 0.45}

    if (df[‘value’].count() < 3):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a minimum of at least three data points for anomaly detection.")
    elif (max_fraction_anomalies <= 0.0 or max_fraction_anomalies >  1.0):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid max fraction of anomalies, 0 < x <= 1.0.")
    elif (sensitivity_score <= 0 or sensitivity_score > 100 ):
        return (df.assign(is_anomaly=False, anomaly_score=0.0), weights, "Must have a valid sensitivity score, 0 < x <= 100.")
    else:
        (df_tested, calculations) = run_tests(df)
        df_scored = score_results(df_tested, weights)
        df_out = determine_outliers(df_scored, sensitivity_score, max_fraction_anomalies)
        return (df_out, weights, { "message": "Ensemble of [mean +/- 3*SD, median +/- 1.5*IQR, median +/- 3*MAD].", "calculations": calculations})
```

Thanks to:  Andy Huber.

### Listing 7-12

The print copy of Listing 7-12 is missing a line at the end of the `run_tests(df)` function.  The corrected version is as follows:

```python
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
    max_num_outliers = math.floor(b['len'] / 3)
    df['gesd'] = check_gesd(col, max_num_outliers)
    tests_run['gesd'] = 1
else:
  diagnostics["Extended tests"] = "Did not run extended tests because the dataset was not normal and could not be normalized."

diagnostics["Tests Run"] = tests_run

```

Thanks to:  Andy Huber.
