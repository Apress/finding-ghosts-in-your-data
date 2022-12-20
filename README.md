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

**NO ERRATA**