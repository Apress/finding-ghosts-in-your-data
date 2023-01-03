import streamlit as st
import requests
import pandas as pd
import json
import plotly.express as px
import plotly.graph_objects as go
import ast

st.set_page_config(layout="wide")

@st.cache
def process(server_url, method, sensitivity_score, max_fraction_anomalies, debug, input_data_set):
    full_server_url = f"{server_url}/{method}?sensitivity_score={sensitivity_score}&max_fraction_anomalies={max_fraction_anomalies}&debug={debug}"
    r = requests.post(
        full_server_url,
        data=input_data_set,
        headers={"Content-Type": "application/json"}
    )
    return r

# Used as a helper method for creating lists from JSON.
@st.cache
def convert_univariate_list_to_json(univariate_str):
    # Remove brackets if they exist.
    univariate_str = univariate_str.replace('[', '').replace(']', '')
    univariate_list = univariate_str.split(',')
    df = pd.DataFrame(univariate_list, columns={"value"})
    df["key"] = ""
    return df.to_json(orient="records")

@st.cache
def convert_multivariate_list_to_json(multivariate_str):
    mv_ast = ast.literal_eval(multivariate_str)
    return json.dumps([{"key": k, "vals": v} for idx,[k,v] in enumerate(mv_ast)])

@st.cache
def convert_single_time_series_list_to_json(time_series_str):
    mv_ast = ast.literal_eval(time_series_str)
    return json.dumps([{"key": k, "dt":dt, "value": v} for idx,[k,dt,v] in enumerate(mv_ast)])

@st.cache
def convert_multi_time_series_list_to_json(time_series_str):
    mv_ast = ast.literal_eval(time_series_str)
    return json.dumps([{"key": k, "series_key":sk, "dt":dt, "value": v} for idx,[k,sk,dt,v] in enumerate(mv_ast)])

def main():
    st.write(
    """
    # Finding Ghosts in Your Data

    This is an outlier detection application based on the book Finding Ghosts in Your Data (Apress, 2022).  The purpose of this site is to provide a simple interface for interacting with the outlier detection API we build over the course of the book.

    ## Instructions
    First, select the method you wish to use for outlier detection.  Then, enter the dataset you wish to process.  This dataset should be posted as a JSON array with the appropriate attributes.  The specific attributes you need to enter will depend on the method you chose above.

    If you switch between methods, you will see a sample dataset corresponding to the expected structure of the data.  Follow that pattern for your data.
    """
    )

    server_url = "http://localhost/detect"
    method = st.selectbox(label="Choose the method you wish to use.", options = ("univariate", "multivariate", "timeseries/single", "timeseries/multiple"))
    sensitivity_score = st.slider(label = "Choose a sensitivity score.", min_value=1, max_value=100, value=50)
    max_fraction_anomalies = st.slider(label = "Choose a max fraction of anomalies.", min_value=0.01, max_value=1.0, value=0.1)
    debug = st.checkbox(label="Run in Debug mode?")
    convert_to_json = st.checkbox(label="Convert data in list to JSON format?  If you check this box, enter data as a comma-separated list of values.")
    
    if method == "univariate":
        starting_data_set = """[
        {"key": "1","value": 1},
        {"key": "2", "value": 2},
        {"key": "3", "value": 3},
        {"key": "4", "value": 4},
        {"key": "5", "value": 5},
        {"key": "6", "value": 6},
        {"key": "8", "value": 95}
    ]"""
    elif method == "multivariate":
        starting_data_set = """[
        {"key":1,"vals":[22.46, 17.69, 8.04, 14.11]},
        {"key":2,"vals":[22.56, 17.69, 8.04, 14.11]},
        {"key":3,"vals":[22.66, 17.69, 8.04, 14.11]},
        {"key":4,"vals":[22.76, 17.69, 8.04, 14.11]},
        {"key":5,"vals":[22.896, 17.69, 8.04, 14.11]},
        {"key":6,"vals":[22.9, 22.69, 8.04, 14.11]},
        {"key":7,"vals":[22.06, 17.69, 8.04, 14.11]},
        {"key":8,"vals":[22.16, 17.69, 9.15, 14.11]},
        {"key":9,"vals":[22.26, 17.69, 8.04, 14.11]},
        {"key":10,"vals":[22.36, 178.69, 8.04, 14.11]},
        {"key":11,"vals":[22.46, 17.69, 8.04, 14.11]},
        {"key":12,"vals":[22.56, 17.69, 8.04, 14.11]},
        {"key":13,"vals":[22.66, 17.69, 8.04, 14.11]},
        {"key":14,"vals":[22.76, 17.69, 8.04, 14.11]},
        {"key":15,"vals":[22.86, 17.69, 8.04, 14.11]},
        {"key":16,"vals":[22.76, 17.69, 8.04, 14.11]},
        {"key":17,"vals":[22.66, 17.69, 8.04, 14.11]},
        {"key":18,"vals":[22.56, 17.69, 8.04, 14.11]},
        {"key":19,"vals":[22.46, 17.69, 8.04, 14.11]},
        {"key":20,"vals":[22.36, 17.69, 8.04, 14.11]},
        {"key":21,"vals":[22.26, 17.69, 8.04, 14.11]}
    ]"""
    elif method == "timeseries/single":
        starting_data_set = """[
        {"key": "188", "dt": "2021-01-06T17:00:00Z", "value": 18.360001},
        {"key": "189", "dt": "2021-01-07T17:00:00Z", "value": 18.08},
        {"key": "190", "dt": "2021-01-08T17:00:00Z", "value": 17.690001},
        {"key": "191", "dt": "2021-01-11T17:00:00Z", "value": 19.940001},
        {"key": "192", "dt": "2021-01-12T17:00:00Z", "value": 19.950001},
        {"key": "193", "dt": "2021-01-13T17:00:00Z", "value": 31.4},
        {"key": "194", "dt": "2021-01-14T17:00:00Z", "value": 39.91},
        {"key": "195", "dt": "2021-01-15T17:00:00Z", "value": 35.5},
        {"key": "196", "dt": "2021-01-19T17:00:00Z", "value": 39.360001},
        {"key": "197", "dt": "2021-01-20T17:00:00Z", "value": 39.119999},
        {"key": "198", "dt": "2021-01-21T17:00:00Z", "value": 43.029999},
        {"key": "199", "dt": "2021-01-22T17:00:00Z", "value": 65.010002},
        {"key": "200", "dt": "2021-01-25T17:00:00Z", "value": 76.790001},
        {"key": "201", "dt": "2021-01-26T17:00:00Z", "value": 147.979996},
        {"key": "202", "dt": "2021-01-27T17:00:00Z", "value": 347.51001},
        {"key": "203", "dt": "2021-01-28T17:00:00Z", "value": 193.600006},
        {"key": "204", "dt": "2021-01-29T17:00:00Z", "value": 325},
        {"key": "205", "dt": "2021-02-01T17:00:00Z", "value": 225},
        {"key": "206", "dt": "2021-02-02T17:00:00Z", "value": 90}
    ]"""
    elif method == "timeseries/multiple":
        starting_data_set = """[
        {"key": "k1",  "series_key": "s1", "dt": "2021-12-11T08:00:00Z", "value": 14.3},
        {"key": "k2",  "series_key": "s1", "dt": "2021-12-11T09:00:00Z", "value": 15.3},
        {"key": "k3",  "series_key": "s1", "dt": "2021-12-11T10:00:00Z", "value": 15.8},
        {"key": "k4",  "series_key": "s1", "dt": "2021-12-11T11:00:00Z", "value": 16.2},
        {"key": "k5",  "series_key": "s1", "dt": "2021-12-11T12:00:00Z", "value": 16.4},
        {"key": "k6",  "series_key": "s1", "dt": "2021-12-11T13:00:00Z", "value": 16.5},
        {"key": "k7",  "series_key": "s1", "dt": "2021-12-11T14:00:00Z", "value": 16.3},
        {"key": "k8",  "series_key": "s1", "dt": "2021-12-11T15:00:00Z", "value": 16.0},
        {"key": "k9",  "series_key": "s1", "dt": "2021-12-11T16:00:00Z", "value": 15.5},
        {"key": "k10", "series_key": "s1", "dt": "2021-12-11T17:00:00Z", "value": 15.1},
        {"key": "k11", "series_key": "s1", "dt": "2021-12-11T18:00:00Z", "value": 14.6},
        {"key": "k12", "series_key": "s1", "dt": "2021-12-11T19:00:00Z", "value": 14.4},
        {"key": "k13", "series_key": "s1", "dt": "2021-12-11T20:00:00Z", "value": 14.1},
        {"key": "k14", "series_key": "s1", "dt": "2021-12-11T21:00:00Z", "value": 13.9},
        {"key": "k15", "series_key": "s1", "dt": "2021-12-11T22:00:00Z", "value": 13.7},
        {"key": "k16", "series_key": "s1", "dt": "2021-12-11T23:00:00Z", "value": 190.8},
        {"key": "k17", "series_key": "s1", "dt": "2021-12-12T00:00:00Z", "value": 193.7},
        {"key": "k1a", "series_key": "s2", "dt": "2021-12-11T08:00:00Z", "value": 24.3},
        {"key": "k2a", "series_key": "s2", "dt": "2021-12-11T09:00:00Z", "value": 25.3},
        {"key": "k3a", "series_key": "s2", "dt": "2021-12-11T10:00:00Z", "value": 25.8},
        {"key": "k4a", "series_key": "s2", "dt": "2021-12-11T11:00:00Z", "value": 26.2},
        {"key": "k5a", "series_key": "s2", "dt": "2021-12-11T12:00:00Z", "value": 26.4},
        {"key": "k6a", "series_key": "s2", "dt": "2021-12-11T13:00:00Z", "value": 26.5},
        {"key": "k7a", "series_key": "s2", "dt": "2021-12-11T14:00:00Z", "value": 26.3},
        {"key": "k8a", "series_key": "s2", "dt": "2021-12-11T15:00:00Z", "value": 26.0},
        {"key": "k9a", "series_key": "s2", "dt": "2021-12-11T16:00:00Z", "value": 25.5},
        {"key": "k10a","series_key": "s2", "dt": "2021-12-11T17:00:00Z", "value": 25.1},
        {"key": "k11a","series_key": "s2", "dt": "2021-12-11T18:00:00Z", "value": 24.6},
        {"key": "k12a","series_key": "s2", "dt": "2021-12-11T19:00:00Z", "value": 24.4},
        {"key": "k13a","series_key": "s2", "dt": "2021-12-11T20:00:00Z", "value": 4.1},
        {"key": "k14a","series_key": "s2", "dt": "2021-12-11T21:00:00Z", "value": 213.9},
        {"key": "k15a","series_key": "s2", "dt": "2021-12-11T22:00:00Z", "value": 23.7},
        {"key": "k16a","series_key": "s2", "dt": "2021-12-11T23:00:00Z", "value": 17.8},
        {"key": "k17a","series_key": "s2", "dt": "2021-12-12T00:00:00Z", "value": 183.7},
        {"key": "k1b", "series_key": "s3", "dt": "2021-12-11T08:00:00Z", "value": 28.3},
        {"key": "k2b", "series_key": "s3", "dt": "2021-12-11T09:00:00Z", "value": 29.3},
        {"key": "k3b", "series_key": "s3", "dt": "2021-12-11T10:00:00Z", "value": 29.8},
        {"key": "k4b", "series_key": "s3", "dt": "2021-12-11T11:00:00Z", "value": 30.2},
        {"key": "k5b", "series_key": "s3", "dt": "2021-12-11T12:00:00Z", "value": 22.4},
        {"key": "k6b", "series_key": "s3", "dt": "2021-12-11T13:00:00Z", "value": 24.5},
        {"key": "k7b", "series_key": "s3", "dt": "2021-12-11T14:00:00Z", "value": 28.3},
        {"key": "k8b", "series_key": "s3", "dt": "2021-12-11T15:00:00Z", "value": 21.0},
        {"key": "k9b", "series_key": "s3", "dt": "2021-12-11T16:00:00Z", "value": 25.5},
        {"key": "k10b","series_key": "s3", "dt": "2021-12-11T17:00:00Z", "value": 30.1},
        {"key": "k11b","series_key": "s3", "dt": "2021-12-11T18:00:00Z", "value": 33.6},
        {"key": "k12b","series_key": "s3", "dt": "2021-12-11T19:00:00Z", "value": 32.4},
        {"key": "k13b","series_key": "s3", "dt": "2021-12-11T20:00:00Z", "value": 19.1},
        {"key": "k14b","series_key": "s3", "dt": "2021-12-11T21:00:00Z", "value": 122.9},
        {"key": "k15b","series_key": "s3", "dt": "2021-12-11T22:00:00Z", "value": 23.7},
        {"key": "k16b","series_key": "s3", "dt": "2021-12-11T23:00:00Z", "value": 215.8},
        {"key": "k17b","series_key": "s3", "dt": "2021-12-12T00:00:00Z", "value": 298.7}
    ]"""
    else:
        starting_data_set = "Select a method."
    input_data = st.text_area(label = "Data to process (in JSON format):", value=starting_data_set, height=300)

    if st.button(label="Detect!"):
        if method=="univariate" and convert_to_json:
            input_data = convert_univariate_list_to_json(input_data)
        if method=="multivariate" and convert_to_json:
            input_data = convert_multivariate_list_to_json(input_data)
        if method == "timeseries/single" and convert_to_json:
            input_data = convert_single_time_series_list_to_json(input_data)
        if method == "timeseries/multiple" and convert_to_json:
            input_data = convert_multi_time_series_list_to_json(input_data)
        resp = process(server_url, method, sensitivity_score, max_fraction_anomalies, debug, input_data)
        res = json.loads(resp.content)
        df = pd.DataFrame(res['anomalies'])

        if method=="univariate":
            st.header('Anomaly score per data point')
            colors = {True: '#481567', False: '#3CBB75'}
            g = px.scatter(df, x=df["value"], y=df["anomaly_score"], color=df["is_anomaly"], color_discrete_map=colors,
                        symbol=df["is_anomaly"], symbol_sequence=['square', 'circle'],
                        hover_data=["sds", "mads", "iqrs", "grubbs", "gesd", "dixon", "gaussian_mixture"])
            st.plotly_chart(g, use_container_width=True)


            tbl = df[['key', 'value', 'anomaly_score', 'is_anomaly', 'sds', 'mads', 'iqrs', 'grubbs', 'gesd', 'dixon', 'gaussian_mixture']]
            st.write(tbl)

            if debug:
                col11, col12 = st.columns(2)

                with col11:                
                    st.header('Debug weights')
                    st.write(res['debug_weights'])

                with col12:
                    st.header("Tests Run")
                    st.write(res['debug_details']['Test diagnostics']['Tests Run'])
                    if "Extended tests" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Extended tests'])
                    if "Gaussian mixture test" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Gaussian mixture test'])

                col21, col22 = st.columns(2)

                with col21:
                    st.header("Base Calculations")
                    st.write(res['debug_details']['Test diagnostics']['Base calculations'])

                with col22:
                    st.header("Fitted Calculations")
                    if "Fitted calculations" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Fitted calculations'])

                col31, col32 = st.columns(2)

                with col31:
                    st.header("Initial Normality Checks")
                    if "Initial normality checks" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Initial normality checks'])

                with col32:
                    st.header("Fitted Normality Checks")
                    if "Fitted Lambda" in res['debug_details']['Test diagnostics']:
                        st.write(f"Fitted Lambda = {res['debug_details']['Test diagnostics']['Fitted Lambda']}")
                    if "Fitted normality checks" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']['Fitted normality checks'])
                    if "Fitting Status" in res['debug_details']['Test diagnostics']:
                        st.write(res['debug_details']['Test diagnostics']["Fitting Status"])

                st.header("Full Debug Details")
                st.json(res['debug_details'])
        elif method=="multivariate":
            st.header('Anomaly score per data point')
            colors = {True: '#481567', False: '#3CBB75'}
            df = df.sort_values(by=['anomaly_score'], ascending=False)
            g = px.bar(df, x=df["key"], y=df["anomaly_score"], color=df["is_anomaly"], color_discrete_map=colors,
                        hover_data=["vals", "anomaly_score_cof", "anomaly_score_loci", "anomaly_score_copod"], log_y=True)
            st.plotly_chart(g, use_container_width=True)


            tbl = df[['key', 'vals', 'anomaly_score', 'is_anomaly', 'anomaly_score_cof', 'anomaly_score_loci', 'anomaly_score_copod']]
            st.write(tbl)

            if debug:
                col11, col12 = st.columns(2)

                with col11:                
                    st.header("Tests Run")
                    st.write(res['debug_details']['Tests run'])
                    st.write(res['debug_details']['Test diagnostics'])

                with col12:
                    st.header("Outlier Determinants")
                    st.write(res['debug_details']['Outlier determination'])

                st.header("Full Debug Details")
                st.json(res['debug_details'])
        elif method=="timeseries/single":
            st.header('Anomaly score per data point')
            colors = {True: '#481567', False: '#3CBB75'}
            l = px.line(df, x=df["dt"], y=df["value"], markers=False)
            l.update_traces(line=dict(color = 'rgba(50,50,50,0.2)'))
            s = px.scatter(df, x=df["dt"], y=df["value"], color=df["is_anomaly"], color_discrete_map=colors,
                        symbol=df["is_anomaly"], symbol_sequence=['square', 'circle'],
                        hover_data=["anomaly_score"])
            g = go.Figure(data=l.data + s.data)
            st.plotly_chart(g, use_container_width=True)

            tbl = df[['key', 'dt', 'value', 'anomaly_score', 'is_anomaly']]
            st.write(tbl)

            if debug:
                col11, col12 = st.columns(2)

                with col11:                
                    st.header("Test diagnostics")
                    st.write(res['debug_details']['Test diagnostics'])

                with col12:
                    st.header("Outlier determination")
                    st.write(res['debug_details']['Outlier determination'])

                st.header("Full Debug Details")
                st.json(res['debug_details'])
        elif method=="timeseries/multiple":
            st.header('Anomaly score per segment')
            df_mean = df.groupby("dt", as_index=False).mean("value")
            df_mean['series_key'] = "mean"
            ml = px.line(df_mean, x=df_mean["dt"], y=df_mean["value"], color=df_mean["series_key"], markers=False)
            ml.update_traces(line=dict(color = 'rgba(20,20,20,0.45)'))
            l = px.line(df, x=df["dt"], y=df["value"], color=df["series_key"], markers=False,
                color_discrete_sequence=px.colors.qualitative.Safe)
            # Render mode SVG is the default up to 1000 data points.  After that, the default is WebGL, which does
            # not include hover data. Because multi-series time series may
            # exceed 1000 points, we explicitly include it here so we can see the hover data.
            # Note that this may slow down graph loading for large graphs.
            s = px.scatter(df, x=df["dt"], y=df["value"], color=df["series_key"], 
                        symbol=df["is_anomaly"], symbol_sequence=['square', 'circle'],
                        hover_data=["key", "anomaly_score", "is_anomaly", "sax_score", "diffstd_score", "segment_number"], render_mode="svg",
                        color_discrete_sequence=px.colors.qualitative.Safe)
            s.update_traces(marker_size=9, showlegend=False)
            g = go.Figure(data=l.data + s.data + ml.data)
            st.plotly_chart(g, use_container_width=True)

            tbl = df[['key', 'series_key', 'dt', 'segment_number', 'value', 'sax_distance', 'diffstd_distance', 'sax_score', 'diffstd_score', 'anomaly_score', 'is_anomaly']]
            st.write(tbl)

            if debug:
                col11, col12 = st.columns(2)

                with col11:                
                    st.header("Test diagnostics")
                    st.write(res['debug_details']['Test diagnostics'])

                with col12:
                    st.header("Outlier determination")
                    st.write(res['debug_details']['Outlier scoring'])
                    st.write(res['debug_details']['Outlier determination'])

                st.header("Full Debug Details")
                st.json(res['debug_details'])


if __name__ == "__main__":
    main()