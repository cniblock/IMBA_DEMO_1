# STATS19 Intelligence Platform

UK road casualty intelligence app using [STATS19](https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) open data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data

STATS19 data is included in the repo under `datasets/` (UK Gov open data). The app works out of the box after cloning.

## Pages

- Executive Overview — metrics, charts, operational alerts
- GeoRisk Map — map view by severity
- Risk Factors — hazard profiles, trunk roads
- Vehicle Intelligence — make/model, manoeuvre analysis
- Casualty Intelligence — priority queue, pedestrian analysis
- Data Quality & Refresh Status
