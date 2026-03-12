# STATS19 Intelligence Platform

UK road casualty intelligence app using [STATS19](https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data) open data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data setup

Place STATS19 CSVs in `datasets/`:
- `dft-road-casualty-statistics-collision-last-5-years.csv`
- `dft-road-casualty-statistics-vehicle-last-5-years.csv`
- `dft-road-casualty-statistics-casualty-last-5-years.csv`
- (Optional) provisional 2025 versions and `Local_Authority_Districts_(April_2025)_Names_and_Codes_in_the_UK_v2.csv`

Download from [data.gov.uk](https://data.gov.uk/dataset/cb7ae6f0-4be6-4935-9277-47e5ce24a11f/road-safety-data).

## Pages

- Executive Overview — metrics, charts, operational alerts
- GeoRisk Map — map view by severity
- Risk Factors — hazard profiles, trunk roads
- Vehicle Intelligence — make/model, manoeuvre analysis
- Casualty Intelligence — priority queue, pedestrian analysis
- Data Quality & Refresh Status
