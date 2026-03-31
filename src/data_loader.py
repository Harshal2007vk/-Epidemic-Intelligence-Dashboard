import os
import time
import pandas as pd

CACHE_PATH = "data/merged_cache.parquet"

JHU_BASE = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/"
OWID_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"

def load_jhu_confirmed():
    url = JHU_BASE + "time_series_covid19_confirmed_global.csv"
    df = pd.read_csv(url)
    id_cols = ['Province/State', 'Country/Region', 'Lat', 'Long']
    df = df.melt(id_vars=id_cols, var_name='date', value_name='confirmed')
    df['date'] = pd.to_datetime(df['date'])
    df = df.groupby(['Country/Region', 'date'])['confirmed'].sum().reset_index()
    df.rename(columns={'Country/Region': 'country'}, inplace=True)
    return df

def load_owid():
    cols = [
        'location', 'date',
        'total_vaccinations_per_hundred',
        'total_tests_per_thousand',
        'population',
        'new_cases_smoothed_per_million'
    ]
    df = pd.read_csv(OWID_URL, usecols=cols)
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={'location': 'country'}, inplace=True)
    return df

def load_and_merge(force_refresh=False):
    # If cached file exists and is less than 24h old, use it
    if not force_refresh and os.path.exists(CACHE_PATH):
        age_hours = (time.time() - os.path.getmtime(CACHE_PATH)) / 3600
        if age_hours < 24:
            return pd.read_parquet(CACHE_PATH)

    # Otherwise download fresh
    jhu  = load_jhu_confirmed()
    owid = load_owid()
    merged = pd.merge(jhu, owid, on=['country', 'date'], how='left')

    # Save locally for next run
    os.makedirs("data", exist_ok=True)
    merged.to_parquet(CACHE_PATH, index=False)
    return merged