import pandas as pd
from prophet import Prophet

def forecast_country(df, country, periods=30):
    country_df = (
        df[df['country'] == country][['date', 'cases_7day_avg']]
        .copy()
        .dropna()
        .sort_values('date')
    )
    country_df.columns = ['ds', 'y']

    if len(country_df) < 60:
        return None

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(country_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


def precompute_all_forecasts(df, periods=30):
    counts = df.groupby('country').size()
    eligible = counts[counts >= 60].index.tolist()

    forecasts = {}
    for country in eligible:
        try:
            fc = forecast_country(df, country, periods=periods)
            if fc is not None:
                forecasts[country] = fc
        except Exception:
            pass

    return forecasts