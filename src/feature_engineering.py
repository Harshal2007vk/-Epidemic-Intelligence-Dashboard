import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.sort_values(['country', 'date']).copy()

    df['new_cases'] = (
        df.groupby('country')['confirmed']
        .diff()
        .fillna(0)
        .clip(lower=0)
    )

    df['cases_7day_avg'] = (
        df.groupby('country')['new_cases']
        .transform(lambda x: x.rolling(7, min_periods=1).mean())
    )

    df['growth_rate'] = (
        df.groupby('country')['cases_7day_avg']
        .transform(lambda x: x.pct_change(periods=14).fillna(0))
    )

    df['vax_coverage'] = (
        df['total_vaccinations_per_hundred']
        .fillna(0)
        .clip(0, 100)
    )

    median_tests = df['total_tests_per_thousand'].median()
    df['test_rate'] = df['total_tests_per_thousand'].fillna(median_tests)

    return df