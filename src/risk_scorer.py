import pandas as pd
import numpy as np

def compute_risk_score(df):
    latest = df.groupby('country').last().reset_index()

    gr = latest['growth_rate'].clip(-1, 2)
    gr_norm = (gr + 1) / 3

    vax_norm = 1 - (latest['vax_coverage'] / 100)

    test_norm = 1 - (latest['test_rate'].clip(0, 500) / 500)

    latest['risk_score'] = (
        0.50 * gr_norm +
        0.30 * vax_norm +
        0.20 * test_norm
    ) * 100

    def categorize(score):
        if score >= 60:
            return 'High'
        elif score >= 30:
            return 'Medium'
        return 'Low'

    latest['risk_category'] = latest['risk_score'].apply(categorize)

    return latest[[
        'country', 'risk_score', 'risk_category',
        'growth_rate', 'vax_coverage',
        'cases_7day_avg', 'test_rate'
    ]]