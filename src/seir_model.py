# src/seir_model.py
import numpy as np
import pandas as pd

def run_seir(population, initial_infected, beta, sigma, gamma, days=90):
    """
    SEIR Compartmental Model
    S = Susceptible, E = Exposed, I = Infected, R = Recovered
    
    beta  = transmission rate (how fast it spreads)
    sigma = incubation rate (1/incubation_period, ~1/5 for COVID)
    gamma = recovery rate (1/infectious_period, ~1/10 for COVID)
    """
    S, E, I, R = population - initial_infected, 0, initial_infected, 0
    N = population

    results = []
    for day in range(days):
        new_exposed   = beta * S * I / N
        new_infected  = sigma * E
        new_recovered = gamma * I

        S -= new_exposed
        E += new_exposed  - new_infected
        I += new_infected - new_recovered
        R += new_recovered

        results.append({
            'day': day,
            'Susceptible': max(S, 0),
            'Exposed':     max(E, 0),
            'Infected':    max(I, 0),
            'Recovered':   max(R, 0),
            'R0':          beta / gamma  # basic reproduction number
        })

    return pd.DataFrame(results)


def estimate_beta_from_data(country_df):
    """
    Estimate beta from real case growth rate.
    Uses the early exponential growth phase.
    gamma=0.1 (10-day recovery), sigma=0.2 (5-day incubation) are COVID defaults.
    """
    gamma = 0.1
    recent = country_df['cases_7day_avg'].tail(14)
    growth = recent.pct_change().mean()
    # beta = growth_rate + gamma (derived from SEIR equations)
    beta = max(growth + gamma, gamma * 1.1)
    return round(beta, 4)