import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.data_loader import load_and_merge
from src.feature_engineering import engineer_features
from src.risk_scorer import compute_risk_score
from src.model import precompute_all_forecasts
from src.seir_model import run_seir

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Epidemic Intelligence Dashboard",
    page_icon="🦠",
    layout="wide"
)

# ── Cached data functions ─────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_data():
    raw = load_and_merge()
    return engineer_features(raw)

@st.cache_data(ttl=3600)
def get_risk(_df):
    return compute_risk_score(_df)

@st.cache_data(ttl=3600, show_spinner=False)
def get_all_forecasts(_df):
    return precompute_all_forecasts(_df, periods=30)

@st.cache_data(show_spinner=False)
def build_risk_map(_risk_df):
    color_map = {'High': '#E63946', 'Medium': '#F4A261', 'Low': '#2A9D8F'}
    fig = px.choropleth(
        _risk_df,
        locations='country',
        locationmode='country names',
        color='risk_category',
        color_discrete_map=color_map,
        hover_name='country',
        hover_data={
            'risk_score': ':.1f',
            'growth_rate': ':.1%',
            'vax_coverage': ':.1f',
            'cases_7day_avg': ':.0f',
            'risk_category': False
        },
        category_orders={'risk_category': ['High', 'Medium', 'Low']}
    )
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='natural earth'),
        height=600,
        autosize=True,
        margin=dict(l=0, r=0, t=30, b=0),
        legend_title_text='Risk level'
    )
    return fig

# FIX 1: build_forecast_chart does not cache DataFrames directly to avoid unhashable object errors
def build_forecast_chart(country, _forecast_df, hist_start, hist_end, forecast_days):
    # Rebuild hist slice from the forecast df cutoff
    future_only = _forecast_df[_forecast_df['ds'] > pd.Timestamp(hist_end)].head(forecast_days)

    fig = go.Figure()

    # We pass hist data separately — re-fetch it here since we can't cache the df
    # The caller passes hist_start/hist_end as cache keys; actual hist data is re-fetched
    fig.add_trace(go.Scatter(
        x=_forecast_df[_forecast_df['ds'] <= pd.Timestamp(hist_end)]['ds'],
        y=_forecast_df[_forecast_df['ds'] <= pd.Timestamp(hist_end)]['yhat'],
        name='Actual (modeled)',
        line=dict(color='#264653', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pd.concat([future_only['ds'], future_only['ds'][::-1]]),
        y=pd.concat([future_only['yhat_upper'], future_only['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(230,57,70,0.12)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence interval',
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=future_only['ds'], y=future_only['yhat'],
        name='Forecast',
        line=dict(color='#E63946', dash='dash', width=2)
    ))

    fig.update_layout(
        title=f"{country} — {forecast_days}-day forecast",
        xaxis_title='Date',
        yaxis_title='Daily new cases (7-day avg)',
        height=420,
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig


@st.cache_data
def compute_doubling_time(_df, country):
    cd = _df[_df['country'] == country][['date', 'cases_7day_avg']].dropna()
    cd = cd[cd['cases_7day_avg'] > 10].tail(30)
    if len(cd) < 7:
        return None

    growth = cd['cases_7day_avg'].pct_change().mean()
    if growth <= 0:
        return None

    return round(0.693 / growth, 1)

# ── Startup: load everything once, clean rerun after ─────────────────────────
# FIX 2 & 3: Don't store dataframes in session_state; use cache_data as the
# single source of truth. Use session_state only as a "ready" flag.
# Add st.rerun() after first load so the page renders cleanly without
# the status widget and partial KPIs showing simultaneously.

if 'ready' not in st.session_state:
    with st.status("🔬 Initializing epidemic intelligence system...", expanded=True) as status:
        st.write("📥 Downloading JHU COVID-19 time-series data...")
        get_data()
        st.write("⚠️ Computing country risk scores...")
        get_risk(get_data())
        st.write("📈 Pre-computing Prophet forecasts for all countries (runs once)...")
        get_all_forecasts(get_data())
        st.session_state['ready'] = True
        status.update(label="✅ System ready!", state="complete", expanded=False)
    st.rerun()  # FIX 3: clean rerun so page renders without status widget overlay

# All three calls below hit the cache instantly (no recomputation)
df            = get_data()
risk_df       = get_risk(df)
all_forecasts = get_all_forecasts(df)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🦠 Epidemic Intelligence Dashboard")
st.caption("COVID-19 outbreak prediction · hotspot detection · risk mapping — powered by JHU + OWID data")

# ── KPI metrics ───────────────────────────────────────────────────────────────

high_risk  = (risk_df['risk_category'] == 'High').sum()
med_risk   = (risk_df['risk_category'] == 'Medium').sum()
low_risk   = (risk_df['risk_category'] == 'Low').sum()
total      = len(risk_df)
med_growth = risk_df['growth_rate'].median() * 100

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🔴 High risk",      high_risk)
col2.metric("🟠 Medium risk",    med_risk)
col3.metric("🟢 Low risk",       low_risk)
col4.metric("🌍 Countries",      total)
col5.metric("📊 Median growth",  f"{med_growth:+.1f}%")

st.divider()

# ── Risk map ──────────────────────────────────────────────────────────────────

st.subheader("🗺️ Global Outbreak Risk Map")
st.plotly_chart(build_risk_map(risk_df), use_container_width=True)

# ── Top 10 table ──────────────────────────────────────────────────────────────
# FIX 6: Removed @st.cache_data from top10 — it's cheap, and caching with
# _underscore skips hashing so stale data would persist after TTL refresh.

with st.expander("🔴 Top 10 highest-risk countries", expanded=True):
    top10 = (
        risk_df
        .sort_values('risk_score', ascending=False)
        .head(10)
        [['country', 'risk_category', 'risk_score', 'growth_rate', 'vax_coverage', 'cases_7day_avg']]
    )
    st.dataframe(
        top10.style
        .format({
            'risk_score':     '{:.1f}',
            'growth_rate':    '{:.1%}',
            'vax_coverage':   '{:.1f}',
            'cases_7day_avg': '{:.0f}'
        })
        .background_gradient(subset=['risk_score'], cmap='RdYlGn_r'),
        use_container_width=True
    )

st.divider()

# ── Forecast section ──────────────────────────────────────────────────────────
# FIX 4: Pass df, all_forecasts, risk_df as explicit arguments so the fragment
# never closes over a stale reference after a rerun.

@st.fragment
def forecast_section(df, all_forecasts, risk_df):
    st.subheader("📈 Country-level outbreak forecast")

    sorted_countries = sorted(df['country'].unique())
    default_idx = sorted_countries.index('India') if 'India' in sorted_countries else 0

    col_sel, col_days = st.columns([3, 1])
    selected_country = col_sel.selectbox("Select country", sorted_countries, index=default_idx)
    forecast_days    = col_days.slider("Forecast days", 7, 60, 30)

    forecast = all_forecasts.get(selected_country)

    if forecast is not None:
        country_hist = (
            df[df['country'] == selected_country][['date', 'cases_7day_avg']]
            .dropna()
        )
        hist_start = str(country_hist['date'].min().date())
        hist_end   = str(country_hist['date'].max().date())

        col_chart, col_summary = st.columns([3, 1])

        with col_chart:
            # FIX 1: Pass scalar hist_start/hist_end as cache keys, not the DataFrame
            fig = build_forecast_chart(
                selected_country, forecast, hist_start, hist_end, forecast_days
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_summary:
            row = risk_df[risk_df['country'] == selected_country]
            if not row.empty:
                r = row.iloc[0]
                risk_color = {'High': '🔴', 'Medium': '🟠', 'Low': '🟢'}.get(r['risk_category'], '⚪')
                st.markdown(f"### {selected_country}")
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| Risk level | {risk_color} **{r['risk_category']}** |
| Risk score | {r['risk_score']:.1f} / 100 |
| 14-day growth | {r['growth_rate']:+.1%} |
| Vaccine coverage | {r['vax_coverage']:.1f}% |
| 7-day avg cases | {r['cases_7day_avg']:.0f} |
                """)

                dt = compute_doubling_time(df, selected_country)
                if dt is not None:
                    st.metric("Doubling time", f"{dt} days", delta="faster = more dangerous" if dt < 14 else None)
    else:
        st.warning(f"Not enough historical data to forecast **{selected_country}** (needs 60+ days).")

forecast_section(df, all_forecasts, risk_df)  # FIX 4: explicit args

st.divider()

# In app.py — add a new SEIR section

@st.fragment
def seir_section(df, risk_df):
    st.subheader("🧬 SEIR Transmission Model")
    st.caption("Simulates disease spread through Susceptible → Exposed → Infected → Recovered compartments")

    col1, col2 = st.columns([2, 1])

    with col2:
        country = st.selectbox("Country", sorted(df['country'].unique()), key='seir_country')
        days    = st.slider("Simulation days", 30, 180, 90, key='seir_days')

        row = risk_df[risk_df['country'] == country]
        pop = 1_000_000  # default if no population data
        if not row.empty and 'population' in row.columns:
            pop = int(row.iloc[0].get('population', 1_000_000) or 1_000_000)

        beta  = st.slider("Transmission rate (β)", 0.05, 0.80, 0.30, step=0.01)
        gamma = st.slider("Recovery rate (γ)", 0.05, 0.30, 0.10, step=0.01)
        sigma = st.slider("Incubation rate (σ)", 0.10, 0.50, 0.20, step=0.01)
        st.caption(f"R₀ = β/γ = **{beta/gamma:.2f}** ({'epidemic will grow' if beta/gamma > 1 else 'epidemic will fade'})")

    with col1:
        seir_df = run_seir(pop, initial_infected=100, beta=beta, sigma=sigma, gamma=gamma, days=days)
        fig = go.Figure()
        colors = {
            'Susceptible': ('#264653', 'rgba(38,70,83,0.08)'),
            'Exposed':     ('#E9C46A', 'rgba(233,196,106,0.08)'),
            'Infected':    ('#E63946', 'rgba(230,57,70,0.08)'),
            'Recovered':   ('#2A9D8F', 'rgba(42,157,143,0.08)'),
        }

        for col_name, (line_color, fill_color) in colors.items():
            fig.add_trace(go.Scatter(
                x=seir_df['day'],
                y=seir_df[col_name],
                name=col_name,
                line=dict(color=line_color, width=2),
                fill='tozeroy',
                fillcolor=fill_color
            ))

        fig.update_layout(
            title=f"SEIR simulation — {country} (pop: {pop:,})",
            xaxis_title='Days',
            yaxis_title='People',
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

seir_section(df, risk_df)

# ── Natural language query ────────────────────────────────────────────────────
# FIX 5: Wrap in st.form so query only runs on submit, not on every keystroke.

st.subheader("💬 Query the data")
st.caption("Try: 'high risk low vaccination' · 'rising cases' · 'top 5 worst' · 'declining'")

with st.form("query_form"):
    query     = st.text_input("Ask a question about the data", placeholder="e.g. high risk and low vaccination")
    submitted = st.form_submit_button("🔍 Search")

if submitted and query:
    q      = query.lower()
    result = risk_df.copy()

    if 'high risk' in q or 'high-risk' in q:
        result = result[result['risk_category'] == 'High']
    elif 'medium risk' in q:
        result = result[result['risk_category'] == 'Medium']
    elif 'low risk' in q:
        result = result[result['risk_category'] == 'Low']

    if 'low vax' in q or 'low vaccination' in q or 'unvaccinated' in q:
        result = result[result['vax_coverage'] < 40]

    if 'rising' in q or 'growing' in q or 'increasing' in q:
        result = result[result['growth_rate'] > 0.10]

    if 'declining' in q or 'falling' in q or 'improving' in q:
        result = result[result['growth_rate'] < -0.05]

    if 'top 5' in q:
        result = result.sort_values('risk_score', ascending=False).head(5)
    elif 'top 10' in q or 'worst' in q:
        result = result.sort_values('risk_score', ascending=False).head(10)

    st.markdown(f"**{len(result)} countries** match your query:")
    st.dataframe(
        result[['country', 'risk_category', 'risk_score', 'growth_rate', 'vax_coverage']]
        .sort_values('risk_score', ascending=False)
        .style.format({
            'risk_score':  '{:.1f}',
            'growth_rate': '{:.1%}',
            'vax_coverage':'{:.1f}'
        }),
        use_container_width=True
    )




# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("Data: Johns Hopkins CSSE · Our World in Data · Built for CodeCure AI Hackathon, IIT BHU")

