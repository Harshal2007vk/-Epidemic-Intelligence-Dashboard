# 🦠 Epidemic Intelligence Dashboard
### CodeCure AI Hackathon — SPIRIT 2026, IIT (BHU) Varanasi | Track C: Epidemic Spread Prediction

> An AI-powered epidemic intelligence system that predicts COVID-19 outbreak risk, models disease transmission, and provides actionable public health insights — built on real-world epidemiological data from Johns Hopkins CSSE and Our World in Data.

---

## 🎯 Problem it solves

Infectious disease outbreaks spread faster than public health systems can respond. This system provides:
- **Early warning** — detects rising case trends before they become outbreaks
- **Hotspot identification** — ranks 200+ countries by outbreak risk in real time
- **Transmission modeling** — simulates how disease spreads through a population using the SEIR compartmental model
- **30-day forecasts** — predicts future case counts using Facebook Prophet time-series model

---

## 🖥️ Live dashboard features

| Feature | Description |
|---|---|
| 🗺️ Global risk map | Color-coded choropleth — High / Medium / Low risk by country |
| 📈 Prophet forecast | 30-day case count prediction with confidence intervals |
| 🧬 SEIR model | Interactive transmission simulation (adjust β, γ, σ in real time) |
| 📊 Risk scoring | Composite score: growth rate + vaccination coverage + testing rate |
| 💬 NL query | Ask "high risk low vaccination" → instant filtered results |
| ⏱️ Doubling time | Biological indicator of how fast an outbreak is accelerating |

---

## 🧪 Epidemiological methodology

### Risk score formula
```
Risk Score = (0.50 × growth_rate_norm) + (0.30 × vaccine_gap) + (0.20 × testing_gap)
```
- **Growth rate (50%)** — 14-day % change in 7-day average cases. Strongest real-time signal.
- **Vaccine gap (30%)** — `1 - (vaccinations_per_hundred / 100)`. Unvaccinated population = higher susceptibility.
- **Testing gap (20%)** — Low testing → undercounting → hidden burden. Proxy for surveillance quality.

### SEIR transmission model
Classical compartmental model used in epidemiology:
- **S** (Susceptible) → **E** (Exposed) → **I** (Infected) → **R** (Recovered)
- R₀ = β/γ: if R₀ > 1, epidemic grows; if R₀ < 1, it fades
- Default COVID parameters: σ = 0.2 (5-day incubation), γ = 0.1 (10-day recovery)

### Prophet forecasting
- Meta's time-series model handles COVID's irregular seasonality and trend breaks
- Trained on 7-day rolling average (removes weekly reporting noise)
- Outputs: predicted cases + 95% confidence interval for next 30 days

---

## 📦 Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3.10+ |
| Dashboard | Streamlit ≥ 1.33 |
| Forecasting | Prophet (Meta) |
| Visualization | Plotly Express + Graph Objects |
| Data processing | Pandas, NumPy |
| Transmission model | Custom SEIR (NumPy) |

---

## 📊 Datasets used

| Dataset | Source | Usage |
|---|---|---|
| JHU COVID-19 Time Series | Johns Hopkins CSSE | Daily cases, deaths — primary forecasting data |
| Our World in Data COVID-19 | OWID | Vaccination, testing rates — risk score enrichment |

---

## 🚀 Installation & setup

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/epidemic-dashboard.git
cd epidemic-dashboard

# 2. Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

**First run:** Downloads datasets and pre-computes forecasts (~5–10 min, runs once)  
**Second run:** Loads from local cache — starts in under 3 seconds

---

## 📁 Project structure

```
epidemic_dashboard/
├── app.py                      # Main Streamlit dashboard
├── requirements.txt
├── README.md
├── data/
│   └── merged_cache.parquet    # Auto-generated local cache
└── src/
    ├── __init__.py
    ├── data_loader.py          # JHU + OWID data pipeline
    ├── feature_engineering.py  # Rolling averages, growth rates
    ├── risk_scorer.py          # Composite risk scoring
    ├── model.py                # Prophet forecasting
    └── seir_model.py           # SEIR transmission model
```

---

## 🔬 Biological insights

- **India** shows Medium risk (51.7/100) with +145% 14-day growth — suggests a rising wave requiring surveillance
- **65 countries** currently classified as High risk, predominantly in Africa and parts of Asia where vaccine coverage remains below 40%
- **R₀ analysis**: With default COVID parameters (β=0.3, γ=0.1), R₀ = 3.0 — consistent with published Delta/Omicron estimates
- Low testing rates in several High-risk countries suggest true case burden is significantly underestimated

---

## 👥 Team

Built for CodeCure AI Hackathon — SPIRIT 2026, IIT (BHU) Varanasi
