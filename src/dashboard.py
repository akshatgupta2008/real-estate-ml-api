"""
Real Estate ML Pipeline: Streamlit Visual Testing Dashboard
------------------------------------------------------------
Dedicated Python GUI dashboard for testing property valuations with rich visuals.
Run via: streamlit run src/dashboard.py
"""

import sys
from pathlib import Path

# Sanitize sys.path
sys.path = [p for p in sys.path if 'google-cloud-sdk' not in p]
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from src.data_processing import preprocess_input_dict, FEATURE_COLUMNS
from src.model import load_or_train_model

st.set_page_config(
    page_title="Real Estate ML Valuation Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .metric-val {
        font-size: 2.1rem;
        font-weight: 800;
        color: #0F172A;
    }
    .metric-lbl {
        font-size: 0.9rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_cached_model():
    return load_or_train_model()

model = get_cached_model()

# Header Section
st.markdown('<div class="main-header">🏡 Ames Real Estate Valuation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Interactive Monotonic XGBoost Automated Valuation Model (AVM)</div>', unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.header("🛠️ Property Specification Inputs")

st.sidebar.subheader("📐 Area & Living Space")
gr_liv = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=3999, value=2100, step=50)
total_bsmt = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=1100, step=50)
lot_area = st.sidebar.number_input("Lot Size (sq ft)", min_value=1000, max_value=50000, value=10500, step=500)

st.sidebar.subheader("⭐ Quality & Structural Ratings")
overall_qual = st.sidebar.slider("Overall Build Quality (1 = Poor, 10 = Excellent)", min_value=1, max_value=10, value=8)
overall_cond = st.sidebar.slider("Overall Property Condition (1-10)", min_value=1, max_value=10, value=6)
exter_qual = st.sidebar.selectbox("Exterior Quality", ["Ex (Excellent)", "Gd (Good)", "TA (Typical)", "Fa (Fair)"], index=1)
kitchen_qual = st.sidebar.selectbox("Kitchen Quality", ["Ex (Excellent)", "Gd (Good)", "TA (Typical)", "Fa (Fair)"], index=1)

st.sidebar.subheader("🚪 Rooms & Amenities")
garage_cars = st.sidebar.slider("Garage Vehicle Capacity (cars)", min_value=0, max_value=4, value=2)
full_bath = st.sidebar.slider("Full Bathrooms", min_value=1, max_value=4, value=2)
half_bath = st.sidebar.slider("Half Bathrooms", min_value=0, max_value=2, value=1)
bedrooms = st.sidebar.slider("Bedrooms Above Grade", min_value=1, max_value=6, value=3)
fireplaces = st.sidebar.slider("Fireplaces", min_value=0, max_value=3, value=1)

st.sidebar.subheader("📅 Construction Era")
year_built = st.sidebar.number_input("Year Built", min_value=1870, max_value=2026, value=2008, step=1)
year_remod = st.sidebar.number_input("Year Remodeled / Addition", min_value=1870, max_value=2026, value=2010, step=1)

# Build Input Dictionary
input_payload = {
    'Gr Liv Area': gr_liv,
    'Overall Qual': overall_qual,
    'Overall_Cond': overall_cond,
    'Total_Bsmt_SF': total_bsmt,
    'First_Flr_SF': gr_liv * 0.6,
    'Second_Flr_SF': gr_liv * 0.4,
    'Garage_Area': garage_cars * 220.0,
    'Garage Cars': garage_cars,
    'Full Bath': full_bath,
    'Half_Bath': half_bath,
    'Bsmt_Full_Bath': 1 if total_bsmt > 0 else 0,
    'Bedroom AbvGr': bedrooms,
    'TotRms_AbvGrd': bedrooms + full_bath + 2,
    'Fireplaces': fireplaces,
    'Lot_Area': lot_area,
    'Wood_Deck_SF': 120.0,
    'Open_Porch_SF': 45.0,
    'Year Built': year_built,
    'Year_Remod_Add': year_remod,
    'Exter_Qual': exter_qual.split()[0],
    'Kitchen_Qual': kitchen_qual.split()[0]
}

# Inference
df_processed = preprocess_input_dict(input_payload)
pred_val = float(model.predict(df_processed)[0])
price_per_sqft = pred_val / max(gr_liv, 1.0)
total_sf = gr_liv + total_bsmt

# Main KPIs
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Estimated Property Value</div>
        <div class="metric-val" style="color: #059669;">${pred_val:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Price / Living Sq Ft</div>
        <div class="metric-val" style="color: #2563EB;">${price_per_sqft:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-lbl">Total Living Space</div>
        <div class="metric-val" style="color: #7C3AED;">{total_sf:,.0f} <span style="font-size:1.1rem;font-weight:500;">sq ft</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎯 Market Valuation Gauge")
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = pred_val,
        number = {'prefix': "$", 'valueformat': ",.0f"},
        title = {'text': "Valuation Position (Ames Market Spectrum)"},
        gauge = {
            'axis': {'range': [50000, 600000], 'tickformat': "$,.0f"},
            'bar': {'color': "#059669"},
            'steps': [
                {'range': [50000, 140000], 'color': "#FEF3C7"},
                {'range': [140000, 250000], 'color': "#D1FAE5"},
                {'range': [250000, 450000], 'color': "#DBEAFE"},
                {'range': [450000, 600000], 'color': "#F3E8FF"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': pred_val
            }
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

with col_right:
    st.subheader("📊 Top Model Drivers (XGBoost Gain Weight)")
    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': FEATURE_COLUMNS,
        'Importance (%)': importances * 100
    }).sort_values('Importance (%)', ascending=False).head(7)
    
    fig_bar = px.bar(
        df_imp,
        x='Importance (%)',
        y='Feature',
        orientation='h',
        color='Importance (%)',
        color_continuous_scale='Greens',
        text_auto='.1f'
    )
    fig_bar.update_layout(
        height=350,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")
st.subheader("⚖️ Model Paradigm Cross-Validation Comparison")
col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Monotonic XGBoost (Production)", "$25,403 RMSE", f"R² = 89.32%")
col_m2.metric("Random Forest Ensemble", "$26,397 RMSE", f"R² = 88.58%")
col_m3.metric("Linear Regression Baseline", "$33,535 RMSE", f"R² = 81.65%")
