import streamlit as st
import pickle
import numpy as np
from datetime import datetime

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# LOAD MODEL
# =========================
model = pickle.load(open("model.pkl", "rb"))
r2_score = pickle.load(open("r2_score.pkl", "rb"))

# =========================
# DARK UI CSS
# =========================
st.markdown("""
<style>
.stApp {
    background-color: #050816;
    color: white;
}

[data-testid="stSidebar"] {
    background-color: #0B1120;
    border-right: 1px solid #1E293B;
}

h1, h2, h3, h4 {
    color: #00E5FF !important;
}

p, label, div {
    color: white !important;
}

.stButton > button {
    background: linear-gradient(90deg, #00E5FF, #007CF0);
    color: white;
    border-radius: 12px;
    border: none;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #00B8D4, #0057D9);
    color: white;
}

[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #1F2937;
    padding: 15px;
    border-radius: 15px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TITLE
# =========================
st.markdown(
    "<h1 style='text-align:center;'>🚗 AI Car Price Prediction Dashboard</h1>",
    unsafe_allow_html=True
)

# HERO SECTION
st.markdown("""
<div style='
padding:25px;
border-radius:15px;
background: linear-gradient(to right, #0F2027, #203A43, #2C5364);
text-align:center;
margin-bottom:20px;
'>
<h2 style='color:white;'>Machine Learning Based Used Car Valuation System</h2>
<p style='color:white;'>
Predict the estimated resale price of a car using Linear Regression.
</p>
</div>
""", unsafe_allow_html=True)

# =========================
# INPUTS
# =========================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Car Details")

    current_year = datetime.now().year

    year = st.slider("Manufacturing Year", 2000, current_year, 2018)
    km_driven = st.number_input("Kilometers Driven", 0, 500000, 30000)
    mileage = st.number_input("Mileage (km/l)", 5.0, 40.0, 18.0, step=0.1)
    engine = st.number_input("Engine Capacity (cc)", 800, 5000, 1200)

with col2:
    st.subheader("📊 Live Metrics")

    car_age = current_year - year

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Car Age", f"{car_age} years")
    with m2:
        st.metric("Mileage", f"{mileage:.1f} km/l")

    m3, m4 = st.columns(2)
    with m3:
        st.metric("KM Driven", f"{km_driven:,}")
    with m4:
        st.metric("Engine", f"{engine} cc")

# =========================
# PREDICTION
# =========================
if st.button("💰 Predict Car Price"):
    data = np.array([[year, km_driven, mileage, engine]])
    predicted_price = model.predict(data)[0]

    # Avoid negative predictions
    predicted_price = max(predicted_price, 0)

    st.markdown("---")
    st.subheader("🤖 Prediction Result")

    st.success(
        f"Estimated Car Price: ₹ {predicted_price:,.0f}"
    )

# =========================
# MODEL PERFORMANCE
# =========================
st.markdown("---")
st.subheader("📈 Model Performance")

c1, c2 = st.columns(2)

with c1:
    st.metric("Algorithm", "Linear Regression")

with c2:
    st.metric("R² Score", f"{r2_score:.4f}")

# =========================
# ABOUT
# =========================
with st.expander("ℹ️ About This Project"):
    st.write("""
    ### Technologies Used
    - Python
    - Streamlit
    - Scikit-learn
    - Pandas
    - NumPy

    ### Machine Learning Concepts
    - Regression
    - Linear Regression
    - Train-Test Split
    - Model Evaluation (R² Score)
    - Model Serialization with Pickle

    ### Objective
    To predict the resale price of a used car using machine learning.
    """)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Developed by anushka | 4th Semester Machine Learning Project")