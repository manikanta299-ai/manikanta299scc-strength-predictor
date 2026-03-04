import streamlit as st
import numpy as np
import joblib

# ------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------

st.set_page_config(
    page_title="SCC Strength Predictor",
    page_icon="🧱",
    layout="wide"
)

st.title("AI-Based SCC Compressive Strength Predictor")
st.write("Metaheuristic Optimized Extra Trees Models")

# ------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------

imputation = st.selectbox(
    "Select Imputation Method",
    ["KNN", "KNN_PCA", "MICE"]
)

# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

try:
    if imputation == "KNN":
        model = joblib.load("ET_DE_KNN.pkl")
    elif imputation == "KNN_PCA":
        model = joblib.load("ET_DE_KNN_PCA.pkl")
    else:
        model = joblib.load("ET_DE_MICE.pkl")
except:
    st.error("Model file not found. Check repository.")
    st.stop()

# ------------------------------------------------
# SCM TYPE
# ------------------------------------------------

scm = st.selectbox(
    "SCM Type",
    ["RHA","GGBS","MK","FFA","SCBA"]
)

scm_map = {
    "RHA":0,
    "GGBS":1,
    "MK":2,
    "FFA":3,
    "SCBA":4
}

scm_code = scm_map[scm]

st.markdown("---")

# ------------------------------------------------
# INPUT PARAMETERS
# ------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    st.subheader("Mix Design")

    cement = st.number_input("Cementitious Content", value=450.0)
    replacement = st.number_input("Replacement Percentage", value=20.0)
    wb = st.number_input("Water Binder Ratio", value=0.40)
    sp = st.number_input("Superplasticizer %", value=1.5)

    st.subheader("Chemical Properties")

    sio2 = st.number_input("SiO2", value=60.0)
    al2o3 = st.number_input("Al2O3", value=20.0)
    fe2o3 = st.number_input("Fe2O3", value=3.0)
    cao = st.number_input("CaO", value=6.0)
    mgo = st.number_input("MgO", value=2.0)
    loi = st.number_input("LOI", value=3.0)

with col2:

    st.subheader("Material Properties")

    sg = st.number_input("Specific Gravity", value=2.3)

    st.subheader("Fresh Concrete")

    slump = st.number_input("Slump Flow", value=650.0)
    t500 = st.number_input("T500 Time", value=4.0)
    vfunnel = st.number_input("V Funnel Time", value=10.0)
    lbox = st.number_input("L Box Ratio", value=0.85)

    st.subheader("Hardened Properties")

    split = st.number_input("Split Tensile Strength", value=3.5)
    rcpt = st.number_input("RCPT", value=1500.0)

st.markdown("---")

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if st.button("Predict Compressive Strength"):

    features = np.array([[
        scm_code,
        cement,
        replacement,
        wb,
        sp,
        sio2,
        al2o3,
        fe2o3,
        cao,
        mgo,
        loi,
        sg,
        slump,
        t500,
        vfunnel,
        lbox,
        split,
        rcpt
    ]])

    prediction = model.predict(features)[0]

    st.success(f"Predicted Compressive Strength: {prediction:.2f} MPa")
