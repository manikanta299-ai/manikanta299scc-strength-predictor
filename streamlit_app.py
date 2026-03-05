import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------

# PAGE SETTINGS

# ------------------------------------------------

st.set_page_config(page_title="SCC Strength Predictor", page_icon="🧱", layout="wide")

st.title("AI-Based SCC Compressive Strength Predictor")
st.caption("Metaheuristic Optimized Extra Trees Model")

# ------------------------------------------------

# MODEL SELECTION

# ------------------------------------------------

imputation = st.selectbox(
"Select Imputation Method",
["KNN", "KNN_PCA", "MICE"]
)

# ------------------------------------------------

# MODEL FILE MAP

# ------------------------------------------------

model_files = {
"KNN": "models/ET_DE_KNN.pkl",
"KNN_PCA": "models/ET_DE_KNN_PCA.pkl",
"MICE": "models/ET_GA_MICE.pkl"
}

model = joblib.load(model_files[imputation])

st.divider()

# ------------------------------------------------

# MIX DESIGN INPUT

# ------------------------------------------------

st.subheader("Mix Design")

cement = st.number_input("Cementitious Content (kg/m³)", 450.0)
replacement = st.number_input("Replacement Percentage (%)", 20.0)
wb = st.number_input("Water Binder Ratio", 0.40)
sp = st.number_input("Superplasticizer Percentage (%)", 1.5)

st.divider()

# ------------------------------------------------

# CHEMICAL PROPERTIES

# ------------------------------------------------

st.subheader("Chemical Properties")

sio2 = st.number_input("SiO2 (%)", 60.0)
al2o3 = st.number_input("Al2O3 (%)", 20.0)
fe2o3 = st.number_input("Fe2O3 (%)", 3.0)
cao = st.number_input("CaO (%)", 6.0)
mgo = st.number_input("MgO (%)", 2.0)
loi = st.number_input("LOI (%)", 3.0)
sg = st.number_input("Specific Gravity", 2.3)

st.divider()

# ------------------------------------------------

# FRESH PROPERTIES

# ------------------------------------------------

st.subheader("Fresh Concrete Properties")

slump = st.number_input("Slump Flow (mm)", 650.0)
t500 = st.number_input("T500 Time (sec)", 4.0)
vfunnel = st.number_input("V Funnel Time (sec)", 10.0)
lbox = st.number_input("L Box Ratio", 0.85)

st.divider()

# ------------------------------------------------

# HARDENED PROPERTIES

# ------------------------------------------------

st.subheader("Hardened Properties")

split = st.number_input("Split Tensile Strength (MPa)", 3.5)
rcpt = st.number_input("RCPT (Coulombs)", 1500.0)

st.divider()

# ------------------------------------------------

# PREDICTION

# ------------------------------------------------

if st.button("Predict Compressive Strength"):

```
input_data = pd.DataFrame([[
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
]], columns=[
    "Cementitious_Content",
    "Replacement_Percentage",
    "Water_Binder_Ratio",
    "Superplasticizer_Percentage",
    "SiO2",
    "Al2O3",
    "Fe2O3",
    "CaO",
    "MgO",
    "LOI",
    "Specific_Gravity",
    "Slump_Flow",
    "T500_Time",
    "V_Funnel_Time",
    "L_Box_Ratio",
    "Split_Tensile_Strength",
    "RCPT"
])

prediction = model.predict(input_data)[0]

st.success(f"Predicted Compressive Strength = {prediction:.2f} MPa")

st.progress(0.90)

report = pd.DataFrame({
    "Parameter": [
        "Cementitious Content",
        "Replacement %",
        "Water Binder Ratio",
        "Superplasticizer %",
        "Predicted Strength"
    ],
    "Value": [
        cement,
        replacement,
        wb,
        sp,
        prediction
    ]
})

csv = report.to_csv(index=False).encode("utf-8")

st.download_button(
    "Download Prediction Report",
    csv,
    "SCC_prediction_report.csv",
    "text/csv"
)
```
