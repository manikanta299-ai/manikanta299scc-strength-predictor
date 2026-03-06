import streamlit as st
import pandas as pd
import joblib

# Page settings

st.set_page_config(page_title="SCC Strength Predictor", page_icon="🧱", layout="wide")

st.title("AI-Based SCC Compressive Strength Predictor")
st.caption("Metaheuristic Optimized Extra Trees Model")

# ------------------------------

# MODEL SELECTION

# ------------------------------

imputation = st.selectbox(
"Select Imputation Method",
["KNN", "KNN_PCA", "MICE"]
)

model_map = {
"KNN": "models/ET_DE_KNN.pkl",
"KNN_PCA": "models/ET_DE_KNN_PCA.pkl",
"MICE": "models/ET_GA_MICE.pkl"
}

model = joblib.load(model_map[imputation])

# ------------------------------

# SCM TYPE

# ------------------------------

scm = st.selectbox(
"SCM Type",
["RHA", "GGBS", "MK", "FFA", "SCBA"]
)

scm_map = {
"RHA": 0,
"GGBS": 1,
"MK": 2,
"FFA": 3,
"SCBA": 4
}

scm_code = scm_map[scm]

st.divider()

# ------------------------------

# INPUT PARAMETERS

# ------------------------------

cement = st.number_input("Cementitious Content (kg/m³)", value=450.0)
replacement = st.number_input("Replacement Percentage (%)", value=20.0)
wb = st.number_input("Water Binder Ratio", value=0.40)
sp = st.number_input("Superplasticizer (%)", value=1.5)

sio2 = st.number_input("SiO2 (%)", value=60.0)
al2o3 = st.number_input("Al2O3 (%)", value=20.0)
fe2o3 = st.number_input("Fe2O3 (%)", value=3.0)

cao = st.number_input("CaO (%)", value=6.0)
mgo = st.number_input("MgO (%)", value=2.0)
loi = st.number_input("LOI (%)", value=3.0)

sg = st.number_input("Specific Gravity", value=2.3)

slump = st.number_input("Slump Flow (mm)", value=650.0)
t500 = st.number_input("T500 Time (sec)", value=4.0)
vfunnel = st.number_input("V Funnel Time (sec)", value=10.0)
lbox = st.number_input("L Box Ratio", value=0.85)

split = st.number_input("Split Tensile Strength (MPa)", value=3.5)
rcpt = st.number_input("RCPT (Coulombs)", value=1500.0)

st.divider()

# ------------------------------

# PREDICTION

# ------------------------------

predict = st.button("Predict Compressive Strength")

if predict:

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
    rcpt,
    scm_code
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
    "RCPT",
    "SCM_Code"
])

prediction = model.predict(input_data)[0]

st.success(f"Predicted Compressive Strength = {prediction:.2f} MPa")
```
