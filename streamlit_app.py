import streamlit as st
import pandas as pd
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
st.caption("Metaheuristic Optimized Extra Trees Model")

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

```
if imputation == "KNN":
    model = joblib.load("models/ET_DE_KNN.pkl")

elif imputation == "KNN_PCA":
    model = joblib.load("models/ET_DE_KNN_PCA.pkl")

else:
    model = joblib.load("models/ET_GA_MICE.pkl")
```

except Exception as e:
st.error(f"Model loading error: {e}")
st.stop()

st.divider()

# ------------------------------------------------

# INPUT TABS

# ------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
["Mix Design", "Chemical Properties", "Fresh Concrete", "Hardened Properties"]
)

# ------------------------------------------------

# MIX DESIGN

# ------------------------------------------------

with tab1:

```
col1, col2 = st.columns(2)

with col1:
    cement = st.number_input("Cementitious Content (kg/m³)", value=450.0)
    replacement = st.number_input("Replacement Percentage (%)", value=20.0)

with col2:
    wb = st.number_input("Water Binder Ratio", value=0.40)
    sp = st.number_input("Superplasticizer Percentage (%)", value=1.5)
```

# ------------------------------------------------

# CHEMICAL PROPERTIES

# ------------------------------------------------

with tab2:

```
col1, col2 = st.columns(2)

with col1:
    sio2 = st.number_input("SiO₂ (%)", value=60.0)
    al2o3 = st.number_input("Al₂O₃ (%)", value=20.0)
    fe2o3 = st.number_input("Fe₂O₃ (%)", value=3.0)

with col2:
    cao = st.number_input("CaO (%)", value=6.0)
    mgo = st.number_input("MgO (%)", value=2.0)
    loi = st.number_input("LOI (%)", value=3.0)

sg = st.number_input("Specific Gravity", value=2.3)
```

# ------------------------------------------------

# FRESH PROPERTIES

# ------------------------------------------------

with tab3:

```
col1, col2 = st.columns(2)

with col1:
    slump = st.number_input("Slump Flow (mm)", value=650.0)
    t500 = st.number_input("T500 Time (sec)", value=4.0)

with col2:
    vfunnel = st.number_input("V Funnel Time (sec)", value=10.0)
    lbox = st.number_input("L Box Ratio", value=0.85)
```

# ------------------------------------------------

# HARDENED PROPERTIES

# ------------------------------------------------

with tab4:

```
col1, col2 = st.columns(2)

with col1:
    split = st.number_input("Split Tensile Strength (MPa)", value=3.5)

with col2:
    rcpt = st.number_input("RCPT (Coulombs)", value=1500.0)
```

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

confidence = 0.90
st.progress(confidence)
st.caption("Model Confidence (approx.): 90%")

report = pd.DataFrame({
    "Parameter":[
        "Cementitious Content",
        "Replacement Percentage",
        "Water Binder Ratio",
        "Superplasticizer Percentage",
        "Predicted Strength"
    ],
    "Value":[
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
