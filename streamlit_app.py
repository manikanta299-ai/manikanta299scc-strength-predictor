import streamlit as st
import numpy as np
import joblib

model_knn = joblib.load("ET_DE_KNN.pkl")
model_knnpca = joblib.load("ET_DE_KNN_PCA.pkl")
model_mice = joblib.load("ET_DE_MICE.pkl")
# ----------------------------------------------------
# Title
# ----------------------------------------------------

st.title("AI-Based SCC Compressive Strength Predictor")
st.write("Metaheuristic Optimized Extra Trees Models")

# ----------------------------------------------------
# Model Selection
# ----------------------------------------------------

imputation = st.selectbox(
    "Select Imputation Method",
    ["KNN", "KNN_PCA", "MICE"]
)

# ----------------------------------------------------
# Load Correct Model
# ----------------------------------------------------

if imputation == "KNN":
    model = joblib.load("models/ET_DE_KNN.pkl")

elif imputation == "KNN_PCA":
    model = joblib.load("models/ET_DE_KNN_PCA.pkl")

else:
    model = joblib.load("models/ET_GA_MICE.pkl")

# ----------------------------------------------------
# SCM Selection
# ----------------------------------------------------

st.header("SCM Selection")

scm = st.selectbox(
    "Select SCM Type",
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

# ----------------------------------------------------
# Mix Parameters
# ----------------------------------------------------

st.header("Input Mix Parameters")

cement = st.number_input("Cementitious Content")
replacement = st.number_input("Replacement Percentage")
wb = st.number_input("Water Binder Ratio")
sp = st.number_input("Superplasticizer Percentage")

sio2 = st.number_input("SiO2")
al2o3 = st.number_input("Al2O3")
fe2o3 = st.number_input("Fe2O3")
cao = st.number_input("CaO")
mgo = st.number_input("MgO")
loi = st.number_input("LOI")

sg = st.number_input("Specific Gravity")

slump = st.number_input("Slump Flow")
t500 = st.number_input("T500 Time")
vfunnel = st.number_input("V Funnel Time")
lbox = st.number_input("L Box Ratio")

split = st.number_input("Split Tensile Strength")
rcpt = st.number_input("RCPT")

# ----------------------------------------------------
# Prediction
# ----------------------------------------------------

if st.button("Predict Strength"):

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
