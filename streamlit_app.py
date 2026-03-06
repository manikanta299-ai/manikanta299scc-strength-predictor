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
-----------------------------
LOAD MODEL
-----------------------------

model = joblib.load("models/ET_DE_KNN.pkl")

-----------------------------
SCM TYPE
-----------------------------

scm = st.selectbox("SCM Type", ["RHA","GGBS","MK","FFA","SCBA"])

scm_map = {"RHA":0,"GGBS":1,"MK":2,"FFA":3,"SCBA":4}

scm_code = scm_map[scm]

-----------------------------
INPUTS
-----------------------------

cement = st.number_input("Cementitious Content",450.0)
replacement = st.number_input("Replacement %",20.0)
wb = st.number_input("Water Binder Ratio",0.40)
sp = st.number_input("Superplasticizer",1.5)

sio2 = st.number_input("SiO2",60.0)
al2o3 = st.number_input("Al2O3",20.0)
fe2o3 = st.number_input("Fe2O3",3.0)

cao = st.number_input("CaO",6.0)
mgo = st.number_input("MgO",2.0)
loi = st.number_input("LOI",3.0)

sg = st.number_input("Specific Gravity",2.3)

slump = st.number_input("Slump Flow",650.0)
t500 = st.number_input("T500",4.0)
vfunnel = st.number_input("V Funnel",10.0)
lbox = st.number_input("L Box",0.85)

split = st.number_input("Split Tensile",3.5)
rcpt = st.number_input("RCPT",1500.0)

-----------------------------
PREDICT
-----------------------------

if st.button("Predict"):

X = pd.DataFrame([[
    cement,replacement,wb,sp,
    sio2,al2o3,fe2o3,cao,mgo,loi,
    sg,slump,t500,vfunnel,lbox,
    split,rcpt,scm_code
]], columns=[
    "Cementitious_Content",
    "Replacement_Percentage",
    "Water_Binder_Ratio",
    "Superplasticizer_Percentage",
    "SiO2","Al2O3","Fe2O3","CaO",
    "MgO","LOI","Specific_Gravity",
    "Slump_Flow","T500_Time","V_Funnel_Time",
    "L_Box_Ratio","Split_Tensile_Strength",
    "RCPT","SCM_Code"
])

y = model.predict(X)[0]

st.success(f"Predicted Compressive Strength = {y:.2f} MPa")
