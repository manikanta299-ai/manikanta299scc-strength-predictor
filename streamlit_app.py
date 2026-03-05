import streamlit as st
import pandas as pd
import joblib

# PAGE SETTINGS

st.set_page_config(page_title="SCC Strength Predictor", page_icon="🧱", layout="wide")

st.title("AI-Based SCC Compressive Strength Predictor")

# MODEL SELECTION

imputation = st.selectbox("Select Imputation Method", ["KNN","KNN_PCA","MICE"])

model_files = {
"KNN":"models/ET_DE_KNN.pkl",
"KNN_PCA":"models/ET_DE_KNN_PCA.pkl",
"MICE":"models/ET_GA_MICE.pkl"
}

model = joblib.load(model_files[imputation])

st.subheader("Input Mix Parameters")

cement = st.number_input("Cementitious Content",450.0)
replacement = st.number_input("Replacement Percentage",20.0)
wb = st.number_input("Water Binder Ratio",0.40)
sp = st.number_input("Superplasticizer Percentage",1.5)

sio2 = st.number_input("SiO2",60.0)
al2o3 = st.number_input("Al2O3",20.0)
fe2o3 = st.number_input("Fe2O3",3.0)
cao = st.number_input("CaO",6.0)
mgo = st.number_input("MgO",2.0)
loi = st.number_input("LOI",3.0)

sg = st.number_input("Specific Gravity",2.3)

slump = st.number_input("Slump Flow",650.0)
t500 = st.number_input("T500 Time",4.0)
vfunnel = st.number_input("V Funnel Time",10.0)
lbox = st.number_input("L Box Ratio",0.85)

split = st.number_input("Split Tensile Strength",3.5)
rcpt = st.number_input("RCPT",1500.0)

if st.button("Predict Compressive Strength"):
data = pd.DataFrame([[cement,replacement,wb,sp,sio2,al2o3,fe2o3,cao,mgo,loi,sg,slump,t500,vfunnel,lbox,split,rcpt]],
columns=["Cementitious_Content","Replacement_Percentage","Water_Binder_Ratio","Superplasticizer_Percentage","SiO2","Al2O3","Fe2O3","CaO","MgO","LOI","Specific_Gravity","Slump_Flow","T500_Time","V_Funnel_Time","L_Box_Ratio","Split_Tensile_Strength","RCPT"])

```
pred = model.predict(data)[0]

st.success(f"Predicted Compressive Strength = {pred:.2f} MPa")
```
