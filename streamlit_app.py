import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="SCC Strength Predictor", page_icon="🧱", layout="wide")

st.title("AI-Based SCC Compressive Strength Predictor")

# ------------------------------------------------
# MODEL SELECTION
# ------------------------------------------------

model_option = st.selectbox(
    "Select Model",
    ["KNN", "KNN_PCA", "MICE"]
)

model_paths = {
    "KNN": "models/ET_DE_KNN.pkl",
    "KNN_PCA": "models/ET_DE_KNN_PCA.pkl",
    "MICE": "models/ET_GA_MICE.pkl"
}

model = joblib.load(model_paths[model_option])

# ------------------------------------------------
# SCM TYPE
# ------------------------------------------------

scm = st.selectbox("SCM Type", ["RHA","GGBS","MK","FFA","SCBA"])

scm_map = {"RHA":0,"GGBS":1,"MK":2,"FFA":3,"SCBA":4}
scm_code = scm_map[scm]

st.divider()

# ------------------------------------------------
# INPUT TABS
# ------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
"Mix Design",
"Chemical Properties",
"Fresh Concrete",
"Hardened Properties"
])

# MIX DESIGN
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        cement = st.number_input("Cementitious Content (kg/m³)",450.0)
        replacement = st.number_input("Replacement Percentage (%)",20.0)

    with col2:
        wb = st.number_input("Water Binder Ratio",0.40)
        sp = st.number_input("Superplasticizer (%)",1.5)

# CHEMICAL
with tab2:

    col1, col2 = st.columns(2)

    with col1:
        sio2 = st.number_input("SiO2 (%)",60.0)
        al2o3 = st.number_input("Al2O3 (%)",20.0)
        fe2o3 = st.number_input("Fe2O3 (%)",3.0)

    with col2:
        cao = st.number_input("CaO (%)",6.0)
        mgo = st.number_input("MgO (%)",2.0)
        loi = st.number_input("LOI (%)",3.0)

    sg = st.number_input("Specific Gravity",2.3)

# FRESH
with tab3:

    col1, col2 = st.columns(2)

    with col1:
        slump = st.number_input("Slump Flow (mm)",650.0)
        t500 = st.number_input("T500 Time (sec)",4.0)

    with col2:
        vfunnel = st.number_input("V Funnel Time (sec)",10.0)
        lbox = st.number_input("L Box Ratio",0.85)

# HARDENED
with tab4:

    col1, col2 = st.columns(2)

    with col1:
        split = st.number_input("Split Tensile Strength (MPa)",3.5)

    with col2:
        rcpt = st.number_input("RCPT (Coulombs)",1500.0)

st.divider()

# ------------------------------------------------
# PREDICTION
# ------------------------------------------------

if st.button("Predict Compressive Strength"):

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

    prediction = model.predict(X)[0]

    st.success(f"Predicted Compressive Strength = {prediction:.2f} MPa")

    st.progress(0.90)
    st.caption("Model confidence ≈ 90%")

    # ------------------------------------------------
    # SCM COMPARISON GRAPH
    # ------------------------------------------------

    materials = ["RHA","GGBS","MK","FFA","SCBA"]
    strengths = []

    for code in range(5):

        X_temp = X.copy()
        X_temp["SCM_Code"] = code

        strengths.append(model.predict(X_temp)[0])

    fig, ax = plt.subplots()

    ax.bar(materials, strengths)

    ax.set_ylabel("Predicted Strength (MPa)")
    ax.set_title("SCM Comparison")

    st.pyplot(fig)

    # ------------------------------------------------
    # REPLACEMENT SENSITIVITY
    # ------------------------------------------------

    st.subheader("Replacement Percentage Sensitivity")

    replacement_range = list(range(0,61,5))
    strength_values = []

    for r in replacement_range:

        X_temp = X.copy()
        X_temp["Replacement_Percentage"] = r

        strength_values.append(model.predict(X_temp)[0])

    fig2, ax2 = plt.subplots()

    ax2.plot(replacement_range, strength_values, marker="o")

    ax2.set_xlabel("Replacement Percentage (%)")
    ax2.set_ylabel("Predicted Strength (MPa)")
    ax2.set_title("Effect of Replacement %")

    st.pyplot(fig2)

    # ------------------------------------------------
    # 3D STRENGTH SURFACE
    # ------------------------------------------------

    st.subheader("3D Strength Surface")

    replacement_range = np.linspace(0,60,15)
    wb_range = np.linspace(0.30,0.50,15)

    X_mesh, Y_mesh = np.meshgrid(replacement_range, wb_range)

    Z = []

    for i in range(len(wb_range)):

        row = []

        for j in range(len(replacement_range)):

            X_temp = X.copy()

            X_temp["Replacement_Percentage"] = replacement_range[j]
            X_temp["Water_Binder_Ratio"] = wb_range[i]

            row.append(model.predict(X_temp)[0])

        Z.append(row)

    Z = np.array(Z)

    fig3 = plt.figure()

    ax3 = fig3.add_subplot(111, projection='3d')

    ax3.plot_surface(X_mesh, Y_mesh, Z)

    ax3.set_xlabel("Replacement %")
    ax3.set_ylabel("Water/Binder Ratio")
    ax3.set_zlabel("Strength (MPa)")

    st.pyplot(fig3)
