# ==============================================================================
# SCC compressive strength predictor (revised model)
# Untuned Extra Trees + bounded KNN imputation, 15 pre-hardening inputs (set A),
# 90 % / 95 % split-conformal prediction intervals.
#
# Run:  streamlit run app.py
# Needs: models/gui_model.joblib (created by 06_gui_model.py)
# ==============================================================================
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="SCC Strength Predictor", page_icon="🧱", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_package():
    return joblib.load(os.path.join(HERE, "models", "gui_model.joblib"))


pkg = load_package()
feats, labels, ranges = pkg["features"], pkg["labels"], pkg["ranges"]


def impute(x):
    """Same bounded KNN imputation as in the study (training-fitted)."""
    im = pkg["imputer"]
    miss = np.isnan(x)
    z = im["knn"].transform((x - im["mu"]) / im["sd"]) * im["sd"] + im["mu"]
    z = np.clip(z, im["lo"], im["hi"])
    z[~miss] = x[~miss]
    return z


# ----------------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------------
st.title("AI-Based SCC Compressive Strength Predictor")
st.write("28-day compressive strength of self-compacting concrete with FFA, GGBS, MK, RHA or SCBA. "
         "Extra Trees model with KNN imputation of missing inputs and 90% conformal prediction interval.")

with st.sidebar:
    st.header("About the model")
    st.markdown(
        f"- Trained on **{pkg['n_mixes']} mixes** from **{pkg['n_studies']} studies**\n"
        f"- Inputs: mix proportions, SCM properties and fresh properties (15)\n"
        f"- Missing inputs are imputed by KNN (k = 8) from the database\n"
        f"- Cross-validated accuracy: R² = {pkg['oof']['R2']:.2f}, RMSE = {pkg['oof']['RMSE']:.1f} MPa\n"
        f"- Strength range of the database: {pkg['target_range'][0]:.0f}–{pkg['target_range'][1]:.0f} MPa")
    st.info("Valid for mixes within the database ranges. For a new material source or laboratory, "
            "calibrate with a few trial mixes.")
    level = st.radio("Prediction interval", ["90%", "95%"], horizontal=True)

# ----------------------------------------------------------------------------------
# SCM selection and typical values
# ----------------------------------------------------------------------------------
SCM_PROPS = [f for f in feats if labels[f][2] == "SCM properties"]


def prefill():
    """Callback: write the typical properties of the selected SCM into the input fields."""
    if st.session_state.get("fill"):
        typ = pkg["scm_typical"].get(st.session_state["scm"], {})
        for f in SCM_PROPS:
            st.session_state[f] = typ.get(f)


st.header("SCM selection")
c1, c2 = st.columns([1, 2])
with c1:
    scm = st.selectbox("Select SCM type", ["FFA", "GGBS", "MK", "RHA", "SCBA"], key="scm", on_change=prefill)
with c2:
    st.write("")
    st.checkbox("Pre-fill SCM properties with typical values of the selected SCM from the database",
                key="fill", on_change=prefill)
st.caption("The SCM type is not a model input; each SCM is described by its oxide composition, LOI and "
           "specific gravity.")

# ----------------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------------
st.header("Input mix parameters")
st.caption("Fields marked * are required. Other fields may be left empty; they are imputed from the database.")
values = {}
groups = ["Mix proportions", "SCM properties", "Fresh properties"]
cols = st.columns(3)
for col, grp in zip(cols, groups):
    with col:
        st.subheader(grp)
        for f in [f for f in feats if labels[f][2] == grp]:
            name, unit, _ = labels[f]
            lo, hi = ranges[f]
            req = f in pkg["required"]
            if f not in st.session_state:
                st.session_state[f] = None
            values[f] = st.number_input(
                f"{name}{' *' if req else ''} ({unit})", min_value=0.0,
                format="%.3f" if hi < 10 else "%.1f", placeholder=f"{lo:g}–{hi:g}",
                help=f"Database range: {lo:g}–{hi:g} {unit}", key=f)

# ----------------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------------
if st.button("Predict strength", type="primary"):
    missing_req = [labels[f][0] for f in pkg["required"] if values[f] is None]
    if missing_req:
        st.error("Please enter: " + ", ".join(missing_req))
        st.stop()

    x = np.array([[np.nan if values[f] is None else float(values[f]) for f in feats]])
    xf = impute(x)
    pred = float(pkg["model"].predict(xf)[0])
    q = pkg["q"]["90" if level == "90%" else "95"]
    lo_pi, hi_pi = max(0.0, pred - q), pred + q

    r1, r2 = st.columns(2)
    r1.metric("Predicted 28-day compressive strength", f"{pred:.1f} MPa")
    r2.metric(f"{level} prediction interval", f"{lo_pi:.1f} – {hi_pi:.1f} MPa")

    out_of_range = [f"{labels[f][0]} = {values[f]:g} (range {ranges[f][0]:g}–{ranges[f][1]:g})"
                    for f in feats if values[f] is not None
                    and not (ranges[f][0] <= values[f] <= ranges[f][1])]
    if out_of_range:
        st.warning("Outside the database range (extrapolation, lower reliability): " + "; ".join(out_of_range))
    n_imp = int(np.isnan(x).sum())
    if n_imp:
        st.info(f"{n_imp} input(s) were left empty and imputed from the database.")

    with st.expander("Inputs used by the model"):
        st.dataframe(pd.DataFrame({
            "Input": [labels[f][0] for f in feats],
            "Unit": [labels[f][1] for f in feats],
            "Entered": ["" if values[f] is None else f"{values[f]:g}" for f in feats],
            "Used": [f"{v:.3g}" for v in xf[0]],
            "Database range": [f"{ranges[f][0]:g}–{ranges[f][1]:g}" for f in feats],
        }), hide_index=True, use_container_width=True)

st.divider()
st.caption("The interval is a split-conformal interval calibrated on cross-validated residuals; it covers about "
           "90% (or 95%) of new mixes similar to those in the database.")
