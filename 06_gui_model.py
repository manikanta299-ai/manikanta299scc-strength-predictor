# ==============================================================================
# 06_gui_model.py
# Step 6: final model package for the Streamlit GUI (revised manuscript, Fig. 22)
#
#        python 06_gui_model.py            (about 1-2 min)
#
# Model   : untuned Extra Trees (500 trees, library defaults) + bounded KNN
#           imputation (k = 8), 15 inputs of set A, fitted on all 160 mixes.
# Interval: split-conformal 90 % and 95 % intervals from 5-fold out-of-fold
#           residuals (KNN refitted inside each fold).
# Output  : OUT_DIR\06_gui\models\gui_model.joblib  (sklearn + numpy objects only,
#           so the GUI does not need scc_common.py)
#           OUT_DIR\06_gui\app.py, requirements.txt, README.txt are written
#           next to it if those files are present in this folder.
# ==============================================================================
import os
import shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold

import scc_common as sc

GUI_FILES = ["app.py", "requirements.txt", "README.txt"]

LABELS = {
    "Cementitious_Content": ("Binder content", "kg/m³", "Mix proportions"),
    "Replacement_Percentage": ("SCM replacement", "%", "Mix proportions"),
    "Water_Binder_Ratio": ("Water–binder ratio", "–", "Mix proportions"),
    "Superplasticizer_Percentage": ("Superplasticiser dosage", "% of binder", "Mix proportions"),
    "SiO2": ("SiO₂ of SCM", "%", "SCM properties"),
    "Al2O3": ("Al₂O₃ of SCM", "%", "SCM properties"),
    "Fe2O3": ("Fe₂O₃ of SCM", "%", "SCM properties"),
    "CaO": ("CaO of SCM", "%", "SCM properties"),
    "MgO": ("MgO of SCM", "%", "SCM properties"),
    "LOI": ("Loss on ignition of SCM", "%", "SCM properties"),
    "Specific_Gravity": ("Specific gravity of SCM", "–", "SCM properties"),
    "Slump_Flow": ("Slump flow", "mm", "Fresh properties"),
    "T500_Time": ("T₅₀₀ time", "s", "Fresh properties"),
    "V_Funnel_Time": ("V-funnel time", "s", "Fresh properties"),
    "L_Box_Ratio": ("L-box ratio", "–", "Fresh properties"),
}
REQUIRED = ["Cementitious_Content", "Replacement_Percentage", "Water_Binder_Ratio"]


def main():
    cfg = sc.get_config("Step 6: GUI model package")
    out = os.path.join(cfg["OUT_DIR"], "06_gui")
    os.makedirs(os.path.join(out, "models"), exist_ok=True)

    df = sc.load_modelling_data(cfg, verbose=False)
    feats = sc.FEATURE_SETS["A"]
    T, M = cfg["TARGET"], cfg["SCM_COL"]
    X = df[feats].to_numpy(float)
    y = df[T].to_numpy(float)

    # ---- 5-fold out-of-fold residuals for the conformal interval ----------------
    oof = np.full(len(y), np.nan)
    for tr, va in KFold(5, shuffle=True, random_state=cfg["MODEL_SEED"]).split(X):
        imp = sc.make_imputer("KNN", cfg).fit(X[tr])
        et = sc.make_et(sc.ET_BASE_PARAMS, cfg).fit(imp.transform(X[tr]), y[tr])
        oof[va] = et.predict(imp.transform(X[va]))
    res = np.abs(y - oof)
    q90, q95 = sc.conformal_q(res, 0.90), sc.conformal_q(res, 0.95)
    m = sc.metrics(y, oof)
    print(f"5-fold OOF: R2 {m['R2']:.3f}, RMSE {m['RMSE']:.2f} MPa | 90% half-width {q90:.2f} MPa, "
          f"95% half-width {q95:.2f} MPa")

    # ---- final model on all mixes ---------------------------------------------------
    imp = sc.make_imputer("KNN", cfg).fit(X)
    et = sc.make_et(sc.ET_BASE_PARAMS, cfg).fit(imp.transform(X), y)

    # typical SCM properties (median of the database) to pre-fill the GUI
    scm_typical = {}
    for scm, g in df.groupby(M):
        scm_typical[scm] = {f: (None if np.isnan(v) else float(round(v, 3)))
                            for f, v in g[feats].median(numeric_only=True).items()}

    pkg = {
        "features": feats,
        "labels": LABELS,
        "required": REQUIRED,
        "imputer": {"mu": imp.mu_, "sd": imp.sd_, "lo": imp.lo_, "hi": imp.hi_, "knn": imp.knn_},
        "model": et,
        "q": {"90": float(q90), "95": float(q95)},
        "ranges": {f: (float(np.nanmin(df[f])), float(np.nanmax(df[f]))) for f in feats},
        "scm_typical": scm_typical,
        "scm_counts": df[M].value_counts().to_dict(),
        "n_mixes": int(len(df)), "n_studies": int(df[cfg["STUDY_COL"]].nunique()),
        "target_range": (float(y.min()), float(y.max())),
        "oof": {"R2": float(m["R2"]), "RMSE": float(m["RMSE"])},
    }
    path = os.path.join(out, "models", "gui_model.joblib")
    joblib.dump(pkg, path)
    print("Saved:", path)

    # copy the GUI files that sit next to this script
    here = os.path.dirname(os.path.abspath(__file__))
    for f in GUI_FILES:
        if os.path.exists(os.path.join(here, f)):
            shutil.copy(os.path.join(here, f), os.path.join(out, f))
            print("Copied:", f)

    # ---- self-check: GUI-style prediction of the first mix ----------------------------
    x = X[:1].copy()
    z = pkg["imputer"]["knn"].transform((x - imp.mu_) / imp.sd_) * imp.sd_ + imp.mu_
    z = np.clip(z, imp.lo_, imp.hi_)
    z[~np.isnan(x)] = x[~np.isnan(x)]
    p1, p2 = et.predict(z)[0], et.predict(imp.transform(x))[0]
    print(f"Self-check (GUI imputation = pipeline imputation): {p1:.3f} vs {p2:.3f} MPa")
    print(f'\nStart the GUI:  cd "{out}"  then  streamlit run app.py')


if __name__ == "__main__":
    main()
