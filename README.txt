SCC Strength Predictor (revised model)

Predicts the 28-day compressive strength of self-compacting concrete containing
FFA, GGBS, MK, RHA or SCBA, with a 90% or 95% conformal prediction interval.

Model: untuned Extra Trees (500 trees) with bounded KNN imputation (k = 8),
15 inputs available before hardened-concrete testing, trained on 160 mixes
from 31 studies. Empty inputs (except binder content, replacement and W/B)
are imputed from the database.

Files
  app.py                   Streamlit application
  models/gui_model.joblib  model package created by 06_gui_model.py
  requirements.txt         Python packages

Run
  pip install -r requirements.txt
  streamlit run app.py

Validity: mixes within the database ranges. For a new material source or
laboratory, calibrate with a few trial mixes.
