# Network Intrusion Detection Using Machine Learning Models
Group 14 (Lhoreneil I. Jose, Julian Carlo Cabaobao) | CST9L | The University of Mindanao

Random Forest classifier that labels NSL-KDD network connections as Normal, DoS, Probe, R2L, or U2R.

| Result (KDDTest+) | Value |
|---|---|
| Accuracy | 73.88% |
| Macro F1 | 50.44% |
| Normal flagged as attack | 2.73% |
| Attacks detected | 57.92% |

## Files
- `Group14_NSL_KDD_Random_Forest.ipynb` - full pipeline (cleaning, transformation, engineering, selection, training, evaluation, demo)
- `app.py` - Streamlit application (`streamlit run app.py`)
- `nsl_kdd_random_forest.joblib` - trained model (scikit-learn 1.8.0)
- `sample_connections.csv` - 150 real test records for the app
- `data/` - NSL-KDD `KDDTrain+.txt` and `KDDTest+.txt` (source: https://www.kaggle.com/datasets/hassan06/nslkdd)
- `figures/` - confusion matrix, feature importance, conceptual framework

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```
Limitations: offline benchmark data only; R2L and U2R attacks are mostly missed (see paper).
