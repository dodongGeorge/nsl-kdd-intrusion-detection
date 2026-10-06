import numpy as np, pandas as pd, joblib, streamlit as st

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️", layout="wide")
CAT = ["protocol_type", "service", "flag"]
ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]
FEATURES = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
 "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
 "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
 "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
 "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
 "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate"]

@st.cache_resource
def load():
    b = joblib.load("nsl_kdd_random_forest.joblib")
    return b["model"], b["selected_columns"]
model, selected = load()

def engineer(d):
    d = d.copy()
    d["total_bytes"] = d.src_bytes + d.dst_bytes
    d["bytes_ratio"] = d.src_bytes / (d.dst_bytes + 1)
    d["log_src_bytes"] = np.log1p(d.src_bytes); d["log_dst_bytes"] = np.log1p(d.dst_bytes); d["log_duration"] = np.log1p(d.duration)
    d["failed_login_flag"] = (d.num_failed_logins > 0).astype(int)
    d["srv_count_ratio"] = d.srv_count / (d["count"] + 1)
    d["error_rate_sum"] = d.serror_rate + d.rerror_rate
    d["dst_host_error_sum"] = d.dst_host_serror_rate + d.dst_host_rerror_rate
    return d

def predict(df):
    df = df[FEATURES].copy()
    for c in CAT: df[c] = df[c].astype(str).str.strip().str.lower()
    d = engineer(df)
    X = pd.concat([d.drop(columns=CAT).reset_index(drop=True), pd.get_dummies(d[CAT]).reset_index(drop=True)], axis=1)
    X = X.reindex(columns=selected, fill_value=0).astype(float)
    p = model.predict_proba(X)
    out = pd.DataFrame(p, columns=model.classes_)[ORDER]
    out.insert(0, "Predicted class", model.classes_[p.argmax(axis=1)])
    out.insert(1, "Confidence", p.max(axis=1))
    return out

st.title("🛡️ Network Intrusion Detection")
st.caption("Random Forest classifier trained on NSL-KDD | Group 14: Lhoreneil I. Jose and Julian Carlo Cabaobao | University of Mindanao")
t1, t2, t3 = st.tabs(["Try a sample connection", "Upload connections (CSV)", "Model performance"])

with t1:
    sample = pd.read_csv("sample_connections.csv")
    st.write("Pick a real connection record from the NSL-KDD test file and see how the model classifies it.")
    i = st.selectbox("Sample connection", sample.index,
        format_func=lambda k: f"#{k}: {sample.protocol_type[k]} / {sample.service[k]} / {sample.flag[k]} (src_bytes={sample.src_bytes[k]})")
    row = sample.iloc[[i]]
    res = predict(row).iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted class", res["Predicted class"]); c2.metric("Confidence", f"{res['Confidence']*100:.1f}%"); c3.metric("Actual class", row.actual_class.iloc[0])
    if res["Predicted class"] == row.actual_class.iloc[0]: st.success("Correct prediction")
    else: st.warning("Incorrect prediction (R2L and U2R attacks are often missed; see Model performance)")
    st.bar_chart(res[ORDER].astype(float))
    with st.expander("Show the 41 feature values"): st.dataframe(row[FEATURES].T.astype(str))

with t2:
    st.write("Upload a CSV with the 41 NSL-KDD feature columns (with a header row using the standard feature names).")
    st.download_button("Download a sample CSV", pd.read_csv("sample_connections.csv").to_csv(index=False), "sample_connections.csv")
    up = st.file_uploader("CSV file", type="csv")
    if up:
        df = pd.read_csv(up)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing: st.error(f"Missing columns: {missing}")
        else:
            out = pd.concat([df.reset_index(drop=True), predict(df)], axis=1)
            st.dataframe(out[["Predicted class", "Confidence"] + [c for c in df.columns if c not in ("actual_class",)]].head(500))
            st.bar_chart(out["Predicted class"].value_counts())

with t3:
    a, b, c, d = st.columns(4)
    a.metric("Accuracy", "73.88%"); b.metric("Macro F1", "50.44%"); c.metric("Normal flagged as attack", "2.73%"); d.metric("Attacks detected", "57.92%")
    st.table(pd.DataFrame({"Precision": ["63.63%","96.06%","85.60%","80.00%","70.00%"], "Recall": ["97.27%","76.49%","61.63%","0.14%","10.45%"],
        "F1-score": ["76.93%","85.16%","71.66%","0.28%","18.18%"]}, index=ORDER))
    x, y = st.columns(2)
    x.image("figures/cm.png", caption="Confusion matrix (KDDTest+)"); y.image("figures/imp.png", caption="Top 15 feature importances")
    st.info("The model is strong on Normal and DoS traffic but misses most R2L and U2R attacks (rare in training; 17 attack types appear only in the test set).")
