"""Lab 04 - Phish or Legit? (Streamlit UI). Run:  streamlit run app.py"""
import re

import numpy as np
import pandas as pd
import streamlit as st

from src.pipeline import load_data, split, make_pipeline, predict, word_weights, contributions

st.set_page_config(page_title="Phish or Legit?", page_icon="🎣")
st.title("Phish or Legit?")
st.warning("This tool supports a human decision and does not block messages by itself. The data is synthetic.")
st.caption("TF-IDF + classifier on a SYNTHETIC dataset (fictional brands). "
           "Labels are 'phishing-style' vs 'legit', not real-world ground truth.")

EXAMPLES = {
    "Phishing - correctly detected (Northbank)":
        "Hello, We noticed activity on your Northbank account. Confirm your details. "
        "Please reply as soon as you can. https://xcen0-login.info/at5g",
    "Phishing - correctly detected (BlueFreight)":
        "ALERT: BlueFreight: your package is waiting. Confirm your details. "
        "This is your final notice. http://secure-2kcy-verify.com/j8n1ws",
    "Legit - correctly classified (order)":
        "Your order #96192 has shipped.",
    "Legit - correctly classified (reminder)":
        "Reminder: class starts tomorrow at 9. Open the official app to view it.",
    "MISSED phishing (false negative)":
        "Your Civic Savings account has a new notice. http://bit.ly/tp99jp",
    "FALSE ALARM on legit (false positive)":
        "Your invoice of $21.12 is ready. Update your payment info. https://docs.example.com/d/k5isac",
}


@st.cache_resource
def get_models():
    df = load_data()
    Xtr, Xte, ytr, yte = split(df)
    models = {
        "Logistic regression": make_pipeline("lr").fit(Xtr, ytr),
        "Logistic regression + sigmoid calibration": make_pipeline("lr_cal").fit(Xtr, ytr),
    }
    return models, Xte, yte


models, Xte, yte = get_models()
lr = models["Logistic regression"]          # used for the explanation (needs coefficients)

# ---------------- sidebar: settings ----------------
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              help="Predict PHISHING when the score is >= threshold. "
                                   "Higher = fewer false alarms but more missed phishing.")
model_name = st.sidebar.selectbox("Classifier", list(models))
model = models[model_name]

# ---------------- input ----------------
choice = st.selectbox("Example messages", ["(write my own)"] + list(EXAMPLES))
default = EXAMPLES.get(choice, "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd")
text = st.text_area("Message to classify:", value=default, height=110, key=f"txt-{choice}")

if st.button("Classify", type="primary"):
    try:
        _, score = predict(model, text)
    except ValueError as err:
        st.error(str(err))
        st.stop()

    label = "phish" if score >= threshold else "legit"
    if label == "phish":
        st.error(f"⚠️ Looks like PHISHING-style text  ·  score {score:.2f} ≥ threshold {threshold:.2f}")
    else:
        st.success(f"✅ Looks legit  ·  score {score:.2f} < threshold {threshold:.2f}")
    st.progress(min(max(score, 0.0), 1.0))
    st.caption("The score is NOT guaranteed to be a true probability unless calibration was checked.")

    # highlighted text
    weights = word_weights(lr, text)
    parts = []
    for tok in re.findall(r"\w+|\W+", text):
        w = weights.get(tok.lower(), 0.0)
        if tok.strip() and re.match(r"\w+", tok) and abs(w) > 0.3:
            parts.append(f":red[**{tok}**]" if w > 0 else f":green[**{tok}**]")
        else:
            parts.append(tok)
    st.markdown("".join(parts))
    st.caption("red = pushes toward phishing, green = pushes toward legit (strong words only)")

    # per-feature explanation (logistic regression)
    contrib = contributions(lr, text)
    up = [(f, v) for f, v in contrib.items() if v > 0][:6]
    down = [(f, v) for f, v in contrib.items() if v < 0][:6]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Increase the score")
        st.table(pd.DataFrame({"feature": [f for f, _ in up], "log-odds": [round(v, 2) for _, v in up]}))
    with c2:
        st.subheader("Decrease the score")
        st.table(pd.DataFrame({"feature": [f for f, _ in down], "log-odds": [round(v, 2) for _, v in down]}))
    st.caption("Explanation = TF-IDF value × logistic-regression coefficient (uncalibrated model).")

# ---------------- optional: precision / recall vs threshold ----------------
with st.expander("Precision / recall versus threshold (test set)"):
    scores = model.predict_proba(Xte)[:, 1]
    rows = []
    for t in np.arange(0.1, 0.95, 0.05):
        pred = scores >= t
        tp = int(((pred == 1) & (yte.values == 1)).sum())
        prec = tp / pred.sum() if pred.sum() else float("nan")
        rec = tp / (yte.values == 1).sum()
        rows.append({"threshold": round(t, 2), "precision": prec, "recall": rec})
    st.line_chart(pd.DataFrame(rows).set_index("threshold"))