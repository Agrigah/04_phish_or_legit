"""Lab 04 - Phish or Legit? (Streamlit UI + extensions). Run:  streamlit run app.py"""
import csv, html, io, os, re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, precision_recall_fscore_support

from src.pipeline import load_data, split, make_pipeline, predict, word_weights, contributions

st.set_page_config(page_title="Phish or Legit?", page_icon="🎣", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp {
  background: linear-gradient(135deg, #0b1020, #14123a, #0a2a3a, #0b1020);
  background-size: 400% 400%; animation: bg 18s ease infinite;
}
@keyframes bg { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }
@keyframes rise { from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none} }
@keyframes pop { 0%{opacity:0; transform:scale(.92)} 60%{transform:scale(1.03)} 100%{opacity:1; transform:scale(1)} }
@keyframes grow { from{width:0} }
@keyframes shine { to{background-position:200% center} }
@keyframes pulseR { 0%,100%{box-shadow:0 0 0 0 rgba(255,77,109,.55)} 50%{box-shadow:0 0 28px 6px rgba(255,77,109,.35)} }
@keyframes pulseG { 0%,100%{box-shadow:0 0 0 0 rgba(0,229,168,.5)} 50%{box-shadow:0 0 28px 6px rgba(0,229,168,.3)} }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-6px)} }

.hero { text-align:center; animation: rise .8s ease both; margin-bottom:.3rem; }
.hero .icon { font-size:3rem; display:inline-block; animation: float 3s ease-in-out infinite; }
.hero h1 {
  font-size:3rem; font-weight:700; margin:0; padding:0;
  background: linear-gradient(90deg,#00e5a8,#4d9bff,#ff4d6d,#00e5a8); background-size:200% auto;
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
  animation: shine 5s linear infinite;
}
.notice {
  animation: rise .9s ease both; background:rgba(255,196,0,.08); border:1px solid rgba(255,196,0,.45);
  border-left:5px solid #ffc400; border-radius:12px; padding:.8rem 1rem; color:#ffe08a; margin:.8rem 0;
}
.sub { color:#8f9bc7; font-size:.85rem; animation: rise 1s ease both; }
.res { border-radius:18px; padding:1.1rem 1.3rem; margin:1rem 0 .4rem; animation: pop .6s ease both; }
.res.phish { background:rgba(255,77,109,.12); border:1px solid #ff4d6d; animation: pop .6s ease both, pulseR 2s ease-in-out .6s infinite; }
.res.legit { background:rgba(0,229,168,.10); border:1px solid #00e5a8; animation: pop .6s ease both, pulseG 2.4s ease-in-out .6s infinite; }
.res .big { font-size:1.5rem; font-weight:700; }
.res .small { color:#c3cbee; font-size:.9rem; }
.gauge { position:relative; height:16px; border-radius:99px; background:rgba(255,255,255,.08); margin:.9rem 0 .2rem; overflow:visible; }
.gauge .fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#00e5a8,#ffc400,#ff4d6d); animation: grow 1.1s cubic-bezier(.2,.8,.2,1) both; }
.gauge .thr { position:absolute; top:-6px; width:3px; height:28px; background:#fff; border-radius:2px; }
.gauge .thr::after { content:'threshold'; position:absolute; top:-16px; left:-22px; font-size:.65rem; color:#fff; }
.hl { line-height:2; padding:.8rem 1rem; border-radius:12px; background:rgba(255,255,255,.04); animation: rise .8s ease both; }
.up { background:rgba(255,77,109,.25); color:#ff9fb2; border-radius:6px; padding:1px 5px; font-weight:700; }
.down { background:rgba(0,229,168,.22); color:#8ff5d6; border-radius:6px; padding:1px 5px; font-weight:700; }
.col-title { font-weight:700; margin:.6rem 0 .3rem; }
.row { margin:.35rem 0; animation: rise .6s ease both; }
.row .lab { display:flex; justify-content:space-between; font-size:.85rem; color:#cfd6f5; }
.row .bar { height:9px; border-radius:99px; margin-top:3px; animation: grow 1s ease both; }
.row .bar.p { background:linear-gradient(90deg,#ff4d6d,#ff8aa0); }
.row .bar.n { background:linear-gradient(90deg,#00e5a8,#7ff3d3); }
div.stButton > button {
  background:linear-gradient(90deg,#00e5a8,#4d9bff); color:#04101c; font-weight:700; border:0;
  border-radius:12px; padding:.6rem 1.6rem; transition: transform .15s ease, box-shadow .15s ease;
}
div.stButton > button:hover { transform:translateY(-3px) scale(1.03); box-shadow:0 8px 24px rgba(77,155,255,.45); color:#04101c; }
section[data-testid="stSidebar"] { border-right:1px solid rgba(255,255,255,.08); }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="icon">🎣</div><h1>Phish or Legit?</h1></div>', unsafe_allow_html=True)
st.markdown('<div class="notice">⚠️ This tool supports a human decision and does not block messages by itself. '
            'The data is synthetic.</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">TF-IDF + classifier on a SYNTHETIC dataset (fictional brands). Labels are '
            "'phishing-style' vs 'legit', not real-world ground truth.</div>", unsafe_allow_html=True)

EXAMPLES = {
    "Phishing - correctly detected (Northbank)":
        "Hello, We noticed activity on your Northbank account. Confirm your details. "
        "Please reply as soon as you can. https://xcen0-login.info/at5g",
    "Phishing - correctly detected (BlueFreight)":
        "ALERT: BlueFreight: your package is waiting. Confirm your details. "
        "This is your final notice. http://secure-2kcy-verify.com/j8n1ws",
    "Legit - correctly classified (order)": "Your order #96192 has shipped.",
    "Legit - correctly classified (reminder)": "Reminder: class starts tomorrow at 9. Open the official app to view it.",
    "MISSED phishing (false negative)": "Your Civic Savings account has a new notice. http://bit.ly/tp99jp",
    "FALSE ALARM on legit (false positive)":
        "Your invoice of $21.12 is ready. Update your payment info. https://docs.example.com/d/k5isac",
    "Obfuscated phishing (N0rthbank, detai1s) - char n-grams help":
        "Hello, We noticed activity on your N0rthbank account. C0nfirm your detai1s. https://xcen0-login.info/at5g",
}
WORD, CHAR = "Logistic regression", "LR + character n-grams (extension)"
LRCAL, NB, NBCAL = "LR + sigmoid calibration", "Naive Bayes (uncalibrated)", "NB + sigmoid calibration"


@st.cache_resource(show_spinner="Training models on the synthetic training set…")
def get_models():
    df = load_data()
    Xtr, Xte, ytr, yte = split(df)
    spec = {WORD: ("lr", "word"), CHAR: ("lr", "char_wb"), LRCAL: ("lr_cal", "word"),
            NB: ("nb", "word"), NBCAL: ("nb_cal", "word")}
    return {k: make_pipeline(kind, analyzer=an).fit(Xtr, ytr) for k, (kind, an) in spec.items()}, Xte, yte


models, Xte, yte = get_models()
lr = models[WORD]

st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              help="PHISHING when score >= threshold. Higher = fewer false alarms, more missed phishing.")
model_name = st.sidebar.selectbox("Classifier", list(models))
model = models[model_name]
expl_model = model if model_name in (WORD, CHAR) else lr     # explanations need the coefficients
st.sidebar.caption("Explanations use the logistic-regression coefficients"
                   + (" (character n-grams)." if model_name == CHAR else " (word model)."))

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Classify", "📊 Calibration", "📁 CSV batch", "📨 SMS Spam test"])

# ============================ TAB 1: classify + feedback ============================
with tab1:
    choice = st.selectbox("Example messages", ["(write my own)"] + list(EXAMPLES))
    default = EXAMPLES.get(choice, "Northbank: your account has been locked. Verify now http://bit.ly/ab12cd")
    text = st.text_area("Message to classify:", value=default, height=110, key=f"txt-{choice}")
    if st.button("🔍 Classify", type="primary"):
        st.session_state["analyzed"] = text

    def bars(items, kind):
        if not items:
            return '<div class="sub">none</div>'
        mx = max(abs(v) for _, v in items) or 1
        return "".join(
            f'<div class="row" style="animation-delay:{i*0.08:.2f}s"><div class="lab"><span>{html.escape(f)}</span>'
            f'<span>{v:+.2f}</span></div><div class="bar {kind}" style="width:{abs(v)/mx*100:.0f}%;'
            f'animation-delay:{i*0.08:.2f}s"></div></div>' for i, (f, v) in enumerate(items))

    msg = st.session_state.get("analyzed")
    if msg:
        try:
            _, score = predict(model, msg)
        except ValueError as err:
            st.error(str(err)); st.stop()
        is_phish = score >= threshold                      # live: re-evaluated when the slider moves
        cls, icon = ("phish", "🚨") if is_phish else ("legit", "✅")
        st.markdown(
            f'<div class="res {cls}"><div class="big">{icon} {"Looks like PHISHING-style text" if is_phish else "Looks legit"}</div>'
            f'<div class="small">score {score:.2f} {"≥" if is_phish else "<"} threshold {threshold:.2f}</div>'
            f'<div class="gauge"><div class="fill" style="width:{score*100:.0f}%"></div>'
            f'<div class="thr" style="left:{threshold*100:.0f}%"></div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="sub">The score is NOT guaranteed to be a true probability unless calibration was checked '
                    '(see the Calibration tab).</div>', unsafe_allow_html=True)

        w = word_weights(lr, msg)
        parts = []
        for tok in re.findall(r"\w+|\W+", msg):
            wt, safe = w.get(tok.lower(), 0.0), html.escape(tok)
            parts.append(f'<span class="{"up" if wt > 0 else "down"}">{safe}</span>'
                         if tok.strip() and re.match(r"\w+", tok) and abs(wt) > 0.3 else safe)
        st.markdown(f'<div class="col-title">Message analysis</div><div class="hl">{"".join(parts)}</div>'
                    '<div class="sub">red = pushes toward phishing, green = toward legit (strong words only)</div>',
                    unsafe_allow_html=True)
        contrib = contributions(expl_model, msg)
        up = [(f, v) for f, v in contrib.items() if v > 0][:6]
        down = [(f, v) for f, v in contrib.items() if v < 0][:6]
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="col-title">🔺 Increase the score</div>{bars(up, "p")}', unsafe_allow_html=True)
        c2.markdown(f'<div class="col-title">🔻 Decrease the score</div>{bars(down, "n")}', unsafe_allow_html=True)
        st.markdown('<div class="sub">Contribution to the log-odds = TF-IDF value × coefficient. Read the numbers: '
                    'each column has its own bar scale.</div>', unsafe_allow_html=True)

        # ---- feedback button ----
        st.divider()
        if st.button("🚩 Misclassified - log this message for retraining"):
            row = {"time": datetime.now().isoformat(timespec="seconds"), "text": msg, "score": round(score, 3),
                   "threshold": threshold, "model": model_name, "predicted": cls,
                   "suggested_label": "legit" if is_phish else "phish"}
            st.session_state.setdefault("feedback", []).append(row)
            try:   # local log (ephemeral on Streamlit Cloud, hence the download below)
                new = not os.path.exists("feedback.csv")
                with open("feedback.csv", "a", newline="", encoding="utf-8") as f:
                    wr = csv.DictWriter(f, fieldnames=list(row)); new and wr.writeheader(); wr.writerow(row)
            except OSError:
                pass
            st.toast("Feedback logged. Thank you!", icon="🚩")
    fb = st.session_state.get("feedback", [])
    if fb:
        st.caption(f"{len(fb)} message(s) flagged this session. Nothing is retrained automatically.")
        st.download_button("⬇️ Download feedback.csv", pd.DataFrame(fb).to_csv(index=False), "feedback.csv", "text/csv")

# ============================ TAB 2: calibration ============================
with tab2:
    st.write("A **reliability diagram** compares the predicted score with the observed share of phishing. "
             "The closer to the dotted diagonal, the more the score behaves like a probability. "
             "Sigmoid (Platt) calibration is fitted with 5-fold cross-validation on the training set.")
    cal_models = [WORD, LRCAL, NB, NBCAL]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    fig.patch.set_alpha(0); ax.set_facecolor("#141b38")
    ax.plot([0, 1], [0, 1], ":", color="white", label="perfect")
    rows = []
    for name, col in zip(cal_models, ["#4d9bff", "#00e5a8", "#ff4d6d", "#ffc400"]):
        p = models[name].predict_proba(Xte)[:, 1]
        fr, mp = calibration_curve(yte, p, n_bins=8, strategy="quantile")
        ax.plot(mp, fr, "o-", color=col, label=name)
        rows.append({"model": name, "Brier score": round(brier_score_loss(yte, p), 3)})
    ax.set_xlabel("mean predicted score", color="white"); ax.set_ylabel("observed share of phishing", color="white")
    ax.tick_params(colors="white"); ax.legend(fontsize=7)
    st.pyplot(fig)
    st.table(pd.DataFrame(rows).set_index("model"))
    st.caption("Lower Brier = better probabilities. Naive Bayes is typically over-confident; calibration improves it.")

# ============================ TAB 3: CSV batch ============================
with tab3:
    st.write("Upload a CSV with a **text** column (or the first column is used). Messages with score ≥ threshold "
             f"(**{threshold:.2f}**, current model) are flagged. Nothing is blocked or deleted.")
    up = st.file_uploader("CSV of messages", type=["csv"], key="batch")
    if up:
        d = pd.read_csv(up).head(5000)
        col = "text" if "text" in d.columns else d.columns[0]
        d["score"] = model.predict_proba(d[col].fillna("").astype(str))[:, 1].round(3)
        d["flag"] = np.where(d["score"] >= threshold, "phish", "legit")
        flagged = d[d["flag"] == "phish"]
        st.write(f"{len(flagged)} flagged out of {len(d)} messages (max 5,000 rows).")
        st.dataframe(flagged, use_container_width=True)
        st.download_button("⬇️ Export flagged messages (CSV)", flagged.to_csv(index=False), "flagged.csv", "text/csv")

# ============================ TAB 4: UCI SMS Spam ============================
with tab4:
    st.warning("The UCI SMS Spam collection is a **spam** dataset, not a phishing dataset, and the models here were "
               "trained on synthetic messages: low scores are expected (domain shift).")
    st.write("Upload `SMSSpamCollection` (tab-separated: label, text) from the UCI repository, or the Kaggle CSV "
             "(columns v1, v2). The current model and threshold are applied to it.")
    up2 = st.file_uploader("SMS Spam Collection", type=["txt", "csv", "tsv", ""], key="sms")
    if up2:
        raw = up2.getvalue().decode("latin-1")
        if "\t" in raw.splitlines()[0]:
            s = pd.read_csv(io.StringIO(raw), sep="\t", header=None, names=["label", "text"], quoting=csv.QUOTE_NONE)
        else:
            s = pd.read_csv(io.StringIO(raw)).iloc[:, :2]; s.columns = ["label", "text"]
        s = s[s["label"].isin(["ham", "spam"])].dropna()
        y = (s["label"] == "spam").astype(int).values
        pred = (model.predict_proba(s["text"].astype(str))[:, 1] >= threshold).astype(int)
        p_, r_, f_, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        st.write(f"{len(s)} messages, {y.mean():.1%} spam. Precision **{p_:.3f}**, recall **{r_:.3f}**, F1 **{f_:.3f}**")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        st.table(pd.DataFrame(cm, index=["true ham", "true spam"], columns=["pred legit", "pred phish"]))

with st.expander("📈 Precision / recall versus threshold (test set)"):
    scores = model.predict_proba(Xte)[:, 1]
    rows = []
    for t in np.arange(0.1, 0.95, 0.05):
        pr = scores >= t; tp = int(((pr == 1) & (yte.values == 1)).sum())
        rows.append({"threshold": round(t, 2), "precision": tp / pr.sum() if pr.sum() else float("nan"),
                     "recall": tp / (yte.values == 1).sum()})
    st.line_chart(pd.DataFrame(rows).set_index("threshold"))