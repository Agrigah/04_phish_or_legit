# Phish or Legit?

A small web application that classifies a short message (SMS / email) as **phishing** or **legitimate**,
shows the classifier score, and explains the decision with per-feature contributions.

**Live app:** https://04phishorlegit-h3qdirzcqetmdibxux4mpt.streamlit.app

> This tool supports a human decision and does not block messages by itself. The data is synthetic.

## What it does
- Text box + **Classify** button: predicted class and classifier score.
- **Decision-threshold slider**: the predicted class changes live.
- **Explanation**: words that increase / decrease the score (TF-IDF value x logistic-regression coefficient).
- **6 example messages**: 2 phishing, 2 legitimate, 1 missed phishing (false negative), 1 false alarm (false positive).
- Optional: classifier selector (logistic regression / sigmoid-calibrated) and a precision/recall vs threshold plot.

## Method
- Data: 2,000 synthetic messages, 20 % phishing (fictional brands).
- Stratified split 1,500 train / 500 test (seed 42), done **before** any fitting.
- Features: TF-IDF on word unigrams + bigrams (URLs, amounts and numbers replaced by placeholder tokens).
- Models: Multinomial Naive Bayes and logistic regression (L2), fitted inside one scikit-learn `Pipeline` on the training set only.

| Classifier | Precision | Recall | F1 | Brier |
|---|---|---|---|---|
| Naive Bayes | 0.611 | 0.800 | 0.693 | 0.104 |
| Logistic Regression | 0.747 | 0.710 | 0.728 | 0.071 |
| NB + calibration | 0.694 | 0.750 | 0.721 | 0.086 |
| LR + char n-grams (extension) | 0.777 | 0.730 | 0.753 | 0.064 |

(test set, threshold 0.5)

## Run locally
```bash
pip install -r requirements.txt
python -m src.evaluate      # metrics, threshold trade-off, failures -> results/
streamlit run app.py        # open http://localhost:8501
```

## Project structure
```
app.py                 Streamlit interface
src/pipeline.py        preprocessing, models, prediction, explanations
src/evaluate.py        evaluation script
data/messages.csv      synthetic dataset (data/make_data.py generates it)
results/               metrics.json, failures.txt, plots, run log
.streamlit/config.toml dark theme
```

## Limits
The data is synthetic; real phishing evolves and real-world performance would be lower. Once the discriminative
features are known, an attacker can avoid them, so the model needs regular retraining. Both error types matter:
false positives hurt legitimate senders, false negatives hurt victims.
