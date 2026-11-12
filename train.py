"""Train fast provision-type classifiers on train.csv and test them on provisions.csv.

Two variants:
  tfidf - word + character n-grams -> logistic regression (no neural network)
  e5    - multilingual-e5-small sentence embeddings -> logistic regression

Usage: python train.py
Writes report_trained.html, predictions_trained.csv and model_<variant>.joblib.
"""
import html
import time
import warnings

warnings.simplefilter("ignore")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline, make_union

LABELS = ["requirement", "recommendation", "permission", "possibility", "statement"]
E5 = "intfloat/multilingual-e5-small"


class E5Encoder:
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(E5)
        self.model = AutoModel.from_pretrained(E5).eval()

    def encode(self, texts, batch=32):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tok(["query: " + t for t in texts[i:i + batch]], padding=True, truncation=True,
                           max_length=128, return_tensors="pt")
            with torch.inference_mode():
                h = self.model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1)
            emb = (h * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(emb, dim=-1).numpy())
        return np.vstack(out)


def tfidf_clf():
    return make_pipeline(
        make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
                   TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        LogisticRegression(C=20, max_iter=2000))


def cv_accuracy(clf, X, y):
    pred = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return (pred == y).mean()


def pct(x):
    return f"{100 * x:.1f}"


def report(results, test, n_train):
    e = html.escape
    summary = "".join(
        f"<tr><td>{e(r['name'])}</td><td class=num>{pct(r['cv'])}%</td>"
        f"<td class=num>{pct(r['acc'])}% ({int(r['ok'].sum())} / {len(test)})</td>"
        + "".join(f"<td class=num>{pct(r['ok'][test.lang == l].mean())}%</td>" for l in ("de", "en"))
        + f"<td class=num>{r['ms']:.1f}</td></tr>" for r in results)
    sections = ""
    for r in results:
        cm = pd.crosstab(test.label, r["pred"]).reindex(index=LABELS, columns=LABELS, fill_value=0)
        cm_rows = "".join(
            f"<tr><th>{t}</th>" + "".join(
                f"<td class='num {'diag' if t == p else ('off' if cm.loc[t, p] else '')}'>{cm.loc[t, p]}</td>"
                for p in LABELS) + "</tr>" for t in LABELS)
        rows = ""
        for i, rec in enumerate(test.itertuples()):
            ok = r["pred"][i] == rec.label
            cells = "".join(
                f"<td class='num prob {'pred' if l == r['pred'][i] else ''} {'true' if l == rec.label else ''}' "
                f"style='--p:{r['proba'][i][j]:.3f}'>{pct(r['proba'][i][j])}</td>" for j, l in enumerate(LABELS))
            rows += (f"<tr class='{'' if ok else 'miss'}'><td class=num>{rec.id}</td><td>{rec.lang}</td>"
                     f"<td class=text>{e(rec.text)}</td><td>{rec.label}</td>"
                     f"<td>{'✓' if ok else '✗'} {r['pred'][i]}</td>{cells}</tr>")
        sections += f"""<section><h2>{e(r['name'])}</h2>
<div class="big">{pct(r['acc'])}% <span class="sub">on the 50 test sentences, {r['ms']:.1f} ms per sentence on CPU</span></div>
<table><caption>Confusion matrix (rows: true, columns: predicted)</caption><tr><th></th>{''.join(f'<th>{l}</th>' for l in LABELS)}</tr>{cm_rows}</table>
<h3>Per-sentence probabilities, %</h3>
<div class="scroll"><table><tr><th>#</th><th>lang</th><th>sentence</th><th>true</th><th>predicted</th>{''.join(f'<th>{l}</th>' for l in LABELS)}</tr>{rows}</table></div></section>"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trained provision classifiers</title>
<style>
:root {{ --bg:#fff; --fg:#1b1f24; --muted:#667085; --line:#e3e6ea; --bar:37,99,235; --good:#15803d; --bad:#b42318; --missbg:#fef3f2; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#14171c; --fg:#e6e8eb; --muted:#98a2b3; --line:#2b3038; --bar:96,165,250; --good:#4ade80; --bad:#f97066; --missbg:#2a1a19; }}
}}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:1200px; margin:auto; }}
h1 {{ font-size:22px; margin:0 0 4px; }} h2 {{ font-size:18px; margin:0 0 8px; }} h3 {{ font-size:14px; margin:20px 0 6px; }}
section {{ margin-top:32px; padding-top:20px; border-top:2px solid var(--line); }}
p, li {{ color:var(--muted); max-width:80ch; }}
.big {{ font-size:32px; font-weight:600; margin-bottom:12px; }} .sub {{ font-size:14px; font-weight:400; color:var(--muted); }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; }}
caption {{ text-align:left; font-weight:600; padding-bottom:6px; }}
th, td {{ padding:5px 9px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ font-weight:600; color:var(--muted); font-size:12px; white-space:nowrap; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.text {{ min-width:300px; }}
.prob {{ background:rgba(var(--bar), calc(var(--p) * .55)); }}
.prob.pred {{ font-weight:700; }} .prob.true {{ box-shadow:inset 0 0 0 2px var(--good); }}
tr.miss td:nth-child(5) {{ color:var(--bad); font-weight:600; }} tr.miss td.text {{ background:var(--missbg); }}
.diag {{ color:var(--good); font-weight:700; }} .off {{ color:var(--bad); }}
</style></head><body><main>
<h1>Trained provision-type classifiers</h1>
<p>Trained on {n_train} sentences written in the style of DIN standards (German and English, 30 per type and language),
tested on the separate 50 sentences used for the zero-shot runs. Training and test sentences were written by the
same author in the same style, so these numbers are an upper bound for real standard texts.</p>
<div class="scroll"><table><tr><th>model</th><th>5-fold CV on train</th><th>test accuracy</th><th>test DE</th><th>test EN</th><th>ms / sentence (CPU)</th></tr>{summary}</table></div>
<p>In the tables below, cell shading is proportional to the probability; bold = predicted type,
green outline = true type; rows with a wrong prediction are tinted.</p>
{sections}
</main></body></html>"""


def main():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("provisions.csv")
    y = train.label.to_numpy()
    results = []

    # --- tfidf ---
    clf = tfidf_clf()
    cv = cv_accuracy(clf, train.text, y)
    clf.fit(train.text, y)
    t0 = time.perf_counter()
    proba = np.vstack([clf.predict_proba([t]) for t in test.text])  # one at a time, as in production
    ms = 1000 * (time.perf_counter() - t0) / len(test)
    joblib.dump(clf, "model_tfidf.joblib")
    results.append({"name": "TF-IDF n-grams + logistic regression", "key": "tfidf", "cv": cv, "ms": ms,
                    "proba": proba[:, [list(clf.classes_).index(l) for l in LABELS]]})

    # --- e5 embeddings ---
    enc = E5Encoder()
    Xtr = enc.encode(train.text.tolist())
    head = LogisticRegression(C=20, max_iter=2000)
    cv = cv_accuracy(head, Xtr, y)
    head.fit(Xtr, y)
    enc.encode([test.text[0]])  # warm-up
    t0 = time.perf_counter()
    proba = np.vstack([head.predict_proba(enc.encode([t])) for t in test.text])
    ms = 1000 * (time.perf_counter() - t0) / len(test)
    joblib.dump(head, "model_e5.joblib")
    results.append({"name": f"{E5} embeddings + logistic regression", "key": "e5", "cv": cv, "ms": ms,
                    "proba": proba[:, [list(head.classes_).index(l) for l in LABELS]]})

    out = test.copy()
    for r in results:
        r["pred"] = np.array(LABELS)[r["proba"].argmax(1)]
        r["ok"] = pd.Series(r["pred"] == test.label.to_numpy())
        r["acc"] = r["ok"].mean()
        out[f"{r['key']}_pred"] = r["pred"]
        for j, l in enumerate(LABELS):
            out[f"{r['key']}_p_{l}"] = r["proba"][:, j].round(4)
        print(f"{r['name']}: cv={r['cv']:.3f} test={r['acc']:.3f} "
              f"de={r['ok'][test.lang == 'de'].mean():.2f} en={r['ok'][test.lang == 'en'].mean():.2f} "
              f"{r['ms']:.1f} ms/sentence")
        print("  misses:", [(int(i), l, p) for i, l, p in zip(test.id, test.label, r["pred"]) if l != p])
    out.to_csv("predictions_trained.csv", index=False)
    with open("report_trained.html", "w", encoding="utf-8") as f:
        f.write(report(results, test, len(train)))
    print("wrote report_trained.html, predictions_trained.csv")


if __name__ == "__main__":
    main()
