"""Zero-shot provision-type classification of standard sentences with Laya.

Usage: python classify.py [provisions.csv] [report.html]
"""
import html
import os
import sys
import warnings

os.environ.setdefault("USE_TF", "0")  # avoids a known laya/transformers hang when TF is installed

import pandas as pd
from laya import Router

LABELS = ["requirement", "recommendation", "permission", "possibility", "statement"]
# One Laya checkpoint per language: English-only model for English, multilingual for German.
MODEL = {"en": "english", "de": "multilingual"}
LANG_NAME = {"en": "English", "de": "German"}
# Options are keyed by the modal verb of the sentence's language (same order as LABELS);
# this scored best of the formulations tried (74% vs 48% for label-name keys).
CRITERIA = {
    "de": {"muss": "Anforderung, verbindlich", "sollte": "Empfehlung", "darf": "Erlaubnis",
           "kann": "Möglichkeit oder Fähigkeit", "ist": "reine Aussage, Information"},
    "en": {"shall": "requirement, mandatory", "should": "recommendation", "may": "permission",
           "can": "possibility or capability", "is": "plain statement, information"},
}
INSTRUCTIONS = "Which verbal form expresses the meaning of this sentence?"


def classify(df):
    router = Router()
    rows = []
    for rec in df.itertuples():
        crit = CRITERIA[rec.lang]
        to_label = dict(zip(crit, LABELS))
        res = router.predict(rec.text, {"provision_type": {
            "type": "choice", "instructions": INSTRUCTIONS, "criteria": crit}}, model=MODEL[rec.lang])
        ans = res["answers"]["provision_type"]
        rows.append({
            "id": rec.id, "lang": rec.lang, "text": rec.text, "label": rec.label,
            "pred": to_label[ans["choice"]], "model": res["routing"]["model"],
            "truncated": res["usage"]["truncated"],
            **{f"p_{to_label[k]}": p for k, p in ans["probabilities"].items()},
        })
    return pd.DataFrame(rows)


def pct(x):
    return f"{100 * x:.1f}"


def section(sub, lang):
    e = html.escape
    g = sub.groupby("label", sort=False).ok.agg(["sum", "count"])
    type_rows = "".join(
        f"<tr><td>{k}</td><td class=num>{int(r['sum'])} / {int(r['count'])}</td>"
        f"<td class=num>{pct(r['sum'] / r['count'])}%</td></tr>" for k, r in g.iterrows())
    type_html = f"<table><caption>By true type</caption><tr><th>type</th><th>correct</th><th>accuracy</th></tr>{type_rows}</table>"

    cm = pd.crosstab(sub.label, sub.pred).reindex(index=LABELS, columns=LABELS, fill_value=0)
    cm_rows = "".join(
        f"<tr><th>{t}</th>" + "".join(
            f"<td class='num {'diag' if t == p else ('off' if cm.loc[t, p] else '')}'>{cm.loc[t, p]}</td>"
            for p in LABELS) + "</tr>" for t in LABELS)
    cm_html = ("<table><caption>Confusion matrix (rows: true, columns: predicted)</caption><tr><th></th>"
               + "".join(f"<th>{l}</th>" for l in LABELS) + f"</tr>{cm_rows}</table>")

    def prob_cell(r, l):
        p = r[f"p_{l}"]
        cls = " ".join(c for c, on in (("pred", l == r.pred), ("true", l == r.label)) if on)
        return (f"<td class='num prob {cls}' style='--p:{p:.3f}' "
                f"title='{l}: {pct(p)}%'>{pct(p)}</td>")

    sent_rows = "".join(
        f"<tr class='{'' if r.ok else 'miss'}'><td class=num>{r.id}</td>"
        f"<td class=text>{e(r.text)}{' (truncated)' if r.truncated else ''}</td><td>{r.label}</td>"
        f"<td>{'✓' if r.ok else '✗'} {r.pred}</td>"
        + "".join(prob_cell(r, l) for l in LABELS) + "</tr>"
        for _, r in sub.iterrows())

    options = "; ".join(f"{e(k)}: {e(d)}" for k, d in CRITERIA[lang].items())
    return f"""<section>
<h2>{LANG_NAME[lang]} — checkpoint <code>{e(MODEL[lang])}</code></h2>
<div class="big">{pct(sub.ok.mean())}% <span class="sub">accuracy ({int(sub.ok.sum())} / {len(sub)})</span></div>
<p>Options given to the model (key: description): {options}</p>
<div class="grid">{type_html}{cm_html}</div>
<h3>Per-sentence probabilities, %</h3>
<div class="scroll"><table>
<tr><th>#</th><th>sentence</th><th>true</th><th>predicted</th>{''.join(f'<th>{l}</th>' for l in LABELS)}</tr>
{sent_rows}</table></div>
</section>"""


def render(out, cal_warnings):
    out["ok"] = out.pred == out.label
    e = html.escape
    warn_html = "".join(f"<li>{e(w)}</li>" for w in cal_warnings)
    sections = "".join(section(out[out.lang == lang], lang) for lang in MODEL if (out.lang == lang).any())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Laya provision-type report</title>
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
.big {{ font-size:32px; font-weight:600; }} .sub {{ font-size:14px; font-weight:400; color:var(--muted); }}
.grid {{ display:flex; flex-wrap:wrap; gap:24px; align-items:flex-start; }}
.scroll {{ overflow-x:auto; }}
table {{ border-collapse:collapse; }}
caption {{ text-align:left; font-weight:600; padding-bottom:6px; }}
th, td {{ padding:5px 9px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
th {{ font-weight:600; color:var(--muted); font-size:12px; white-space:nowrap; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
.text {{ min-width:300px; }}
.prob {{ background:rgba(var(--bar), calc(var(--p) * .55)); }}
.prob.pred {{ font-weight:700; }}
.prob.true {{ box-shadow:inset 0 0 0 2px var(--good); }}
tr.miss td:nth-child(4) {{ color:var(--bad); font-weight:600; }}
tr.miss td.text {{ background:var(--missbg); }}
.diag {{ color:var(--good); font-weight:700; }} .off {{ color:var(--bad); }}
</style></head><body><main>
<h1>Laya — provision type classification</h1>
<p>Zero-shot, {len(out)} sentences written in the style of DIN standards. Each language is scored by its own
Laya checkpoint and reported separately. One <code>choice</code> question with five options:
“{e(INSTRUCTIONS)}” No fine-tuning, no temperature calibration.</p>
<p>In the tables, cell shading is proportional to the probability; bold = predicted type,
green outline = true type; rows with a wrong prediction are tinted.</p>
{sections}
<section><h2>Notes</h2>
<ul>
<li>Probabilities are the model's softmax over the five options; they sum to 100%.</li>
<li>Laya's model card says the base checkpoints ship over-confident and need temperature fitting on your own data
before the probabilities can be trusted as calibrated.</li>
{warn_html}
<li>The question formulation was chosen as the best of seven tried on these same {len(out)} sentences (range 48–74%),
so the accuracy shown is optimistic; it needs confirming on sentences not used for the choice.</li>
</ul></section>
</main></body></html>"""


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "provisions.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "report.html"
    df = pd.read_csv(src)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = classify(df)
    cal = sorted({str(w.message) for w in caught if "laya" in str(w.message)})
    out.to_csv("predictions.csv", index=False)
    with open(dst, "w", encoding="utf-8") as f:
        f.write(render(out, cal))
    for lang, sub in out.groupby("lang", sort=False):
        print(f"{lang} ({MODEL[lang]}): accuracy {sub.ok.mean():.3f} ({int(sub.ok.sum())}/{len(sub)})")
        print(sub.groupby("label", sort=False).ok.mean().round(3).to_string())
    print(f"wrote {dst} and predictions.csv")


if __name__ == "__main__":
    main()
