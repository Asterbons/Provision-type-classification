"""Compare fast zero-shot classifiers on provisions.csv (accuracy and CPU latency).

Usage: python compare.py
Writes comparison.csv (one row per model x formulation x language).
"""
import os
import time
import warnings

os.environ.setdefault("USE_TF", "0")
warnings.simplefilter("ignore")

import pandas as pd
import torch

LABELS = ["requirement", "recommendation", "permission", "possibility", "statement"]

# Two fixed label formulations per language, in LABELS order.
FORMS = {
    "names": {
        "en": ["requirement", "recommendation", "permission", "possibility", "statement"],
        "de": ["Anforderung", "Empfehlung", "Erlaubnis", "Möglichkeit", "Aussage"],
    },
    "desc": {
        "en": ["a requirement: something is mandatory (shall)",
               "a recommendation: something is advised (should)",
               "a permission: something is allowed (may)",
               "a possibility or capability: something can happen (can)",
               "a plain statement of fact"],
        "de": ["eine Anforderung: etwas ist verbindlich (muss)",
               "eine Empfehlung: etwas wird angeraten (sollte)",
               "eine Erlaubnis: etwas ist zulässig (darf)",
               "eine Möglichkeit oder Fähigkeit: etwas kann eintreten (kann)",
               "eine reine Tatsachenaussage"],
    },
}


def gliclass(model_id):
    from gliclass import GLiClassModel, ZeroShotClassificationPipeline
    from transformers import AutoTokenizer
    pipe = ZeroShotClassificationPipeline(
        GLiClassModel.from_pretrained(model_id), AutoTokenizer.from_pretrained(model_id),
        classification_type="single-label", device="cpu")

    def predict(text, options, lang):
        return max(pipe(text, options, threshold=0.0)[0], key=lambda d: d["score"])["label"]
    return predict


def laya(_):
    from laya import Router
    router = Router()
    model = {"en": "english", "de": "multilingual"}

    def predict(text, options, lang):
        q = {"q": {"type": "choice", "instructions": "What does this sentence from a technical standard express?",
                   "criteria": options}}
        return router.predict(text, q, model=model[lang])["answers"]["q"]["choice"]
    return predict


# (name, loader, model id, languages it is meant for)
MODELS = [
    ("laya (en + multilingual)", laya, None, ("en", "de")),
    ("gliclass-x-base", gliclass, "knowledgator/gliclass-x-base", ("en", "de")),
]


def main():
    df = pd.read_csv("provisions.csv")
    rows = []
    for name, loader, model_id, langs in MODELS:
        try:
            predict = loader(model_id)
        except Exception as ex:  # a model that fails to load should not stop the comparison
            print(f"{name}: LOAD FAILED: {type(ex).__name__}: {ex}", flush=True)
            continue
        sub = df[df.lang.isin(langs)]
        predict(sub.text.iloc[0], FORMS["names"][sub.lang.iloc[0]], sub.lang.iloc[0])  # warm-up
        for form, opts in FORMS.items():
            for lang in langs:
                part = sub[sub.lang == lang]
                t0 = time.perf_counter()
                with torch.inference_mode():
                    preds = [LABELS[opts[lang].index(predict(t, opts[lang], lang))] for t in part.text]
                ms = 1000 * (time.perf_counter() - t0) / len(part)
                ok = pd.Series(preds, index=part.index) == part.label
                row = {"model": name, "form": form, "lang": lang, "acc": ok.mean(), "ms_per_sentence": ms,
                       **{l: ok[part.label == l].mean() for l in LABELS}}
                rows.append(row)
                print(f"{name:32s} {form:5s} {lang} acc={row['acc']:.2f} {ms:6.0f} ms | "
                      + " ".join(f"{l[:4]}={row[l]:.1f}" for l in LABELS), flush=True)
        del predict
    pd.DataFrame(rows).to_csv("comparison.csv", index=False)


if __name__ == "__main__":
    main()
