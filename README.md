# Provision type classification for standards

An experiment: determine the provision type of a sentence from a standard (DIN) quickly, without a generative LLM.

Types (by the verbal forms of DIN 820-2 / ISO/IEC Directives, Part 2):

| Type | German | English |
|---|---|---|
| requirement | muss, ist zu, darf nicht | shall, shall not |
| recommendation | sollte, es wird empfohlen | should, it is recommended |
| permission | darf, braucht nicht, ist zulässig | may, need not, is permitted |
| possibility | kann, lässt sich | can, is able to |
| statement | none of these forms: fact, definition, description | same |

## Results

Tested on the 50 sentences in `provisions.csv` (25 German, 25 English).

| Approach | Accuracy | ms / sentence (CPU) |
|---|---|---|
| TF-IDF n-grams + logistic regression (trained) | 100% | 1.2 |
| multilingual-e5-small embeddings + logistic regression (trained) | 82% | 17 |
| Laya zero-shot, best question formulation | 74% | 95–500 |
| GLiClass zero-shot (gliclass-x-base) | 36–68% | 135–175 |

**How to read these numbers**

- All sentences, training and test alike, are written in the style of standards rather than taken from real standards, and by the same author. The 100% for TF-IDF is therefore an upper bound; cross-validation on the training set gives 84–86%. Nothing has been tested on real standards.
- The TF-IDF model has essentially learned the verbal forms and fails on phrasings it has not seen ("is prohibited", "es ist darauf zu achten").
- The Laya question formulation (74%) was picked as the best of seven on the same 50 sentences; without that selection it scores 48–66%.
- Zero-shot models depend heavily on how the labels are worded: the spread for a single model reaches 30 points.
- NLI-based zero-shot models (bge-m3-zeroshot-v2.0, mDeBERTa-v3-base-xnli, deberta-v3-large-zeroshot-v2.0) were tried and dropped: 36–72% accuracy at 2.5–12 s per sentence, because they need one forward pass per label. gliclass-large-v3.0 was dropped too (English only, 16–44%).
- TypeSafe Jev and Microsoft Decision-1 were not tested: both are closed, paid APIs.

## Files

| File | What it is |
|---|---|
| `provisions.csv` | 50 test sentences with their types |
| `train.csv` | 300 training sentences (30 per type and language), no overlap with the test set |
| `classify.py` | Laya zero-shot run → `report.html`, `predictions.csv` |
| `compare.py` | accuracy and speed comparison of the fast zero-shot classifiers (Laya, GLiClass) → `comparison.csv` |
| `train.py` | trains two fast classifiers → `report_trained.html`, `predictions_trained.csv`, `model_*.joblib` |
| `report.html`, `report_trained.html` | reports with the probability of each type for each sentence |

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

`gliclass` is needed only for `compare.py`, `laya` for `classify.py` and `compare.py`.

## Running

```bash
.venv\Scripts\python train.py      # train and test, takes seconds
.venv\Scripts\python classify.py   # Laya zero-shot
.venv\Scripts\python compare.py    # Laya vs GLiClass zero-shot
```

On first run the models are downloaded from Hugging Face: about 470 MB for `train.py`, about 1.5 GB for `classify.py`, about 2.6 GB for `compare.py`.

`classify.py` accepts custom paths: `python classify.py input.csv report.html`. The input CSV has the columns `id`, `lang` (`de` or `en`), `label`, `text`.

## Next steps

Plan agreed in discussion. The numbers are working estimates, not measurements.

### Target setup

A small classifier handles most sentences; a router sends the low-confidence 5–10% to an LLM hosted in Azure. The LLM is also used once, offline, to pre-label training data.

### Data

- Available: about 300 German and 300 English NISO STS documents used for local tests, out of 40–50 thousand standards in total.
- Set aside about 50 documents for testing only. Training sentences come from the remaining documents, so that near-identical sentences from one standard never land in both sets.
- The unit is the sentence, not the document. A script extracts all sentences together with language, standard and section type.

### Labelling

| Set | Size | How it is labelled |
|---|---|---|
| Test | 500–600 sentences from the held-out documents | By hand and blind, without seeing any predicted label. Never used for training. |
| Training | 5,000–10,000 sentences | Rules (spaCy) and the Azure LLM label independently; matching labels are accepted. |

Manual checks on the training set, about 1,000–1,500 sentences in total:

- All sentences where rules and LLM disagree.
- An audit of the agreed labels: 50–100 random sentences per type, checked carefully. Zero errors in 300 means an error rate below 1% at 95% confidence; around 15 errors in 300 means the labels cannot be trusted yet. This is the check for cases where rules and LLM are wrong in the same way.
- Optional: a quick pass through the training set grouped by type and verbal form, where outliers stand out.

Errors found in the audit are usually systematic (for example "darf nicht" or "kann entfallen"), so they are fixed in the rule or the prompt rather than sentence by sentence.

The test set has to be clean because everything is measured against it. The training set tolerates a few percent of label noise; a systematic error in it shows up as a drop for one type on the test set.

Limits: a rare construction may be missing from 600 random sentences, so rare types (permission, possibility) are topped up deliberately. 600 sentences give roughly ±3–4 percentage points on overall accuracy and much less certainty per rare type.

### Models

- Build the rule baseline on verbal forms first. It is the lower bound any model has to beat, a free rough labeller and an explanation of each decision.
- Fine-tune a small multilingual encoder with a five-class head directly in `transformers`. The classes are fixed, so models that take labels at request time (Laya, GLiClass) add little; they stay as candidates, not the main path. The e5 result above used a frozen encoder with only a linear layer trained.
- Do not train inside spaCy: its text classifier is either a bag of words (same as the TF-IDF model), a CNN on its own vectors, or a wrapper around the same encoder fine-tuning. Keep spaCy for sentence splitting and for grammar-based rules ("ist … zu prüfen", scope of negation). A directly trained model can be wrapped as a spaCy component later.
- Colleagues' earlier spaCy training attempt: look at `meta.json` (`performance`) and `config.cfg` (`[components.textcat.model]`) in their model folder for results and architecture, and check whether labelled sentences from real standards are left over.

### Evaluation

Report per-type precision, recall and F1, and macro-F1 as the headline number, not overall accuracy. Permission and possibility are rare, so a model that never finds them can still reach high accuracy. Missed requirements and false requirements are different errors and likely cost differently.

### Open questions

- Unit of classification on real text: list items that inherit the modal verb from their lead-in sentence, sentences with more than one provision type, tables, formulas and figure captions. Decide before labelling.
- How much the STS section type helps. It narrows the choice (scope, terms and notes are almost always statements) but does not determine the type inside normative sections.
- Which STS version the documents use; needed for the sentence extraction script.
