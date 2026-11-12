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
| Other zero-shot classifiers (GLiClass, NLI models) | 36–72% | 135–12,000 |

**How to read these numbers**

- All sentences, training and test alike, are written in the style of standards rather than taken from real standards, and by the same author. The 100% for TF-IDF is therefore an upper bound; cross-validation on the training set gives 84–86%. Nothing has been tested on real standards.
- The TF-IDF model has essentially learned the verbal forms and fails on phrasings it has not seen ("is prohibited", "es ist darauf zu achten").
- The Laya question formulation (74%) was picked as the best of seven on the same 50 sentences; without that selection it scores 48–66%.
- Zero-shot models depend heavily on how the labels are worded: the spread for a single model reaches 30 points.
- TypeSafe Jev and Microsoft Decision-1 were not tested: both are closed, paid APIs.

## Files

| File | What it is |
|---|---|
| `provisions.csv` | 50 test sentences with their types |
| `train.csv` | 300 training sentences (30 per type and language), no overlap with the test set |
| `classify.py` | Laya zero-shot run → `report.html`, `predictions.csv` |
| `compare.py` | accuracy and speed comparison of zero-shot classifiers → `comparison.csv` |
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
.venv\Scripts\python compare.py    # zero-shot model comparison, slow on CPU
```

On first run the models are downloaded from Hugging Face: about 470 MB for `train.py`, about 1.5 GB for `classify.py`, about 6 GB for `compare.py`.

`classify.py` accepts custom paths: `python classify.py input.csv report.html`. The input CSV has the columns `id`, `lang` (`de` or `en`), `label`, `text`.

## Next steps

1. Test the trained models on sentences from real standards (NISO STS).
2. Build a test set: 500–1,000 random sentences labelled by hand; split training and test by standard, not by sentence.
3. Label further in rounds of 200–300 sentences, starting with those the model is unsure about and with the rare types (permission, possibility).
4. Try fine-tuning the encoder itself rather than only a linear layer on top of its embeddings.
