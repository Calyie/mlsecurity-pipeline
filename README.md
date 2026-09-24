# mlops-pipeline

A small, readable machine-learning pipeline in Python, and security experiments on the models it
trains. The pipeline is a package of plain functions: fetch, load, preprocess, train, evaluate,
save, load. The experiments are notebooks that each ask one question about an attack or a defence
and answer it with a table.

## Layout

| Path | What it is |
|---|---|
| `pipeline/` | The stages, one module each: `data`, `preprocess`, `train`, `evaluate`, `registry`, `plots`, `results`, `attacks` |
| `notebooks/` | The experiments. `00_pipeline_walkthrough` runs the pipeline on four datasets; `TEMPLATE.ipynb` is the structure every experiment follows |
| `experiments/results.csv` | One row per experiment run, written by `log_result()` |
| `tests/` | Tests for every stage; they run in CI on each push |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
pytest -q
jupyter lab notebooks/
```

Drop the extra index URL for a GPU build of PyTorch. The Malimg notebook cell downloads from
Kaggle and needs a token in `~/.kaggle/`.

## Notebooks

| Notebook | Question |
|---|---|
| `00_pipeline_walkthrough` | Does one pipeline handle a regression, a text classifier, an intrusion classifier and an image classifier? |
| `01_label_flipping_blobs` | How fast does a linear classifier degrade as an attacker flips training labels? |

Every experiment states its question, its threat model, its baseline, the attack, the defence,
a results table, and what the result does and does not show.

## Licence

MIT.
