# mlsecurity-pipeline

A small, readable machine-learning pipeline in Python, and security experiments on the models it
trains. The pipeline is a package of plain functions: fetch, load, preprocess, train, evaluate,
save, load. The experiments are notebooks that each ask one question about an attack or a defence
and answer it with a table.

## Layout

| Path | What it is |
|---|---|
| `pipeline/` | The stages, one module each: `data`, `preprocess`, `train`, `evaluate`, `registry`, `plots`, `results`, `attacks` |
| `notebooks/` | The experiments |
| `experiments/results.csv` | One row per experiment run, written by `log_result()` |
| `tests/` | Tests for every stage; they run in CI on each push |


Every experiment states its question, its threat model, its baseline, the attack and the defence.

## Licence

MIT.
