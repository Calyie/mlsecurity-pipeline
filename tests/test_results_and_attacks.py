import numpy as np

from pipeline import flip_labels, load_results, log_result
from pipeline import results as results_module


def test_log_result_appends_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(results_module, "RESULTS_FILE", str(tmp_path / "results.csv"))
    log_result("demo", poison_rate=0.1, accuracy=0.9)
    log_result("demo", poison_rate=0.2, accuracy=0.8)
    log_result("other", note="text")
    df = load_results("demo")
    assert len(df) == 2 and list(df["poison_rate"]) == [0.1, 0.2]
    assert len(load_results()) == 3


def test_flip_labels_flips_exactly_the_requested_share():
    y = np.array([0, 1] * 50)
    poisoned, flipped = flip_labels(y, 0.2, seed=1)
    assert len(flipped) == 20 and len(set(flipped)) == 20
    assert (poisoned[flipped] != y[flipped]).all()
    untouched = np.setdiff1d(np.arange(100), flipped)
    assert (poisoned[untouched] == y[untouched]).all()
    assert (y == np.array([0, 1] * 50)).all()          # the input is not changed


def test_flip_labels_zero_and_bounds():
    y = np.array([0, 1, 0, 1])
    poisoned, flipped = flip_labels(y, 0.0)
    assert len(flipped) == 0 and (poisoned == y).all()
    try:
        flip_labels(y, 1.5)
    except ValueError:
        pass
    else:
        raise AssertionError("a fraction above 1 must be refused")
