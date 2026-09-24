import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pipeline import KDD_COLUMNS, fetch_dataset, load_csv, load_images, prepare_kdd


def test_fetch_refuses_anything_but_https():
    with pytest.raises(ValueError):
        fetch_dataset("http://example.com/data.zip")
    with pytest.raises(ValueError):
        fetch_dataset("not a url")


def test_load_csv_with_and_without_header(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,x\n2,y\n1,x\n")
    df = load_csv(path)
    assert list(df.columns) == ["a", "b"] and len(df) == 2       # the duplicate row is dropped
    path.write_text("1,x\n2,y\n")
    df = load_csv(path, header_names=["first", "second"])
    assert list(df.columns) == ["first", "second"] and len(df) == 2


def test_prepare_kdd_maps_attacks_to_categories():
    df = pd.DataFrame({"duration": [0, 1, 2, 3], "attack": ["normal.", "smurf.", "nmap.", "made_up."]})
    out = prepare_kdd(df)
    assert list(out["attack_category"]) == ["normal", "dos", "probe", "normal"]
    assert "attack" not in out.columns and "attack" in KDD_COLUMNS


def test_load_images_splits_a_class_folder(tmp_path):
    rng = np.random.default_rng(0)
    for label in ("cat", "dog"):
        (tmp_path / "wrapper" / label).mkdir(parents=True)
        for i in range(5):
            Image.fromarray(rng.integers(0, 255, (20, 20, 3), dtype=np.uint8)).save(
                tmp_path / "wrapper" / label / f"{i}.png")
    train_loader, test_loader, n_classes, n_channels = load_images(
        tmp_path, image_size=16, mean=[0.5] * 3, std=[0.5] * 3, batch_size=4)
    assert n_classes == 2 and n_channels == 3
    assert len(train_loader.dataset) == 8 and len(test_loader.dataset) == 2
    images, labels = next(iter(train_loader))
    assert images.shape == (4, 3, 16, 16)
