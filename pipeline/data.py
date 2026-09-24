"""Fetch and load datasets: CSV files (also .gz) and image folders."""
import io
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

DOWNLOAD_DIR = str(Path(__file__).resolve().parents[1] / "downloads")   # at the repository root


def fetch_dataset(url, zipped=True, download_dir=DOWNLOAD_DIR, file_name=None):
    """Download a dataset over HTTPS into `download_dir`.

    A zip archive is extracted into a folder named after the file. Returns the
    path of that folder, or of the downloaded file. If `file_name` is given,
    returns the path of that file inside the extracted folder instead.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("The URL must be a valid HTTPS URL")

    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True)
    name = Path(parsed.path).name or "dataset"

    if zipped:
        extract_path = download_dir / Path(name).stem
        if not extract_path.exists():
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            extract_path.mkdir(parents=True)
            with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
                archive.extractall(extract_path)
            print(f"Downloaded and extracted to {extract_path}")
        else:
            print(f"Using the copy already in {extract_path}")
        if file_name:
            return (extract_path / file_name).resolve()
        return extract_path.resolve()

    file_path = download_dir / name
    if not file_path.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        file_path.write_bytes(response.content)
        print(f"Downloaded to {file_path}")
    else:
        print(f"Using the copy already in {file_path}")
    return file_path.resolve()


def load_csv(path, header_names=None, sep=","):
    """Read a CSV (or .gz CSV) into a DataFrame and drop duplicate rows.

    Pass `header_names` when the file has no header row.
    """
    if header_names:
        df = pd.read_csv(path, sep=sep, header=None, names=header_names)
    else:
        df = pd.read_csv(path, sep=sep)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df


def load_images(folder, image_size, mean, std, batch_size=64, train_ratio=0.8, seed=1337):
    """Load an image folder (one sub-folder per class) into train and test loaders.

    Returns (train_loader, test_loader, n_classes, n_channels). The split is a
    random 80/20 split, seeded, so it is the same on every run.
    """
    folder = Path(folder)
    # descend through single-folder wrappers, which zip archives often add
    while True:
        subdirs = [d for d in folder.iterdir() if d.is_dir()]
        if len(subdirs) == 1:
            folder = subdirs[0]
        else:
            break

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    dataset = ImageFolder(root=folder, transform=transform)
    n_train = int(len(dataset) * train_ratio)
    n_test = len(dataset) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [n_train, n_test], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    n_channels = dataset[0][0].shape[0]
    n_classes = len(dataset.classes)
    print(f"{len(dataset)} images, {n_classes} classes, {n_channels} channel(s); "
          f"{n_train} train / {n_test} test")
    return train_loader, test_loader, n_classes, n_channels


# --- KDD Cup 1999 (network intrusion) ---------------------------------------
# the UCI archive; the zip holds kddcup.data_10_percent.gz, which is the usual training file
KDD_ATTACK_URL = "https://archive.ics.uci.edu/static/public/130/kdd+cup+1999+data.zip"
KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "attack",
]

ATTACK_CATEGORIES = {
    "dos": ["apache2", "back", "land", "neptune", "mailbomb", "pod", "processtable", "smurf",
            "teardrop", "udpstorm", "worm"],
    "probe": ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"],
    "privilege": ["buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm"],
    "access": ["ftp_write", "guess_passwd", "http_tunnel", "imap", "multihop", "named", "phf",
               "sendmail", "snmpgetattack", "snmpguess", "spy", "warezclient", "warezmaster",
               "xclock", "xsnoop"],
}


def prepare_kdd(df):
    """Map the KDD attack names to five categories in a new `attack_category` column.

    The raw file spells labels with a trailing dot ("normal."), which is removed.
    Rows whose attack name is unknown are labelled "normal".
    """
    lookup = {}
    for category, names in ATTACK_CATEGORIES.items():
        for name in names:
            lookup[name] = category
    attack = df["attack"].astype(str).str.rstrip(".")
    df = df.copy()
    df["attack_category"] = attack.map(lookup).fillna("normal")
    return df.drop(columns=["attack"])
