"""Download the M5 competition data into data/raw/.

Usage:  python scripts/download_data.py
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
URL = "https://raw.githubusercontent.com/Nixtla/m5-forecasts/main/datasets/m5.zip"
EXPECTED = ["calendar.csv", "sales_train_evaluation.csv", "sell_prices.csv"]


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    if all((RAW / f).exists() for f in EXPECTED):
        print("Data already present, nothing to do.")
        return 0

    archive = RAW / "m5.zip"
    print(f"Downloading M5 (~50 MB) from {URL}")
    urllib.request.urlretrieve(URL, archive)

    print("Extracting")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(RAW)

    missing = [f for f in EXPECTED if not (RAW / f).exists()]
    if missing:
        print(f"ERROR: expected files missing after extraction: {missing}")
        return 1

    print(f"Done. Files in {RAW}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
