"""Download CUAD and extract each contract to a plain-text file.

CUAD ships as a single large JSON (CUADv1.json) with 510 contracts, each
contract's full text sitting in data[i].paragraphs[0].context. Our loader
wants a folder of .txt files, so this script bridges the two: download the
zip, pull out CUADv1.json, and write one .txt per contract into
data/cuad/full_contract_txt/.

Run once:  uv run python scripts/download_cuad.py
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

CUAD_ZIP_URL = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
OUT_DIR = Path("data/cuad/full_contract_txt")


def safe_filename(title: str) -> str:
    """Turn a contract title into a safe .txt filename."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")
    return f"{name[:120]}.txt"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    work = Path("data/cuad")
    zip_path = work / "data.zip"
    json_path = work / "CUADv1.json"

    if not json_path.exists():
        print(f"Downloading CUAD from {CUAD_ZIP_URL} ...")
        urlretrieve(CUAD_ZIP_URL, zip_path)
        print("Extracting CUADv1.json ...")
        with zipfile.ZipFile(zip_path) as z:
            z.extract("CUADv1.json", path=work)
    else:
        print("CUADv1.json already present, skipping download.")

    print("Writing contracts to text files ...")
    data = json.loads(json_path.read_text(encoding="utf-8"))["data"]
    written = 0
    for contract in data:
        title = contract.get("title", f"contract_{written}")
        paragraphs = contract.get("paragraphs", [])
        if not paragraphs:
            continue
        text = paragraphs[0].get("context", "")
        if not text.strip():
            continue
        (OUT_DIR / safe_filename(title)).write_text(text, encoding="utf-8")
        written += 1

    print(f"Done. Wrote {written} contracts to {OUT_DIR}/")
    print(f"Set CUAD_DIR=data/cuad in your .env (already the default).")


if __name__ == "__main__":
    main()
