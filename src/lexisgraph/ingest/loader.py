"""Load CUAD contracts from disk.

CUAD ships in a couple of shapes. The most common:
  - a folder of plain-text contracts (full_contract_txt/*.txt)
  - and/or a master JSON (CUAD_v1.json) in SQuAD-style format.

This loader handles the plain-text folder (simplest, most reliable) and
falls back to scanning any *.txt under the configured directory. It yields
(source_name, raw_text) pairs for the chunker to consume.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from lexisgraph.config import get_settings


def find_contract_files(cuad_dir: str | None = None) -> list[Path]:
    settings = get_settings()
    root = Path(cuad_dir or settings.cuad_dir)
    if not root.exists():
        raise FileNotFoundError(
            f"CUAD directory not found: {root}. "
            "Set CUAD_DIR in your .env to the folder containing the contracts."
        )

    # Prefer the canonical subfolder if present, else any .txt under root.
    preferred = root / "full_contract_txt"
    search_root = preferred if preferred.exists() else root
    files = sorted(search_root.rglob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt contracts found under {search_root}")
    return files


def load_contracts(
    cuad_dir: str | None = None, limit: int | None = None
) -> Iterator[tuple[str, str]]:
    """Yield (source_filename, text) for each contract, up to `limit`."""
    files = find_contract_files(cuad_dir)
    if limit:
        files = files[:limit]
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if text.strip():
            yield path.name, text
