"""Extract the archived public USGS tables and check their SHA-256 hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=root / "inputs")
    args = parser.parse_args()
    expected = json.loads((root / "input_hashes.json").read_text(encoding="utf-8"))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(root / "data/public_usgs_inputs.zip") as archive:
        if set(archive.namelist()) != set(expected):
            raise ValueError("The input archive does not contain the five declared files")
        for name, digest in expected.items():
            content = archive.read(name)
            if hashlib.sha256(content).hexdigest() != digest:
                raise ValueError(f"Archived input hash mismatch: {name}")
            target = out / name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f"Different existing input; choose a new directory: {target}")
            if not target.exists():
                target.write_bytes(content)
            print(f"Verified {name}")


if __name__ == "__main__":
    main()
