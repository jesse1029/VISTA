"""Check all distributed files against MANIFEST.json."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Manifest paths must stay inside the repository")
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing release file: {relative}")
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise ValueError(f"Release file mismatch: {relative}")
    print(f"PASS: {len(manifest['files'])} distributed files")


if __name__ == "__main__":
    main()
