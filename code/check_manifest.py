"""Check all distributed files against MANIFEST.json."""
import hashlib
import json
from pathlib import Path


def check_release(root=None):
    root = Path(__file__).resolve().parent if root is None else Path(root).resolve()
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
    return len(manifest["files"])


def main():
    print(f"PASS: {check_release()} distributed files")


if __name__ == "__main__":
    main()
