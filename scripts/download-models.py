#!/usr/bin/env python3
"""Download only the pinned artifacts, with resumable HF transfers and SHA256 checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = json.loads((ROOT / "models.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=["all", *MODELS], default="all", nargs="?")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    directory = Path(os.environ.get("DATA_DIR", "/workspace")) / "models"
    directory.mkdir(parents=True, exist_ok=True)
    selected = MODELS if args.backend == "all" else {args.backend: MODELS[args.backend]}
    for name, model in selected.items():
        target = directory / name / model["file"]
        if not args.verify_only:
            from huggingface_hub import hf_hub_download
            hf_hub_download(model["repo"], model["file"], revision=model["revision"],
                            local_dir=directory / name)
        if target.stat().st_size != model["bytes"]:
            raise SystemExit(f"Wrong file size: {target}")
        print(f"Checking SHA256: {target}", flush=True)
        with target.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != model["sha256"]:
            raise SystemExit(f"SHA256 mismatch: {target}; remove the corrupt file and retry")
        print(f"Verified {name}: {digest}", flush=True)


if __name__ == "__main__":
    main()
