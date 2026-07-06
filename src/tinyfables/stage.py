"""The stage contract: every stage writes its artifacts plus a manifest.json
recording the config echo and a SHA-256 per artifact. Two runs of the same
stage with the same config+seed must produce identical `artifacts` maps."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, stage: str, config_obj, artifacts: list[Path]) -> Path:
    manifest = {
        "stage": stage,
        "config": dataclasses.asdict(config_obj),
        "artifacts": {p.name: sha256_file(p) for p in sorted(artifacts)},
        "created_unix": int(time.time()),
    }
    out = out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out
