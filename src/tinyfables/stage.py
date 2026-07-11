"""The stage contract: every stage writes its artifacts plus a manifest.json
recording the config echo and a SHA-256 per artifact. Two runs of the same
stage with the same config+seed must produce identical `artifacts` maps.

The manifest is written LAST, so its presence marks stage completion;
resumable-run tooling relies on this, and future stages must preserve the
ordering (write all other artifacts first, manifest.json last)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_VERSIONED_PACKAGES = [
    "tinyfables",
    "tokenizers",
    "numpy",
    "torch",
    "transformers",
    "matplotlib",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict[str, str]:
    out = {}
    for pkg in _VERSIONED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    return out


def write_manifest(
    out_dir: Path,
    stage: str,
    config_obj,
    artifacts: list[Path],
    inputs: dict[str, Path] | None = None,
) -> Path:
    manifest = {
        "stage": stage,
        "config": dataclasses.asdict(config_obj),
        "artifacts": {p.relative_to(out_dir).as_posix(): sha256_file(p) for p in artifacts},
        "inputs": {name: sha256_file(path) for name, path in (inputs or {}).items()},
        "versions": _versions(),
        "created_unix": int(time.time()),
    }
    out = out_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return out
