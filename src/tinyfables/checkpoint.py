"""Periodic training checkpoints for session-death resume. A checkpoint dir holds
the model (safetensors via save_pretrained), optimizer + AMP scaler + step
(optimizer.pt), and a completion marker (checkpoint_state.json) written LAST —
mirroring the stage-manifest convention, so a checkpoint interrupted mid-write is
never mistaken for complete. On Colab the parent ckpt_dir is Drive-mounted, so
checkpoints outlive the runtime and the same command resumes after a session death."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

_MARKER = "checkpoint_state.json"
_PREFIX = "step_"


def save_checkpoint(model, optimizer, scaler, step: int, ckpt_dir) -> Path:
    d = Path(ckpt_dir) / f"{_PREFIX}{step}"
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    scaler_state = None
    if scaler is not None:
        scaler_dict = scaler.state_dict()
        scaler_state = scaler_dict if scaler_dict else None  # Convert empty dict (disabled scaler) to None
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scaler": scaler_state,
            "step": step,
        },
        d / "optimizer.pt",
    )
    (d / _MARKER).write_text(json.dumps({"step": step}) + "\n")  # LAST = completion marker
    return d


def _complete(d: Path) -> bool:
    return (
        (d / _MARKER).exists()
        and (d / "optimizer.pt").exists()
        and (d / "model.safetensors").exists()
    )


def _complete_checkpoints(ckpt_dir) -> list[tuple[int, Path]]:
    root = Path(ckpt_dir)
    if not root.exists():
        return []
    out: list[tuple[int, Path]] = []
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith(_PREFIX) and _complete(d):
            try:
                out.append((int(d.name[len(_PREFIX) :]), d))
            except ValueError:
                pass
    return sorted(out, key=lambda t: t[0])


def find_latest_checkpoint(ckpt_dir) -> Path | None:
    ckpts = _complete_checkpoints(ckpt_dir)
    return ckpts[-1][1] if ckpts else None


def load_checkpoint(ckpt_path, model_cls, device):
    d = Path(ckpt_path)
    model = model_cls.from_pretrained(d).to(device)
    state = torch.load(d / "optimizer.pt", map_location=device)
    return model, state


def prune_checkpoints(ckpt_dir, keep_last_k: int) -> None:
    if keep_last_k <= 0:
        return
    ckpts = _complete_checkpoints(ckpt_dir)
    for _step, d in ckpts[:-keep_last_k]:
        shutil.rmtree(d, ignore_errors=True)
