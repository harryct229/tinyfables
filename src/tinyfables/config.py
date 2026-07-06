"""Frozen dataclass configs loaded from YAML. Unknown keys are errors so a
typo in an experiment config fails loudly instead of silently using defaults."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass(frozen=True)
class SourceSpec:
    jsonl_path: str | None = None
    hf_dataset: str | None = None
    hf_split: str = "train"
    max_rows: int | None = None

    def __post_init__(self) -> None:
        if (self.jsonl_path is None) == (self.hf_dataset is None):
            raise ValueError("set exactly one of source.jsonl_path or source.hf_dataset")


@dataclass(frozen=True)
class TokenizerConfig:
    source: SourceSpec
    vocab_size: int = 8192
    seed: int = 0
    compare_gpt2: bool = False


@dataclass(frozen=True)
class PrepConfig:
    source: SourceSpec
    tokenizer_dir: str
    window: int = 1024
    seed: int = 0


@dataclass(frozen=True)
class PretrainConfig:
    prep_dir: str
    tokenizer_dir: str
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    n_ctx: int = 1024
    batch_size: int = 16
    steps: int = 1000
    lr: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0
    resume_from: str | None = None
    device: str = "cpu"
    seed: int = 0


def _build(cls: type[T], data: Any) -> T:
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = dict(data)
    if "source" in kwargs:
        kwargs["source"] = _build(SourceSpec, kwargs["source"])
    return cls(**kwargs)


def load_config(path: str | Path, cls: type[T]) -> T:
    with open(path) as f:
        data = yaml.safe_load(f)
    return _build(cls, data or {})
