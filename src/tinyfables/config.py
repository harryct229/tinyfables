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
    paraphrase_bank: str | None = None
    paraphrase_coverage: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.paraphrase_coverage <= 1.0:
            raise ValueError("paraphrase_coverage must be in [0, 1]")
        if self.paraphrase_coverage > 0.0 and self.paraphrase_bank is None:
            raise ValueError("paraphrase_coverage > 0 requires paraphrase_bank")


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
    amp: bool = False
    ckpt_dir: str | None = None
    ckpt_every: int = 0
    keep_last_k: int = 2
    log_every: int = 0
    trackio_project: str | None = None
    trackio_space_id: str | None = None
    run_name: str | None = None
    ckpt_hub_repo: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    tokenizer_dir: str
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    n_ctx: int = 1024
    batch_size: int = 16
    window: int = 1024
    warmup_steps: int = 5
    measure_steps: int = 20
    token_budget: int = 250_000_000
    target_tokens_per_sec: float = 0.0
    amp: bool = True
    device: str = "auto"
    seed: int = 0


@dataclass(frozen=True)
class EvalConfig:
    checkpoint: str
    tokenizer_dir: str
    source: SourceSpec
    paraphrase_bank: str | None = None
    n_ctx: int = 1024
    n_perplexity_rows: int = 500
    n_generations: int = 50
    max_new_tokens: int = 320
    min_new_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 50
    moral_threshold: float = 0.3
    seed: int = 0
    device: str = "auto"


@dataclass(frozen=True)
class PairgenConfig:
    checkpoint: str
    tokenizer_dir: str
    source: SourceSpec
    n_ctx: int = 1024
    n_pairs: int = 2000
    max_new_tokens: int = 320
    min_new_tokens: int = 80
    temperature: float = 0.9
    top_k: int = 50
    seed: int = 0
    device: str = "auto"


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
